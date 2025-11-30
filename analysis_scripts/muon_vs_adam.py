# megatron/core/optimizer/muon.py
from typing import Tuple, Dict
import torch
import math
import torch.distributed as dist

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import math
from typing import Tuple, Dict

from torch.profiler import profile, record_function, ProfilerActivity
import time


# copy from https://github.com/KellerJordan/Muon/tree/master
# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

def normalize_range(range: Tuple[int, int], start):
    return (range[0] - start, range[1] - start)

class MuonDistMeta:

    # which buffer and bucket param belongs to
    buffer_idx: int = 0
    bucket_idx: int = 0
    # param shape after tp
    shape: torch.Size = None
    # param location in global buffer
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    # param location in global buffer (current dp slice)
    local_range: Tuple[int, int] = None

    def __init__(self, buffer_idx: int, bucket_idx: int, shape: torch.Size, global_range: Tuple[int, int], tp_split_dim: int):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self,  local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (start, end) if start < end else (local_buffer_range[0], local_buffer_range[0])

# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr

# copy from https://github.com/KellerJordan/Muon/tree/master and support distributed solution
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, param_groups, lr=2e-2, weight_decay=0.1,
                 matched_adamw_rms=0.2, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_betas=(0.95, 0.95), adamw_eps=1e-8):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,)

        super().__init__(param_groups, defaults)
        self.distributed_mode = False


    def enable_distributed_mode(self, global_buffer_sizes, dist_group, tp_group,
                                dist_metas: Dict[torch.nn.Parameter, MuonDistMeta]):
        """
        enable distributed mode
        Args:
            global_buffer_size: global buffer size
            dist group: optimizer sharding group
            tp group: param tp group
            dist metas: dist metas for all param
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        # The outer loop is for different parameter groups (e.g., weights vs. biases)
        for global_bucket_sizes in global_buffer_sizes: # <--- rename `global_bucket_sizes`
            local_bucket_sizes = []
            local_bucket_ranges = []

            # The inner loop is for the different buckets within a single group
            for (global_bucket_size, bucket_offset) in global_bucket_sizes:
                # calculate the local range for THIS specific bucket
                assert global_bucket_size % world_size == 0
                local_bucket_size = global_bucket_size // world_size
                # Renaming here makes the logic so much clearer
                local_bucket_start = local_bucket_size * rank + bucket_offset
                local_buffer_range = (local_bucket_start, local_bucket_start + local_bucket_size)
                local_bucket_sizes.append(local_bucket_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][dist_meta.bucket_idx]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def step(self):

        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # update muon momentum first
        # `self.param_groups` is already sharded
        for group in self.param_groups:

            if not group.get("use_muon", False):
                continue

            momentum = group['momentum']
            params = group["params"]

            for p in params:

                g = p.grad
                assert g is not None
                # 1-dim grad for distributed mode
                assert self.distributed_mode or g.dim() == 2

                # prepare muon buffer in state
                state = self.state[p]
                if not "muon_buffer" in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)

                # save to ns input
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                ns_inputs[p] = g.bfloat16()

        # rewrite ns_inputs if distributed
        """
        the four-step "acrobatic" journey of the ns_inputs data:

        1.  **DP `all_gather`**: (ZeRO) Gather all the sharded pieces from your data-parallel "column" to re-create your **full TP slice**.
        2.  **TP `all_gather`**: Gather all the TP slices from your tensor-parallel "row" to re-create the **full, 100% complete matrix**.
        3.  *(...Run the math on the full matrix...)*
        4.  **TP `shard`**: Shard the full `update` matrix back down to your **local TP slice**.
        5.  **DP `shard`**: (ZeRO) Shard that TP slice *again* back down to the **local DP/ZeRO slice** that you're responsible for.

        """
        if self.distributed_mode:

            # initialize buffers 
            # hanged the variable nnames to `local_bucket_size` and `global_bucket_size` for clarity
            ns_input_local_buffers = [
                [ torch.empty((local_bucket_size), device=device, dtype=dtype)
                    for local_bucket_size in local_bucket_sizes ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [ torch.empty((global_bucket_size), device=device, dtype=dtype)
                    for (global_bucket_size, bucket_offset) in global_bucket_sizes ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            # looping through all params in local rank, ok.
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                # ceate a reference to `ns_input_local_buffers`
                # the update is in local rank, so we only need one `for` loop
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_range = normalize_range(dist_meta.local_range, local_buffer_range[0]) # local_range in global_range
                # copy data into this `ns_input_local_buffer` memory
                # because dist.all_gather requires a single, physically contiguous block of memory to work efficiently.
                ns_input_local_buffer[local_range[0]:local_range[1]].copy_(ns_input.view(-1))

            # all gather buffers: one bucket at a time. -- the "shipping" phase
            for ns_input_global_buffer, ns_input_local_buffer in zip(ns_input_global_buffers, ns_input_local_buffers):
                for ns_input_global_bucket, ns_input_local_bucket in zip(ns_input_global_buffer, ns_input_local_buffer):
                    dist.all_gather_into_tensor(ns_input_global_bucket, ns_input_local_bucket, group=self.dist_group)

            # overwrite ns input with the `all_gather`-ed `ns_inputs` -- the "unpacking" phase
            # this is the "opposite" of filling ns input data to local buffer
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][dist_meta.bucket_idx][1]
                global_range = normalize_range(dist_meta.global_range, offset)

                #ns_inputs[p] = ns_input_global_buffer[global_range[0]:global_range[1]].view(-1) 
                ## bug fix 👆🏻-- overwrite ns input with the `all_gather`-ed `ns_inputs` -- the "unpacking" phase
                #ns_inputs[p] = ns_input_global_buffer[global_range[0]:global_range[1]].view(-1)
                # Unpack the 1D slice of data
                unpacked_data = ns_input_global_buffer[global_range[0]:global_range[1]]

                # THIS IS THE FIX: Reshape it to its correct 2D shape, not view(-1)
                ns_inputs[p] = unpacked_data.view(dist_meta.shape)    

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # update muon momentum first
        for group in self.param_groups:

            if not group.get('use_muon', False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"] # <-- add this

            for p in params:

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    ns_input_shards = [ torch.empty_like(ns_input) for _ in range(tp_world_size) ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                # this is effectivly "shadding" the newtonschulz-processed update, 
                # and keep only your assigned piece, discarding the rest
                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                # only local dp buffer part
                if self.distributed_mode:
                    # local range in global range
                    # unpacking the tp sharded update to dp sharded update
                    local_range = normalize_range(dist_meta.local_range, dist_meta.global_range[0])
                    update = update.reshape(-1)[local_range[0]:local_range[1]]

                # apply weight decay
                p.data.mul_(1 - lr*weight_decay)

                #  adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            # init step
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step = group['step']
            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            for p in params:

                g = p.grad
                assert g is not None
                state = self.state[p]

                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)


##--------------- tests/unit_tests/test_optimizer_muon.py -----------------
import os

import torch
import torch.distributed as dist

#from megatron.core.optimizer.muon import Muon, MuonDistMeta, normalize_range

def is_rank_0():
    return torch.distributed.get_rank() == 0

def print_rank_0(*args):
    if is_rank_0():
        print(*args)

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def gen_param_and_grads():

    # reset manual seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    dtype = torch.float32

    # gen params
    params = [ torch.randn(shape, device=device, dtype=dtype) for shape in [
            (4096, 4096), (1024, 324), (456, 1024), (676, 876), (128, 128), ] ]

    # gen grads [ [ grad-list ] * step ]
    grads = [ [ torch.randn_like(param) for param in params ] for _ in range(10) ]

    return params, grads

def distribute_params(params, grads, tp_dims, dist_group, tp_group):
    """ 将 param 进行 dist & tp shard, 仅保留自己的一部分 """

    params = params.copy()
    grads = [ step_grads.copy() for step_grads in grads ]

    # tp dist
    tp_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)
    for i, param in enumerate(params):
        tp_dim = tp_dims[i]
        if tp_dim == -1:
            continue
        # Shard the parameter tensor along the `tp_dim` dimension.
        assert param.shape[tp_dim] % tp_size == 0
        local_range_start = param.shape[tp_dim] // tp_size * tp_rank
        # range of the shard based on the rank of the current GOU in the given `tp_group``
        local_range_end = param.shape[tp_dim] // tp_size * (tp_rank + 1)
        # each GPU gets `[local_range_start:local_range_end, :] ` rows or `[:, local_range_start:local_range_end]` columns
        params[i] = param[local_range_start:local_range_end, :] if tp_dim == 0 else \
                    param[:, local_range_start:local_range_end].contiguous()
        # same logic applies to sharding the gradients for the current layer(param)
        for step_grads in grads:
            step_grads[i] = step_grads[i][local_range_start:local_range_end, :] if tp_dim == 0 else \
                            step_grads[i][:, local_range_start:local_range_end].contiguous()

    # distributed
    world_size = dist.get_world_size(dist_group)
    rank = dist.get_rank(dist_group)

    # global as the given DP group
    # "global" here means "global to the TP group's worth of parameters."
    global_buffer_size = sum(param.numel() for param in params)
    local_buffer_size = cdiv(global_buffer_size, world_size)
    # deciding the shard range for this rank
    local_buffer_range = (local_buffer_size * rank, local_buffer_size * (rank + 1))
    # padded global_buffer_size
    global_buffer_size = local_buffer_size * world_size # fix global buffer size

    numel_acc = 0
    dist_params = []
    dist_grads = [[] for _ in grads]
    dist_metas = {}
    for i, param in enumerate(params):

        # gen meta
        # align global buffer index(range) with local buffer index(range)
        # see handwritten diagram for more details
        numel = param.numel()
        dist_meta = MuonDistMeta(0, 0, param.shape, (numel_acc, numel_acc + numel), tp_dims[i])
        dist_meta.set_local_buffer_range(local_buffer_range)
        numel_acc += numel

        # skip if no element in this shard
        if dist_meta.local_range[0] == dist_meta.local_range[1]:
            continue

        # gen param

        # Convert the ABSOLUTE slice range (from the global virtual buffer)
        # into a RELATIVE slice range (local to just this one parameter).
        local_range = normalize_range(dist_meta.local_range, dist_meta.global_range[0]) 
        
        # 1. Flatten the 2D parameter tensor into a 1D vector.
        # 2. Use the relative range to slice out the piece this GPU is responsible for storing.
        dist_param = param.view(-1)[local_range[0]:local_range[1]]
        dist_params.append(dist_param)
        dist_metas[dist_param] = dist_meta

        # gen grad
        # same logoc as the `gen param` scetion
        for step, step_grads in enumerate(grads):
            dist_grad = step_grads[i].view(-1)[local_range[0]:local_range[1]]
            dist_grads[step].append(dist_grad)

    return dist_params, dist_grads, global_buffer_size, dist_metas




def test_muon_dist(dp_size, tp_size):

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert dp_size * tp_size == world_size

    # init dist group
    for i in range(tp_size):
        # decide the tp group based on grod of size `tp_size`
        ranks = range(i, world_size, tp_size)
        group = dist.new_group(ranks)
        # each rank finds its groups
        if rank in ranks:
            # groups are passed as instructions
            dist_group = group
    # init tp group
    for i in range(dp_size):
        ranks = range(i * tp_size, (i + 1) * tp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    print_rank_0("process group initialized")

    params_ref, grads_ref = gen_param_and_grads()
    params_test, grads_test = gen_param_and_grads()
    tp_dims = [0, 1, -1, 1, 0]
    #tp_dims = [1, 0, -1, 0, 1]

    # global_buffer_size is the padded buffer size of the dp group where the current rank belongs to
    params_test, grads_test, global_buffer_size, dist_metas \
         = distribute_params(params_test, grads_test, tp_dims, dist_group, tp_group)

    muon_args = {
        "use_muon": True,
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "weight_decay": 0.1,
    }

    # gen params
    ref_param_groups = [{
        "params": params_ref,
        **muon_args
    }]
    test_param_groups = [{
        "params": params_test,
        **muon_args
    }]

    ref_muon  = Muon(ref_param_groups)
    test_muon = Muon(test_param_groups)
    test_muon.enable_distributed_mode([[(global_buffer_size, 0)]], dist_group, tp_group, dist_metas)

    for step in range(10):

        # add grad
        for i, grad in enumerate(grads_ref[step]):
            params_ref[i].grad = grad.clone()
        for i, grad in enumerate(grads_test[step]):
            params_test[i].grad = grad.clone()
        # step
        ref_muon.step()
        test_muon.step()

        # distribute ref params
        dist_ref_params, _, _, _ = distribute_params(params_ref, [], tp_dims, dist_group, tp_group)
        # verify
        for i, params_x2 in enumerate(zip(dist_ref_params, params_test)):
            assert (params_x2[0] == params_x2[1]).all(), f"rank {rank} param {i} verify failed"
        print_rank_0(f" - step {step} verify passed")

    print_rank_0(f"dist dp = {dp_size} tp = {tp_size} test passed")


from torch.profiler import profile, record_function, ProfilerActivity
#-------------------------- benchmarks/added for benchmark_muon_vs_adam.py -----------------

def gen_param_and_grads():
    # reset manual seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    # Using float32 as input (Muon will cast internally, AdamW uses as is)
    dtype = torch.float32 

    # gen params (LLM Sized)
    params = [ torch.randn(shape, device=device, dtype=dtype) for shape in [
            (4096, 4096), (1024, 324), (456, 1024), (676, 876), (128, 128), ] ]

    # gen grads [ [ grad-list ] * step ]
    grads = [ [ torch.randn_like(param) for param in params ] for _ in range(5) ] # 5 steps is enough

    return params, grads

# backward + optimizer step only

def benchmark_adamw(rank, world_size, steps=5):
    print_rank_0(f"🥊 Starting Round 1: AdamW (Standard DDP Simulation)...")
    params, grads_list = gen_param_and_grads()
    
    # Standard AdamW setup
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    # Warmup
    for p, g in zip(params, grads_list[0]):
        p.grad = g
        # Simulate DDP: All-Reduce gradients
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
    optimizer.step()
    optimizer.zero_grad()

    # Profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("AdamW_Battle"):
            for step in range(steps):
                # 1. Simulate Backward Pass (Gradient Available)
                for i, p in enumerate(params):
                    p.grad = grads_list[step][i]
                
                # 2. Simulate DDP Communication (The cost of AdamW comms)
                with record_function("AdamW_Comm_AllReduce"):
                    for p in params:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
                
                # 3. Optimizer Step (Should be fast/local)
                with record_function("AdamW_Step"):
                    optimizer.step()
                    optimizer.zero_grad()
    
    prof.export_chrome_trace(f"trace_adamw_rank{rank}.json")
    print_rank_0("✅ AdamW Round Finished.")

def setup_process_groups(dp_size, tp_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    for i in range(tp_size):
        ranks = range(i, world_size, tp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            dist_group = group
    
    for i in range(dp_size):
        ranks = range(i * tp_size, (i + 1) * tp_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    
    return dist_group, tp_group

def benchmark_muon(rank, world_size, dp_size, tp_size, steps=5):
    print_rank_0(f"🥊 Starting Round 2: Muon (DP={dp_size}, TP={tp_size})...")
    
    # Setup (same as the OG test, but separate)
    dist_group, tp_group = setup_process_groups(dp_size, tp_size)
    params, grads_list = gen_param_and_grads()
    tp_dims = [0, 1, -1, 1, 0]
    
    params, grads_list, global_buffer_size, dist_metas = \
        distribute_params(params, grads_list, tp_dims, dist_group, tp_group)
    
    muon_args = {
        "use_muon": True,
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "weight_decay": 0.1,
    }
    
    optimizer = Muon([{"params": params, **muon_args}])
    optimizer.enable_distributed_mode(
        [[(global_buffer_size, 0)]], dist_group, tp_group, dist_metas
    )
    
    # Warmup
    for p, g in zip(params, grads_list[0]):
        p.grad = g
    optimizer.step()
    
    # Profile ONLY the optimizer steps (like AdamW)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True) as prof:
        with record_function("Muon_Battle"):
            for step in range(steps):
                # 1. Attach gradients (simulating backward pass)
                with record_function("Muon_Attach_Grads"):
                    for i, p in enumerate(params):
                        p.grad = grads_list[step][i]
                
                # 2. Optimizer Step (THIS is what we want to measure)
                with record_function("Muon_Step"):
                    optimizer.step()
    
    prof.export_chrome_trace(f"trace_muon_dp{dp_size}_tp{tp_size}_rank{rank}.json")
    print_rank_0("✅ Muon Round Finished.")


# -- full setup --


def simulate_fwd_bwd(size=2048, iterations=20):
    """Simulate model forward + backward compute
    Adjust size and iterations to match your real model's compute time
    """
    dummy = torch.randn(size, size, device='cuda')
    for _ in range(iterations):
        dummy = torch.matmul(dummy, dummy)
    torch.cuda.synchronize()

def benchmark_adamw_full_step(rank, world_size, steps=5):
    print_rank_0(f"🥊 AdamW with Forward-Backward Simulation...")
    params, grads_list = gen_param_and_grads()
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    # Warmup
    for p, g in zip(params, grads_list[0]):
        p.grad = g
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
    optimizer.step()
    optimizer.zero_grad()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("AdamW_Full_Training_Step"):
            for step in range(steps):
                # 1. Forward + Backward
                with record_function("FWD_BWD"):
                    simulate_fwd_bwd()
                
                # 2. Gradients available
                with record_function("Attach_Grads"):
                    for i, p in enumerate(params):
                        p.grad = grads_list[step][i]
                
                # 3. DDP Gradient sync
                with record_function("AdamW_AllReduce"):
                    for p in params:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
                
                # 4. Optimizer step
                with record_function("AdamW_Step"):
                    optimizer.step()
                    optimizer.zero_grad()
    
    prof.export_chrome_trace(f"trace_adamw_FULLSTEP_rank{rank}.json")
    print_rank_0("✅ AdamW Full Step Finished.")

def benchmark_muon_full_step(rank, world_size, dp_size, tp_size, steps=5):
    print_rank_0(f"🥊 Muon with Forward-Backward Simulation...")
    
    dist_group, tp_group = setup_process_groups(dp_size, tp_size)
    params, grads_list = gen_param_and_grads()
    tp_dims = [0, 1, -1, 1, 0]
    
    params, grads_list, global_buffer_size, dist_metas = \
        distribute_params(params, grads_list, tp_dims, dist_group, tp_group)
    
    optimizer = Muon([{"params": params, "use_muon": True, "lr": 0.1, 
                       "momentum": 0.9, "nesterov": True, "ns_steps": 5, 
                       "weight_decay": 0.1}])
    optimizer.enable_distributed_mode(
        [[(global_buffer_size, 0)]], dist_group, tp_group, dist_metas
    )
    
    # Warmup
    for p, g in zip(params, grads_list[0]):
        p.grad = g
    optimizer.step()
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("Muon_Full_Training_Step"):
            for step in range(steps):
                # 1. Forward + Backward
                with record_function("FWD_BWD"):
                    simulate_fwd_bwd()
                
                # 2. Gradients available
                with record_function("Attach_Grads"):
                    for i, p in enumerate(params):
                        p.grad = grads_list[step][i]
                
                # 3. Optimizer step (includes Muon's communication)
                with record_function("Muon_Step"):
                    optimizer.step()
    
    prof.export_chrome_trace(f"trace_muon_FULLSTEP_dp{dp_size}_tp{tp_size}_rank{rank}.json")
    print_rank_0("✅ Muon Full Step Finished.")

def run_process(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Test 1: Optimizer-only (what you already have)
    benchmark_adamw(rank, world_size, steps=5)
    benchmark_muon(rank, world_size, dp_size=2, tp_size=2, steps=5)
    
    # Test 2: Full training step (to verify 1-3% claim)
    benchmark_adamw_full_step(rank, world_size, steps=5)
    benchmark_muon_full_step(rank, world_size, dp_size=2, tp_size=2, steps=5)
    
    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 4
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    torch.multiprocessing.spawn(run_process, args=(world_size,), nprocs=world_size, join=True)

    print("✅ All tests passed!") 