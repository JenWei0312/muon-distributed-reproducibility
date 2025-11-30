import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent / 'traces'/'comparison'

def analyze_trace(filename):
    # Make filename relative to script location
    filepath = SCRIPT_DIR / filename
    
    with open(filepath) as f:
        trace = json.load(f)

    
    timings = {
        'nccl_all_reduce': [],
        'nccl_all_gather': [],
        'Muon_Step': [],
        'AdamW_Step': [],
        'Muon_Full_Training_Step': [],
        'AdamW_Full_Training_Step': [],
        'FWD_BWD': [],
        'AdamW_AllReduce': [],
        'Attach_Grads': [],
    }
    
    # Let's see ALL event names first
    all_names = set()
    
    for event in trace['traceEvents']:
        if 'name' in event:
            all_names.add(event['name'])
        
        if 'dur' not in event:
            continue
        name = event.get('name', '')
        dur = event['dur']
        
        # Match any key that contains the name
        for key in timings.keys():
            if key in name:
                timings[key].append(dur)
                break
    
    print(f"\n=== {filename} ===")
    print(f"All unique event names found: {len(all_names)}")
    print("Sample names:", list(all_names)[:20])
    
    print("\n--- Timings ---")
    for key, values in timings.items():
        if values:
            total = sum(values)
            print(f"{key}: {len(values)} calls, total={total/1000:.2f}ms, "
                  f"avg={total/len(values)/1000:.2f}ms")
    
    # Calculate percentages
    if timings['Muon_Step']:
        muon_opt = sum(timings['Muon_Step'])
        muon_total = sum(timings['Muon_Full_Training_Step']) if timings['Muon_Full_Training_Step'] else sum(timings['FWD_BWD']) + muon_opt
        print(f"\n📊 Muon Optimizer: {muon_opt/1000:.2f}ms = {(muon_opt/muon_total)*100:.1f}% of total")
    
    if timings['AdamW_Step']:
        adam_opt = sum(timings['AdamW_Step'])
        adam_total = sum(timings['AdamW_Full_Training_Step']) if timings['AdamW_Full_Training_Step'] else sum(timings['FWD_BWD'])/2 + adam_opt
        print(f"📊 AdamW Optimizer: {adam_opt/1000:.2f}ms = {(adam_opt/adam_total)*100:.1f}% of total\n")


def detailed_comm_analysis(filename):
    # Make filename relative to script location
    filepath = SCRIPT_DIR / filename
    with open(filepath) as f:
        trace = json.load(f)
    
    comm_ops = {
        'all_reduce': [],
        'all_gather': [],
        'reduce_scatter': [],
        'broadcast': [],
    }
    
    for event in trace['traceEvents']:
        if 'dur' not in event:
            continue
        name = event.get('name', '').lower()
        dur = event['dur']
        
        if 'all_reduce' in name or 'allreduce' in name:
            comm_ops['all_reduce'].append(dur)
        elif 'all_gather' in name or 'allgather' in name:
            comm_ops['all_gather'].append(dur)
        elif 'reduce_scatter' in name or 'reducescatter' in name:
            comm_ops['reduce_scatter'].append(dur)
        elif 'broadcast' in name:
            comm_ops['broadcast'].append(dur)
    
    print(f"\n=== Communication Breakdown: {filename} ===")
    total_comm = 0
    for op, times in comm_ops.items():
        if times:
            op_total = sum(times)
            total_comm += op_total
            print(f"{op}: {len(times)} calls, {op_total/1000:.2f}ms total, {op_total/len(times)/1000:.2f}ms avg")
    
    print(f"\nTotal Communication: {total_comm/1000:.2f}ms")
    return total_comm



def main():
    muon_trace_file = 'trace_muon_FULLSTEP_dp2_tp2_rank0.json'
    adam_trace_file = 'trace_adamw_FULLSTEP_rank0.json'

    analyze_trace(muon_trace_file)
    analyze_trace(adam_trace_file)

    muon_comm = detailed_comm_analysis(muon_trace_file)
    adam_comm = detailed_comm_analysis(adam_trace_file)


    print(f"\n📊 Communication Comparison:")
    print(f"Muon comm: {muon_comm/1000:.2f}ms")
    print(f"AdamW comm: {adam_comm/1000:.2f}ms")
    print(f"Ratio: {muon_comm/adam_comm:.2f}x")

if __name__ == "__main__":
    main()