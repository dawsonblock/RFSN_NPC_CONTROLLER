import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def load_logs(log_dir: str):
    log_files = glob.glob(f"{log_dir}/*.jsonl")
    all_events = []
    print(f"Loading {len(log_files)} log files from {log_dir}...")
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    all_events.append(event)
                except json.JSONDecodeError:
                    continue
    return all_events

def analyze_decisions(events):
    """Analyze decision frequency and safety vetoes."""
    action_counts = Counter()
    safety_vetoes = 0
    total_decisions = 0
    
    for e in events:
        if e.get("type") == "ACTION_CHOSEN":
            total_decisions += 1
            data = e.get("data", {})
            action = data.get("npc_action", "unknown")
            action_counts[action] += 1
            
        elif e.get("type") == "SAFETY_EVENT":
            safety_vetoes += 1
            
    print("\n--- Decision Analysis ---")
    print(f"Total Decisions: {total_decisions}")
    print(f"Safety Vetoes: {safety_vetoes}")
    if total_decisions > 0:
        print(f"Veto Rate: {safety_vetoes/total_decisions:.2%}")
        
    print("\nAction Distribution:")
    for action, count in action_counts.most_common():
        print(f"  {action}: {count} ({count/total_decisions:.1%})")

def analyze_rewards(events):
    """Analyze reward trends."""
    rewards = []
    
    for e in events:
        if e.get("type") == "LEARNING_UPDATE":
            data = e.get("data", {})
            r = data.get("reward", 0)
            rewards.append(r)
            
    print("\n--- Reward Analysis ---")
    if rewards:
        print(f"Total Updates: {len(rewards)}")
        print(f"Mean Reward: {statistics.mean(rewards):.4f}")
        if len(rewards) > 1:
            print(f"Reward StdDev: {statistics.stdev(rewards):.4f}")
        print(f"Min/Max: {min(rewards):.4f} / {max(rewards):.4f}")
    else:
        print("No learning updates found.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RFSN Logs")
    parser.add_argument("--logs", type=str, default="Python/data/audit", help="Path to log directory")
    args = parser.parse_args()
    
    events = load_logs(args.logs)
    if not events:
        print("No events found.")
        return
        
    analyze_decisions(events)
    analyze_rewards(events)

if __name__ == "__main__":
    main()
