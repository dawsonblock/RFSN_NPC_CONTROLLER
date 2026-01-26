"""
Offline Log Evaluator: Computes learning metrics from JSONL logs.

Produces actionable reports on:
- Action distribution per abstract state
- Reward mean/variance per action per state
- Exploration ratio over time
- Safety veto frequency and top reasons
- Regret proxy (if scores available)
- Policy drift across versions

Usage:
    python -m tools.evaluate_logs --logdir data/learning --out report.json
"""
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional
from datetime import datetime


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from JSONL file."""
    records = []
    if not path.exists():
        return records
    
    with open(path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records


def join_decisions_rewards(
    decisions: List[Dict], 
    rewards: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Join decisions with their rewards by decision_id.
    
    Returns:
        {decision_id: {decision: {...}, rewards: [...]}}
    """
    joined: Dict[str, Dict[str, Any]] = {}
    
    for d in decisions:
        did = d.get("decision_id")
        if did:
            joined[did] = {"decision": d, "rewards": []}
    
    for r in rewards:
        did = r.get("decision_id")
        if did and did in joined:
            joined[did]["rewards"].append(r)
    
    return joined


def compute_action_distribution(decisions: List[Dict]) -> Dict[str, int]:
    """Compute action distribution across all decisions."""
    return Counter(d.get("chosen_action_id", "unknown") for d in decisions)


def compute_state_action_matrix(
    decisions: List[Dict]
) -> Dict[str, Dict[str, int]]:
    """
    Compute action distribution per abstract state.
    
    Returns:
        {abstract_state_key: {action: count}}
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for d in decisions:
        state = d.get("abstract_state_key", "unknown")
        action = d.get("chosen_action_id", "unknown")
        matrix[state][action] += 1
    
    return {k: dict(v) for k, v in matrix.items()}


def compute_reward_stats(
    joined: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute reward statistics per action per state.
    
    Returns:
        {state: {action: {mean, std, count}}}
    """
    # Collect rewards per (state, action)
    reward_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for data in joined.values():
        d = data["decision"]
        state = d.get("abstract_state_key", "unknown")
        action = d.get("chosen_action_id", "unknown")
        
        for r in data["rewards"]:
            reward_data[state][action].append(r.get("reward", 0.0))
    
    # Compute stats
    stats: Dict[str, Dict[str, Any]] = {}
    for state, actions in reward_data.items():
        stats[state] = {}
        for action, rewards in actions.items():
            if rewards:
                stats[state][action] = {
                    "count": len(rewards),
                    "mean": round(statistics.mean(rewards), 4),
                    "std": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
                    "min": round(min(rewards), 4),
                    "max": round(max(rewards), 4)
                }
    
    return stats


def compute_exploration_ratio(decisions: List[Dict]) -> Dict[str, float]:
    """Compute exploration ratio over time (bucketed by hour)."""
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"explore": 0, "exploit": 0})
    
    for d in decisions:
        ts = d.get("timestamp_ms", 0)
        hour = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:00")
        
        if d.get("exploration"):
            buckets[hour]["explore"] += 1
        else:
            buckets[hour]["exploit"] += 1
    
    ratios = {}
    for hour, counts in sorted(buckets.items()):
        total = counts["explore"] + counts["exploit"]
        ratios[hour] = round(counts["explore"] / max(1, total), 4)
    
    return ratios


def compute_safety_stats(decisions: List[Dict]) -> Dict[str, Any]:
    """Compute safety veto statistics."""
    total = len(decisions)
    vetoed = [d for d in decisions if d.get("safety_vetoed")]
    
    # Count veto reasons
    reasons = Counter(d.get("veto_reason", "unknown") for d in vetoed)
    
    return {
        "total_decisions": total,
        "total_vetoed": len(vetoed),
        "veto_rate": round(len(vetoed) / max(1, total), 4),
        "top_reasons": dict(reasons.most_common(10))
    }


def compute_regret_proxy(joined: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute regret proxy: difference between max score and chosen score.
    
    Only works if decisions include scores.
    """
    regrets = []
    
    for data in joined.values():
        d = data["decision"]
        scores = d.get("scores")
        
        if not scores:
            continue
        
        chosen = d.get("chosen_action_id")
        chosen_score = scores.get(chosen, 0)
        max_score = max(scores.values()) if scores else 0
        
        regret = max_score - chosen_score
        regrets.append(regret)
    
    if not regrets:
        return {"available": False}
    
    return {
        "available": True,
        "mean_regret": round(statistics.mean(regrets), 4),
        "total_regret": round(sum(regrets), 4),
        "count": len(regrets)
    }


def compute_policy_drift(decisions: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compute action distribution per policy version."""
    versions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for d in decisions:
        version = d.get("policy_version", "unknown")
        action = d.get("chosen_action_id", "unknown")
        versions[version][action] += 1
    
    return {v: dict(actions) for v, actions in versions.items()}


def generate_report(logdir: Path) -> Dict[str, Any]:
    """Generate full evaluation report from logs."""
    decisions_path = logdir / "decisions.jsonl"
    rewards_path = logdir / "rewards.jsonl"
    
    decisions = load_jsonl(decisions_path)
    rewards = load_jsonl(rewards_path)
    
    if not decisions:
        return {"error": "No decisions found", "log_path": str(decisions_path)}
    
    joined = join_decisions_rewards(decisions, rewards)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "log_dir": str(logdir),
        "summary": {
            "total_decisions": len(decisions),
            "total_rewards": len(rewards),
            "joined_decisions": len([j for j in joined.values() if j["rewards"]])
        },
        "action_distribution": compute_action_distribution(decisions),
        "state_action_matrix": compute_state_action_matrix(decisions),
        "reward_stats": compute_reward_stats(joined),
        "exploration_ratio": compute_exploration_ratio(decisions),
        "safety_stats": compute_safety_stats(decisions),
        "regret_proxy": compute_regret_proxy(joined),
        "policy_drift": compute_policy_drift(decisions)
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Offline Log Evaluator")
    parser.add_argument(
        "--logdir", 
        type=str, 
        default="data/learning",
        help="Directory containing decisions.jsonl and rewards.jsonl"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="report.json",
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also output CSVs for key metrics"
    )
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    report = generate_report(logdir)
    
    # Save JSON report
    out_path = Path(args.out)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {out_path}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Decisions: {report['summary']['total_decisions']}")
    print(f"Rewards: {report['summary']['total_rewards']}")
    print(f"Joined: {report['summary']['joined_decisions']}")
    print(f"Safety Veto Rate: {report['safety_stats']['veto_rate']:.2%}")
    
    if report['regret_proxy'].get('available'):
        print(f"Mean Regret: {report['regret_proxy']['mean_regret']:.4f}")
    
    print("\nTop Actions:")
    for action, count in sorted(
        report['action_distribution'].items(), 
        key=lambda x: -x[1]
    )[:5]:
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()
