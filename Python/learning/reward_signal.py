import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class RewardChannel:
    """
    Explicit reward channel for external feedback (human or system).
    Logs rewards to an append-only JSONL file for processing.
    """
    
    def __init__(self, log_path: str = "data/rewards.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def submit_reward(self, 
                      reward: float, 
                      source: str,
                      context_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Submit a numeric reward signal.
        
        Args:
            reward: The float value (e.g. 1.0, -1.0)
            source: Identifier for who gave the reward (e.g. "player_menu", "quest_system")
            context_id: Ideally the bandit context ID this applies to (if known)
            metadata: Any extra info
            
        Returns:
            True if logged successfully.
        """
        
        entry = {
            "timestamp": time.time(),
            "reward": reward,
            "source": source,
            "context_id": context_id,
            "metadata": metadata or {}
        }
        
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return True
        except IOError as e:
            print(f"Failed to log reward: {e}")
            return False

    def read_pending_rewards(self) -> list:
        """
        Read rewards, potentially to consume them for batch updates.
        (Implementation depends on how we want to consumer them)
        """
        rewards = []
        if not self.log_path.exists():
            return []
            
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    rewards.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rewards
