"""
RewardChannel: Explicit reward ingestion with decision_id binding.

All rewards MUST reference a valid decision_id to enable:
- Correct reward attribution
- Trace-based credit assignment
- Offline evaluation

Unknown decision_ids are rejected or quarantined.
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from learning.decision_record import get_decision_logger

logger = logging.getLogger("learning.reward")

# Environment flags
STRICT_REWARDS = os.environ.get("STRICT_REWARDS", "1") == "1"


class RewardValidationError(Exception):
    """Raised when reward validation fails."""
    pass


class RewardChannel:
    """
    Explicit reward channel for external feedback (human or system).
    
    Key change from v1.0: Requires decision_id and validates against
    the decision log before accepting rewards.
    """
    
    VERSION = "2.0"
    
    def __init__(
        self,
        log_path: str = "data/learning/rewards.jsonl",
        quarantine_path: str = "data/learning/rewards_quarantine.jsonl"
    ):
        self.log_path = Path(log_path)
        self.quarantine_path = Path(quarantine_path)
        
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get decision logger for validation
        self._decision_logger = get_decision_logger()
    
    def submit_reward(
        self,
        decision_id: str,
        reward: float,
        source: str,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Submit a numeric reward signal.
        
        Args:
            decision_id: The decision this reward applies to (REQUIRED)
            reward: The float value (0-1 expected, will be clamped)
            source: Identifier for who gave the reward 
                    (e.g. "player", "system", "moderation", "test")
            notes: Optional human-readable notes
            metadata: Any extra info
            
        Returns:
            (success, error_message)
            
        Raises:
            RewardValidationError: If STRICT_REWARDS and validation fails
        """
        if not decision_id:
            error = "decision_id is required"
            if STRICT_REWARDS:
                raise RewardValidationError(error)
            return False, error
        
        # Validate decision exists
        if not self._decision_logger.exists(decision_id):
            error = f"Unknown decision_id: {decision_id}"
            self._quarantine_reward(decision_id, reward, source, notes, metadata, error)
            
            if STRICT_REWARDS:
                raise RewardValidationError(error)
            
            logger.warning(f"[REWARD] Quarantined: {error}")
            return False, error
        
        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        entry = {
            "timestamp_ms": int(time.time() * 1000),
            "decision_id": decision_id,
            "reward": reward,
            "source": source,
            "notes": notes,
            "metadata": metadata or {},
            "validated": True
        }
        
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            logger.debug(
                f"[REWARD] decision={decision_id[:8]} reward={reward:.2f} source={source}"
            )
            return True, None
            
        except IOError as e:
            error = f"Failed to log reward: {e}"
            logger.error(error)
            return False, error
    
    def _quarantine_reward(
        self,
        decision_id: str,
        reward: float,
        source: str,
        notes: Optional[str],
        metadata: Optional[Dict[str, Any]],
        reason: str
    ) -> None:
        """Log invalid rewards to quarantine for later inspection."""
        entry = {
            "timestamp_ms": int(time.time() * 1000),
            "decision_id": decision_id,
            "reward": reward,
            "source": source,
            "notes": notes,
            "metadata": metadata or {},
            "quarantine_reason": reason
        }
        
        try:
            with open(self.quarantine_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass  # Best effort
    
    def get_rewards_for_decision(self, decision_id: str) -> List[Dict[str, Any]]:
        """Get all rewards logged for a specific decision."""
        rewards = []
        if not self.log_path.exists():
            return rewards
        
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("decision_id") == decision_id:
                            rewards.append(entry)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        
        return rewards
    
    def read_pending_rewards(self, since_ms: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read rewards for processing.
        
        Args:
            since_ms: Only return rewards after this timestamp
            
        Returns:
            List of reward entries
        """
        rewards = []
        if not self.log_path.exists():
            return rewards
        
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if since_ms is None or entry.get("timestamp_ms", 0) > since_ms:
                            rewards.append(entry)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        
        return rewards
    
    def get_quarantined(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get quarantined rewards for inspection."""
        quarantined = []
        if not self.quarantine_path.exists():
            return quarantined
        
        try:
            with open(self.quarantine_path, "r") as f:
                for line in f:
                    try:
                        quarantined.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        
        return quarantined[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward channel statistics."""
        rewards = self.read_pending_rewards()
        quarantined = self.get_quarantined()
        
        sources = {}
        for r in rewards:
            src = r.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        
        return {
            "version": self.VERSION,
            "total_rewards": len(rewards),
            "total_quarantined": len(quarantined),
            "by_source": sources,
            "strict_mode": STRICT_REWARDS
        }


# Global reward channel (singleton)
_reward_channel: Optional[RewardChannel] = None

def get_reward_channel() -> RewardChannel:
    """Get or create global reward channel."""
    global _reward_channel
    if _reward_channel is None:
        _reward_channel = RewardChannel()
    return _reward_channel
