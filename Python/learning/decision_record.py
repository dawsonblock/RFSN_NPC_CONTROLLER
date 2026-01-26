"""
DecisionRecord: The canonical join key for the learning system.

Every turn produces a DecisionRecord with a unique decision_id.
Rewards, traces, and updates all reference this decision_id.
"""
from __future__ import annotations

import uuid
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger("learning.decision")


@dataclass
class DecisionRecord:
    """
    Immutable record of a policy decision.
    
    This is the join key for:
    - Reward attribution
    - Trace credit assignment
    - Offline evaluation
    - Policy rollback analysis
    """
    
    # Core identifiers
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Context identifiers
    npc_id: str = ""
    session_id: str = ""
    
    # State representation
    raw_state_hash: str = ""          # SHA256 of raw state dict
    abstract_state_key: str = ""       # Categorical abstraction
    abstraction_version: str = "1.0"   # Version for stability
    
    # Decision details
    candidate_actions: List[str] = field(default_factory=list)
    chosen_action_id: str = ""
    
    # Policy metadata
    policy_name: str = "policy_owner"
    policy_version: str = "1.0"
    
    # Safety and exploration
    safety_vetoed: bool = False
    veto_reason: Optional[str] = None
    exploration: bool = False
    
    # Optional scoring details (for counterfactual analysis)
    scores: Optional[Dict[str, float]] = None
    
    # Trace linkage
    prior_decision_id: Optional[str] = None  # Links to previous decision in session
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON logging."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        """Deserialize from dict."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DecisionRecord":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @staticmethod
    def compute_state_hash(state: Dict[str, Any]) -> str:
        """Compute stable hash of raw state for reproducibility."""
        # Sort keys for determinism
        canonical = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def get_trace_key(self) -> str:
        """Key for trace lookup: (npc, session, decision)."""
        return f"{self.npc_id}:{self.session_id}:{self.decision_id}"


@dataclass
class PolicyDecision:
    """
    Result of policy selection, wraps DecisionRecord with action metadata.
    """
    record: DecisionRecord
    action: Any  # The actual action object (NPCAction, ActionMode, etc.)
    confidence: float = 0.5
    
    @property
    def decision_id(self) -> str:
        return self.record.decision_id
    
    @property
    def action_id(self) -> str:
        return self.record.chosen_action_id


class DecisionLogger:
    """
    Append-only logger for DecisionRecords.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("data/learning/decisions.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast lookup (decision_id -> line number)
        self._index: Dict[str, int] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Build index from existing log file."""
        if not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        if 'decision_id' in data:
                            self._index[data['decision_id']] = i
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load decision index: {e}")
    
    def log(self, record: DecisionRecord) -> None:
        """Append decision record to log."""
        try:
            with open(self.log_path, 'a') as f:
                f.write(record.to_json() + '\n')
            self._index[record.decision_id] = len(self._index)
            logger.debug(f"[DECISION LOG] {record.decision_id} -> {record.chosen_action_id}")
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    def exists(self, decision_id: str) -> bool:
        """Check if decision_id exists in index."""
        return decision_id in self._index
    
    def get(self, decision_id: str) -> Optional[DecisionRecord]:
        """Retrieve decision record by ID (slow, reads file)."""
        if decision_id not in self._index:
            return None
        
        line_num = self._index[decision_id]
        try:
            with open(self.log_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == line_num:
                        return DecisionRecord.from_json(line.strip())
        except Exception as e:
            logger.error(f"Failed to get decision {decision_id}: {e}")
        return None
    
    def get_session_decisions(self, session_id: str, npc_id: str, limit: int = 100) -> List[DecisionRecord]:
        """Get recent decisions for a session/npc pair."""
        decisions = []
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('session_id') == session_id and data.get('npc_id') == npc_id:
                            decisions.append(DecisionRecord.from_dict(data))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to get session decisions: {e}")
        
        return decisions[-limit:]  # Return most recent


# Global decision logger (singleton pattern)
_decision_logger: Optional[DecisionLogger] = None

def get_decision_logger() -> DecisionLogger:
    """Get or create the global decision logger."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger
