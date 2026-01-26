"""
ActionTracer: N-step credit assignment using decision_ids.

Maintains a rolling trace of recent decisions to enable backward
credit propagation when delayed rewards arrive.
"""
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import time
import logging

logger = logging.getLogger("learning.trace")


class ActionTracer:
    """
    Maintains a short rolling history of decisions to enable N-step credit assignment.
    
    Key change from v1.0: Now stores decision_ids instead of just action names,
    enabling reliable reward attribution via the decision log.
    """
    
    VERSION = "2.0"
    
    def __init__(self, trace_length: int = 5, gamma: float = 0.9):
        """
        Args:
            trace_length: How many steps back to attribute reward.
            gamma: Decay factor for past decisions.
        """
        self.trace_length = trace_length
        self.gamma = gamma
        
        # Per-session traces: (npc_id, session_id) -> deque of trace entries
        self._traces: Dict[Tuple[str, str], deque] = {}
    
    def _get_trace(self, npc_id: str, session_id: str) -> deque:
        """Get or create trace for a session."""
        key = (npc_id, session_id)
        if key not in self._traces:
            self._traces[key] = deque(maxlen=self.trace_length)
        return self._traces[key]
    
    def record_step(
        self,
        decision_id: str,
        npc_id: str,
        session_id: str,
        abstract_state_key: str,
        action_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a decision step.
        
        Args:
            decision_id: The unique decision identifier (join key)
            npc_id: NPC identifier
            session_id: Session identifier
            abstract_state_key: Abstracted state key
            action_id: The chosen action
            metadata: Optional additional metadata
        """
        trace = self._get_trace(npc_id, session_id)
        
        entry = {
            "timestamp_ms": int(time.time() * 1000),
            "decision_id": decision_id,
            "abstract_state_key": abstract_state_key,
            "action_id": action_id,
            "metadata": metadata or {}
        }
        trace.append(entry)
        
        logger.debug(
            f"[TRACE RECORD] session={session_id[:8]} decision={decision_id[:8]} "
            f"action={action_id} trace_len={len(trace)}"
        )
    
    def propagate_reward(
        self,
        decision_id: str,
        npc_id: str,
        session_id: str,
        reward: float
    ) -> List[Dict[str, Any]]:
        """
        Distribute a received reward backward through the trace.
        
        Args:
            decision_id: The decision that received the reward
            npc_id: NPC identifier
            session_id: Session identifier
            reward: The reward value to propagate
            
        Returns:
            List of updates to apply: [{decision_id, action_id, discounted_reward}]
        """
        trace = self._get_trace(npc_id, session_id)
        updates = []
        
        # Find the decision in trace
        decision_idx = None
        for i, entry in enumerate(trace):
            if entry["decision_id"] == decision_id:
                decision_idx = i
                break
        
        if decision_idx is None:
            logger.warning(f"[TRACE] Decision {decision_id[:8]} not found in trace")
            return []
        
        # Propagate backwards from the decision
        current_discount = 1.0
        
        for i in range(decision_idx, -1, -1):
            step = trace[i]
            
            # Geometric decay attribution
            attributed_reward = reward * current_discount
            
            updates.append({
                "decision_id": step["decision_id"],
                "abstract_state_key": step["abstract_state_key"],
                "action_id": step["action_id"],
                "reward": attributed_reward,
                "original_timestamp_ms": step["timestamp_ms"]
            })
            
            current_discount *= self.gamma
        
        logger.debug(
            f"[TRACE PROPAGATE] decision={decision_id[:8]} reward={reward:.2f} "
            f"updates={len(updates)}"
        )
        
        return updates
    
    def find_decision(
        self,
        decision_id: str,
        npc_id: str,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find a decision in the trace by ID."""
        trace = self._get_trace(npc_id, session_id)
        for entry in trace:
            if entry["decision_id"] == decision_id:
                return entry
        return None
    
    def get_recent(
        self,
        npc_id: str,
        session_id: str,
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """Get the most recent N entries from the trace."""
        trace = self._get_trace(npc_id, session_id)
        return list(trace)[-count:]
    
    def clear_session(self, npc_id: str, session_id: str) -> None:
        """Clear trace for a specific session."""
        key = (npc_id, session_id)
        if key in self._traces:
            self._traces[key].clear()
    
    def clear_all(self) -> None:
        """Clear all traces."""
        self._traces.clear()
    
    # Backward compatibility aliases
    def clear(self) -> None:
        """Alias for clear_all (backward compat)."""
        self.clear_all()
