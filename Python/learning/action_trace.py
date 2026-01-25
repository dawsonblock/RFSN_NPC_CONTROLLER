from collections import deque
from typing import Dict, Any, List, Optional
import time

class ActionTracer:
    """
    Maintains a short rolling history of actions to enable N-step credit assignment.
    Not a full RL replay buffer; strictly for tracing recent causality.
    """
    
    def __init__(self, trace_length: int = 5, gamma: float = 0.9):
        """
        Args:
            trace_length: How many steps back to attribute reward.
            gamma: Decay factor for past actions.
        """
        self.trace_length = trace_length
        self.gamma = gamma
        self._trace: deque = deque(maxlen=trace_length)
        
    def record_step(self, 
                    context_id: str, 
                    action: str, 
                    state_snapshot: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record a decision step.
        """
        entry = {
            "timestamp": time.time(),
            "context_id": context_id,
            "action": action,
            "state_snapshot": state_snapshot,
            "metadata": metadata or {}
        }
        self._trace.append(entry)
        
    def propagate_reward(self, reward: float) -> List[Dict[str, Any]]:
        """
        Distribute a received reward backward through the trace.
        Returns a list of updates to apply: [{context_id, action, discounted_reward}]
        """
        updates = []
        current_discount = 1.0
        
        # Iterate backwards
        for i in range(len(self._trace) - 1, -1, -1):
            step = self._trace[i]
            
            # Simple geometric decay attribution
            attributed_reward = reward * current_discount
            
            updates.append({
                "context_id": step["context_id"],
                "action": step["action"],
                "reward": attributed_reward,
                "original_timestamp": step["timestamp"]
            })
            
            current_discount *= self.gamma
            
        return updates

    def clear(self):
        self._trace.clear()
