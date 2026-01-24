"""
Temporal Memory: Short-term experience buffer for anticipatory action selection.

Enables NPCs to "remember" recent state-action-outcome tuples and bias
future decisions based on what worked in similar states.

This is NOT a neural net—it's a lightweight similarity-weighted recency buffer.
"""
from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Deque
import logging

from world_model import NPCAction, StateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single recorded experience (state, action, reward, timestamp)."""
    state: StateSnapshot
    action: NPCAction
    reward: float  # [0, 1] normalized reward
    timestamp: float = field(default_factory=time.time)
    
    def age_seconds(self) -> float:
        """Time since this experience was recorded."""
        return time.time() - self.timestamp


class TemporalMemory:
    """
    Short-term memory that biases action selection based on recent outcomes.
    
    Key insight: the bandit learns long-term frequencies, but temporal memory
    provides short-term anticipation—"this situation feels like 2 turns ago."
    
    Algorithm:
    1. For each recent experience, compute similarity to current state
    2. Weight by recency (exponential decay)
    3. Accumulate adjustment per action based on (reward - 0.5) * weight
    4. Return adjustments to bias bandit priors
    """
    
    def __init__(
        self,
        max_size: int = 50,
        decay_rate: float = 0.95,
        adjustment_scale: float = 0.1,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize temporal memory.
        
        Args:
            max_size: Maximum experiences to keep
            decay_rate: Per-second decay (0.95 = slow decay, 0.85 = faster forgetting)
            adjustment_scale: Max adjustment magnitude (0.1 = weak signal)
            similarity_threshold: Minimum similarity to consider (0.3 = fairly loose)
        """
        self.experiences: Deque[Experience] = deque(maxlen=max_size)
        self.decay_rate = decay_rate
        self.adjustment_scale = adjustment_scale
        self.similarity_threshold = similarity_threshold
        
        logger.info(
            f"[TemporalMemory] Initialized (max_size={max_size}, "
            f"decay={decay_rate}, scale={adjustment_scale})"
        )
    
    def record(
        self,
        state: StateSnapshot,
        action: NPCAction,
        reward: float
    ) -> None:
        """
        Record an experience after action selection and outcome.
        
        Args:
            state: The state when action was taken
            action: The action that was taken
            reward: Normalized reward [0, 1]
        """
        exp = Experience(
            state=state,
            action=action,
            reward=max(0.0, min(1.0, reward)),  # Clamp to [0, 1]
            timestamp=time.time()
        )
        self.experiences.append(exp)
        
        logger.debug(
            f"[TemporalMemory] Recorded: action={action.value}, "
            f"reward={reward:.2f}, buffer_size={len(self.experiences)}"
        )
    
    def get_prior_adjustments(
        self,
        current_state: StateSnapshot
    ) -> Dict[NPCAction, float]:
        """
        Compute prior adjustments based on recent experiences.
        
        Returns a dict of action -> adjustment value:
        - Positive: boost this action (worked recently in similar state)
        - Negative: decay this action (failed recently in similar state)
        
        Adjustments are bounded to [-adjustment_scale, +adjustment_scale].
        """
        if not self.experiences:
            return {}
        
        now = time.time()
        raw_adjustments: Dict[NPCAction, float] = {}
        total_weight_per_action: Dict[NPCAction, float] = {}
        
        for exp in self.experiences:
            # Compute similarity (inverse of distance)
            distance = current_state.distance_to(exp.state)
            similarity = 1.0 / (1.0 + distance)
            
            # Skip if too dissimilar
            if similarity < self.similarity_threshold:
                continue
            
            # Compute recency weight (exponential decay per second)
            age_seconds = now - exp.timestamp
            recency_weight = math.pow(self.decay_rate, age_seconds / 10.0)  # Decay per 10s
            
            # Combined weight
            weight = similarity * recency_weight
            
            # Reward signal centered at 0.5 (neutral)
            # reward > 0.5 → positive signal (action worked)
            # reward < 0.5 → negative signal (action failed)
            signal = (exp.reward - 0.5) * 2.0  # Scale to [-1, +1]
            
            # Accumulate
            if exp.action not in raw_adjustments:
                raw_adjustments[exp.action] = 0.0
                total_weight_per_action[exp.action] = 0.0
            
            raw_adjustments[exp.action] += weight * signal
            total_weight_per_action[exp.action] += weight
        
        # Normalize and clamp to adjustment scale
        adjustments: Dict[NPCAction, float] = {}
        for action, raw in raw_adjustments.items():
            total_weight = total_weight_per_action[action]
            if total_weight > 0:
                normalized = raw / total_weight
                clamped = max(-self.adjustment_scale, min(self.adjustment_scale, normalized * self.adjustment_scale))
                adjustments[action] = clamped
        
        if adjustments:
            logger.debug(
                f"[TemporalMemory] Prior adjustments: "
                f"{', '.join(f'{a.value}={v:+.3f}' for a, v in adjustments.items())}"
            )
        
        return adjustments
    
    def get_similar_experiences(
        self,
        current_state: StateSnapshot,
        top_k: int = 5
    ) -> List[Experience]:
        """
        Get the top-K most similar experiences to current state.
        Useful for debugging and introspection.
        """
        if not self.experiences:
            return []
        
        scored = []
        for exp in self.experiences:
            distance = current_state.distance_to(exp.state)
            similarity = 1.0 / (1.0 + distance)
            scored.append((similarity, exp))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:top_k]]
    
    def clear(self) -> None:
        """Clear all experiences."""
        self.experiences.clear()
        logger.info("[TemporalMemory] Cleared all experiences")
    
    def __len__(self) -> int:
        return len(self.experiences)
    
    def stats(self) -> Dict[str, any]:
        """Get statistics about current memory state."""
        if not self.experiences:
            return {"size": 0, "oldest_age_s": 0, "action_counts": {}}
        
        action_counts: Dict[str, int] = {}
        for exp in self.experiences:
            action_counts[exp.action.value] = action_counts.get(exp.action.value, 0) + 1
        
        oldest = min(exp.timestamp for exp in self.experiences)
        
        return {
            "size": len(self.experiences),
            "oldest_age_s": time.time() - oldest,
            "action_counts": action_counts
        }
