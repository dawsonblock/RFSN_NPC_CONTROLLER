"""
Contextual Bandit with LinUCB for RFSN NPC Action Selection.

Replaces coarse BanditKey with feature vectors for better generalization.
Uses online linear regression with UCB exploration.
"""
from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from world_model import NPCAction, PlayerSignal


class ConversationPhase(Enum):
    """Phase of the conversation."""
    EARLY = "early"      # First 2 turns
    MID = "mid"          # Turns 3-6
    LATE = "late"        # Turn 7+


@dataclass
class BanditContext:
    """
    Rich context for contextual bandit decision.
    Captures mood, affinity, player signal, conversation state.
    """
    mood_bucket: str          # hostile/neutral/friendly
    affinity_bucket: str      # neg/neutral/pos  
    player_signal: str        # from PlayerSignal
    last_action: Optional[str]  # previous NPC action
    conversation_phase: str   # early/mid/late
    has_safety_flag: bool     # from IntentGate
    turn_count: int = 0
    
    # Feature dimension constants
    MOOD_BUCKETS = ["hostile", "neutral", "friendly"]
    AFFINITY_BUCKETS = ["neg", "neutral", "pos"]
    SIGNALS = [s.value for s in PlayerSignal]
    PHASES = ["early", "mid", "late"]
    
    @classmethod
    def from_state(
        cls,
        mood: str,
        affinity: float,
        player_signal: PlayerSignal,
        last_action: Optional[NPCAction],
        turn_count: int,
        has_safety_flag: bool = False
    ) -> "BanditContext":
        """Create context from state components."""
        # Mood bucket
        mood_lower = mood.lower()
        if any(h in mood_lower for h in ["hostile", "angry", "furious", "fearful"]):
            mood_bucket = "hostile"
        elif any(f in mood_lower for f in ["friendly", "happy", "joyful", "warm"]):
            mood_bucket = "friendly"
        else:
            mood_bucket = "neutral"
        
        # Affinity bucket
        if affinity <= -0.2:
            affinity_bucket = "neg"
        elif affinity >= 0.2:
            affinity_bucket = "pos"
        else:
            affinity_bucket = "neutral"
        
        # Conversation phase
        if turn_count <= 2:
            phase = "early"
        elif turn_count <= 6:
            phase = "mid"
        else:
            phase = "late"
        
        return cls(
            mood_bucket=mood_bucket,
            affinity_bucket=affinity_bucket,
            player_signal=player_signal.value,
            last_action=last_action.value if last_action else None,
            conversation_phase=phase,
            has_safety_flag=has_safety_flag,
            turn_count=turn_count
        )
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert context to feature vector for LinUCB.
        
        Features:
        - One-hot mood bucket (3)
        - One-hot affinity bucket (3)
        - One-hot player signal (covered by hash)
        - One-hot conversation phase (3)
        - Binary safety flag (1)
        - Turn count normalized (1)
        - Bias term (1)
        """
        features = []
        
        # One-hot mood (3 dims)
        for bucket in self.MOOD_BUCKETS:
            features.append(1.0 if self.mood_bucket == bucket else 0.0)
        
        # One-hot affinity (3 dims)
        for bucket in self.AFFINITY_BUCKETS:
            features.append(1.0 if self.affinity_bucket == bucket else 0.0)
        
        # One-hot phase (3 dims)
        for phase in self.PHASES:
            features.append(1.0 if self.conversation_phase == phase else 0.0)
        
        # Binary safety flag (1 dim)
        features.append(1.0 if self.has_safety_flag else 0.0)
        
        # Normalized turn count (1 dim)
        features.append(min(1.0, self.turn_count / 10.0))
        
        # Bias term (1 dim)
        features.append(1.0)
        
        return np.array(features, dtype=np.float64)
    
    @classmethod
    def feature_dim(cls) -> int:
        """Return dimension of feature vector."""
        return 3 + 3 + 3 + 1 + 1 + 1  # mood + affinity + phase + safety + turn + bias


@dataclass
class LinUCBArm:
    """
    LinUCB arm with online linear regression.
    
    Maintains A matrix (d x d) and b vector (d) for each arm.
    θ = A^-1 * b is the weight vector.
    UCB = θᵀx + α * sqrt(xᵀ A^-1 x)
    """
    dim: int
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    n: int = 0
    
    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.dim, dtype=np.float64)
        if self.b is None:
            self.b = np.zeros(self.dim, dtype=np.float64)
    
    def get_ucb(self, x: np.ndarray, alpha: float = 1.0) -> float:
        """
        Compute UCB score for this arm given context x.
        
        UCB = θᵀx + α * sqrt(xᵀ A^-1 x)
        """
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        
        # Expected reward
        expected = theta @ x
        
        # Exploration bonus
        uncertainty = alpha * math.sqrt(x @ A_inv @ x)
        
        return expected + uncertainty
    
    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Update arm parameters with observed reward.
        
        A = A + xxᵀ
        b = b + r*x
        """
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "dim": self.dim,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "n": self.n
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinUCBArm":
        """Deserialize from dict."""
        arm = cls(dim=data["dim"])
        arm.A = np.array(data["A"], dtype=np.float64)
        arm.b = np.array(data["b"], dtype=np.float64)
        arm.n = data.get("n", 0)
        return arm


class ContextualBandit:
    """
    LinUCB contextual bandit for NPC action selection.
    
    Uses feature vectors instead of coarse buckets for better generalization.
    Maintains separate LinUCB arm per action.
    """
    
    DEFAULT_ALPHA = 0.5  # Exploration parameter
    
    def __init__(
        self,
        path: Optional[Path] = None,
        alpha: float = DEFAULT_ALPHA,
        dim: Optional[int] = None
    ):
        self.path = path or Path("data/learning/contextual_bandit.json")
        self.alpha = alpha
        self.dim = dim or BanditContext.feature_dim()
        
        # Arms: action_name -> LinUCBArm
        self._arms: Dict[str, LinUCBArm] = {}
        
        # Counterfactual log for analysis
        self._counterfactual_log: List[Dict] = []
        self._max_log_size = 1000
        
        self.load()
    
    def _get_arm(self, action: NPCAction) -> LinUCBArm:
        """Get or create arm for action."""
        name = action.value
        if name not in self._arms:
            self._arms[name] = LinUCBArm(dim=self.dim)
        return self._arms[name]
    
    def select(
        self,
        context: BanditContext,
        candidates: List[NPCAction],
        log_counterfactual: bool = True
    ) -> NPCAction:
        """
        Select best action for context using LinUCB.
        
        Args:
            context: Rich context features
            candidates: Available actions
            log_counterfactual: Whether to log all candidate scores
            
        Returns:
            Selected action
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        x = context.to_feature_vector()
        
        # Compute UCB for each candidate
        scores: List[tuple[NPCAction, float]] = []
        for action in candidates:
            arm = self._get_arm(action)
            ucb = arm.get_ucb(x, self.alpha)
            scores.append((action, ucb))
        
        # Sort by UCB score (descending)
        scores.sort(key=lambda t: t[1], reverse=True)
        
        # Log counterfactual (top-K)
        if log_counterfactual and len(scores) > 1:
            self._log_counterfactual(context, scores[:5])
        
        return scores[0][0]
    
    def update(
        self,
        context: BanditContext,
        action: NPCAction,
        reward: float
    ) -> None:
        """
        Update arm with observed reward.
        
        Args:
            context: Context when action was taken
            action: Action that was taken
            reward: Observed reward [0, 1]
        """
        x = context.to_feature_vector()
        reward = max(0.0, min(1.0, reward))  # Clamp
        
        arm = self._get_arm(action)
        arm.update(x, reward)
    
    def _log_counterfactual(
        self,
        context: BanditContext,
        scores: List[tuple[NPCAction, float]]
    ) -> None:
        """Log top-K candidates for counterfactual analysis."""
        entry = {
            "context": {
                "mood": context.mood_bucket,
                "affinity": context.affinity_bucket,
                "signal": context.player_signal,
                "phase": context.conversation_phase
            },
            "candidates": [
                {"action": a.value, "score": round(s, 4)}
                for a, s in scores
            ],
            "chosen": scores[0][0].value
        }
        
        self._counterfactual_log.append(entry)
        
        # Trim log
        if len(self._counterfactual_log) > self._max_log_size:
            self._counterfactual_log = self._counterfactual_log[-self._max_log_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        stats = {}
        for name, arm in self._arms.items():
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            stats[name] = {
                "trials": arm.n,
                "weights_norm": float(np.linalg.norm(theta)),
                "top_weight_indices": [int(i) for i in np.argsort(np.abs(theta))[-3:]]
            }
        return stats
    
    def save(self) -> None:
        """Persist bandit state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "alpha": self.alpha,
            "dim": self.dim,
            "arms": {name: arm.to_dict() for name, arm in self._arms.items()},
            "counterfactual_log": self._counterfactual_log[-100:]  # Latest 100
        }
        
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        import os
        os.replace(tmp_path, self.path)
    
    def load(self) -> None:
        """Load bandit state from disk."""
        if not self.path.exists():
            return
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.alpha = data.get("alpha", self.DEFAULT_ALPHA)
            self.dim = data.get("dim", self.dim)
            
            self._arms = {
                name: LinUCBArm.from_dict(arm_data)
                for name, arm_data in data.get("arms", {}).items()
            }
            
            self._counterfactual_log = data.get("counterfactual_log", [])
            
        except Exception as e:
            print(f"Error loading contextual bandit: {e}")
            self._arms = {}
