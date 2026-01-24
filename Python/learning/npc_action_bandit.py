"""
NPC Action Bandit: Thompson Sampling learner for action selection.
Learns which NPCAction to choose per state context bucket.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from world_model import NPCAction, PlayerSignal, StateSnapshot


@dataclass(frozen=True)
class BanditKey:
    """
    Coarse bucket so learning generalizes without exploding state space.
    """
    combat_bucket: str
    relationship_bucket: str
    threat_bucket: str

    @classmethod
    def from_state(cls, s: StateSnapshot, sig: PlayerSignal) -> "BanditKey":
        # Discretize into coarse buckets to reduce state explosion
        
        # Combat bucket
        c_bucket = "combat" if s.combat_active else "social"
        
        # Relationship bucket
        # "hostile" (very_neg, neg), "neutral" (neutral), "friendly" (pos, very_pos)
        aff = float(s.affinity)
        if aff <= -0.2:
            r_bucket = "hostile"
        elif aff >= 0.2:
            r_bucket = "friendly"
        else:
            r_bucket = "neutral"
            
        # Threat bucket
        # armed if aggression detected (ATTACK, THREATEN) vs unarmed (others)
        t_bucket = "unarmed"
        if sig in (PlayerSignal.ATTACK, PlayerSignal.THREATEN):
            t_bucket = "armed"

        return cls(
            combat_bucket=c_bucket,
            relationship_bucket=r_bucket,
            threat_bucket=t_bucket,
        )

    def to_str(self) -> str:
        return f"{self.combat_bucket}:{self.relationship_bucket}:{self.threat_bucket}"


def _recover_tmp_files(path: Path):
    """Recover from a crash where .tmp exists but main file is missing"""
    tmp_path = path.with_suffix(".json.tmp")
    # If main file missing but tmp exists, recover tmp
    if not path.exists() and tmp_path.exists():
        import os
        os.replace(tmp_path, path)
        print(f"Recovered bandit data from {tmp_path}")



class NPCActionBandit:
    """
    Thompson sampling bandit with Beta priors per (key, action).
    Rewards must be in [0, 1].
    """
    
    # Prior blending constants
    PRIOR_BLEND_MIN = -0.05  # Min bump from scorer priors
    PRIOR_BLEND_MAX = 0.05   # Max bump from scorer priors
    PRIOR_SCALE = 20.0        # Scale factor to normalize scorer values into bump range
    
    # Adaptive exploration constants
    EXPLORE_START = 0.20      # Initial exploration rate (20%)
    EXPLORE_MIN = 0.02        # Minimum exploration rate (2%)
    EXPLORE_DECAY_TRIALS = 100  # Trials to reach minimum

    def __init__(self, path: Optional[Path] = None, min_trials_before_exploit: int = 3):
        self.path = path or Path("data/learning/npc_action_bandit.json")
        self.min_trials_before_exploit = min_trials_before_exploit
        self._arms: Dict[str, Dict[str, Dict[str, float]]] = {}  # key -> action -> {alpha,beta}
        self.load()

    def _get_arm(self, key: str, action: NPCAction) -> Dict[str, float]:
        k = key
        a = action.value
        if k not in self._arms:
            self._arms[k] = {}
        if a not in self._arms[k]:
            # mild prior: alpha=1, beta=1 (uniform)
            self._arms[k][a] = {"alpha": 1.0, "beta": 1.0, "n": 0.0}
        return self._arms[k][a]

    @staticmethod
    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x
    
    def get_exploration_rate(self, key: str) -> float:
        """
        Calculate adaptive exploration rate for a context.
        
        As the bandit gains more experience in a context, exploration decreases.
        Uses exponential decay from EXPLORE_START to EXPLORE_MIN.
        
        Args:
            key: Context bucket key
            
        Returns:
            Exploration rate between EXPLORE_MIN and EXPLORE_START
        """
        if key not in self._arms:
            return self.EXPLORE_START
        
        # Sum total trials across all actions in this context
        total_trials = sum(arm.get("n", 0) for arm in self._arms[key].values())
        
        # Exponential decay: rate = min + (start - min) * exp(-trials / decay_rate)
        decay_factor = math.exp(-total_trials / self.EXPLORE_DECAY_TRIALS)
        rate = self.EXPLORE_MIN + (self.EXPLORE_START - self.EXPLORE_MIN) * decay_factor
        
        return rate
    
    def get_context_stats(self, key: str) -> Dict[str, any]:
        """Get statistics for a context bucket."""
        if key not in self._arms:
            return {"total_trials": 0, "actions": {}, "exploration_rate": self.EXPLORE_START}
        
        action_stats = {}
        for action_name, arm in self._arms[key].items():
            n = arm.get("n", 0)
            alpha = arm.get("alpha", 1)
            beta = arm.get("beta", 1)
            # Compute mean of Beta distribution
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            action_stats[action_name] = {
                "trials": n,
                "mean_reward": round(mean, 3),
                "alpha": round(alpha, 2),
                "beta": round(beta, 2)
            }
        
        return {
            "total_trials": sum(arm.get("n", 0) for arm in self._arms[key].values()),
            "actions": action_stats,
            "exploration_rate": round(self.get_exploration_rate(key), 3)
        }

    def select(
        self,
        key: str,
        candidates: List[NPCAction],
        priors: Optional[Dict[NPCAction, float]] = None,
        explore_bias: float = None,  # Now optional - uses adaptive rate if None
        temporal_adjustments: Optional[Dict[NPCAction, float]] = None,
    ) -> NPCAction:
        """
        Select an action among candidates.
        
        Args:
            key: Bandit key (context bucket)
            candidates: List of candidate actions
            priors: Optional scorer values (any real number). Used as weak tie-breaker.
            explore_bias: Exploration rate (if None, uses adaptive rate based on experience)
            temporal_adjustments: Optional adjustments from TemporalMemory.
                Positive = action worked recently in similar state.
                Negative = action failed recently in similar state.
        """
        if not candidates:
            raise ValueError("No candidates provided to bandit.select()")

        # Use adaptive exploration rate if not explicitly provided
        actual_explore_rate = explore_bias if explore_bias is not None else self.get_exploration_rate(key)
        
        # Early exploration
        if random.random() < actual_explore_rate:
            return random.choice(candidates)

        best_a: Optional[NPCAction] = None
        best_score: float = -1e9

        for a in candidates:
            arm = self._get_arm(key, a)
            alpha = arm["alpha"]
            beta = arm["beta"]
            n = arm["n"]

            # if we have almost no data, keep it a bit exploratory
            sample = random.betavariate(alpha, beta)

            # weak prior blending: normalize scorer prior into a small bump
            bump = 0.0
            if priors and a in priors:
                p = priors[a]
                bump = max(self.PRIOR_BLEND_MIN, min(self.PRIOR_BLEND_MAX, p / self.PRIOR_SCALE))

            # Temporal intuition: bias based on recent experiences
            # This is the "anticipation" mechanism - adjust before selection
            temporal_bump = 0.0
            if temporal_adjustments and a in temporal_adjustments:
                temporal_bump = temporal_adjustments[a]  # Already clamped by TemporalMemory

            # discourage exploitation if no trials yet
            cold_start_penalty = 0.0
            if n < self.min_trials_before_exploit:
                cold_start_penalty = -0.03 * (self.min_trials_before_exploit - n)

            score = sample + bump + temporal_bump + cold_start_penalty
            if score > best_score:
                best_score = score
                best_a = a

        return best_a or candidates[0]

    def update(self, key: str, action: NPCAction, reward_01: float) -> None:
        r = self._clamp01(float(reward_01))
        arm = self._get_arm(key, action)

        # Beta update: treat reward as fractional success
        arm["alpha"] += r
        arm["beta"] += (1.0 - r)
        arm["n"] += 1.0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: save to .tmp, then rename
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"arms": self._arms}, f, indent=2, sort_keys=True)
        
        import os
        os.replace(tmp_path, self.path)

    def load(self) -> None:
        # Attempt recovery first
        _recover_tmp_files(self.path)
        
        if not self.path.exists():
            self._arms = {}
            return
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._arms = data.get("arms", {}) or {}
        except Exception as e:
            print(f"Corrupted bandit data at {self.path}: {e}")
            # if corrupted, start fresh (better than crashing prod)
            self._arms = {}

