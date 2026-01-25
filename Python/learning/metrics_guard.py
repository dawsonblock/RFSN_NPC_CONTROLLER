"""
Metrics Guard for Learning Regression Detection (v1.3).

Monitors rolling metrics and protects against learning drift.
Auto-reduces learning rate or freezes updates on threshold breach.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    timestamp: float
    reward: float
    latency_ms: float
    was_corrected: bool
    was_blocked: bool
    npc_name: str


@dataclass 
class RollingWindow:
    """Rolling window for metric aggregation."""
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float) -> None:
        self.values.append(value)
    
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def count(self) -> int:
        return len(self.values)


class MetricsGuard:
    """
    Regression guard for learning systems.
    
    Monitors:
    - Correction rate (user corrections)
    - Block rate (safety blocks)
    - Average reward
    - Average latency
    
    Actions:
    - Reduce learning rate if drift detected
    - Freeze updates if severe drift
    """
    
    # Thresholds
    CORRECTION_RATE_WARN = 0.15    # 15% correction rate triggers warning
    CORRECTION_RATE_FREEZE = 0.25  # 25% triggers freeze
    BLOCK_RATE_WARN = 0.10         # 10% block rate triggers warning
    BLOCK_RATE_FREEZE = 0.20       # 20% triggers freeze
    REWARD_DROP_WARN = 0.15        # 15% reward drop triggers warning
    LATENCY_SPIKE_WARN = 2.0       # 2x latency increase triggers warning
    
    # Learning rate adjustments
    NORMAL_LEARNING_RATE = 1.0
    REDUCED_LEARNING_RATE = 0.3
    FROZEN_LEARNING_RATE = 0.0
    
    def __init__(self, path: Optional[Path] = None, window_size: int = 100):
        self.path = path or Path("data/learning/metrics_guard.json")
        self.window_size = window_size
        
        # Rolling windows
        self._rewards = RollingWindow()
        self._corrections = RollingWindow()
        self._blocks = RollingWindow()
        self._latencies = RollingWindow()
        
        # Baseline (computed from first N turns)
        self._baseline_reward: Optional[float] = None
        self._baseline_latency: Optional[float] = None
        self._baseline_samples = 50
        
        # State
        self._learning_rate = self.NORMAL_LEARNING_RATE
        self._freeze_until: Optional[float] = None
        self._warnings: List[Dict[str, Any]] = []
        
        # Turn history (for analysis)
        self._turn_history: deque = deque(maxlen=500)
        
        self.load()
    
    def record_turn(
        self,
        reward: float,
        latency_ms: float,
        was_corrected: bool,
        was_blocked: bool,
        npc_name: str = ""
    ) -> None:
        """
        Record metrics for a completed turn.
        
        Args:
            reward: Observed reward [0, 1]
            latency_ms: First token latency
            was_corrected: User corrected/repeated
            was_blocked: Safety system blocked
            npc_name: NPC identifier
        """
        turn = TurnMetrics(
            timestamp=time.time(),
            reward=reward,
            latency_ms=latency_ms,
            was_corrected=was_corrected,
            was_blocked=was_blocked,
            npc_name=npc_name
        )
        
        self._turn_history.append(turn)
        
        # Update rolling windows
        self._rewards.add(reward)
        self._corrections.add(1.0 if was_corrected else 0.0)
        self._blocks.add(1.0 if was_blocked else 0.0)
        self._latencies.add(latency_ms)
        
        # Update baseline
        if self._baseline_reward is None and self._rewards.count() >= self._baseline_samples:
            self._baseline_reward = self._rewards.mean()
            self._baseline_latency = self._latencies.mean()
        
        # Check for drift
        self._check_drift()
    
    def _check_drift(self) -> None:
        """Check for drift and adjust learning rate."""
        if self._rewards.count() < 20:
            return  # Need enough data
        
        # Check if freeze expired
        if self._freeze_until and time.time() > self._freeze_until:
            self._freeze_until = None
            self._learning_rate = self.REDUCED_LEARNING_RATE
            self._log_warning("freeze_expired", "Learning unfrozen, using reduced rate")
        
        # Already frozen?
        if self._freeze_until:
            return
        
        correction_rate = self._corrections.mean()
        block_rate = self._blocks.mean()
        avg_reward = self._rewards.mean()
        avg_latency = self._latencies.mean()
        
        # Check correction rate
        if correction_rate >= self.CORRECTION_RATE_FREEZE:
            self._freeze_learning(
                "high_correction_rate",
                f"Correction rate {correction_rate:.1%} >= {self.CORRECTION_RATE_FREEZE:.1%}"
            )
            return
        elif correction_rate >= self.CORRECTION_RATE_WARN:
            self._reduce_learning(
                "elevated_correction_rate",
                f"Correction rate {correction_rate:.1%}"
            )
        
        # Check block rate
        if block_rate >= self.BLOCK_RATE_FREEZE:
            self._freeze_learning(
                "high_block_rate",
                f"Block rate {block_rate:.1%} >= {self.BLOCK_RATE_FREEZE:.1%}"
            )
            return
        elif block_rate >= self.BLOCK_RATE_WARN:
            self._reduce_learning(
                "elevated_block_rate",
                f"Block rate {block_rate:.1%}"
            )
        
        # Check reward drop (if baseline exists)
        if self._baseline_reward and self._baseline_reward > 0:
            reward_drop = (self._baseline_reward - avg_reward) / self._baseline_reward
            if reward_drop >= self.REWARD_DROP_WARN:
                self._reduce_learning(
                    "reward_degradation",
                    f"Reward dropped {reward_drop:.1%} from baseline"
                )
        
        # Check latency spike
        if self._baseline_latency and self._baseline_latency > 0:
            latency_ratio = avg_latency / self._baseline_latency
            if latency_ratio >= self.LATENCY_SPIKE_WARN:
                self._log_warning(
                    "latency_spike",
                    f"Latency {latency_ratio:.1f}x baseline"
                )
    
    def _reduce_learning(self, reason: str, details: str) -> None:
        """Reduce learning rate."""
        if self._learning_rate > self.REDUCED_LEARNING_RATE:
            self._learning_rate = self.REDUCED_LEARNING_RATE
            self._log_warning(reason, f"Reduced learning rate: {details}")
    
    def _freeze_learning(self, reason: str, details: str) -> None:
        """Freeze learning for a period."""
        self._learning_rate = self.FROZEN_LEARNING_RATE
        self._freeze_until = time.time() + 300  # 5 minute freeze
        self._log_warning(reason, f"FROZEN learning: {details}")
    
    def _log_warning(self, category: str, message: str) -> None:
        """Log a warning."""
        warning = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "message": message
        }
        self._warnings.append(warning)
        
        # Keep last 50 warnings
        if len(self._warnings) > 50:
            self._warnings = self._warnings[-50:]
        
        print(f"[MetricsGuard] {category}: {message}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate multiplier."""
        return self._learning_rate
    
    def is_frozen(self) -> bool:
        """Check if learning is frozen."""
        return self._freeze_until is not None and time.time() < self._freeze_until
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            "learning_rate": self._learning_rate,
            "is_frozen": self.is_frozen(),
            "correction_rate": round(self._corrections.mean(), 3),
            "block_rate": round(self._blocks.mean(), 3),
            "avg_reward": round(self._rewards.mean(), 3),
            "avg_latency_ms": round(self._latencies.mean(), 1),
            "baseline_reward": self._baseline_reward,
            "baseline_latency_ms": self._baseline_latency,
            "total_turns": len(self._turn_history),
            "recent_warnings": self._warnings[-5:]
        }
    
    def reset_baseline(self) -> None:
        """Reset baseline to current averages."""
        if self._rewards.count() >= 10:
            self._baseline_reward = self._rewards.mean()
            self._baseline_latency = self._latencies.mean()
            self._learning_rate = self.NORMAL_LEARNING_RATE
    
    def save(self) -> None:
        """Persist guard state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "baseline_reward": self._baseline_reward,
            "baseline_latency": self._baseline_latency,
            "learning_rate": self._learning_rate,
            "freeze_until": self._freeze_until,
            "warnings": self._warnings[-20:]
        }
        
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        import os
        os.replace(tmp_path, self.path)
    
    def load(self) -> None:
        """Load guard state."""
        if not self.path.exists():
            return
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._baseline_reward = data.get("baseline_reward")
            self._baseline_latency = data.get("baseline_latency")
            self._learning_rate = data.get("learning_rate", self.NORMAL_LEARNING_RATE)
            self._freeze_until = data.get("freeze_until")
            self._warnings = data.get("warnings", [])
            
        except Exception as e:
            print(f"Error loading metrics guard: {e}")
