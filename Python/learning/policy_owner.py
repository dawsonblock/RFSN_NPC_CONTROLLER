"""
PolicyOwner: SINGLE authoritative policy entrypoint.

This module owns ALL action selection and update logic in production.
No other module may make policy decisions at runtime.

Control flow:
    State → Abstraction → Safety Filter → PolicyOwner.select_action() 
    → Execution → Logging → (Optional Reward) → PolicyOwner.update()
"""
from __future__ import annotations

import os
import random
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from learning.decision_record import (
    DecisionRecord, PolicyDecision, get_decision_logger
)
from learning.state_abstraction import StateAbstractor
from learning.contextual_bandit import ContextualBandit, BanditContext
from learning.metrics_guard import MetricsGuard
from world_model import NPCAction, PlayerSignal

logger = logging.getLogger("learning.policy_owner")

# Environment flags
STRICT_LEARNING = os.environ.get("STRICT_LEARNING", "0") == "1"
LEARNING_DISABLED = os.environ.get("LEARNING_DISABLED", "0") == "1"


@dataclass
class AbstractContext:
    """
    Abstracted context for policy selection.
    Produced by StateAbstractor, consumed by PolicyOwner.
    """
    npc_id: str
    session_id: str
    abstract_state_key: str
    raw_state_hash: str
    
    # Rich features for bandit
    mood_bucket: str = "neutral"
    affinity_bucket: str = "neutral"
    player_signal: str = "question"
    last_action: Optional[str] = None
    turn_count: int = 0
    has_safety_flag: bool = False
    
    # Trace linkage
    prior_decision_id: Optional[str] = None


class PolicyOwner:
    """
    SINGLE authoritative policy for action selection and updates.
    
    This is the ONLY module that decides actions in production.
    All other bandit/policy modules are baselines or testing utilities.
    
    Guarantees:
    - Every selection produces a DecisionRecord with decision_id
    - Every update requires a valid decision_id
    - Learning can be disabled for deterministic replay
    """
    
    VERSION = "1.0"
    NAME = "policy_owner_linucb"
    
    def __init__(
        self,
        bandit_path: Optional[Path] = None,
        seed: Optional[int] = None,
        alpha: float = 0.5
    ):
        # Core bandit (LinUCB by default)
        self._bandit = ContextualBandit(
            path=bandit_path or Path("data/learning/policy_owner_bandit.json"),
            alpha=alpha
        )
        
        # State abstractor for deterministic key generation
        self._abstractor = StateAbstractor()
        
        # Metrics guard for regression detection
        self._metrics_guard = MetricsGuard()
        
        # Decision logger
        self._decision_logger = get_decision_logger()
        
        # Deterministic seeding
        self._seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Track session -> last_decision_id for trace linking
        self._session_last_decision: Dict[str, str] = {}
        
        logger.info(f"PolicyOwner initialized (version={self.VERSION}, strict={STRICT_LEARNING})")
    
    def select_action(
        self,
        context: AbstractContext,
        candidates: List[NPCAction],
        safety_filter: Optional[Callable[[NPCAction], bool]] = None
    ) -> PolicyDecision:
        """
        Select an action from candidates using the bandit policy.
        
        Args:
            context: Abstracted context with state features
            candidates: List of candidate actions
            safety_filter: Optional filter to veto unsafe actions
            
        Returns:
            PolicyDecision with DecisionRecord and chosen action
            
        Invariant: ALWAYS produces a DecisionRecord, even on veto.
        """
        if not candidates:
            candidates = [NPCAction.EXPLAIN]  # Fallback
        
        # Build bandit context from abstract context
        bandit_ctx = BanditContext(
            mood_bucket=context.mood_bucket,
            affinity_bucket=context.affinity_bucket,
            player_signal=context.player_signal,
            last_action=context.last_action,
            conversation_phase=self._get_phase(context.turn_count),
            has_safety_flag=context.has_safety_flag,
            turn_count=context.turn_count
        )
        
        # Apply safety filter to candidates
        safe_candidates = candidates
        safety_vetoed = False
        veto_reason = None
        
        if safety_filter:
            safe_candidates = [a for a in candidates if safety_filter(a)]
            if len(safe_candidates) < len(candidates):
                safety_vetoed = True
                vetoed = set(candidates) - set(safe_candidates)
                veto_reason = f"Safety filter removed: {[a.value for a in vetoed]}"
                logger.debug(f"[POLICY] Safety filter vetoed: {veto_reason}")
        
        if not safe_candidates:
            safe_candidates = [NPCAction.EXPLAIN]  # Ultimate fallback
            safety_vetoed = True
            veto_reason = "All candidates vetoed, using EXPLAIN fallback"
        
        # Select action using bandit
        exploration = False
        if LEARNING_DISABLED:
            # Deterministic: always pick first candidate (sorted by value)
            safe_candidates.sort(key=lambda a: a.value)
            chosen_action = safe_candidates[0]
        else:
            # Use bandit selection
            chosen_action = self._bandit.select(
                bandit_ctx, 
                safe_candidates,
                log_counterfactual=True
            )
            # Check if this was exploration (for logging)
            exploration = self._was_exploration(bandit_ctx, chosen_action, safe_candidates)
        
        # Get scores for counterfactual logging
        scores = self._get_scores(bandit_ctx, safe_candidates)
        
        # Create decision record
        record = DecisionRecord(
            npc_id=context.npc_id,
            session_id=context.session_id,
            raw_state_hash=context.raw_state_hash,
            abstract_state_key=context.abstract_state_key,
            abstraction_version=self._abstractor.VERSION,
            candidate_actions=[a.value for a in candidates],
            chosen_action_id=chosen_action.value,
            policy_name=self.NAME,
            policy_version=self.VERSION,
            safety_vetoed=safety_vetoed,
            veto_reason=veto_reason,
            exploration=exploration,
            scores=scores,
            prior_decision_id=context.prior_decision_id
        )
        
        # Log decision
        self._decision_logger.log(record)
        
        # Update session trace
        session_key = f"{context.npc_id}:{context.session_id}"
        self._session_last_decision[session_key] = record.decision_id
        
        logger.debug(
            f"[POLICY SELECT] decision_id={record.decision_id[:8]} "
            f"action={chosen_action.value} exploration={exploration}"
        )
        
        return PolicyDecision(
            record=record,
            action=chosen_action,
            confidence=scores.get(chosen_action.value, 0.5) if scores else 0.5
        )
    
    def update(
        self,
        decision: PolicyDecision,
        reward: Optional[float],
        meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update policy with reward for a decision.
        
        Args:
            decision: The PolicyDecision that was executed
            reward: Reward value (0-1), or None to skip update
            meta: Optional metadata (source, notes, etc.)
            
        Returns:
            True if update was applied, False if rejected
            
        Invariant: Update requires valid decision_id in log.
        """
        if LEARNING_DISABLED:
            logger.debug(f"[POLICY UPDATE] Skipped (LEARNING_DISABLED)")
            return False
        
        if reward is None:
            if STRICT_LEARNING:
                logger.debug(f"[POLICY UPDATE] Rejected (STRICT mode, no reward)")
                return False
            return True  # No-op success
        
        # Validate decision exists
        record = decision.record
        if not self._decision_logger.exists(record.decision_id):
            logger.warning(f"[POLICY UPDATE] Rejected: unknown decision_id={record.decision_id}")
            return False
        
        # Get learning rate from metrics guard
        learning_rate = self._metrics_guard.get_learning_rate()
        if learning_rate <= 0:
            logger.debug(f"[POLICY UPDATE] Frozen by metrics guard")
            return False
        
        # Clamp reward
        reward = max(0.0, min(1.0, reward))
        
        # Rebuild bandit context
        bandit_ctx = BanditContext(
            mood_bucket="neutral",  # Will be overridden from abstract key
            affinity_bucket="neutral",
            player_signal="question",
            last_action=None,
            conversation_phase="mid",
            has_safety_flag=record.safety_vetoed
        )
        
        # Parse abstract state key to restore context
        self._restore_context_from_key(bandit_ctx, record.abstract_state_key)
        
        # Apply update with learning rate
        scaled_reward = reward * learning_rate
        action = NPCAction(record.chosen_action_id)
        self._bandit.update(bandit_ctx, action, scaled_reward)
        
        # Record metrics
        source = meta.get("source", "implicit") if meta else "implicit"
        was_corrected = source == "user_correction"
        self._metrics_guard.record_turn(
            reward=reward,
            latency_ms=0,
            was_corrected=was_corrected,
            was_blocked=record.safety_vetoed,
            npc_name=record.npc_id
        )
        
        logger.debug(
            f"[POLICY UPDATE] decision_id={record.decision_id[:8]} "
            f"reward={reward:.2f} scaled={scaled_reward:.2f}"
        )
        
        return True
    
    def get_last_decision_id(self, npc_id: str, session_id: str) -> Optional[str]:
        """Get the last decision_id for a session (for trace linking)."""
        session_key = f"{npc_id}:{session_id}"
        return self._session_last_decision.get(session_key)
    
    def _get_phase(self, turn_count: int) -> str:
        """Map turn count to conversation phase."""
        if turn_count <= 2:
            return "early"
        elif turn_count <= 6:
            return "mid"
        else:
            return "late"
    
    def _was_exploration(
        self, 
        ctx: BanditContext, 
        chosen: NPCAction, 
        candidates: List[NPCAction]
    ) -> bool:
        """Check if selection was exploration (not argmax)."""
        if len(candidates) <= 1:
            return False
        
        # Get scores and check if chosen was argmax
        scores = []
        x = ctx.to_feature_vector()
        for action in candidates:
            arm = self._bandit._get_arm(action)
            ucb = arm.get_ucb(x, self._bandit.alpha)
            scores.append((action, ucb))
        
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[0][0] != chosen
    
    def _get_scores(
        self, 
        ctx: BanditContext, 
        candidates: List[NPCAction]
    ) -> Dict[str, float]:
        """Get UCB scores for all candidates."""
        scores = {}
        x = ctx.to_feature_vector()
        for action in candidates:
            arm = self._bandit._get_arm(action)
            ucb = arm.get_ucb(x, self._bandit.alpha)
            scores[action.value] = round(ucb, 4)
        return scores
    
    def _restore_context_from_key(self, ctx: BanditContext, key: str) -> None:
        """Parse abstract state key to restore context fields."""
        # Key format: "LOCATION:loc|QUEST_STATE:qs|RELATIONSHIP:rel|THREAT_LEVEL:tl"
        parts = key.split("|")
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                if k == "RELATIONSHIP":
                    ctx.affinity_bucket = v if v in ["neg", "neutral", "pos"] else "neutral"
                elif k == "THREAT_LEVEL":
                    ctx.has_safety_flag = (v == "high")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            "version": self.VERSION,
            "name": self.NAME,
            "bandit_stats": self._bandit.get_stats(),
            "metrics": self._metrics_guard.get_stats(),
            "learning_rate": self._metrics_guard.get_learning_rate(),
            "strict_mode": STRICT_LEARNING,
            "learning_disabled": LEARNING_DISABLED
        }
    
    def save(self) -> None:
        """Persist bandit state."""
        self._bandit.save()


# Global policy owner instance (singleton)
_policy_owner: Optional[PolicyOwner] = None

def get_policy_owner() -> PolicyOwner:
    """Get or create the global PolicyOwner instance."""
    global _policy_owner
    if _policy_owner is None:
        _policy_owner = PolicyOwner()
    return _policy_owner


def reset_policy_owner() -> None:
    """Reset the global PolicyOwner (for testing)."""
    global _policy_owner
    _policy_owner = None
