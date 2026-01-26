"""
CGW Integration: Cognitive Global Workspace gate for NPC attention control.

Wraps ThalamusGate + CGWRuntime to provide:
- Attention arbitration for policy decisions
- Forced signal bypass for safety/urgency
- Single-slot attended content tracking
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

from cgw_ssl_guard import (
    SimpleEventBus,
    ThalamusGate,
    CGWRuntime,
    Runtime,
    ForcedAttentionOverride,
    OverrideResult,
    SerialityMonitor,
    Candidate,
    ForcedCandidate,
    SelectionReason,
    SelfModel,
    CGWState,
)

from world_model import NPCAction

logger = logging.getLogger("learning.cgw")


@dataclass
class AttentionCandidate:
    """Wrapper for policy decision as CGW candidate."""
    decision_id: str
    npc_id: str
    action: NPCAction
    score: float
    urgency: float = 0.0
    surprise: float = 0.0
    
    def to_candidate(self) -> Candidate:
        """Convert to CGW Candidate."""
        payload = f"{self.npc_id}:{self.action.value}:{self.decision_id}".encode()
        return Candidate(
            slot_id=self.decision_id,
            source_module=f"policy:{self.npc_id}",
            content_payload=payload,
            saliency=self.score,
            urgency=self.urgency,
            surprise=self.surprise,
        )


class CGWManager:
    """
    Manages CGW attention gate for NPC dialogue.
    
    Key responsibilities:
    1. Submit policy decisions as candidates
    2. Inject forced signals for safety/urgency
    3. Select winning candidate per cycle
    4. Track attended content history
    """
    
    def __init__(self, self_model_provider: Optional[Callable[[], SelfModel]] = None):
        self.event_bus = SimpleEventBus()
        self.gate = ThalamusGate(self.event_bus)
        self.cgw = CGWRuntime(self.event_bus)
        
        # Default self model provider
        self._self_model_provider = self_model_provider or self._default_self_model
        
        # Create runtime
        self.runtime = Runtime(
            gate=self.gate,
            cgw=self.cgw,
            event_bus=self.event_bus,
            self_model_provider=self._self_model_provider,
        )
        
        # Forced override executor
        self.override = ForcedAttentionOverride(self.runtime, self.event_bus)
        
        # Seriality monitor (multiple commits per cycle = hard failure)
        self.seriality_monitor = SerialityMonitor()
        self.event_bus.on("CGW_COMMIT", self.seriality_monitor.on_commit)
        
        # Event logging
        self.event_bus.on("GATE_SELECTION", self._log_selection)
        self.event_bus.on("CGW_COMMIT", self._log_commit)
        
        logger.info("[CGW] Manager initialized")
    
    def _default_self_model(self) -> SelfModel:
        """Default self model with empty goals."""
        return SelfModel(
            goals=["respond_appropriately"],
            active_intentions=["engage_player"],
            confidence_estimates={"dialogue": 0.8},
        )
    
    def _log_selection(self, event: Any) -> None:
        """Log gate selection events."""
        logger.debug(f"[CGW] SELECTION: slot={event.slot_id}, reason={event.reason.value}, forced={event.winner_is_forced}")
    
    def _log_commit(self, event: Dict[str, Any]) -> None:
        """Log CGW commit events."""
        logger.debug(f"[CGW] COMMIT: cycle={event.get('cycle_id')}, slot={event.get('slot_id')}, forced={event.get('forced')}")
    
    def submit_candidate(self, candidate: AttentionCandidate) -> None:
        """
        Submit a policy decision as CGW candidate.
        
        Args:
            candidate: Wrapped policy decision
        """
        cgw_candidate = candidate.to_candidate()
        self.gate.submit_candidate(cgw_candidate)
        logger.debug(f"[CGW] Submitted candidate: {candidate.decision_id} ({candidate.action.value})")
    
    def inject_forced(
        self, 
        source: str, 
        npc_action: NPCAction, 
        reason: str = "FORCED_OVERRIDE"
    ) -> str:
        """
        Inject a forced signal that bypasses competition.
        
        Args:
            source: Module/system injecting the signal (e.g., "safety_gate")
            npc_action: The action to force
            reason: Reason for forced override
            
        Returns:
            slot_id of the forced signal
        """
        payload = f"forced:{npc_action.value}".encode()
        slot_id = self.gate.inject_forced_signal(
            source_module=source,
            content_payload=payload,
            reason=reason,
        )
        logger.info(f"[CGW] Injected forced signal: {slot_id} ({npc_action.value}) reason={reason}")
        return slot_id
    
    def tick(self) -> bool:
        """
        Run one CGW cycle.
        
        Returns:
            True if a winner was selected
        """
        return self.runtime.tick()
    
    def select_and_commit(self) -> tuple[Optional[NPCAction], Optional[str], bool]:
        """
        Run selection and commit in one call.
        
        Returns:
            (selected_action, decision_id, is_forced)
        """
        # Run one cycle
        had_winner = self.tick()
        
        if not had_winner:
            return None, None, False
        
        # Get current attended content
        state = self.cgw.get_current_state()
        if state is None:
            return None, None, False
        
        # Parse payload to extract action
        payload = state.attended_content.payload_bytes.decode()
        is_forced = state.causal_trace.forced_override
        decision_id = state.content_id()
        
        # Parse action from payload
        if payload.startswith("forced:"):
            action_str = payload.split(":")[1]
        else:
            # Format: npc_id:action:decision_id
            parts = payload.split(":")
            action_str = parts[1] if len(parts) > 1 else None
        
        action = None
        if action_str:
            try:
                action = NPCAction(action_str)
            except ValueError:
                logger.warning(f"[CGW] Unknown action in payload: {action_str}")
        
        return action, decision_id, is_forced
    
    def get_current_state(self) -> Optional[CGWState]:
        """Get current CGW state."""
        return self.cgw.get_current_state()
    
    def execute_forced_override(self, max_cycles: int = 5) -> OverrideResult:
        """
        Execute a forced attention override test.
        
        Args:
            max_cycles: Maximum cycles to wait
            
        Returns:
            OverrideResult with success/failure details
        """
        return self.override.execute(max_wait_cycles=max_cycles)
    
    def has_forced_pending(self) -> bool:
        """Check if there are forced signals waiting."""
        return len(self.gate.forced_queue) > 0
    
    def clear_candidates(self) -> None:
        """Clear all pending candidates (not forced)."""
        self.gate.candidates.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CGW statistics."""
        return {
            "gate_cycle": self.gate.cycle_counter,
            "cgw_cycle": self.cgw.cycle_counter,
            "forced_queue_size": len(self.gate.forced_queue),
            "candidate_count": len(self.gate.candidates),
            "history_size": len(self.cgw.state_history),
        }


# Global singleton
_cgw_manager: Optional[CGWManager] = None

def get_cgw_manager() -> CGWManager:
    """Get or create global CGW manager."""
    global _cgw_manager
    if _cgw_manager is None:
        _cgw_manager = CGWManager()
    return _cgw_manager

def reset_cgw_manager() -> None:
    """Reset global CGW manager (for testing)."""
    global _cgw_manager
    _cgw_manager = None
