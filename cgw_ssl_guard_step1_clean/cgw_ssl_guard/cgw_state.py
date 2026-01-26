from __future__ import annotations

from collections import deque
from typing import Optional, Union
import hashlib
import time

from .types import (
    AttendedContent, CausalTrace, CGWState, Candidate, ForcedCandidate,
    SelectionReason, SelfModel
)

class CGWRuntime:
    """Workspace runtime with atomic state swap."""

    def __init__(self, event_bus) -> None:
        self.current_state: Optional[CGWState] = None
        self.state_history = deque(maxlen=1000)
        self.cycle_counter: int = 0
        self.event_bus = event_bus

    def update(self, winner: Union[ForcedCandidate, Candidate], reason: SelectionReason, self_model: SelfModel) -> CGWState:
        self.cycle_counter += 1

        if isinstance(winner, ForcedCandidate):
            slot_id = winner.slot_id
            payload = winner.content_payload
            source = winner.source_module
            forced = True
            score = None
        else:
            slot_id = winner.slot_id
            payload = winner.content_payload
            source = winner.source_module
            forced = False
            score = winner.score()

        attended = AttendedContent(
            slot_id=slot_id,
            payload_hash=hashlib.sha256(payload).hexdigest(),
            payload_bytes=payload,
            source_module=source,
            timestamp=time.time(),
        )

        trace = CausalTrace(
            winner_reason=reason,
            winner_score=score,
            losers=[],
            forced_override=forced,
        )

        new_state = CGWState(
            cycle_id=self.cycle_counter,
            timestamp=time.time(),
            attended_content=attended,
            causal_trace=trace,
            self_model=self_model,
        )

        old = self.current_state
        self.current_state = new_state
        if old is not None:
            self.state_history.append(old)

        self.event_bus.emit("CGW_COMMIT", {
            "cycle_id": new_state.cycle_id,
            "slot_id": new_state.content_id(),
            "reason": new_state.causal_trace.winner_reason.name,
            "forced": new_state.causal_trace.forced_override,
            "timestamp": new_state.timestamp,
        })
        return new_state

    def get_current_state(self) -> Optional[CGWState]:
        return self.current_state
