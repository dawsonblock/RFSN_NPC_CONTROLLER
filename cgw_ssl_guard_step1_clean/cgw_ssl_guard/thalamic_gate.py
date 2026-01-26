from __future__ import annotations

from typing import List, Optional, Union
from collections import deque
import time
import uuid

from .types import Candidate, ForcedCandidate, SelectionEvent, SelectionReason

class ThalamusGate:
    """Arbiter selecting which signal enters CGW."""

    def __init__(self, event_bus) -> None:
        self.forced_queue = deque()  # deque[ForcedCandidate]
        self.candidates: List[Candidate] = []
        self.cycle_counter: int = 0
        self.event_bus = event_bus

        self.max_candidates_per_cycle = 20
        self.competition_cooldown_ms = 100
        self.last_selection_time = 0.0

    def inject_forced_signal(self, source_module: str, content_payload: bytes, reason: str = "FORCED_OVERRIDE") -> str:
        slot_id = f"forced_{uuid.uuid4().hex[:12]}"
        forced = ForcedCandidate(
            slot_id=slot_id,
            source_module=source_module,
            content_payload=content_payload,
            timestamp=time.time(),
            reason=reason,
        )
        self.forced_queue.append(forced)
        self.event_bus.emit("FORCED_INJECTION", {
            "slot_id": slot_id,
            "source": source_module,
            "timestamp": time.time(),
            "queue_depth": len(self.forced_queue),
        })
        return slot_id

    def submit_candidate(self, candidate: Candidate) -> None:
        if len(self.candidates) >= self.max_candidates_per_cycle:
            self.candidates.sort(key=lambda c: c.score(), reverse=True)
            self.candidates = self.candidates[: self.max_candidates_per_cycle - 1]
        self.candidates.append(candidate)

    def _emit_selection_event(self, slot_id: str, reason: SelectionReason, forced_queue_size: int,
                             losers: List[str], winner_is_forced: bool) -> None:
        event = SelectionEvent(
            cycle_id=self.cycle_counter,
            slot_id=slot_id,
            reason=reason,
            timestamp=time.time(),
            forced_queue_size=forced_queue_size,
            losers=losers,
            winner_is_forced=winner_is_forced,
        )
        self.event_bus.emit("GATE_SELECTION", event)

    def select_winner(self) -> tuple[Optional[Union[ForcedCandidate, Candidate]], SelectionReason]:
        self.cycle_counter += 1

        # Forced bypass
        if self.forced_queue:
            winner: ForcedCandidate = self.forced_queue.popleft()
            losers = [c.slot_id for c in self.candidates]
            self._emit_selection_event(
                slot_id=winner.slot_id,
                reason=SelectionReason.FORCED_OVERRIDE,
                forced_queue_size=len(self.forced_queue),
                losers=losers,
                winner_is_forced=True,
            )
            self.candidates.clear()
            return winner, SelectionReason.FORCED_OVERRIDE

        # Normal competition
        if self.candidates:
            now = time.time()
            if (now - self.last_selection_time) < (self.competition_cooldown_ms / 1000.0):
                return None, SelectionReason.COMPETITION

            self.candidates.sort(key=lambda c: c.score(), reverse=True)
            winner = self.candidates[0]
            losers = [c.slot_id for c in self.candidates[1:]]

            if winner.urgency > 0.8:
                reason = SelectionReason.URGENCY
            elif winner.surprise > 0.8:
                reason = SelectionReason.SURPRISE
            else:
                reason = SelectionReason.COMPETITION

            self._emit_selection_event(
                slot_id=winner.slot_id,
                reason=reason,
                forced_queue_size=0,
                losers=losers,
                winner_is_forced=False,
            )
            self.candidates.clear()
            self.last_selection_time = now
            return winner, reason

        return None, SelectionReason.COMPETITION
