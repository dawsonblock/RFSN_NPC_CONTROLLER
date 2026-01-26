from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .runtime import Runtime
from .types import SelectionEvent, SelectionReason

@dataclass
class OverrideProbe:
    content_payload: bytes
    expected_displacement_cycles: int = 3

@dataclass
class OverrideResult:
    success: bool
    latency_cycles: int
    original_slot_id: Optional[str]
    probe_slot_id: str

    selection_failed: bool
    commit_failed: bool
    wrong_commit: bool
    overtaken_by_other_forced: bool
    queue_contaminated: bool
    timeout: bool

    selection_event: Optional[SelectionEvent] = None
    commit_event: Optional[Dict[str, Any]] = None

class EventCollector:
    def __init__(self) -> None:
        self.selections: List[SelectionEvent] = []
        self.commits: List[Dict[str, Any]] = []

    def on_gate_selection(self, event: SelectionEvent) -> None:
        self.selections.append(event)

    def on_cgw_commit(self, event: Dict[str, Any]) -> None:
        self.commits.append(event)

    def find_selection(self, slot_id: str) -> Optional[SelectionEvent]:
        for e in self.selections:
            if e.slot_id == slot_id:
                return e
        return None

    def find_commit(self, slot_id: str) -> Optional[Dict[str, Any]]:
        for e in self.commits:
            if e.get("slot_id") == slot_id:
                return e
        return None

class ForcedAttentionOverride:
    def __init__(self, runtime: Runtime, event_bus) -> None:
        self.runtime = runtime
        self.gate = runtime.gate
        self.cgw = runtime.cgw
        self.event_bus = event_bus

    def execute(self, max_wait_cycles: int = 5) -> OverrideResult:
        if len(self.gate.forced_queue) > 0:
            return OverrideResult(
                success=False,
                latency_cycles=0,
                original_slot_id=None,
                probe_slot_id="N/A",
                selection_failed=False,
                commit_failed=False,
                wrong_commit=False,
                overtaken_by_other_forced=False,
                queue_contaminated=True,
                timeout=False,
            )

        original_state = self.cgw.get_current_state()
        original_slot_id = original_state.content_id() if original_state else None

        collector = EventCollector()
        self.event_bus.on("GATE_SELECTION", collector.on_gate_selection)
        self.event_bus.on("CGW_COMMIT", collector.on_cgw_commit)

        probe = OverrideProbe(content_payload=b"FORCED_DUMMY_TASK_ARITHMETIC_CHAIN", expected_displacement_cycles=3)
        target_slot_id = self.gate.inject_forced_signal("OVERRIDE_TESTER", probe.content_payload, reason="FORCED_OVERRIDE")

        start_cycle = int(self.cgw.cycle_counter)
        selection_event = None
        commit_event = None

        for _ in range(int(max_wait_cycles)):
            self.runtime.tick()
            if selection_event is None:
                selection_event = collector.find_selection(target_slot_id)
            if commit_event is None:
                commit_event = collector.find_commit(target_slot_id)
            if selection_event and commit_event:
                break

        selection_failed = selection_event is None
        commit_failed = commit_event is None

        overtaken = False
        if selection_failed and collector.selections:
            last = collector.selections[-1]
            if last.winner_is_forced and last.slot_id != target_slot_id:
                overtaken = True

        wrong_commit = False
        if commit_event is not None and commit_event.get("slot_id") != target_slot_id:
            wrong_commit = True

        current_state = self.cgw.get_current_state()
        current_slot_id = current_state.content_id() if current_state else None

        success = (not selection_failed and not commit_failed and not wrong_commit and not overtaken and current_slot_id == target_slot_id)

        if commit_event and "cycle_id" in commit_event:
            latency = int(commit_event["cycle_id"]) - start_cycle
        else:
            latency = int(max_wait_cycles)

        timeout = latency >= int(max_wait_cycles)

        return OverrideResult(
            success=success,
            latency_cycles=latency,
            original_slot_id=original_slot_id,
            probe_slot_id=target_slot_id,
            selection_failed=selection_failed,
            commit_failed=commit_failed,
            wrong_commit=wrong_commit,
            overtaken_by_other_forced=overtaken,
            queue_contaminated=False,
            timeout=timeout,
            selection_event=selection_event,
            commit_event=commit_event,
        )
