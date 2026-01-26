from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

from .thalamic_gate import ThalamusGate
from .cgw_state import CGWRuntime
from .types import SelfModel

@dataclass
class Runtime:
    gate: ThalamusGate
    cgw: CGWRuntime
    event_bus: Any
    self_model_provider: Callable[[], SelfModel]

    def tick(self) -> bool:
        winner, reason = self.gate.select_winner()
        if winner is None:
            return False
        self_model = self.self_model_provider()
        self.cgw.update(winner, reason, self_model)
        return True

    def run_cycles(self, n: int) -> int:
        active = 0
        for _ in range(n):
            if self.tick():
                active += 1
        return active
