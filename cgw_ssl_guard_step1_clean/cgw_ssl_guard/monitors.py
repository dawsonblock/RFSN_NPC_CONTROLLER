from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class SerialityMonitor:
    commits_per_cycle: Dict[int, int] = field(default_factory=dict)

    def on_commit(self, event: Dict[str, Any]) -> None:
        cid = int(event["cycle_id"])
        self.commits_per_cycle[cid] = self.commits_per_cycle.get(cid, 0) + 1

    def check_seriality_violation(self, cycle_id: int) -> bool:
        return self.commits_per_cycle.get(int(cycle_id), 0) > 1
