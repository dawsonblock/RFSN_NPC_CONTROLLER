from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

class SelectionReason(Enum):
    FORCED_OVERRIDE = "FORCED_OVERRIDE"
    COMPETITION = "COMPETITION"
    URGENCY = "URGENCY"
    SURPRISE = "SURPRISE"

@dataclass
class Candidate:
    """A signal competing for workspace entry."""
    slot_id: str
    source_module: str
    content_payload: bytes
    saliency: float
    urgency: float = 0.0
    surprise: float = 0.0
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def score(self) -> float:
        return (0.5 * float(self.saliency) +
                0.3 * float(self.urgency) +
                0.2 * float(self.surprise))

@dataclass
class ForcedCandidate:
    """A forced signal that bypasses competition entirely."""
    slot_id: str
    source_module: str
    content_payload: bytes
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    reason: str = "FORCED_OVERRIDE"

@dataclass
class SelectionEvent:
    """Emitted when gate selects a winner."""
    cycle_id: int
    slot_id: str
    reason: SelectionReason
    timestamp: float
    forced_queue_size: int
    losers: List[str]
    winner_is_forced: bool

    def total_candidates(self) -> int:
        return 1 + len(self.losers)

@dataclass(frozen=True)
class AttendedContent:
    """Single authoritative slot."""
    slot_id: str
    payload_hash: str
    payload_bytes: bytes
    source_module: str
    timestamp: float

@dataclass
class CausalTrace:
    winner_reason: SelectionReason
    winner_score: Optional[float]
    losers: List[str] = field(default_factory=list)
    forced_override: bool = False

@dataclass
class SelfModel:
    goals: List[str] = field(default_factory=list)
    active_intentions: List[str] = field(default_factory=list)
    confidence_estimates: Dict[str, float] = field(default_factory=dict)

    def delta_magnitude(self, other: "SelfModel") -> float:
        goal_diff = set(self.goals) ^ set(other.goals)
        intent_diff = set(self.active_intentions) ^ set(other.active_intentions)
        keys = set(self.confidence_estimates) | set(other.confidence_estimates)
        conf_diff = sum(abs(self.confidence_estimates.get(k, 0.0) - other.confidence_estimates.get(k, 0.0)) for k in keys)
        return float(len(goal_diff) + len(intent_diff)) + float(conf_diff)

@dataclass
class CGWState:
    cycle_id: int
    timestamp: float
    attended_content: AttendedContent
    causal_trace: CausalTrace
    self_model: SelfModel

    def content_id(self) -> str:
        return self.attended_content.slot_id

    def content_hash(self) -> str:
        return self.attended_content.payload_hash
