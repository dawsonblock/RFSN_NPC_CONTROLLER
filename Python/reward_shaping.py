from collections import deque
from typing import Dict

class RewardAccumulator:
    """
    Collects micro-rewards over a short window and emits a shaped reward.
    This avoids temporal credit assignment while reducing reward sparsity.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def add(self, value: float, reason: str = ""):
        self.buffer.append(value)

    def emit(self) -> float:
        if not self.buffer:
            return 0.0
        total = sum(self.buffer)
        self.buffer.clear()
        return total
