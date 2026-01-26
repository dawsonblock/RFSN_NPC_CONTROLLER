from __future__ import annotations
from typing import Any, Callable, DefaultDict, List
from collections import defaultdict

class SimpleEventBus:
    """Minimal synchronous event bus."""
    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def on(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subs[topic].append(handler)

    def emit(self, topic: str, payload: Any) -> None:
        for h in list(self._subs.get(topic, [])):
            h(payload)
