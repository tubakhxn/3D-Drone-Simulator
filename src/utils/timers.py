"""Utility timing helpers."""
from __future__ import annotations

from collections import deque
from time import perf_counter


class FrameTimer:
    """Tracks frame durations and exposes a rolling FPS estimate."""

    def __init__(self, window: int = 90) -> None:
        self.window = window
        self.samples = deque(maxlen=window)
        self._last = perf_counter()

    def tick(self) -> float:
        """Record the time since the previous tick and return FPS."""
        now = perf_counter()
        delta = now - self._last
        self._last = now
        self.samples.append(delta)
        if not self.samples or delta <= 0:
            return 0.0
        avg = sum(self.samples) / len(self.samples)
        return 1.0 / avg if avg else 0.0
