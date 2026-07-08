"""Small shared helpers for the transport backends.

Pulled out of base.py so that module holds only the TransportBridge contract and
the concrete stand-ins. Two unrelated pieces live here: StreamFreshness (a
thread-safe last-receipt clock for link health) and _SimpleRate (a sleep-based
rate limiter for non-ROS loops).
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Iterable


class StreamFreshness:
    """Thread-safe last-receipt clock shared by live transports to report link health.

    A transport marks it whenever a message lands on any of its subscriptions; consumers
    read seconds_since() to decide whether the stream is still online. Replay/synthetic
    sources never mark it, so they report no freshness (None) rather than a fake age.
    """

    def __init__(self) -> None:
        self._last_monotonic = 0.0
        self._lock = threading.Lock()

    def mark(self) -> None:
        """Stamp the current monotonic time as the moment a message was received."""
        with self._lock:
            self._last_monotonic = time.monotonic()

    def seconds_since(self) -> float | None:
        """Seconds since the last mark(), or None if nothing has been marked yet."""
        with self._lock:
            if self._last_monotonic == 0.0:
                return None
            return time.monotonic() - self._last_monotonic


class ImageRateTracker:
    """Track per-camera image receive rates and report the minimum Hz."""

    def __init__(self, sample_limit: int = 20) -> None:
        self._sample_limit = max(2, sample_limit)
        self._stamps: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def mark(self, key: str, timestamp: float | None = None) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            stamps = self._stamps.setdefault(
                key, collections.deque(maxlen=self._sample_limit)
            )
            if stamps and stamp <= stamps[-1]:
                return
            stamps.append(stamp)

    def mark_many(self, keys: Iterable[str], timestamp: float | None = None) -> None:
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        for key in keys:
            self.mark(str(key), stamp)

    def min_hz(self, keys: Iterable[str] | None = None) -> float | None:
        with self._lock:
            selected = tuple(keys) if keys is not None else tuple(self._stamps)
            if not selected:
                return None
            rates: list[float] = []
            for key in selected:
                stamps = self._stamps.get(str(key))
                if stamps is None or len(stamps) < 2:
                    return None
                duration = stamps[-1] - stamps[0]
                if duration <= 0:
                    return None
                rates.append((len(stamps) - 1) / duration)
        return min(rates) if rates else None


class _SimpleRate:
    """A time.sleep-based rate limiter (no ROS dependency)."""

    def __init__(self, hz: float) -> None:
        self._period = 1.0 / max(hz, 1e-6)
        self._last = time.monotonic()

    def sleep(self) -> None:
        """Block until one period has elapsed since the previous sleep() return."""
        now = time.monotonic()
        remaining = self._period - (now - self._last)
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()
