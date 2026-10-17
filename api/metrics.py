"""Thread-safe request metrics (counts and latency percentiles)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list (0 if empty)."""
    if not sorted_values:
        return 0.0
    index = int(round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


class MetricsRegistry:
    """Thread-safe request counters and latency windows.

    Args:
        window: How many recent latency samples to keep per endpoint.
    """

    def __init__(self, window: int = 2000) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._count: defaultdict[str, int] = defaultdict(int)
        self._total_ms: defaultdict[str, float] = defaultdict(float)
        self._max_ms: defaultdict[str, float] = defaultdict(float)
        self._window: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._status: defaultdict[str, int] = defaultdict(int)

    def record(self, endpoint: str, status: int, duration_ms: float) -> None:
        """Record one finished request."""
        with self._lock:
            self._count[endpoint] += 1
            self._total_ms[endpoint] += duration_ms
            self._max_ms[endpoint] = max(self._max_ms[endpoint], duration_ms)
            self._window[endpoint].append(duration_ms)
            self._status[str(status)] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        with self._lock:
            endpoints: dict[str, dict[str, float]] = {}
            for name, count in self._count.items():
                values = sorted(self._window[name])
                endpoints[name] = {
                    "count": count,
                    "avg_ms": round(self._total_ms[name] / count, 3),
                    "p50_ms": round(_percentile(values, 50), 3),
                    "p95_ms": round(_percentile(values, 95), 3),
                    "max_ms": round(self._max_ms[name], 3),
                }
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "total_requests": sum(self._count.values()),
                "status_codes": dict(self._status),
                "endpoints": endpoints,
            }
