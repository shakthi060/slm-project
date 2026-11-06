from __future__ import annotations

import statistics
import threading
from collections import deque

from src.evaluation.baseline import percentile


class MetricsStore:
    """Thread-safe metrics for POST /chat requests."""

    def __init__(self, latency_window: int = 500) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=max(1, latency_window))
        self._total = 0
        self._successful = 0
        self._failed = 0

    def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self._total += 1
            self._successful += 1
            self._latencies.append(float(latency_ms))

    def record_failure(self, latency_ms: float | None = None) -> None:
        with self._lock:
            self._total += 1
            self._failed += 1
            if latency_ms is not None:
                self._latencies.append(float(latency_ms))

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self._latencies)
            return {
                "total_requests": self._total,
                "successful_requests": self._successful,
                "failed_requests": self._failed,
                "average_latency_ms": round(statistics.mean(latencies), 3)
                if latencies
                else 0.0,
                "p50_latency_ms": round(percentile(latencies, 0.50), 3),
                "p95_latency_ms": round(percentile(latencies, 0.95), 3),
                "latency_samples": len(latencies),
            }
