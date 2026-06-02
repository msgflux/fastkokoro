from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class SpeechMetrics:
    requests: int = 0
    streaming_requests: int = 0
    errors: int = 0
    chunks: int = 0
    bytes: int = 0
    latency_seconds_total: float = 0.0
    first_chunk_latency_seconds_total: float = 0.0
    first_chunk_observations: int = 0


@dataclass
class RequestMetrics:
    requests: int = 0
    latency_seconds_total: float = 0.0
    by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_path: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = RequestMetrics()
        self._speech = SpeechMetrics()

    def record_request(
        self, path: str, status_code: int, latency_seconds: float
    ) -> None:
        with self._lock:
            self._requests.requests += 1
            self._requests.latency_seconds_total += latency_seconds
            self._requests.by_status[str(status_code)] += 1
            self._requests.by_path[path] += 1

    def record_speech(
        self,
        *,
        streaming: bool,
        latency_seconds: float,
        first_chunk_latency_seconds: float | None = None,
        chunks: int = 0,
        bytes_count: int = 0,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._speech.requests += 1
            self._speech.streaming_requests += int(streaming)
            self._speech.errors += int(error)
            self._speech.chunks += chunks
            self._speech.bytes += bytes_count
            self._speech.latency_seconds_total += latency_seconds
            if first_chunk_latency_seconds is not None:
                self._speech.first_chunk_latency_seconds_total += (
                    first_chunk_latency_seconds
                )
                self._speech.first_chunk_observations += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests_average = average(
                self._requests.latency_seconds_total,
                self._requests.requests,
            )
            speech_average = average(
                self._speech.latency_seconds_total,
                self._speech.requests,
            )
            first_chunk_average = average(
                self._speech.first_chunk_latency_seconds_total,
                self._speech.first_chunk_observations,
            )
            return {
                "http": {
                    "requests": self._requests.requests,
                    "latency_seconds_total": self._requests.latency_seconds_total,
                    "latency_seconds_avg": requests_average,
                    "by_status": dict(self._requests.by_status),
                    "by_path": dict(self._requests.by_path),
                },
                "speech": {
                    "requests": self._speech.requests,
                    "streaming_requests": self._speech.streaming_requests,
                    "errors": self._speech.errors,
                    "chunks": self._speech.chunks,
                    "bytes": self._speech.bytes,
                    "latency_seconds_total": self._speech.latency_seconds_total,
                    "latency_seconds_avg": speech_average,
                    "first_chunk_latency_seconds_total": (
                        self._speech.first_chunk_latency_seconds_total
                    ),
                    "first_chunk_latency_seconds_avg": first_chunk_average,
                    "first_chunk_observations": self._speech.first_chunk_observations,
                },
            }


def average(total: float, count: int) -> float:
    if count == 0:
        return 0.0
    return total / count
