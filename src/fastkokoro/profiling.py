from __future__ import annotations

import cProfile
import io
import pstats
import re
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastkokoro.config import Settings


class Profiler:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.profile
        self.output_dir = settings.profile_dir
        self.profile_warmup = settings.profile_warmup
        self.profile_requests = settings.profile_requests
        self._lock = Lock()
        self._sequence = 0

    @contextmanager
    def capture(self, label: str, *, enabled: bool) -> Iterator[Path | None]:
        if not self.enabled or not enabled:
            yield None
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile = cProfile.Profile()
        profile.enable()
        path = self._allocate_path(label)
        try:
            yield path
        finally:
            profile.disable()
            profile.dump_stats(str(path))
            self._write_summary(profile, path.with_suffix(".txt"))

    def snapshot(self) -> dict[str, Any]:
        recent_profiles: list[str] = []
        if self.output_dir.exists():
            recent_profiles = [
                path.name
                for path in sorted(
                    self.output_dir.glob("*.prof"),
                    key=lambda candidate: candidate.stat().st_mtime,
                    reverse=True,
                )[:10]
            ]
        return {
            "enabled": self.enabled,
            "dir": str(self.output_dir),
            "warmup": self.profile_warmup,
            "requests": self.profile_requests,
            "recent_profiles": recent_profiles,
        }

    def _allocate_path(self, label: str) -> Path:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        slug = slugify(label)
        return self.output_dir / f"{sequence:04d}-{slug}-{timestamp}.prof"

    def _write_summary(self, profile: cProfile.Profile, path: Path) -> None:
        stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        path.write_text(stream.getvalue())


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "profile"
