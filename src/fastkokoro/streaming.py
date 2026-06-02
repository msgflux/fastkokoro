from __future__ import annotations

import re
from collections.abc import Iterable

from fastkokoro.config import SAMPLE_RATE

BYTES_PER_PCM_SAMPLE = 2


def split_sentences(text: str) -> list[str]:
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text)]
    return [segment for segment in segments if segment]


def split_phrases(text: str) -> list[str]:
    segments = [segment.strip() for segment in re.split(r"(?<=[,;:!?])\s+", text)]
    return [segment for segment in segments if segment]


def split_pcm_frames(
    audio: bytes,
    frame_ms: int,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> Iterable[bytes]:
    frame_size = max(
        BYTES_PER_PCM_SAMPLE,
        int(sample_rate * BYTES_PER_PCM_SAMPLE * frame_ms / 1000),
    )
    frame_size -= frame_size % BYTES_PER_PCM_SAMPLE
    for index in range(0, len(audio), frame_size):
        yield audio[index : index + frame_size]
