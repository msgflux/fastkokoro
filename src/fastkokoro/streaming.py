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


def split_chunks(
    text: str,
    *,
    max_chars: int,
    max_words: int,
) -> list[str]:
    return split_scheduled_chunks(
        text,
        initial_max_chars=max_chars,
        initial_max_words=max_words,
        max_chars=max_chars,
        max_words=max_words,
    )


def split_scheduled_chunks(
    text: str,
    *,
    initial_max_chars: int,
    initial_max_words: int,
    max_chars: int,
    max_words: int,
) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    current_chars = 0
    segment_index = 0

    for word in text.split():
        segment_max_chars, segment_max_words = _segment_limits(
            segment_index,
            initial_max_chars=initial_max_chars,
            initial_max_words=initial_max_words,
            max_chars=max_chars,
            max_words=max_words,
        )
        projected_chars = current_chars + len(word) + (1 if current else 0)
        should_flush = current and (
            projected_chars > segment_max_chars or len(current) >= segment_max_words
        )
        if should_flush:
            segments.append(" ".join(current))
            segment_index += 1
            current = []
            current_chars = 0

        current.append(word)
        current_chars += len(word) + (1 if len(current) > 1 else 0)

        if word[-1:] in {".", ",", ";", ":", "!", "?"}:
            segments.append(" ".join(current))
            segment_index += 1
            current = []
            current_chars = 0

    if current:
        segments.append(" ".join(current))

    return segments


def _segment_limits(
    index: int,
    *,
    initial_max_chars: int,
    initial_max_words: int,
    max_chars: int,
    max_words: int,
) -> tuple[int, int]:
    multiplier = 2**index
    return (
        min(max_chars, initial_max_chars * multiplier),
        min(max_words, initial_max_words * multiplier),
    )


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
