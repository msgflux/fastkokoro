from __future__ import annotations

from io import BytesIO
from typing import Literal

import numpy as np
import soundfile as sf
from kokoro_onnx import trim_audio as kokoro_trim_audio
from numba import njit

AudioFormat = Literal["pcm", "wav", "mp3", "opus", "flac"]


@njit(cache=True)
def _encode_pcm_jit(samples: np.ndarray) -> np.ndarray:
    output = np.empty(samples.shape[0], dtype=np.int16)
    for index in range(samples.shape[0]):
        value = samples[index]
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        output[index] = np.int16(value * 32767.0)
    return output


@njit(cache=True)
def _trim_bounds_jit(samples: np.ndarray, top_db: float) -> tuple[int, int]:
    if samples.shape[0] == 0:
        return 0, 0

    peak = 0.0
    for value in samples:
        magnitude = value if value >= 0.0 else -value
        if magnitude > peak:
            peak = magnitude

    if peak <= 0.0:
        return 0, 0

    threshold = peak * (10.0 ** (-top_db / 20.0))
    start = 0
    end = samples.shape[0]

    while start < end:
        value = samples[start]
        magnitude = value if value >= 0.0 else -value
        if magnitude > threshold:
            break
        start += 1

    if start == end:
        return 0, 0

    end -= 1
    while end >= start:
        value = samples[end]
        magnitude = value if value >= 0.0 else -value
        if magnitude > threshold:
            end += 1
            break
        end -= 1

    if end < start:
        end = start
    return start, end


def _encode_pcm(samples: np.ndarray, *, use_pcm_jit: bool) -> bytes:
    samples_f32 = np.asarray(samples, dtype=np.float32)
    if use_pcm_jit:
        pcm = _encode_pcm_jit(samples_f32)
        return pcm.astype("<i2", copy=False).tobytes()

    clipped = np.clip(samples_f32, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def trim_audio_part(
    samples: np.ndarray, *, use_jit: bool, top_db: float = 60.0
) -> np.ndarray:
    samples_f32 = np.asarray(samples, dtype=np.float32)
    if not use_jit or samples_f32.ndim != 1:
        trimmed, _ = kokoro_trim_audio(samples_f32)
        return trimmed

    start, end = _trim_bounds_jit(samples_f32, top_db)
    return samples_f32[start:end]


def encode_audio(
    samples: np.ndarray,
    sample_rate: int,
    audio_format: AudioFormat,
    *,
    use_pcm_jit: bool = False,
) -> bytes:
    if audio_format == "pcm":
        return _encode_pcm(samples, use_pcm_jit=use_pcm_jit)

    subtype = None
    container = audio_format.upper()
    if audio_format == "wav":
        container = "WAV"
        subtype = "PCM_16"
    elif audio_format == "mp3":
        container = "MP3"
    elif audio_format == "opus":
        container = "OGG"
        subtype = "OPUS"
    elif audio_format == "flac":
        container = "FLAC"

    with BytesIO() as buffer:
        sf.write(buffer, samples, sample_rate, format=container, subtype=subtype)
        return buffer.getvalue()


def media_type(audio_format: AudioFormat) -> str:
    return {
        "pcm": "audio/pcm",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
    }[audio_format]
