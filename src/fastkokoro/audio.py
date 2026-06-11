from __future__ import annotations

from io import BytesIO
from typing import Literal

import numpy as np
import soundfile as sf
from numba import njit

from fastkokoro.kokoro import trim_audio as kokoro_trim_audio

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
    if samples_f32.ndim != 1:
        trimmed, _ = kokoro_trim_audio(samples_f32)
        return trimmed

    trimmed, _ = kokoro_trim_audio(samples_f32, top_db=top_db, use_jit=use_jit)
    return trimmed


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
