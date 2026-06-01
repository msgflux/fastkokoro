from __future__ import annotations

from io import BytesIO
from typing import Literal

import numpy as np
import soundfile as sf

AudioFormat = Literal["pcm", "wav", "mp3", "opus", "flac"]


def encode_audio(
    samples: np.ndarray, sample_rate: int, audio_format: AudioFormat
) -> bytes:
    if audio_format == "pcm":
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767).astype("<i2").tobytes()

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
