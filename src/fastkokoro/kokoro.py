from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import espeakng_loader
import numpy as np
import phonemizer
from numba import njit
from numpy.typing import NDArray
from phonemizer.backend.espeak.wrapper import EspeakWrapper

MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000

DEFAULT_VOCAB = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    '"': 11,
    "(": 12,
    ")": 13,
    "“": 14,
    "”": 15,
    " ": 16,
    "̃": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "ꭧ": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177,
}


@dataclass
class EspeakConfig:
    lib_path: str | None = None
    data_path: str | None = None


class Tokenizer:
    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab: dict | None = None,
    ):
        self.vocab = vocab or DEFAULT_VOCAB
        config = espeak_config or EspeakConfig()
        if not config.data_path:
            config.data_path = espeakng_loader.get_data_path()
        if not config.lib_path:
            config.lib_path = espeakng_loader.get_library_path()

        env_library = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
        if env_library:
            config.lib_path = env_library

        try:
            ctypes.cdll.LoadLibrary(config.lib_path)
        except Exception as exc:
            fallback = ctypes.util.find_library(
                "espeak-ng"
            ) or ctypes.util.find_library("espeak")
            if not fallback:
                raise RuntimeError(_espeak_error_info()) from exc
            try:
                ctypes.cdll.LoadLibrary(fallback)
            except Exception as fallback_exc:
                raise RuntimeError(f"{fallback_exc}: {_espeak_error_info()}") from exc
            config.lib_path = fallback

        EspeakWrapper.set_data_path(config.data_path)
        EspeakWrapper.set_library(config.lib_path)

    @staticmethod
    def normalize_text(text: str) -> str:
        return text.strip()

    def tokenize(self, phonemes: str) -> list[int]:
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes"
            )
        return [token for token in map(self.vocab.get, phonemes) if token is not None]

    def phonemize(self, text: str, lang: str = "en-us", norm: bool = True) -> str:
        if norm:
            text = self.normalize_text(text)
        phonemes = phonemizer.phonemize(
            text,
            lang,
            preserve_punctuation=True,
            with_stress=True,
        )
        phonemes = "".join(symbol for symbol in phonemes if symbol in self.vocab)
        return phonemes.strip()


class Kokoro:
    def __init__(
        self,
        voices_path: str | Path,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
    ):
        self.voices = np.load(voices_path)
        self.tokenizer = Tokenizer(espeak_config, vocab=self._load_vocab(vocab_config))

    @classmethod
    def from_session(
        cls,
        session: Any,
        voices_path: str | Path,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
    ) -> Kokoro:
        del session
        return cls(voices_path, espeak_config, vocab_config)

    def _load_vocab(self, vocab_config: dict | str | None) -> dict:
        if isinstance(vocab_config, str):
            with open(vocab_config, encoding="utf-8") as fp:
                config = json.load(fp)
                return config["vocab"]
        if isinstance(vocab_config, dict):
            return vocab_config["vocab"]
        return {}

    def get_voice_style(self, name: str) -> NDArray[np.float32]:
        return self.voices[name]

    def get_voices(self) -> list[str]:
        return sorted(self.voices.keys())


def trim_audio(
    samples: np.ndarray,
    *,
    top_db: float = 60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    use_jit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    samples_f32 = np.asarray(samples, dtype=np.float32)
    if samples_f32.ndim != 1:
        raise ValueError("trim_audio only supports mono audio")
    if samples_f32.size == 0:
        return samples_f32, np.asarray([0, 0])

    if use_jit:
        start, end = _trim_bounds_frame_rms_jit(
            samples_f32,
            top_db,
            frame_length,
            hop_length,
        )
    else:
        start, end = _trim_bounds_frame_rms_numpy(
            samples_f32,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    return samples_f32[start:end], np.asarray([start, end])


def _trim_bounds_frame_rms_numpy(
    samples: np.ndarray,
    *,
    top_db: float,
    frame_length: int,
    hop_length: int,
) -> tuple[int, int]:
    non_silent = _signal_to_frame_nonsilent(
        samples,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
    )
    nonzero = np.flatnonzero(non_silent)
    if nonzero.size == 0:
        return 0, 0
    start = int(nonzero[0] * hop_length)
    end = min(samples.shape[-1], int((nonzero[-1] + 1) * hop_length))
    return start, end


@njit(cache=True)
def _trim_bounds_frame_rms_jit(
    samples: np.ndarray,
    top_db: float,
    frame_length: int,
    hop_length: int,
) -> tuple[int, int]:
    sample_count = samples.shape[0]
    if sample_count == 0:
        return 0, 0

    pad = frame_length // 2
    padded_count = sample_count + (pad * 2)
    if padded_count < frame_length:
        return 0, 0

    frame_count = 1 + (padded_count - frame_length) // hop_length
    rms = np.empty(frame_count, dtype=np.float32)
    ref = 0.0

    for frame_index in range(frame_count):
        frame_start = frame_index * hop_length
        power = 0.0
        for offset in range(frame_length):
            padded_index = frame_start + offset
            sample_index = padded_index - pad
            value = 0.0
            if 0 <= sample_index < sample_count:
                value = samples[sample_index]
            power += value * value
        frame_rms = (power / frame_length) ** 0.5
        rms[frame_index] = frame_rms
        if frame_rms > ref:
            ref = frame_rms

    if ref <= 0.0:
        return 0, 0

    amin = 1e-5
    ref_value = ref if ref > amin else amin
    first = -1
    last = -1
    for frame_index in range(frame_count):
        magnitude = rms[frame_index]
        if magnitude < amin:
            magnitude = amin
        db = 20.0 * np.log10(magnitude / ref_value)
        if db > -top_db:
            if first < 0:
                first = frame_index
            last = frame_index

    if first < 0:
        return 0, 0

    start = first * hop_length
    end = (last + 1) * hop_length
    if end > sample_count:
        end = sample_count
    return start, end


def _signal_to_frame_nonsilent(
    samples: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    top_db: float,
) -> np.ndarray:
    rms = _rms(samples, frame_length=frame_length, hop_length=hop_length)
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    ref = float(np.max(rms))
    if ref <= 0.0:
        return np.zeros_like(rms, dtype=bool)
    db = 20.0 * np.log10(np.maximum(1e-5, rms) / max(1e-5, ref))
    return db > -top_db


def _rms(samples: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    padded = np.pad(samples, (frame_length // 2, frame_length // 2), mode="constant")
    if padded.shape[0] < frame_length:
        return np.zeros(0, dtype=np.float32)
    frame_count = 1 + (padded.shape[0] - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))


def _espeak_error_info() -> str:
    return (
        "Failed to load espeak-ng. Please install espeak-ng system wide or set "
        "PHONEMIZER_ESPEAK_LIBRARY. "
        f"Environment: {platform.platform()} ({platform.release()}) | {sys.version}"
    )
