from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_REPO = "nvidia/kokoro-82M-onnx-opt"
DEFAULT_MODEL_FILE = "kokoro-82m-v1.0.onnx"
DEFAULT_VOICES_FILE = "voices.bin"
DEFAULT_VOICES_INDEX_FILE = "voices.txt"
DEFAULT_VOICE = "af_heart"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8880
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class Settings:
    model_repo: str
    model_file: str
    voices_file: str
    voices_index_file: str
    model_path: Path | None
    voices_path: Path | None
    cache_dir: Path
    default_voice: str
    default_lang: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        model_path = os.getenv("FASTKOKORO_MODEL_PATH")
        voices_path = os.getenv("FASTKOKORO_VOICES_PATH")
        cache_dir = os.getenv("FASTKOKORO_CACHE_DIR")

        return cls(
            model_repo=os.getenv("FASTKOKORO_MODEL_REPO", DEFAULT_MODEL_REPO),
            model_file=os.getenv("FASTKOKORO_MODEL_FILE", DEFAULT_MODEL_FILE),
            voices_file=os.getenv("FASTKOKORO_VOICES_FILE", DEFAULT_VOICES_FILE),
            voices_index_file=os.getenv(
                "FASTKOKORO_VOICES_INDEX_FILE", DEFAULT_VOICES_INDEX_FILE
            ),
            model_path=Path(model_path).expanduser() if model_path else None,
            voices_path=Path(voices_path).expanduser() if voices_path else None,
            cache_dir=Path(cache_dir or "~/.cache/fastkokoro").expanduser(),
            default_voice=os.getenv("FASTKOKORO_DEFAULT_VOICE", DEFAULT_VOICE),
            default_lang=os.getenv("FASTKOKORO_DEFAULT_LANG", "en-us"),
            host=os.getenv("FASTKOKORO_HOST", DEFAULT_HOST),
            port=int(os.getenv("FASTKOKORO_PORT", str(DEFAULT_PORT))),
        )
