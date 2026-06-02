from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_REPO = "nvidia/kokoro-82M-onnx-opt"
DEFAULT_MODEL_FILE = "kokoro-82m-v1.0.onnx"
DEFAULT_VOICES_FILE = "voices.bin"
DEFAULT_VOICES_INDEX_FILE = "voices.txt"
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "en-us"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8880
DEFAULT_ONNX_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_WARMUP_TEXT = "hello"
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
    onnx_providers: tuple[str, ...]
    onnx_auto_providers: bool
    onnx_intra_op_num_threads: int | None
    onnx_inter_op_num_threads: int | None
    warmup: bool
    warmup_text: str

    @classmethod
    def from_env(cls) -> Settings:
        model_path = os.getenv("FASTKOKORO_MODEL_PATH")
        voices_path = os.getenv("FASTKOKORO_VOICES_PATH")
        cache_dir = os.getenv("FASTKOKORO_CACHE_DIR")
        providers = os.getenv("FASTKOKORO_ONNX_PROVIDERS")

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
            default_lang=os.getenv("FASTKOKORO_DEFAULT_LANG", DEFAULT_LANG),
            host=os.getenv("FASTKOKORO_HOST", DEFAULT_HOST),
            port=int(os.getenv("FASTKOKORO_PORT", str(DEFAULT_PORT))),
            onnx_providers=parse_csv(providers) or DEFAULT_ONNX_PROVIDERS,
            onnx_auto_providers=parse_bool(os.getenv("FASTKOKORO_ONNX_AUTO_PROVIDERS")),
            onnx_intra_op_num_threads=parse_optional_int(
                os.getenv("FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS")
            ),
            onnx_inter_op_num_threads=parse_optional_int(
                os.getenv("FASTKOKORO_ONNX_INTER_OP_NUM_THREADS")
            ),
            warmup=parse_bool(os.getenv("FASTKOKORO_WARMUP"), default=True),
            warmup_text=os.getenv("FASTKOKORO_WARMUP_TEXT", DEFAULT_WARMUP_TEXT),
        )


def parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)
