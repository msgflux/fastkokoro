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
DEFAULT_ONNX_INTRA_OP_NUM_THREADS = min(4, os.cpu_count() or 1)
DEFAULT_ONNX_INTER_OP_NUM_THREADS = 1
DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL = "all"
DEFAULT_ONNX_IO_BINDING = True
DEFAULT_ONNX_IO_BINDING_DEVICE = "auto"
DEFAULT_ONNX_WEIGHT_ONLY_NBITS = None
DEFAULT_ONNX_WEIGHT_ONLY_BLOCK_SIZE = 128
DEFAULT_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL = 4
DEFAULT_ONNX_WEIGHT_ONLY_SYMMETRIC = True
DEFAULT_WARMUP_TEXT = "hello"
DEFAULT_STREAM_STRATEGY = "phrase"
DEFAULT_STREAM_AUDIO_FRAME_MS = 200
SAMPLE_RATE = 24000
STREAM_STRATEGIES = {"kokoro", "phrase", "sentence"}
ONNX_GRAPH_OPTIMIZATION_LEVELS = {"disable", "basic", "extended", "all"}
ONNX_IO_BINDING_DEVICES = {"auto", "cpu", "cuda"}
ONNX_WEIGHT_ONLY_NBITS = {4, 8}


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
    onnx_graph_optimization_level: str
    onnx_io_binding: bool
    onnx_io_binding_device: str
    onnx_weight_only_nbits: int | None
    onnx_weight_only_block_size: int
    onnx_weight_only_accuracy_level: int
    onnx_weight_only_symmetric: bool
    warmup: bool
    warmup_text: str
    stream_strategy: str
    stream_audio_frame_ms: int

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
                os.getenv(
                    "FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS",
                    str(DEFAULT_ONNX_INTRA_OP_NUM_THREADS),
                )
            ),
            onnx_inter_op_num_threads=parse_optional_int(
                os.getenv(
                    "FASTKOKORO_ONNX_INTER_OP_NUM_THREADS",
                    str(DEFAULT_ONNX_INTER_OP_NUM_THREADS),
                )
            ),
            onnx_graph_optimization_level=parse_onnx_graph_optimization_level(
                os.getenv(
                    "FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL",
                    DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL,
                )
            ),
            onnx_io_binding=parse_bool(
                os.getenv("FASTKOKORO_ONNX_IO_BINDING"),
                default=DEFAULT_ONNX_IO_BINDING,
            ),
            onnx_io_binding_device=parse_onnx_io_binding_device(
                os.getenv(
                    "FASTKOKORO_ONNX_IO_BINDING_DEVICE",
                    DEFAULT_ONNX_IO_BINDING_DEVICE,
                )
            ),
            onnx_weight_only_nbits=parse_onnx_weight_only_nbits(
                os.getenv("FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS")
            ),
            onnx_weight_only_block_size=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_ONNX_WEIGHT_ONLY_BLOCK_SIZE",
                    str(DEFAULT_ONNX_WEIGHT_ONLY_BLOCK_SIZE),
                ),
                name="FASTKOKORO_ONNX_WEIGHT_ONLY_BLOCK_SIZE",
            ),
            onnx_weight_only_accuracy_level=parse_non_negative_int(
                os.getenv(
                    "FASTKOKORO_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL",
                    str(DEFAULT_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL),
                ),
                name="FASTKOKORO_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL",
            ),
            onnx_weight_only_symmetric=parse_bool(
                os.getenv("FASTKOKORO_ONNX_WEIGHT_ONLY_SYMMETRIC"),
                default=DEFAULT_ONNX_WEIGHT_ONLY_SYMMETRIC,
            ),
            warmup=parse_bool(os.getenv("FASTKOKORO_WARMUP"), default=True),
            warmup_text=os.getenv("FASTKOKORO_WARMUP_TEXT", DEFAULT_WARMUP_TEXT),
            stream_strategy=parse_stream_strategy(
                os.getenv("FASTKOKORO_STREAM_STRATEGY", DEFAULT_STREAM_STRATEGY)
            ),
            stream_audio_frame_ms=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_AUDIO_FRAME_MS",
                    str(DEFAULT_STREAM_AUDIO_FRAME_MS),
                ),
                name="FASTKOKORO_STREAM_AUDIO_FRAME_MS",
            ),
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


def parse_positive_int(value: str | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def parse_non_negative_int(value: str | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be zero or greater")
    return parsed


def parse_stream_strategy(value: str) -> str:
    parsed = value.strip().lower()
    if parsed not in STREAM_STRATEGIES:
        choices = ", ".join(sorted(STREAM_STRATEGIES))
        raise ValueError(f"FASTKOKORO_STREAM_STRATEGY must be one of: {choices}")
    return parsed


def parse_onnx_graph_optimization_level(value: str) -> str:
    parsed = value.strip().lower()
    if parsed not in ONNX_GRAPH_OPTIMIZATION_LEVELS:
        choices = ", ".join(sorted(ONNX_GRAPH_OPTIMIZATION_LEVELS))
        raise ValueError(
            f"FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL must be one of: {choices}"
        )
    return parsed


def parse_onnx_io_binding_device(value: str) -> str:
    parsed = value.strip().lower()
    if parsed not in ONNX_IO_BINDING_DEVICES:
        choices = ", ".join(sorted(ONNX_IO_BINDING_DEVICES))
        raise ValueError(f"FASTKOKORO_ONNX_IO_BINDING_DEVICE must be one of: {choices}")
    return parsed


def parse_onnx_weight_only_nbits(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return DEFAULT_ONNX_WEIGHT_ONLY_NBITS
    parsed = int(value)
    if parsed not in ONNX_WEIGHT_ONLY_NBITS:
        choices = ", ".join(str(choice) for choice in sorted(ONNX_WEIGHT_ONLY_NBITS))
        raise ValueError(f"FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS must be one of: {choices}")
    return parsed
