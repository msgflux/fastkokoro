from __future__ import annotations

import json
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
DEFAULT_ONNX_INTRA_OP_NUM_THREADS = min(6, os.cpu_count() or 1)
DEFAULT_ONNX_INTER_OP_NUM_THREADS = 1
DEFAULT_ONNX_GRAPH_OPTIMIZATION_LEVEL = "all"
DEFAULT_ONNX_LOG_SEVERITY_LEVEL = 3
DEFAULT_ONNX_IO_BINDING = True
DEFAULT_ONNX_IO_BINDING_DEVICE = "auto"
DEFAULT_ONNX_WEIGHT_ONLY_NBITS = None
DEFAULT_ONNX_WEIGHT_ONLY_BLOCK_SIZE = 128
DEFAULT_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL = 4
DEFAULT_ONNX_WEIGHT_ONLY_SYMMETRIC = True
DEFAULT_ONNX_ADAIN_FUSION = False
DEFAULT_ONNX_CONV_ADAIN_FUSION = False
DEFAULT_WARMUP_MULTI_SHAPE = False
DEFAULT_ONNX_TTFC_SHAPE_BUCKETS = (6, 7, 8, 9, 10, 11, 12, 16, 24)
DEFAULT_JIT = True
DEFAULT_WARMUP_TEXT = "hello"
DEFAULT_STREAM_STRATEGY = "chunk"
DEFAULT_STREAM_AUDIO_FRAME_MS = 200
DEFAULT_STREAM_MAX_SEGMENT_CHARS = 32
DEFAULT_STREAM_MAX_SEGMENT_WORDS = 2
DEFAULT_STREAM_SCHEDULE_MAX_SEGMENT_CHARS = 96
DEFAULT_STREAM_SCHEDULE_MAX_SEGMENT_WORDS = 12
DEFAULT_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS = 48
DEFAULT_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS = 4
DEFAULT_CORS_ALLOW_ORIGINS = ("*",)
DEFAULT_CORS_ALLOW_METHODS = ("GET", "POST", "OPTIONS")
DEFAULT_CORS_ALLOW_HEADERS = ("*",)
SAMPLE_RATE = 24000
STREAM_STRATEGIES = {"chunk", "kokoro", "phrase", "sentence"}
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
    onnx_provider_options: dict[str, dict[str, str]]
    onnx_auto_providers: bool
    onnx_intra_op_num_threads: int | None
    onnx_inter_op_num_threads: int | None
    onnx_graph_optimization_level: str
    onnx_log_severity_level: int
    onnx_io_binding: bool
    onnx_io_binding_device: str
    onnx_weight_only_nbits: int | None
    onnx_weight_only_block_size: int
    onnx_weight_only_accuracy_level: int
    onnx_weight_only_symmetric: bool
    onnx_adain_fusion: bool
    onnx_adain_model_path: Path | None
    onnx_adain_custom_op_library: Path | None
    onnx_conv_adain_fusion: bool
    onnx_conv_adain_model_path: Path | None
    onnx_conv_adain_custom_op_library: Path | None
    warmup_multi_shape: bool
    onnx_ttfc_shape_buckets: tuple[int, ...]
    jit: bool
    warmup: bool
    warmup_text: str
    stream_strategy: str
    stream_audio_frame_ms: int
    stream_max_segment_chars: int
    stream_max_segment_words: int
    stream_schedule_max_segment_chars: int
    stream_schedule_max_segment_words: int
    stream_cpu_schedule_max_segment_chars: int
    stream_cpu_schedule_max_segment_words: int
    cors_allow_origins: tuple[str, ...]
    cors_allow_methods: tuple[str, ...]
    cors_allow_headers: tuple[str, ...]
    cors_allow_credentials: bool

    @classmethod
    def from_env(cls) -> Settings:
        model_path = os.getenv("FASTKOKORO_MODEL_PATH")
        voices_path = os.getenv("FASTKOKORO_VOICES_PATH")
        cache_dir = os.getenv("FASTKOKORO_CACHE_DIR")
        providers = os.getenv("FASTKOKORO_ONNX_PROVIDERS")
        provider_options = os.getenv("FASTKOKORO_ONNX_PROVIDER_OPTIONS")
        adain_model_path = os.getenv("FASTKOKORO_ONNX_ADAIN_MODEL_PATH")
        adain_custom_op_library = os.getenv("FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY")
        conv_adain_model_path = os.getenv("FASTKOKORO_ONNX_CONV_ADAIN_MODEL_PATH")
        conv_adain_custom_op_library = os.getenv(
            "FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY"
        )
        cors_allow_origins = os.getenv("FASTKOKORO_CORS_ALLOW_ORIGINS")
        cors_allow_methods = os.getenv("FASTKOKORO_CORS_ALLOW_METHODS")
        cors_allow_headers = os.getenv("FASTKOKORO_CORS_ALLOW_HEADERS")

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
            onnx_provider_options=parse_provider_options(provider_options),
            onnx_auto_providers=parse_bool(
                os.getenv("FASTKOKORO_ONNX_AUTO_PROVIDERS"),
                default=True,
            ),
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
            onnx_log_severity_level=parse_onnx_log_severity_level(
                os.getenv(
                    "FASTKOKORO_ONNX_LOG_SEVERITY_LEVEL",
                    str(DEFAULT_ONNX_LOG_SEVERITY_LEVEL),
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
            onnx_adain_fusion=parse_bool(
                os.getenv("FASTKOKORO_ONNX_ADAIN_FUSION"),
                default=DEFAULT_ONNX_ADAIN_FUSION,
            ),
            onnx_adain_model_path=(
                Path(adain_model_path).expanduser() if adain_model_path else None
            ),
            onnx_adain_custom_op_library=(
                Path(adain_custom_op_library).expanduser()
                if adain_custom_op_library
                else None
            ),
            onnx_conv_adain_fusion=parse_bool(
                os.getenv("FASTKOKORO_ONNX_CONV_ADAIN_FUSION"),
                default=DEFAULT_ONNX_CONV_ADAIN_FUSION,
            ),
            onnx_conv_adain_model_path=(
                Path(conv_adain_model_path).expanduser()
                if conv_adain_model_path
                else None
            ),
            onnx_conv_adain_custom_op_library=(
                Path(conv_adain_custom_op_library).expanduser()
                if conv_adain_custom_op_library
                else None
            ),
            warmup_multi_shape=parse_bool(
                os.getenv("FASTKOKORO_WARMUP_MULTI_SHAPE"),
                default=DEFAULT_WARMUP_MULTI_SHAPE,
            ),
            onnx_ttfc_shape_buckets=parse_int_csv(
                os.getenv("FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS"),
                name="FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS",
            )
            or DEFAULT_ONNX_TTFC_SHAPE_BUCKETS,
            jit=parse_bool(
                os.getenv("FASTKOKORO_JIT"),
                default=DEFAULT_JIT,
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
            stream_max_segment_chars=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_MAX_SEGMENT_CHARS",
                    str(DEFAULT_STREAM_MAX_SEGMENT_CHARS),
                ),
                name="FASTKOKORO_STREAM_MAX_SEGMENT_CHARS",
            ),
            stream_max_segment_words=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_MAX_SEGMENT_WORDS",
                    str(DEFAULT_STREAM_MAX_SEGMENT_WORDS),
                ),
                name="FASTKOKORO_STREAM_MAX_SEGMENT_WORDS",
            ),
            stream_schedule_max_segment_chars=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_CHARS",
                    str(DEFAULT_STREAM_SCHEDULE_MAX_SEGMENT_CHARS),
                ),
                name="FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_CHARS",
            ),
            stream_schedule_max_segment_words=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_WORDS",
                    str(DEFAULT_STREAM_SCHEDULE_MAX_SEGMENT_WORDS),
                ),
                name="FASTKOKORO_STREAM_SCHEDULE_MAX_SEGMENT_WORDS",
            ),
            stream_cpu_schedule_max_segment_chars=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS",
                    str(DEFAULT_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS),
                ),
                name="FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_CHARS",
            ),
            stream_cpu_schedule_max_segment_words=parse_positive_int(
                os.getenv(
                    "FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS",
                    str(DEFAULT_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS),
                ),
                name="FASTKOKORO_STREAM_CPU_SCHEDULE_MAX_SEGMENT_WORDS",
            ),
            cors_allow_origins=parse_csv(cors_allow_origins)
            or DEFAULT_CORS_ALLOW_ORIGINS,
            cors_allow_methods=parse_csv(cors_allow_methods)
            or DEFAULT_CORS_ALLOW_METHODS,
            cors_allow_headers=parse_csv(cors_allow_headers)
            or DEFAULT_CORS_ALLOW_HEADERS,
            cors_allow_credentials=parse_bool(
                os.getenv("FASTKOKORO_CORS_ALLOW_CREDENTIALS")
            ),
        )


def parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_provider_options(value: str | None) -> dict[str, dict[str, str]]:
    if value is None or value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("FASTKOKORO_ONNX_PROVIDER_OPTIONS must be a JSON object")

    options: dict[str, dict[str, str]] = {}
    for provider, provider_options in parsed.items():
        if not isinstance(provider, str) or not isinstance(provider_options, dict):
            raise ValueError(
                "FASTKOKORO_ONNX_PROVIDER_OPTIONS must map provider names to objects"
            )
        options[provider] = {
            str(key): str(option_value)
            for key, option_value in provider_options.items()
        }
    return options


def parse_int_csv(value: str | None, *, name: str) -> tuple[int, ...]:
    if not value:
        return ()
    parsed = []
    for item in value.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        integer = int(candidate)
        if integer <= 0:
            raise ValueError(f"{name} must be positive")
        parsed.append(integer)
    return tuple(sorted(set(parsed)))


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


def parse_onnx_log_severity_level(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 4:
        raise ValueError("FASTKOKORO_ONNX_LOG_SEVERITY_LEVEL must be between 0 and 4")
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
