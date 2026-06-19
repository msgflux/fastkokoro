from __future__ import annotations

import logging
import re
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

from fastkokoro.assets import resolve_model_path, resolve_voices_path
from fastkokoro.audio import AudioFormat, encode_audio, trim_audio_part, trim_audio_tail
from fastkokoro.config import Settings
from fastkokoro.fixed_shape_experiments import resolve_ttfc_attention_mask_model_path
from fastkokoro.graph_fusion import (
    resolve_adain_fused_model_path,
    resolve_conv_adain_fused_model_path,
)
from fastkokoro.kokoro import MAX_PHONEME_LENGTH, SAMPLE_RATE, Kokoro
from fastkokoro.onnx import create_session
from fastkokoro.quantization import resolve_quantized_model_path
from fastkokoro.streaming import (
    split_pcm_frames,
    split_phrases,
    split_scheduled_chunks,
    split_sentences,
)
from fastkokoro.voices import normalize_language, validate_voice_language

logger = logging.getLogger("uvicorn.error")


def _require_ort():
    global ort
    if ort is None:
        try:
            import onnxruntime as runtime
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install `fastkokoro[gpu]` "
                "to use CUDA IOBinding."
            ) from exc
        ort = runtime
    return ort


def _resolve_static_input_width(inputs, input_name: str) -> int:
    for item in inputs:
        if item.name != input_name:
            continue
        shape = getattr(item, "shape", None)
        if shape is None or len(shape) < 2:
            return MAX_PHONEME_LENGTH + 2
        width = shape[1]
        if isinstance(width, int) and width > 0:
            return width
        return MAX_PHONEME_LENGTH + 2
    return MAX_PHONEME_LENGTH + 2


OUTPUT_BUFFER_POOL_SIZES = (8192, 16384, 32768, 65536)
PAUSE_TAG_PATTERN = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]", re.IGNORECASE)
PHONEME_PUNCTUATION = ".,!?;:\u2026\u2014"
PHONEME_BREAK_PRIORITY = ("!.?\u2026", ":;", ",\u2014")

_PHONEMIZE_CACHE_MAXSIZE = 128
TTFC_SYNTHETIC_PHONEME_PATTERNS = (
    "a",
    "i",
    "o",
    "m",
    "s",
    "t",
    "r",
    "la",
    "ma",
    "sa",
    "oi",
    "ai",
    "ou",
    "um",
)


@dataclass
class OnnxInputBuffers:
    token_ids: np.ndarray
    attention_mask: np.ndarray
    speed_float32: np.ndarray
    speed_int32: np.ndarray


@dataclass(frozen=True)
class OnnxSessionProfile:
    input_names: frozenset[str]
    output_name: str
    token_input_name: str
    token_input_width: int


@dataclass(frozen=True)
class TextControlSegment:
    text: str = ""
    pause_seconds: float | None = None


class FastKokoro:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_path = resolve_quantized_model_path(
            resolve_model_path(self.settings),
            self.settings,
        )
        self.model_path = resolve_adain_fused_model_path(self.model_path, self.settings)
        self.model_path = resolve_conv_adain_fused_model_path(
            self.model_path, self.settings
        )
        self.model_path = resolve_ttfc_attention_mask_model_path(
            self.model_path,
            self.settings,
        )
        self.voices_path = resolve_voices_path(self.settings)
        self.session = create_session(self.model_path, self.settings)
        self.ttfc_model_path = self.settings.onnx_ttfc_model_path or self.model_path
        self.ttfc_session = None
        if self.settings.onnx_ttfc_model_path is not None:
            self.ttfc_session = create_session(self.ttfc_model_path, self.settings)
        elif self.settings.onnx_ttfc_warm_session:
            self.ttfc_session = create_session(self.model_path, self.settings)
        self.kokoro = Kokoro.from_session(self.session, str(self.voices_path))
        self._voices = tuple(self.kokoro.get_voices())
        self._voice_set = frozenset(self._voices)
        self._voice_styles = {
            voice: self.kokoro.get_voice_style(voice) for voice in self._voices
        }
        self._onnx_profile = self._build_onnx_session_profile(self.session)
        self._ttfc_onnx_profile = (
            self._build_onnx_session_profile(self.ttfc_session)
            if self.ttfc_session is not None
            else None
        )
        self._onnx_input_names = self._onnx_profile.input_names
        self._onnx_output_name = self._onnx_profile.output_name
        self._token_input_name = self._onnx_profile.token_input_name
        self._token_input_width = self._onnx_profile.token_input_width
        self._onnx_input_buffers = threading.local()
        self._output_buffers = threading.local()
        self._phonemize_cache: dict[tuple[str, str], str] = {}
        self._warm_ttfc_shape_buckets()
        self._warm_ttfc_session()
        logger.info(
            "fastkokoro engine initialized: model_repo=%s model_file=%s "
            "model_path=%s ttfc_model_path=%s voices_path=%s active_providers=%s "
            "default_voice=%s default_lang=%s warmup=%s stream_strategy=%s "
            "stream_audio_frame_ms=%s",
            self.settings.model_repo,
            self.settings.model_file,
            self.model_path,
            self.ttfc_model_path if self.ttfc_session is not None else None,
            self.voices_path,
            self.session.get_providers(),
            self.settings.default_voice,
            self.settings.default_lang,
            self.settings.warmup,
            self.settings.stream_strategy,
            self.settings.stream_audio_frame_ms,
        )

    def voices(self) -> list[str]:
        return list(self._voices)

    def _build_onnx_session_profile(self, session) -> OnnxSessionProfile:
        input_names = frozenset(item.name for item in session.get_inputs())
        token_input_name = "input_ids" if "input_ids" in input_names else "tokens"
        return OnnxSessionProfile(
            input_names=input_names,
            output_name=session.get_outputs()[0].name,
            token_input_name=token_input_name,
            token_input_width=_resolve_static_input_width(
                session.get_inputs(),
                token_input_name,
            ),
        )

    def _onnx_profile_for_session(self, session) -> OnnxSessionProfile:
        if session is self.ttfc_session and self._ttfc_onnx_profile is not None:
            return self._ttfc_onnx_profile
        return self._onnx_profile

    def _phonemize_cached(self, text: str, lang: str) -> str:
        key = (text, lang)
        cache = self._phonemize_cache
        if key in cache:
            return cache[key]
        result = self.kokoro.tokenizer.phonemize(text, lang)
        if len(cache) >= _PHONEMIZE_CACHE_MAXSIZE:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result

    def warmup(self) -> None:
        self.create(
            self.settings.warmup_text,
            voice=self.settings.default_voice,
            response_format="pcm",
            lang=self.settings.default_lang,
        )

    def resolve_request(self, voice: str | None, lang: str | None) -> tuple[str, str]:
        resolved_voice = voice or self.settings.default_voice
        resolved_lang = normalize_language(
            lang, resolved_voice, self.settings.default_lang
        )
        validate_voice_language(resolved_voice, resolved_lang, self._voice_set)
        return resolved_voice, resolved_lang

    def create(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: AudioFormat = "mp3",
        lang: str | None = None,
    ) -> bytes:
        resolved_voice, resolved_lang = self.resolve_request(voice, lang)

        return self._create_resolved(
            text,
            voice=resolved_voice,
            speed=speed,
            response_format=response_format,
            lang=resolved_lang,
        )

    def _create_resolved(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        response_format: AudioFormat,
        lang: str,
    ) -> bytes:
        samples, sample_rate = self._create_samples(
            text,
            voice=self._voice_styles[voice],
            speed=speed,
            lang=lang,
        )
        return encode_audio(
            samples,
            sample_rate,
            response_format,
            use_pcm_jit=self.settings.jit,
        )

    def _create_samples(
        self,
        text: str,
        *,
        voice: np.ndarray,
        speed: float,
        lang: str,
        is_phonemes: bool = False,
        trim: bool = True,
        session=None,
    ) -> tuple[np.ndarray, int]:
        assert 0.5 <= speed <= 2.0, "Speed should be between 0.5 and 2.0"

        segments = (
            [TextControlSegment(text=text)]
            if is_phonemes
            else split_text_control_segments(text)
        )
        if not segments:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        initial_size = self._select_output_buffer_size(len(segments) * 4096)
        merged = self._acquire_output_buffer(initial_size)
        merged_length = 0
        for segment in segments:
            if segment.pause_seconds is not None:
                audio_parts = [silence_samples(segment.pause_seconds)]
            else:
                phonemes = (
                    segment.text
                    if is_phonemes
                    else self._phonemize_cached(segment.text, lang)
                )
                audio_parts = []
                for phoneme_batch in split_phonemes_for_model(phonemes):
                    profile = self._onnx_profile_for_session(session or self.session)
                    for onnx_batch in self._split_for_onnx_token_width(
                        phoneme_batch,
                        profile=profile,
                    ):
                        audio_parts.append(
                            self._run_onnx_audio(
                                onnx_batch,
                                voice,
                                speed,
                                session=session,
                            )
                        )

            for audio_part in audio_parts:
                if trim and segment.pause_seconds is None:
                    audio_part = self._trim_audio_part(audio_part)
                required = merged_length + len(audio_part)
                if required > len(merged):
                    merged = self._grow_output_buffer(merged, merged_length, required)
                merged[merged_length:required] = audio_part
                merged_length = required

        return merged[:merged_length], SAMPLE_RATE

    def _trim_audio_part(self, audio_part: np.ndarray) -> np.ndarray:
        return trim_audio_part(audio_part, use_jit=self.settings.jit)

    def _run_onnx_audio(
        self,
        phonemes: str,
        voice: np.ndarray,
        speed: float,
        *,
        session=None,
    ) -> np.ndarray:
        session = session or self.session
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        token_ids = self.kokoro.tokenizer.tokenize(phonemes)
        assert len(token_ids) <= MAX_PHONEME_LENGTH, (
            f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad "
            "token 0 at the start & end"
        )

        profile = self._onnx_profile_for_session(session)
        inputs = self._build_onnx_inputs(token_ids, voice, speed, profile=profile)
        if self.settings.onnx_io_binding:
            audio = self._run_onnx_audio_iobinding(
                inputs,
                session=session,
                output_name=profile.output_name,
            )
        else:
            audio = session.run(None, inputs)[0]
        return trim_audio_tail(
            audio,
            sample_rate=SAMPLE_RATE,
            trim_ms=self.settings.runtime_tail_trim_ms,
            fade_ms=self.settings.runtime_tail_fade_ms,
        )

    def _build_onnx_inputs(
        self,
        token_ids: list[int],
        voice: np.ndarray,
        speed: float,
        *,
        profile: OnnxSessionProfile | None = None,
    ) -> dict[str, np.ndarray]:
        profile = profile or self._onnx_profile
        buffers = self._get_onnx_input_buffers()
        token_count = len(token_ids)
        token_width = (
            profile.token_input_width
            if "attention_mask" in profile.input_names
            else token_count + 2
        )
        if token_width > buffers.token_ids.shape[1]:
            raise ValueError(
                f"ONNX token input width {token_width} exceeds buffer width "
                f"{buffers.token_ids.shape[1]}"
            )
        if token_count + 2 > token_width:
            raise ValueError(
                f"Tokenized input length {token_count + 2} exceeds fixed ONNX "
                f"input width {token_width}"
            )

        buffers.token_ids[:, :token_width] = 0
        buffers.attention_mask[:, :token_width] = 0
        buffers.token_ids[0, 0] = 0
        buffers.token_ids[0, 1 : token_count + 1] = token_ids
        buffers.token_ids[0, token_count + 1] = 0
        buffers.attention_mask[0, : token_count + 2] = 1

        token_input = buffers.token_ids[:, :token_width]
        style = voice[token_count]

        if profile.token_input_name == "input_ids":
            buffers.speed_int32[0] = speed
            inputs = {
                "input_ids": token_input,
                "style": np.array(style, dtype=np.float32),
                "speed": buffers.speed_int32,
            }
        else:
            buffers.speed_float32[0] = speed
            inputs = {
                "tokens": token_input,
                "style": style,
                "speed": buffers.speed_float32,
            }

        if "attention_mask" in profile.input_names:
            inputs["attention_mask"] = buffers.attention_mask[:, :token_width]

        return inputs

    def _split_for_onnx_token_width(
        self,
        phonemes: str,
        *,
        profile: OnnxSessionProfile | None = None,
    ) -> list[str]:
        profile = profile or self._onnx_profile
        if "attention_mask" not in profile.input_names:
            return [phonemes]

        max_tokens = profile.token_input_width - 2
        if max_tokens <= 0:
            raise ValueError("Fixed ONNX token input width must leave room for pads")
        if len(self.kokoro.tokenizer.tokenize(phonemes)) <= max_tokens:
            return [phonemes]

        batches: list[str] = []
        current = ""
        for piece in phonemes.split():
            candidate = piece if not current else f"{current} {piece}"
            if len(self.kokoro.tokenizer.tokenize(candidate)) <= max_tokens:
                current = candidate
                continue
            if current:
                batches.append(current)
                current = ""
            if len(self.kokoro.tokenizer.tokenize(piece)) <= max_tokens:
                current = piece
            else:
                batches.extend(self._split_oversized_token_piece(piece, max_tokens))

        if current:
            batches.append(current)
        return batches

    def _split_oversized_token_piece(self, piece: str, max_tokens: int) -> list[str]:
        batches: list[str] = []
        current = ""
        for char in piece:
            candidate = current + char
            if len(self.kokoro.tokenizer.tokenize(candidate)) <= max_tokens:
                current = candidate
                continue
            if current:
                batches.append(current)
            current = char
        if current:
            batches.append(current)
        return batches

    def _can_warm_token_count(
        self,
        token_count: int,
        voice: np.ndarray,
        *,
        profile: OnnxSessionProfile | None = None,
    ) -> bool:
        profile = profile or self._onnx_profile
        if token_count <= 0 or token_count > MAX_PHONEME_LENGTH:
            return False
        if token_count + 2 > profile.token_input_width:
            return False
        return token_count < len(voice)

    def _warm_ttfc_shape_buckets(self) -> None:
        if not self.settings.warmup_multi_shape:
            return
        if not self.settings.onnx_ttfc_shape_buckets:
            return

        voice = self._voice_styles[self.settings.default_voice]
        lang = self.settings.default_lang
        warmed: list[int] = []

        for bucket in self.settings.onnx_ttfc_shape_buckets:
            token_count = bucket - 2
            if not self._can_warm_token_count(
                token_count,
                voice,
                profile=self._onnx_profile,
            ):
                continue
            inputs = self._build_onnx_inputs(
                [0] * token_count,
                voice,
                1.0,
                profile=self._onnx_profile,
            )
            self.session.run(None, inputs)
            warmed.append(bucket)

        if self.settings.stream_strategy in {"chunk", "phrase", "sentence"}:
            strategy_buckets = self._warm_streaming_first_segments(voice, lang)
            warmed.extend(strategy_buckets)

        if warmed:
            logger.info("Warmed ONNX TTFC shape buckets: buckets=%s", warmed)

    def _warm_ttfc_session(self) -> None:
        if self.ttfc_session is None:
            return
        if not self.settings.onnx_ttfc_shape_buckets:
            return

        voice = self._voice_styles[self.settings.default_voice]
        profile = self._ttfc_onnx_profile or self._onnx_profile
        warmed: list[int] = []
        for bucket in self.settings.onnx_ttfc_shape_buckets:
            token_count = bucket - 2
            if not self._can_warm_token_count(token_count, voice, profile=profile):
                continue
            inputs = self._build_onnx_inputs(
                [0] * token_count,
                voice,
                1.0,
                profile=profile,
            )
            self.ttfc_session.run(None, inputs)
            warmed.append(bucket)

        for text in self.settings.onnx_ttfc_warm_texts:
            try:
                phonemes = self._phonemize_cached(
                    text,
                    self.settings.default_lang,
                )
                for batch in split_phonemes_for_model(phonemes):
                    for onnx_batch in self._split_for_onnx_token_width(
                        batch,
                        profile=profile,
                    ):
                        token_ids = self.kokoro.tokenizer.tokenize(onnx_batch)
                        if not self._can_warm_token_count(
                            len(token_ids),
                            voice,
                            profile=profile,
                        ):
                            continue
                        inputs = self._build_onnx_inputs(
                            token_ids,
                            voice,
                            1.0,
                            profile=profile,
                        )
                        self.ttfc_session.run(None, inputs)
            except Exception:
                logger.exception("Failed to warm TTFC session sample text: %s", text)

        warmed_signatures = self._warm_ttfc_token_signatures(voice, profile)
        warmed.extend(warmed_signatures)

        if warmed:
            logger.info("Warmed dedicated ONNX TTFC session: buckets=%s", warmed)

    def _warm_ttfc_token_signatures(
        self,
        voice: np.ndarray,
        profile: OnnxSessionProfile,
    ) -> list[int]:
        warmed: list[int] = []
        for token_count in self.settings.onnx_ttfc_warm_token_counts:
            if not self._can_warm_token_count(token_count, voice, profile=profile):
                continue
            for token_ids in self._synthetic_token_signatures(token_count):
                if not self._can_warm_token_count(
                    len(token_ids),
                    voice,
                    profile=profile,
                ):
                    continue
                inputs = self._build_onnx_inputs(
                    token_ids,
                    voice,
                    1.0,
                    profile=profile,
                )
                self.ttfc_session.run(None, inputs)
                if token_count not in warmed:
                    warmed.append(token_count)
        return warmed

    def _synthetic_token_signatures(self, token_count: int) -> list[list[int]]:
        signatures: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for pattern in TTFC_SYNTHETIC_PHONEME_PATTERNS:
            phonemes = self._repeat_phoneme_pattern(pattern, token_count)
            token_ids = self.kokoro.tokenizer.tokenize(phonemes)
            if len(token_ids) != token_count:
                continue
            key = tuple(token_ids)
            if key in seen:
                continue
            seen.add(key)
            signatures.append(token_ids)
            if len(signatures) >= 4:
                break
        return signatures

    def _repeat_phoneme_pattern(self, pattern: str, token_count: int) -> str:
        repeated = (pattern * ((token_count // len(pattern)) + 1))[:token_count]
        return repeated

    def _warm_streaming_first_segments(self, voice: np.ndarray, lang: str) -> list[int]:
        warmed: list[int] = []
        sample_texts = [
            "Ola,",
            "Hello,",
            "Hola,",
            "Bonjour,",
            "Ciao,",
        ]
        for text in sample_texts:
            try:
                phonemes = self.kokoro.tokenizer.phonemize(text, lang)
                batches = split_phonemes_for_model(phonemes)
                for batch in batches:
                    tokens = self.kokoro.tokenizer.tokenize(batch)
                    token_count = len(tokens)
                    if not self._can_warm_token_count(
                        token_count,
                        voice,
                        profile=self._onnx_profile,
                    ):
                        continue
                    inputs = self._build_onnx_inputs(
                        [0] * token_count,
                        voice,
                        1.0,
                        profile=self._onnx_profile,
                    )
                    self.session.run(None, inputs)
                    bucket = token_count + 2
                    if bucket not in warmed:
                        warmed.append(bucket)
            except Exception:
                continue
        return warmed

    def _run_onnx_audio_iobinding(
        self,
        inputs: dict[str, np.ndarray],
        *,
        session=None,
        output_name: str | None = None,
    ) -> np.ndarray:
        session = session or self.session
        output_name = output_name or self._onnx_profile_for_session(session).output_name
        device = self._resolve_iobinding_device(session=session)
        if device == "cuda":
            try:
                return self._run_onnx_audio_cuda_iobinding(
                    inputs,
                    session=session,
                    output_name=output_name,
                )
            except RuntimeError:
                logger.exception("CUDA IOBinding failed; falling back to CPU IOBinding")
                return self._run_onnx_audio_cpu_iobinding(
                    inputs,
                    session=session,
                    output_name=output_name,
                )
        return self._run_onnx_audio_cpu_iobinding(
            inputs,
            session=session,
            output_name=output_name,
        )

    def _resolve_iobinding_device(self, *, session=None) -> str:
        session = session or self.session
        configured = self.settings.onnx_io_binding_device
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            return "cuda"
        if "CUDAExecutionProvider" in session.get_providers():
            return "cuda"
        return "cpu"

    def _run_onnx_audio_cpu_iobinding(
        self,
        inputs: dict[str, np.ndarray],
        *,
        session=None,
        output_name: str | None = None,
    ) -> np.ndarray:
        session = session or self.session
        output_name = output_name or self._onnx_profile_for_session(session).output_name
        binding = session.io_binding()
        for name, value in inputs.items():
            binding.bind_cpu_input(name, value)
        binding.bind_output(output_name)
        session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def _run_onnx_audio_cuda_iobinding(
        self,
        inputs: dict[str, np.ndarray],
        *,
        session=None,
        output_name: str | None = None,
    ) -> np.ndarray:
        session = session or self.session
        output_name = output_name or self._onnx_profile_for_session(session).output_name
        runtime = _require_ort()
        binding = session.io_binding()
        for name, value in inputs.items():
            ortvalue = runtime.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
            binding.bind_ortvalue_input(name, ortvalue)
        binding.bind_output(output_name, device_type="cpu")
        session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def _get_onnx_input_buffers(self) -> OnnxInputBuffers:
        buffers = getattr(self._onnx_input_buffers, "buffers", None)
        if buffers is None:
            buffers = OnnxInputBuffers(
                token_ids=np.zeros((1, MAX_PHONEME_LENGTH + 2), dtype=np.int64),
                attention_mask=np.zeros(
                    (1, MAX_PHONEME_LENGTH + 2),
                    dtype=np.int64,
                ),
                speed_float32=np.ones(1, dtype=np.float32),
                speed_int32=np.ones(1, dtype=np.int32),
            )
            self._onnx_input_buffers.buffers = buffers
        return buffers

    def _get_output_buffer_pool(self) -> dict[int, np.ndarray]:
        pool = getattr(self._output_buffers, "buffers", None)
        if pool is None:
            pool = {}
            self._output_buffers.buffers = pool
        return pool

    def _select_output_buffer_size(self, required: int) -> int:
        for size in OUTPUT_BUFFER_POOL_SIZES:
            if size >= required:
                return size
        return 1 << (required - 1).bit_length()

    def _acquire_output_buffer(self, size: int) -> np.ndarray:
        pool = self._get_output_buffer_pool()
        buffer = pool.get(size)
        if buffer is None or len(buffer) < size:
            buffer = np.empty(size, dtype=np.float32)
            pool[size] = buffer
        return buffer

    def _grow_output_buffer(
        self, source: np.ndarray, source_length: int, required: int
    ) -> np.ndarray:
        target_size = self._select_output_buffer_size(required)
        grown = self._acquire_output_buffer(target_size)
        grown[:source_length] = source[:source_length]
        return grown

    async def create_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: AudioFormat = "pcm",
        lang: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        resolved_voice, resolved_lang = self.resolve_request(voice, lang)

        if self.settings.stream_strategy == "kokoro":
            for segment in split_text_control_segments(text):
                if segment.pause_seconds is not None:
                    audio = encode_audio(
                        silence_samples(segment.pause_seconds),
                        SAMPLE_RATE,
                        response_format,
                        use_pcm_jit=self.settings.jit,
                    )
                    if response_format == "pcm":
                        for frame in split_pcm_frames(
                            audio,
                            self.settings.stream_audio_frame_ms,
                        ):
                            yield frame
                    else:
                        yield audio
                    continue

                samples, sample_rate = self._create_samples(
                    segment.text,
                    voice=self._voice_styles[resolved_voice],
                    speed=speed,
                    lang=resolved_lang,
                )
                yield encode_audio(
                    samples.astype(np.float32),
                    sample_rate,
                    response_format,
                    use_pcm_jit=self.settings.jit,
                )
            return

        used_ttfc_session = False
        for segment in self._stream_text_control_segments(text):
            if segment.pause_seconds is None:
                ttfc_session = self.ttfc_session if not used_ttfc_session else None
                if ttfc_session is not None:
                    samples, sample_rate = self._create_samples(
                        segment.text,
                        voice=self._voice_styles[resolved_voice],
                        speed=speed,
                        lang=resolved_lang,
                        session=ttfc_session,
                    )
                    used_ttfc_session = True
                    audio = encode_audio(
                        samples,
                        sample_rate,
                        response_format,
                        use_pcm_jit=self.settings.jit,
                    )
                else:
                    audio = self._create_resolved(
                        segment.text,
                        voice=resolved_voice,
                        speed=speed,
                        response_format=response_format,
                        lang=resolved_lang,
                    )
            else:
                audio = encode_audio(
                    silence_samples(segment.pause_seconds),
                    SAMPLE_RATE,
                    response_format,
                    use_pcm_jit=self.settings.jit,
                )
            if response_format != "pcm":
                yield audio
                continue

            for frame in split_pcm_frames(
                audio,
                self.settings.stream_audio_frame_ms,
            ):
                yield frame

    def _stream_schedule_limits(self) -> tuple[int, int]:
        providers = set(self.session.get_providers())
        if {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & providers:
            return (
                self.settings.stream_schedule_max_segment_chars,
                self.settings.stream_schedule_max_segment_words,
            )
        return (
            self.settings.stream_cpu_schedule_max_segment_chars,
            self.settings.stream_cpu_schedule_max_segment_words,
        )

    def _stream_text_control_segments(self, text: str) -> list[TextControlSegment]:
        segments: list[TextControlSegment] = []
        for segment in split_text_control_segments(text):
            if segment.pause_seconds is not None:
                segments.append(segment)
                continue

            if self.settings.stream_strategy == "chunk":
                max_chars, max_words = self._stream_schedule_limits()
                text_segments = split_scheduled_chunks(
                    segment.text,
                    initial_max_chars=self.settings.stream_max_segment_chars,
                    initial_max_words=self.settings.stream_max_segment_words,
                    max_chars=max_chars,
                    max_words=max_words,
                )
            elif self.settings.stream_strategy == "phrase":
                text_segments = split_phrases(segment.text)
            elif self.settings.stream_strategy == "adaptive":
                text_segments = []
                providers = set(self.session.get_providers())
                has_gpu = bool(
                    {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & providers
                )
                adaptive_max = (
                    self.settings.stream_adaptive_max_chars
                    if has_gpu
                    else self.settings.stream_adaptive_cpu_max_chars
                )
                for sentence in split_sentences(segment.text):
                    if len(sentence) <= adaptive_max:
                        text_segments.append(sentence)
                    else:
                        text_segments.extend(split_phrases(sentence))
            else:
                text_segments = split_sentences(segment.text)

            segments.extend(TextControlSegment(text=item) for item in text_segments)
        return segments


def silence_samples(seconds: float) -> np.ndarray:
    sample_count = max(0, int(seconds * SAMPLE_RATE))
    return np.zeros(sample_count, dtype=np.float32)


def split_text_control_segments(text: str) -> list[TextControlSegment]:
    segments: list[TextControlSegment] = []
    parts = PAUSE_TAG_PATTERN.split(text)
    for index, part in enumerate(parts):
        if index % 2 == 0:
            stripped = part.strip()
            if stripped:
                segments.append(TextControlSegment(text=stripped))
            continue

        seconds = float(part)
        if seconds > 0:
            segments.append(TextControlSegment(pause_seconds=seconds))
    return segments


def split_phonemes_for_model(phonemes: str) -> list[str]:
    parts = re.split(r"([.,!?;:\u2026\u2014])", phonemes)
    batches: list[str] = []
    current_batch = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current_batch) + len(part) + 1 >= MAX_PHONEME_LENGTH:
            if current_batch:
                batches.extend(_split_oversized_phoneme_batch(current_batch.strip()))
                current_batch = part
            else:
                batches.extend(_split_oversized_phoneme_batch(part.strip()))
                current_batch = ""
            continue
        if part in PHONEME_PUNCTUATION:
            current_batch += part
        else:
            if current_batch:
                current_batch += " "
            current_batch += part

    if current_batch:
        batches.extend(_split_oversized_phoneme_batch(current_batch.strip()))
    return batches


def _split_oversized_phoneme_batch(batch: str) -> list[str]:
    if len(batch) <= MAX_PHONEME_LENGTH:
        return [batch]

    output: list[str] = []
    remaining = batch
    while len(remaining) > MAX_PHONEME_LENGTH:
        boundary = _find_phoneme_split_boundary(remaining, MAX_PHONEME_LENGTH)
        output.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        output.append(remaining)
    return output


def _find_phoneme_split_boundary(text: str, limit: int) -> int:
    window = text[:limit]
    for punctuation_group in PHONEME_BREAK_PRIORITY:
        boundary = max(window.rfind(char) for char in punctuation_group)
        if boundary >= 0:
            return boundary + 1
    whitespace = window.rfind(" ")
    if whitespace > 0:
        return whitespace + 1
    return limit
