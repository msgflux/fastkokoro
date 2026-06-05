from __future__ import annotations

import logging
import re
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from kokoro_onnx import MAX_PHONEME_LENGTH, SAMPLE_RATE, Kokoro, trim_audio

from fastkokoro.assets import resolve_model_path, resolve_voices_path
from fastkokoro.audio import AudioFormat, encode_audio
from fastkokoro.config import Settings
from fastkokoro.graph_fusion import resolve_adain_fused_model_path
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


@dataclass
class OnnxInputBuffers:
    token_ids: np.ndarray
    speed_float32: np.ndarray
    speed_int32: np.ndarray


class FastKokoro:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_path = resolve_quantized_model_path(
            resolve_model_path(self.settings),
            self.settings,
        )
        self.model_path = resolve_adain_fused_model_path(self.model_path, self.settings)
        self.voices_path = resolve_voices_path(self.settings)
        self.session = create_session(self.model_path, self.settings)
        self.kokoro = Kokoro.from_session(self.session, str(self.voices_path))
        self._voices = tuple(self.kokoro.get_voices())
        self._voice_set = frozenset(self._voices)
        self._voice_styles = {
            voice: self.kokoro.get_voice_style(voice) for voice in self._voices
        }
        self._onnx_input_names = frozenset(
            item.name for item in self.session.get_inputs()
        )
        self._onnx_output_name = self.session.get_outputs()[0].name
        self._token_input_name = (
            "input_ids" if "input_ids" in self._onnx_input_names else "tokens"
        )
        self._onnx_input_buffers = threading.local()
        logger.info(
            "fastkokoro engine initialized: model_repo=%s model_file=%s "
            "model_path=%s voices_path=%s active_providers=%s "
            "default_voice=%s default_lang=%s warmup=%s stream_strategy=%s "
            "stream_audio_frame_ms=%s",
            self.settings.model_repo,
            self.settings.model_file,
            self.model_path,
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
        return encode_audio(samples, sample_rate, response_format)

    def _create_samples(
        self,
        text: str,
        *,
        voice: np.ndarray,
        speed: float,
        lang: str,
        is_phonemes: bool = False,
        trim: bool = True,
    ) -> tuple[np.ndarray, int]:
        assert 0.5 <= speed <= 2.0, "Speed should be between 0.5 and 2.0"

        phonemes = text if is_phonemes else self.kokoro.tokenizer.phonemize(text, lang)
        audio_parts = []
        for phoneme_batch in split_phonemes_for_model(phonemes):
            audio_part = self._run_onnx_audio(phoneme_batch, voice, speed)
            if trim:
                audio_part, _ = trim_audio(audio_part)
            audio_parts.append(audio_part)

        if not audio_parts:
            return np.array([], dtype=np.float32), SAMPLE_RATE
        return np.concatenate(audio_parts), SAMPLE_RATE

    def _run_onnx_audio(
        self,
        phonemes: str,
        voice: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        token_ids = self.kokoro.tokenizer.tokenize(phonemes)
        assert len(token_ids) <= MAX_PHONEME_LENGTH, (
            f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad "
            "token 0 at the start & end"
        )

        inputs = self._build_onnx_inputs(token_ids, voice, speed)
        if self.settings.onnx_io_binding:
            return self._run_onnx_audio_iobinding(inputs)
        return self.session.run(None, inputs)[0]

    def _build_onnx_inputs(
        self,
        token_ids: list[int],
        voice: np.ndarray,
        speed: float,
    ) -> dict[str, np.ndarray]:
        buffers = self._get_onnx_input_buffers()
        token_count = len(token_ids)
        buffers.token_ids[0, 0] = 0
        buffers.token_ids[0, 1 : token_count + 1] = token_ids
        buffers.token_ids[0, token_count + 1] = 0

        token_input = buffers.token_ids[:, : token_count + 2]
        style = voice[token_count]

        if self._token_input_name == "input_ids":
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

        return inputs

    def _run_onnx_audio_iobinding(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        device = self._resolve_iobinding_device()
        if device == "cuda":
            try:
                return self._run_onnx_audio_cuda_iobinding(inputs)
            except RuntimeError:
                logger.exception("CUDA IOBinding failed; falling back to CPU IOBinding")
                return self._run_onnx_audio_cpu_iobinding(inputs)
        return self._run_onnx_audio_cpu_iobinding(inputs)

    def _resolve_iobinding_device(self) -> str:
        configured = self.settings.onnx_io_binding_device
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            return "cuda"
        if "CUDAExecutionProvider" in self.session.get_providers():
            return "cuda"
        return "cpu"

    def _run_onnx_audio_cpu_iobinding(
        self, inputs: dict[str, np.ndarray]
    ) -> np.ndarray:
        binding = self.session.io_binding()
        for name, value in inputs.items():
            binding.bind_cpu_input(name, value)
        binding.bind_output(self._onnx_output_name)
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def _run_onnx_audio_cuda_iobinding(
        self, inputs: dict[str, np.ndarray]
    ) -> np.ndarray:
        binding = self.session.io_binding()
        for name, value in inputs.items():
            ortvalue = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
            binding.bind_ortvalue_input(name, ortvalue)
        binding.bind_output(self._onnx_output_name, device_type="cpu")
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def _get_onnx_input_buffers(self) -> OnnxInputBuffers:
        buffers = getattr(self._onnx_input_buffers, "buffers", None)
        if buffers is None:
            buffers = OnnxInputBuffers(
                token_ids=np.zeros((1, MAX_PHONEME_LENGTH + 2), dtype=np.int64),
                speed_float32=np.ones(1, dtype=np.float32),
                speed_int32=np.ones(1, dtype=np.int32),
            )
            self._onnx_input_buffers.buffers = buffers
        return buffers

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
            stream = self.kokoro.create_stream(
                text,
                voice=resolved_voice,
                speed=speed,
                lang=resolved_lang,
            )
            async for samples, sample_rate in stream:
                yield encode_audio(
                    samples.astype(np.float32),
                    sample_rate,
                    response_format,
                )
            return

        if self.settings.stream_strategy == "chunk":
            max_chars, max_words = self._stream_schedule_limits()
            segments = split_scheduled_chunks(
                text,
                initial_max_chars=self.settings.stream_max_segment_chars,
                initial_max_words=self.settings.stream_max_segment_words,
                max_chars=max_chars,
                max_words=max_words,
            )
        elif self.settings.stream_strategy == "phrase":
            segments = split_phrases(text)
        else:
            segments = split_sentences(text)

        for segment in segments:
            audio = self._create_resolved(
                segment,
                voice=resolved_voice,
                speed=speed,
                response_format=response_format,
                lang=resolved_lang,
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


def split_phonemes_for_model(phonemes: str) -> list[str]:
    parts = re.split(r"([.,!?;])", phonemes)
    batches: list[str] = []
    current_batch = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current_batch) + len(part) + 1 >= MAX_PHONEME_LENGTH:
            if current_batch:
                batches.append(current_batch.strip())
            current_batch = part
            continue
        if part in ".,!?;":
            current_batch += part
        else:
            if current_batch:
                current_batch += " "
            current_batch += part

    if current_batch:
        batches.append(current_batch.strip())
    return batches
