from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass

import numpy as np

from fastkokoro.audio import encode_audio
from fastkokoro.engine import FastKokoro
from fastkokoro.streaming import (
    split_pcm_frames,
    split_phrases,
    split_scheduled_chunks,
    split_sentences,
)
from scripts.benchmark_corpus import corpus_choices, get_texts


@dataclass
class BenchmarkResult:
    strategy: str
    text_name: str
    text_chars: int
    chunks: int
    bytes: int
    first_chunk_latency_seconds: float
    total_latency_seconds: float
    active_providers: list[str]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="p")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", choices=corpus_choices(), default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-frame-ms", type=int, default=200)
    parser.add_argument("--json-array", action="store_true")
    args = parser.parse_args()

    engine = FastKokoro()
    if args.warmup:
        engine.warmup()

    selected_texts = (
        {args.text: get_texts(args.text)}
        if args.text
        else {name: get_texts(name) for name in corpus_choices()}
    )
    results = []
    for text_name, texts in selected_texts.items():
        for variant_index, text in enumerate(texts, start=1):
            variant_name = f"{text_name}:{variant_index}"
            measurements = (
            await measure(
                "kokoro_create_stream",
                variant_name,
                text,
                kokoro_create_stream(engine, text, args.voice, args.lang, args.speed),
                engine,
            ),
            await measure(
                "sentence_segments",
                variant_name,
                text,
                sentence_segment_stream(
                    engine, text, args.voice, args.lang, args.speed
                ),
                engine,
            ),
            await measure(
                f"phrase_segments_{args.audio_frame_ms}ms_frames",
                variant_name,
                text,
                framed_phrase_segment_stream(
                    engine,
                    text,
                    args.voice,
                    args.lang,
                    args.speed,
                    args.audio_frame_ms,
                ),
                engine,
            ),
            await measure(
                f"chunk_segments_{args.audio_frame_ms}ms_frames",
                variant_name,
                text,
                framed_chunk_segment_stream(
                    engine,
                    text,
                    args.voice,
                    args.lang,
                    args.speed,
                    args.audio_frame_ms,
                ),
                engine,
            ),
            await measure(
                f"sentence_segments_{args.audio_frame_ms}ms_frames",
                variant_name,
                text,
                framed_sentence_segment_stream(
                    engine,
                    text,
                    args.voice,
                    args.lang,
                    args.speed,
                    args.audio_frame_ms,
                ),
                engine,
            ),
        )
            for result in measurements:
                if args.json_array:
                    results.append(result)
                else:
                    print(json.dumps(asdict(result)), flush=True)

    if args.json_array:
        print(json.dumps([asdict(result) for result in results], indent=2))


async def measure(
    strategy: str,
    text_name: str,
    text: str,
    stream: AsyncGenerator[bytes, None],
    engine: FastKokoro,
) -> BenchmarkResult:
    start = time.perf_counter()
    first_chunk_latency = None
    chunks = 0
    bytes_count = 0
    async for chunk in stream:
        now = time.perf_counter()
        if first_chunk_latency is None:
            first_chunk_latency = now - start
        chunks += 1
        bytes_count += len(chunk)

    total_latency = time.perf_counter() - start
    return BenchmarkResult(
        strategy=strategy,
        text_name=text_name,
        text_chars=len(text),
        chunks=chunks,
        bytes=bytes_count,
        first_chunk_latency_seconds=first_chunk_latency or 0.0,
        total_latency_seconds=total_latency,
        active_providers=engine.session.get_providers(),
    )


async def sentence_segment_stream(
    engine: FastKokoro, text: str, voice: str, lang: str, speed: float
) -> AsyncGenerator[bytes, None]:
    for segment in split_sentences(text):
        yield engine.create(
            segment,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        )


async def framed_sentence_segment_stream(
    engine: FastKokoro,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    frame_ms: int,
) -> AsyncGenerator[bytes, None]:
    for segment in split_sentences(text):
        audio = engine.create(
            segment,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        )
        for frame in split_pcm_frames(audio, frame_ms):
            yield frame


async def framed_phrase_segment_stream(
    engine: FastKokoro,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    frame_ms: int,
) -> AsyncGenerator[bytes, None]:
    for segment in split_phrases(text):
        audio = engine.create(
            segment,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        )
        for frame in split_pcm_frames(audio, frame_ms):
            yield frame


async def framed_chunk_segment_stream(
    engine: FastKokoro,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    frame_ms: int,
) -> AsyncGenerator[bytes, None]:
    max_chars, max_words = engine._stream_schedule_limits()
    for segment in split_scheduled_chunks(
        text,
        initial_max_chars=engine.settings.stream_max_segment_chars,
        initial_max_words=engine.settings.stream_max_segment_words,
        max_chars=max_chars,
        max_words=max_words,
    ):
        audio = engine.create(
            segment,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        )
        for frame in split_pcm_frames(audio, frame_ms):
            yield frame


async def kokoro_create_stream(
    engine: FastKokoro, text: str, voice: str, lang: str, speed: float
) -> AsyncGenerator[bytes, None]:
    resolved_voice, resolved_lang = engine.resolve_request(voice, lang)
    stream = engine.kokoro.create_stream(
        text,
        voice=resolved_voice,
        speed=speed,
        lang=resolved_lang,
    )
    async for samples, sample_rate in stream:
        yield encode_audio(samples.astype(np.float32), sample_rate, "pcm")


if __name__ == "__main__":
    asyncio.run(main())
