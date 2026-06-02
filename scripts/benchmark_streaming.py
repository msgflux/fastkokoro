from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import asdict, dataclass

from fastkokoro.engine import FastKokoro

TEXTS = {
    "short": "Ola, tudo bem?",
    "medium": (
        "Ola, tudo bem? Este e um teste de sintese de voz em portugues brasileiro. "
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total de geracao."
    ),
    "long": (
        "Ola, tudo bem? Este e um teste de sintese de voz em portugues brasileiro. "
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total de geracao. "
        "Para streaming em uma interface de terminal, o ideal e entregar audio cedo, "
        "sem esperar o texto inteiro ser processado. Por isso este benchmark compara "
        "a estrategia atual do kokoro onnx com alternativas baseadas em segmentacao "
        "de sentencas e frames de audio no lado do servidor."
    ),
}


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
    parser.add_argument("--text", choices=TEXTS, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-frame-ms", type=int, default=200)
    parser.add_argument("--json-array", action="store_true")
    args = parser.parse_args()

    engine = FastKokoro()
    if args.warmup:
        engine.warmup()

    selected_texts = {args.text: TEXTS[args.text]} if args.text else TEXTS
    results = []
    for text_name, text in selected_texts.items():
        measurements = (
            await measure(
                "kokoro_create_stream",
                text_name,
                text,
                engine.create_stream(
                    text,
                    voice=args.voice,
                    lang=args.lang,
                    speed=args.speed,
                    response_format="pcm",
                ),
                engine,
            ),
            await measure(
                "sentence_segments",
                text_name,
                text,
                sentence_segment_stream(
                    engine, text, args.voice, args.lang, args.speed
                ),
                engine,
            ),
            await measure(
                f"sentence_segments_{args.audio_frame_ms}ms_frames",
                text_name,
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


def split_sentences(text: str) -> list[str]:
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text)]
    return [segment for segment in segments if segment]


def split_pcm_frames(audio: bytes, frame_ms: int) -> Iterable[bytes]:
    sample_rate = 24000
    bytes_per_sample = 2
    frame_size = max(1, int(sample_rate * bytes_per_sample * frame_ms / 1000))
    frame_size -= frame_size % bytes_per_sample
    for index in range(0, len(audio), frame_size):
        yield audio[index : index + frame_size]


if __name__ == "__main__":
    asyncio.run(main())
