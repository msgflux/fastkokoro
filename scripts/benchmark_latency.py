from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, replace

from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro

TEXTS = {
    "tiny": "Ola.",
    "short": "Ola, tudo bem?",
    "medium": (
        "Ola, tudo bem? Este e um teste de sintese de voz em portugues brasileiro. "
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total de geracao."
    ),
}


@dataclass(frozen=True)
class LatencyRun:
    strategy: str
    text_name: str
    text_chars: int
    iteration: int
    chunks: int
    bytes: int
    first_chunk_latency_seconds: float
    total_latency_seconds: float
    intra_op_num_threads: int | None
    inter_op_num_threads: int | None
    active_providers: list[str]


@dataclass(frozen=True)
class LatencySummary:
    strategy: str
    text_name: str
    text_chars: int
    iterations: int
    first_chunk_avg_seconds: float
    first_chunk_p50_seconds: float
    first_chunk_min_seconds: float
    total_avg_seconds: float
    total_p50_seconds: float
    total_min_seconds: float
    chunks_avg: float
    bytes_avg: float
    intra_op_num_threads: int | None
    inter_op_num_threads: int | None
    active_providers: list[str]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="p")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", choices=TEXTS, default="short")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intra-op", type=int, default=None)
    parser.add_argument("--inter-op", type=int, default=None)
    parser.add_argument(
        "--ort-default-threads",
        action="store_true",
        help="Use ONNX Runtime's own default thread settings.",
    )
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()

    base_settings = Settings.from_env()
    settings = base_settings
    if args.ort_default_threads:
        settings = replace(
            settings,
            onnx_intra_op_num_threads=None,
            onnx_inter_op_num_threads=None,
        )
    if args.intra_op is not None:
        settings = replace(settings, onnx_intra_op_num_threads=args.intra_op)
    if args.inter_op is not None:
        settings = replace(settings, onnx_inter_op_num_threads=args.inter_op)
    text = TEXTS[args.text]
    engine = FastKokoro(settings)
    if args.warmup:
        engine.warmup()

    runs = []
    for strategy, runner in (
        ("create_pcm", create_pcm),
        ("stream_sentence", stream_sentence),
        ("stream_kokoro", stream_kokoro),
    ):
        for iteration in range(1, args.iterations + 1):
            run = await measure(
                strategy,
                args.text,
                text,
                iteration,
                runner(engine, text, args.voice, args.lang, args.speed),
                engine,
                settings,
            )
            runs.append(run)
            if args.json_lines:
                print(json.dumps(asdict(run)), flush=True)

    for summary in summarize(runs):
        print(json.dumps(asdict(summary)), flush=True)


async def measure(
    strategy: str,
    text_name: str,
    text: str,
    iteration: int,
    stream: AsyncGenerator[bytes, None],
    engine: FastKokoro,
    settings: Settings,
) -> LatencyRun:
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
    return LatencyRun(
        strategy=strategy,
        text_name=text_name,
        text_chars=len(text),
        iteration=iteration,
        chunks=chunks,
        bytes=bytes_count,
        first_chunk_latency_seconds=first_chunk_latency or 0.0,
        total_latency_seconds=total_latency,
        intra_op_num_threads=settings.onnx_intra_op_num_threads,
        inter_op_num_threads=settings.onnx_inter_op_num_threads,
        active_providers=engine.session.get_providers(),
    )


async def create_pcm(
    engine: FastKokoro, text: str, voice: str, lang: str, speed: float
) -> AsyncGenerator[bytes, None]:
    yield engine.create(
        text,
        voice=voice,
        lang=lang,
        speed=speed,
        response_format="pcm",
    )


async def stream_sentence(
    engine: FastKokoro, text: str, voice: str, lang: str, speed: float
) -> AsyncGenerator[bytes, None]:
    original_strategy = engine.settings.stream_strategy
    engine.settings = replace(engine.settings, stream_strategy="sentence")
    try:
        async for chunk in engine.create_stream(
            text,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        ):
            yield chunk
    finally:
        engine.settings = replace(engine.settings, stream_strategy=original_strategy)


async def stream_kokoro(
    engine: FastKokoro, text: str, voice: str, lang: str, speed: float
) -> AsyncGenerator[bytes, None]:
    original_strategy = engine.settings.stream_strategy
    engine.settings = replace(engine.settings, stream_strategy="kokoro")
    try:
        async for chunk in engine.create_stream(
            text,
            voice=voice,
            lang=lang,
            speed=speed,
            response_format="pcm",
        ):
            yield chunk
    finally:
        engine.settings = replace(engine.settings, stream_strategy=original_strategy)


def summarize(runs: list[LatencyRun]) -> list[LatencySummary]:
    grouped: dict[tuple[str, str], list[LatencyRun]] = {}
    for run in runs:
        grouped.setdefault((run.strategy, run.text_name), []).append(run)

    summaries = []
    for group in grouped.values():
        first_chunk = [run.first_chunk_latency_seconds for run in group]
        total = [run.total_latency_seconds for run in group]
        chunks = [run.chunks for run in group]
        bytes_count = [run.bytes for run in group]
        first = group[0]
        summaries.append(
            LatencySummary(
                strategy=first.strategy,
                text_name=first.text_name,
                text_chars=first.text_chars,
                iterations=len(group),
                first_chunk_avg_seconds=statistics.fmean(first_chunk),
                first_chunk_p50_seconds=statistics.median(first_chunk),
                first_chunk_min_seconds=min(first_chunk),
                total_avg_seconds=statistics.fmean(total),
                total_p50_seconds=statistics.median(total),
                total_min_seconds=min(total),
                chunks_avg=statistics.fmean(chunks),
                bytes_avg=statistics.fmean(bytes_count),
                intra_op_num_threads=first.intra_op_num_threads,
                inter_op_num_threads=first.inter_op_num_threads,
                active_providers=first.active_providers,
            )
        )
    return summaries


if __name__ == "__main__":
    asyncio.run(main())
