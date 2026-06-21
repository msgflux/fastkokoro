#!/usr/bin/env python3
"""
TTFC Benchmark - PCM output. Works on CPU and GPU.

Usage:
  # GPU (Colab/Docker)
  python scripts/benchmark_gpu_ttfc.py --text short --iterations 5

  # CPU (force CPUExecutionProvider)
  FASTKOKORO_ONNX_AUTO_PROVIDERS=false \
    python scripts/benchmark_gpu_ttfc.py --text short --iterations 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from statistics import fmean, median

from fastkokoro.engine import FastKokoro
from fastkokoro.streaming import split_pcm_frames, split_phrases, split_sentences
from scripts.benchmark_corpus import corpus_choices, get_text


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


async def measure(stream, engine, text, strategy, text_name):
    start = time.perf_counter()
    first_chunk_latency = None
    chunks = bytes_count = 0
    async for chunk in stream:
        now = time.perf_counter()
        if first_chunk_latency is None:
            first_chunk_latency = now - start
        chunks += 1
        bytes_count += len(chunk)
    total = time.perf_counter() - start
    return BenchmarkResult(
        strategy,
        text_name,
        len(text),
        chunks,
        bytes_count,
        first_chunk_latency or 0.0,
        total,
        engine.session.get_providers(),
    )


def make_stream(
    engine: FastKokoro,
    strategy: str,
    text: str,
    voice: str,
    lang: str,
    speed: float,
    frame_ms: int,
):
    resolved_voice, resolved_lang = engine.resolve_request(voice, lang)

    if strategy == "chunk":
        from fastkokoro.streaming import split_scheduled_chunks

        c, w = engine._stream_schedule_limits()
        segments = split_scheduled_chunks(
            text,
            initial_max_chars=engine.settings.stream_max_segment_chars,
            initial_max_words=engine.settings.stream_max_segment_words,
            max_chars=c,
            max_words=w,
        )
    elif strategy == "phrase":
        segments = split_phrases(text)
    elif strategy == "adaptive":
        segments = []
        providers = set(engine.session.get_providers())
        has_gpu = bool(
            {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & providers
        )
        adaptive_max = (
            engine.settings.stream_adaptive_max_chars
            if has_gpu
            else engine.settings.stream_adaptive_cpu_max_chars
        )
        for sentence in split_sentences(text):
            if len(sentence) <= adaptive_max:
                segments.append(sentence)
            else:
                segments.extend(split_phrases(sentence))
    else:
        segments = split_sentences(text)

    async def gen():
        for seg in segments:
            audio = engine._create_resolved(
                seg,
                voice=resolved_voice,
                speed=speed,
                response_format="pcm",
                lang=resolved_lang,
            )
            for frame in split_pcm_frames(audio, frame_ms):
                yield frame

    return gen()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", choices=corpus_choices(), default="short")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--frame-ms", type=int, default=200)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    engine = FastKokoro()
    providers = engine.session.get_providers()
    if args.warmup:
        engine.warmup()

    label = f"{args.frame_ms}ms_pcm"

    print(flush=True)
    print(f"Active providers: {providers}", flush=True)
    first_text = get_text(args.text, 0)
    print(
        f"Text corpus: [{args.text}] variants rotating, "
        f"first chars={len(first_text)}",
        flush=True,
    )
    print(f"Voice={args.voice} Lang={args.lang} Speed={args.speed}", flush=True)
    print(f"Iterations={args.iterations} Frame={args.frame_ms}ms", flush=True)
    print(flush=True)

    all_results = []

    for strategy in ["sentence", "adaptive", "phrase", "chunk"]:
        print(f"===== {strategy.upper()} =====", flush=True)
        for i in range(args.iterations):
            text = get_text(args.text, i)
            stream = make_stream(
                engine,
                strategy,
                text,
                args.voice,
                args.lang,
                args.speed,
                args.frame_ms,
            )
            r = await measure(
                stream,
                engine,
                text,
                f"{strategy}_{label}",
                args.text,
            )
            all_results.append(r)
            if args.json:
                print(json.dumps(asdict(r)), flush=True)
            print(
                f"  [{i + 1}/{args.iterations}]  "
                f"TTFC={r.first_chunk_latency_seconds:.4f}s  "
                f"Total={r.total_latency_seconds:.4f}s  "
                f"Chunks={r.chunks}  Bytes={r.bytes}",
                flush=True,
            )

        sr = [r for r in all_results if r.strategy.startswith(strategy)]
        ttfcs = [r.first_chunk_latency_seconds for r in sr]
        tots = [r.total_latency_seconds for r in sr]
        print(
            f"  > AVG:   TTFC={fmean(ttfcs):.4f}s  Total={fmean(tots):.4f}s",
            flush=True,
        )
        print(
            f"  > P50:   TTFC={median(ttfcs):.4f}s  Total={median(tots):.4f}s",
            flush=True,
        )
        print(f"  > MIN:   TTFC={min(ttfcs):.4f}s  Total={min(tots):.4f}s", flush=True)
        print(flush=True)

    if not args.json:
        print("========== FINAL SUMMARY ==========", flush=True)
        for strategy in ["sentence", "adaptive", "phrase", "chunk"]:
            sr = [r for r in all_results if r.strategy.startswith(strategy)]
            ttfcs = [r.first_chunk_latency_seconds for r in sr]
            tots = [r.total_latency_seconds for r in sr]
            print(
                f"{strategy:12s}  "
                f"TTFC avg={fmean(ttfcs):.4f}s  "
                f"p50={median(ttfcs):.4f}s  "
                f"min={min(ttfcs):.4f}s  |  "
                f"Total avg={fmean(tots):.4f}s  "
                f"p50={median(tots):.4f}s",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
