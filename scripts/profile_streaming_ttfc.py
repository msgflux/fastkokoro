from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

from kokoro_onnx import SAMPLE_RATE, trim_audio

from fastkokoro.audio import encode_audio
from fastkokoro.engine import FastKokoro, split_phonemes_for_model
from fastkokoro.streaming import split_pcm_frames, split_phrases, split_sentences

TEXTS = {
    "tiny": "Ola.",
    "short": "Ola, tudo bem?",
    "medium": (
        "Ola, tudo bem? Este e um teste de sintese de voz em portugues brasileiro. "
        "Estamos medindo a latencia ate o primeiro chunk e o tempo total de geracao."
    ),
}


@dataclass(frozen=True)
class StreamingProfileRun:
    text_name: str
    strategy: str
    text_chars: int
    first_segment_chars: int
    first_segment_phonemes: int
    first_segment_tokens: int
    first_segment_audio_samples: int
    first_chunk_bytes: int
    chunks: int
    bytes: int
    resolve_seconds: float
    split_text_seconds: float
    phonemize_seconds: float
    split_phonemes_seconds: float
    tokenize_seconds: float
    build_inputs_seconds: float
    inference_seconds: float
    trim_seconds: float
    encode_seconds: float
    first_frame_seconds: float
    first_chunk_latency_seconds: float
    remaining_segments_seconds: float
    total_seconds: float
    active_providers: list[str]


@dataclass(frozen=True)
class StreamingProfileSummary:
    text_name: str
    strategy: str
    iterations: int
    text_chars: int
    first_segment_chars: int
    first_segment_phonemes: int
    first_segment_tokens: int
    first_chunk_latency_avg_seconds: float
    first_chunk_latency_p50_seconds: float
    inference_avg_seconds: float
    inference_p50_seconds: float
    phonemize_avg_seconds: float
    tokenize_avg_seconds: float
    trim_avg_seconds: float
    encode_avg_seconds: float
    remaining_segments_avg_seconds: float
    total_avg_seconds: float
    total_p50_seconds: float
    chunks_avg: float
    active_providers: list[str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="pf_dora")
    parser.add_argument("--lang", default="p")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", choices=TEXTS, default="short")
    parser.add_argument("--strategy", choices=("phrase", "sentence"), default="phrase")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()

    engine = FastKokoro()
    if args.warmup:
        engine.warmup()

    runs = [
        profile_once(
            engine,
            TEXTS[args.text],
            text_name=args.text,
            strategy=args.strategy,
            voice_name=args.voice,
            lang=args.lang,
            speed=args.speed,
        )
        for _ in range(args.iterations)
    ]
    if args.json_lines:
        for run in runs:
            print(json.dumps(asdict(run)), flush=True)
    print(json.dumps(asdict(summarize(runs))), flush=True)


def profile_once(
    engine: FastKokoro,
    text: str,
    *,
    text_name: str,
    strategy: str,
    voice_name: str,
    lang: str,
    speed: float,
) -> StreamingProfileRun:
    start = time.perf_counter()
    voice, resolved_lang = engine.resolve_request(voice_name, lang)
    resolved = time.perf_counter()

    segments = split_phrases(text) if strategy == "phrase" else split_sentences(text)
    split_text = time.perf_counter()
    first_segment = segments[0] if segments else ""

    phonemes = engine.kokoro.tokenizer.phonemize(first_segment, resolved_lang)
    phonemized = time.perf_counter()

    phoneme_batches = split_phonemes_for_model(phonemes)
    split_phonemes = time.perf_counter()
    first_batch = phoneme_batches[0] if phoneme_batches else ""

    tokens = engine.kokoro.tokenizer.tokenize(first_batch)
    tokenized = time.perf_counter()

    inputs = engine._build_onnx_inputs(tokens, engine._voice_styles[voice], speed)
    inputs_built = time.perf_counter()

    audio = engine.session.run(None, inputs)[0]
    inferred = time.perf_counter()

    audio, _ = trim_audio(audio)
    trimmed = time.perf_counter()

    encoded = encode_audio(audio, SAMPLE_RATE, "pcm")
    encoded_at = time.perf_counter()

    first_segment_chunks = list(
        split_pcm_frames(encoded, engine.settings.stream_audio_frame_ms)
    )
    first_chunk = first_segment_chunks[0] if first_segment_chunks else b""
    first_frame = time.perf_counter()

    remaining_start = time.perf_counter()
    remaining_chunks = 0
    remaining_bytes = 0
    for segment in segments[1:]:
        audio_bytes = engine._create_resolved(
            segment,
            voice=voice,
            speed=speed,
            response_format="pcm",
            lang=resolved_lang,
        )
        for chunk in split_pcm_frames(
            audio_bytes,
            engine.settings.stream_audio_frame_ms,
        ):
            remaining_chunks += 1
            remaining_bytes += len(chunk)
    finished = time.perf_counter()

    return StreamingProfileRun(
        text_name=text_name,
        strategy=strategy,
        text_chars=len(text),
        first_segment_chars=len(first_segment),
        first_segment_phonemes=len(phonemes),
        first_segment_tokens=len(tokens),
        first_segment_audio_samples=len(audio),
        first_chunk_bytes=len(first_chunk),
        chunks=len(first_segment_chunks) + remaining_chunks,
        bytes=sum(len(chunk) for chunk in first_segment_chunks) + remaining_bytes,
        resolve_seconds=resolved - start,
        split_text_seconds=split_text - resolved,
        phonemize_seconds=phonemized - split_text,
        split_phonemes_seconds=split_phonemes - phonemized,
        tokenize_seconds=tokenized - split_phonemes,
        build_inputs_seconds=inputs_built - tokenized,
        inference_seconds=inferred - inputs_built,
        trim_seconds=trimmed - inferred,
        encode_seconds=encoded_at - trimmed,
        first_frame_seconds=first_frame - encoded_at,
        first_chunk_latency_seconds=first_frame - start,
        remaining_segments_seconds=finished - remaining_start,
        total_seconds=finished - start,
        active_providers=engine.session.get_providers(),
    )


def summarize(runs: list[StreamingProfileRun]) -> StreamingProfileSummary:
    first = runs[0]
    return StreamingProfileSummary(
        text_name=first.text_name,
        strategy=first.strategy,
        iterations=len(runs),
        text_chars=first.text_chars,
        first_segment_chars=first.first_segment_chars,
        first_segment_phonemes=first.first_segment_phonemes,
        first_segment_tokens=first.first_segment_tokens,
        first_chunk_latency_avg_seconds=statistics.fmean(
            run.first_chunk_latency_seconds for run in runs
        ),
        first_chunk_latency_p50_seconds=statistics.median(
            run.first_chunk_latency_seconds for run in runs
        ),
        inference_avg_seconds=statistics.fmean(run.inference_seconds for run in runs),
        inference_p50_seconds=statistics.median(run.inference_seconds for run in runs),
        phonemize_avg_seconds=statistics.fmean(run.phonemize_seconds for run in runs),
        tokenize_avg_seconds=statistics.fmean(run.tokenize_seconds for run in runs),
        trim_avg_seconds=statistics.fmean(run.trim_seconds for run in runs),
        encode_avg_seconds=statistics.fmean(run.encode_seconds for run in runs),
        remaining_segments_avg_seconds=statistics.fmean(
            run.remaining_segments_seconds for run in runs
        ),
        total_avg_seconds=statistics.fmean(run.total_seconds for run in runs),
        total_p50_seconds=statistics.median(run.total_seconds for run in runs),
        chunks_avg=statistics.fmean(run.chunks for run in runs),
        active_providers=first.active_providers,
    )


if __name__ == "__main__":
    main()
