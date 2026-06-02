# Streaming Benchmarks

Benchmarks must run with warmup enabled. This is the default behavior in
`fastkokoro`, and avoids mixing model/session initialization with request
latency.

Run the benchmark script:

```bash
uv run python scripts/benchmark_streaming.py --text medium
```

Run every built-in text case:

```bash
uv run python scripts/benchmark_streaming.py
```

The benchmark reports one JSON line per strategy:

- `kokoro_create_stream`: current `kokoro-onnx` streaming implementation.
- `sentence_segments`: splits text into sentences and synthesizes one sentence at
  a time.
- `sentence_segments_200ms_frames`: splits text into sentences, then slices the
  generated PCM output into 200 ms frames before yielding.
- `phrase_segments_200ms_frames`: splits text on phrase punctuation such as
  commas, then slices the generated PCM output into 200 ms frames before
  yielding.

The production server now uses the sentence strategy by default through
`FASTKOKORO_STREAM_STRATEGY=sentence`. The upstream behavior remains available
with `FASTKOKORO_STREAM_STRATEGY=kokoro`. Interactive clients can use
`FASTKOKORO_STREAM_STRATEGY=phrase` when lower TTFC is more important than total
generation time.

Important fields:

- `first_chunk_latency_seconds`: time to first emitted chunk.
- `total_latency_seconds`: total synthesis time for the strategy.
- `chunks`: number of yielded chunks.
- `active_providers`: ONNX Runtime providers used by the session.

## Initial CPU Observations

These local CPU measurements were exploratory and should be repeated without
parallel benchmark runs before publishing formal numbers.

Short text, 14 characters:

| Strategy | Chunks | First chunk | Total |
| --- | ---: | ---: | ---: |
| `kokoro_create_stream` | 1 | 3.62s | 3.62s |
| `sentence_segments` | 1 | 2.55s | 2.55s |
| `sentence_segments_200ms_frames` | 6 | 2.64s | 2.64s |

Medium text, 149 characters, rerun with warmup enabled after sentence streaming
was added:

| Strategy | Chunks | First chunk | Total |
| --- | ---: | ---: | ---: |
| `kokoro_create_stream` | 1 | 9.75s | 9.75s |
| `sentence_segments` | 3 | 1.57s | 10.77s |
| `sentence_segments_200ms_frames` | 47 | 1.84s | 11.56s |

The current `kokoro-onnx` stream emits one chunk for these short and medium
inputs, so time to first chunk is effectively total latency. Sentence-level
segmentation improves perceived streaming latency by yielding earlier, while
frame slicing improves playback granularity after the first sentence is ready.

On the local 8-CPU machine used for these measurements, setting
`FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS=4` and
`FASTKOKORO_ONNX_INTER_OP_NUM_THREADS=1` reduced short-text p50 latency from
about 1.59s with ONNX Runtime's default thread settings to about 1.27s.
