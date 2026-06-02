# Latency Benchmarks

`scripts/benchmark_latency.py` measures warmed request latency with repeated
iterations. It is intended for CPU/GPU tuning where p50, minimum latency, and
outliers are more useful than a single run.

Run the default short-text latency benchmark:

```bash
uv run python scripts/benchmark_latency.py --text short --iterations 5 --warmup
```

Compare ONNX Runtime's own thread defaults with fastkokoro's latency-oriented
defaults:

```bash
uv run python scripts/benchmark_latency.py --text short --iterations 5 --warmup --ort-default-threads
uv run python scripts/benchmark_latency.py --text short --iterations 5 --warmup
```

Try explicit CPU thread settings:

```bash
uv run python scripts/benchmark_latency.py --text short --iterations 5 --warmup --intra-op 4 --inter-op 1
```

Compare streaming segmentation strategies:

```bash
FASTKOKORO_STREAM_STRATEGY=sentence uv run python scripts/benchmark_latency.py --text medium --iterations 3 --warmup
FASTKOKORO_STREAM_STRATEGY=phrase uv run python scripts/benchmark_latency.py --text medium --iterations 3 --warmup
```

## Local CPU Notes

On the local 8-CPU notebook, `intra_op=4` and `inter_op=1` gave the best
short-text latency in the first tuning pass.

Short text, 14 characters:

| Thread config | Strategy | p50 first chunk | min first chunk |
| --- | --- | ---: | ---: |
| ORT default | `stream_sentence` | 1.59s | 1.59s |
| `intra=4`, `inter=1` | `stream_sentence` | 1.28s | 1.22s |
| `intra=4`, `inter=1` | `stream_kokoro` | 1.48s | 1.26s |
| `intra=4`, `inter=1` | `stream_phrase` | 0.69s | 0.61s |

Medium text, 149 characters:

| Thread config | Strategy | p50 first chunk | p50 total |
| --- | --- | ---: | ---: |
| `intra=4`, `inter=1` | `stream_phrase` | 0.80s | 18.02s |
| `intra=4`, `inter=1` | `stream_sentence` | 1.19s | 16.74s |
| `intra=4`, `inter=1` | `stream_kokoro` | 9.46s | 9.46s |

The default `fastkokoro` ONNX Runtime CPU settings now use `intra_op=min(4, CPU
count)` and `inter_op=1`. These settings favor low single-request latency. Set
`FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` and
`FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` explicitly for hardware-specific tuning,
or set them to empty values to use ONNX Runtime's own defaults.

`stream_phrase` is the lowest-latency CPU option in these local measurements and
is appropriate for interactive clients that value TTFC. It can increase total
generation time because more model calls are made.
