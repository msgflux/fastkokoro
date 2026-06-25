# fastkokoro

Lightweight OpenAI-compatible Kokoro TTS server powered by ONNX Runtime.

`fastkokoro` runs the 82M-parameter Kokoro text-to-speech model with low startup
overhead, fast local inference, and a small dependency footprint. It supports CPU
and GPU execution through ONNX Runtime providers, including CUDA, TensorRT, and
other providers when the matching runtime package is installed. The default
model is the fixed-bucket streaming export:
`msgflux/Kokoro-82M-streaming-onnx`.

The default checkpoint is `onnx/kokoro-82m-streaming-b24-fp16.onnx`.

## Demo

Watch a short demo with `fastkokoro`:

https://github.com/user-attachments/assets/b978ad87-59fa-4743-8369-08dbda20c2fc

## Install

Install one ONNX Runtime extra for inference. CPU:

```bash
uv add 'fastkokoro[cpu]'
# or
pip install 'fastkokoro[cpu]'
```

GPU, on platforms supported by `onnxruntime-gpu`:

```bash
uv add 'fastkokoro[gpu]'
# or
pip install 'fastkokoro[gpu]'
```

Legacy CUDA 11.8 / cuDNN8 GPU environments can use the pinned legacy extra:

```bash
uv add 'fastkokoro[gpu-legacy]'
# or
pip install 'fastkokoro[gpu-legacy]'
```

The base `fastkokoro` package intentionally does not install ONNX Runtime.
Starting the engine without either extra raises an explicit install error.
PCM JIT acceleration with Numba is included by default.

## Run

```bash
fastkokoro
```

The server starts on `http://0.0.0.0:8880` by default.

## From Source

Clone the repository and install the local CPU development environment:

```bash
git clone https://github.com/msgflux/fastkokoro.git
cd fastkokoro
uv sync --extra cpu
```

Run the server from source:

```bash
uv run fastkokoro
```

For GPU development environments, use the GPU extra instead:

```bash
uv sync --extra gpu
```

## Docker

Use the published CPU image from Docker Hub:

```bash
docker run -p 8880:8880 msgflux/fastkokoro:cpu
```

Use the published GPU image with NVIDIA Container Toolkit:

```bash
docker run --gpus all -p 8880:8880 msgflux/fastkokoro:gpu
```

Use the TensorRT image when the host has a compatible NVIDIA driver and you want
TensorRT EP with engine caching:

```bash
docker run --gpus all -p 8880:8880 -v fastkokoro-models:/models msgflux/fastkokoro:0.4.0-tensorrt
```

Published tags:

| Tag | Description |
| --- | --- |
| `cpu`, `latest-cpu` | Latest CPU image |
| `gpu`, `latest-gpu` | Alias for the latest CUDA 12.6/cuDNN9 GPU image |
| `gpu-cuda12.6-cudnn9`, `latest-gpu-cuda12.6-cudnn9` | Latest CUDA 12.6/cuDNN9 GPU image |
| `gpu-legacy`, `latest-gpu-legacy` | Alias for the CUDA 11.8/cuDNN8 GPU image |
| `gpu-cuda11.8-cudnn8`, `latest-gpu-cuda11.8-cudnn8` | Latest CUDA 11.8/cuDNN8 GPU image |
| `tensorrt`, `latest-tensorrt` | Alias for the latest TensorRT image |
| `tensorrt-25.06`, `latest-tensorrt-25.06` | TensorRT 25.06 image with ONNX Runtime 1.22 |
| `0.4.0-cpu`, `0.4-cpu` | Versioned CPU image |
| `0.4.0-gpu`, `0.4-gpu` | Versioned CUDA 12.6/cuDNN9 GPU image alias |
| `0.4.0-gpu-cuda12.6-cudnn9`, `0.4-gpu-cuda12.6-cudnn9` | Versioned CUDA 12.6/cuDNN9 GPU image |
| `0.4.0-gpu-legacy`, `0.4-gpu-legacy` | Versioned CUDA 11.8/cuDNN8 GPU image alias |
| `0.4.0-gpu-cuda11.8-cudnn8`, `0.4-gpu-cuda11.8-cudnn8` | Versioned CUDA 11.8/cuDNN8 GPU image |
| `0.4.0-tensorrt`, `0.4-tensorrt` | Versioned TensorRT image alias |
| `0.4.0-tensorrt-25.06`, `0.4-tensorrt-25.06` | Versioned TensorRT 25.06 image |

Build and run the CPU image locally:

```bash
docker build -f docker/Dockerfile.cpu -t fastkokoro:cpu .
docker run -p 8880:8880 fastkokoro:cpu
```

Build and run the GPU image locally:

```bash
docker build -f docker/Dockerfile.gpu -t fastkokoro:gpu .
docker run --gpus all -p 8880:8880 fastkokoro:gpu
```

For older NVIDIA drivers or GPUs that do not work with the current CUDA 12
image, build the CUDA 11.8/cuDNN8 legacy image:

```bash
docker build -f docker/Dockerfile.gpu-legacy -t fastkokoro:gpu-legacy .
docker run --gpus all -p 8880:8880 fastkokoro:gpu-legacy
```

Build and run the TensorRT image locally:

```bash
docker build -f docker/Dockerfile.tensorrt -t fastkokoro:tensorrt .
docker run --gpus all -p 8880:8880 -v fastkokoro-models:/models fastkokoro:tensorrt
```

The TensorRT Dockerfile uses NVIDIA TensorRT 25.06 with ONNX Runtime GPU 1.22.x
and defaults to `TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`.
The standard GPU image uses CUDA 12.6/cuDNN9, while `gpu-legacy` keeps the CUDA
11.8/cuDNN8 build for older hosts.

TensorRT builds an engine the first time it sees a model, bucket, GPU
architecture, and provider option set. Keep `/models` mounted so
`/models/trt-cache` persists; otherwise startup will pay the TensorRT engine
build cost again.

For short conversational phrases where natural continuity matters more than
minimum TTFC, serve the b48 checkpoint and keep the default segment env values.
This lets the adaptive scheduler scale the first scheduled chunk from the loaded
bucket:

```bash
docker run --gpus all -p 8880:8880 -v fastkokoro-models:/models \
  -e FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b48-fp16.onnx \
  -e FASTKOKORO_STREAM_MAX_SEGMENT_CHARS=24 \
  -e FASTKOKORO_STREAM_MAX_SEGMENT_WORDS=2 \
  msgflux/fastkokoro:0.4.0-tensorrt
```

b48 increases token capacity, but it still has a fixed waveform budget per model
call. Keep phrase chunks short enough to fit the vocoder output window; for long
paragraphs, let `adaptive` split the text instead of forcing whole sentences into
one inference.

The CUDA 11.8 legacy image is kept for older hosts, but TensorRT EP support is
published only through the TensorRT 25.06 image. Current Python 3.12 ONNX
Runtime GPU wheels expect TensorRT 10 libraries for TensorRT EP.

Local measurements on a GTX 1650 (SM75), using the b24 streaming model,
`FASTKOKORO_STREAM_STRATEGY=adaptive`, `response_format=pcm`, and HTTP
streaming:

| Scenario | Provider | TTFC p50 |
| --- | --- | ---: |
| Medium/long adaptive streaming | CUDA EP | 58.7 ms |
| Medium/long adaptive streaming | TensorRT EP | 46.1 ms |

These numbers exclude first-time TensorRT engine build. On the same host, first
engine build took minutes; persist `/models/trt-cache` for production.

Environment variables:

| Variable | Default |
| --- | --- |
| `FASTKOKORO_HOST` | `0.0.0.0` |
| `FASTKOKORO_PORT` | `8880` |
| `FASTKOKORO_MODEL_REPO` | `msgflux/Kokoro-82M-streaming-onnx` |
| `FASTKOKORO_MODEL_FILE` | `onnx/kokoro-82m-streaming-b24-fp16.onnx` |
| `FASTKOKORO_MODEL_PATH` | unset; downloads from Hugging Face |
| `FASTKOKORO_VOICES_FILE` | `voices.npz` |
| `FASTKOKORO_VOICES_INDEX_FILE` | `voices.txt` |
| `FASTKOKORO_VOICES_PATH` | unset; downloads from Hugging Face |
| `FASTKOKORO_DEFAULT_VOICE` | `af_heart` |
| `FASTKOKORO_DEFAULT_LANG` | `en-us` |
| `FASTKOKORO_WARMUP` | `true` |
| `FASTKOKORO_WARMUP_TEXT` | `Hello there. This is a warmup request for streaming speech generation.` |
| `FASTKOKORO_WARMUP_REQUEST` | `false` |
| `FASTKOKORO_STREAM_STRATEGY` | `adaptive` |
| `FASTKOKORO_STREAM_AUDIO_FRAME_MS` | `200` |
| `FASTKOKORO_STREAM_MAX_SEGMENT_CHARS` | `24` |
| `FASTKOKORO_STREAM_MAX_SEGMENT_WORDS` | `2` |
| `FASTKOKORO_ONNX_PROVIDERS` | `CPUExecutionProvider` |
| `FASTKOKORO_ONNX_PROVIDER_OPTIONS` | unset |
| `FASTKOKORO_ONNX_AUTO_PROVIDERS` | `false` |
| `FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` | `min(6, CPU count)` |
| `FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` | `1` |
| `FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL` | `all` |
| `FASTKOKORO_ONNX_LOG_SEVERITY_LEVEL` | `3` |
| `FASTKOKORO_ONNX_IO_BINDING` | `true` |
| `FASTKOKORO_ONNX_IO_BINDING_DEVICE` | `auto` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS` | unset; disabled |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_BLOCK_SIZE` | `128` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL` | `4` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_SYMMETRIC` | `true` |
| `FASTKOKORO_JIT` | `true` |
| `FASTKOKORO_PROFILE` | `false` |
| `FASTKOKORO_PROFILE_DIR` | `FASTKOKORO_CACHE_DIR/profiles` |
| `FASTKOKORO_PROFILE_WARMUP` | `false` unless `FASTKOKORO_PROFILE=true` |
| `FASTKOKORO_PROFILE_REQUESTS` | `false` unless `FASTKOKORO_PROFILE=true` |
| `FASTKOKORO_ONNX_ADAIN_FUSION` | `false` |
| `FASTKOKORO_ONNX_ADAIN_MODEL_PATH` | unset; generated under cache |
| `FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY` | unset |
| `FASTKOKORO_CORS_ALLOW_ORIGINS` | `*` |
| `FASTKOKORO_CORS_ALLOW_METHODS` | `GET,POST,OPTIONS` |
| `FASTKOKORO_CORS_ALLOW_HEADERS` | `*` |
| `FASTKOKORO_CORS_ALLOW_CREDENTIALS` | `false` |

`FASTKOKORO_WARMUP=true` runs a short synthesis during startup. This makes the
server take a little longer to become ready, but avoids paying most of the first
request latency on the first user request.

Set `FASTKOKORO_WARMUP_REQUEST=true` to run an in-process startup request through
the same streaming speech endpoint flow and consume the first chunk.

The default b24 streaming model is the balanced option for sub-60 ms local TTFC
without over-fragmenting medium text. TensorRT EP can reduce TTFC below 50 ms
after its engine cache is built. For minimum TTFC with very short chunks, use
b16; for larger chunks, use b32 or b48:

```bash
FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b16-fp16.onnx
FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b24-fp16.onnx
FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b32-fp16.onnx
FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b48-fp16.onnx
```

Bucket size controls the maximum token width of each model call. Two positions
are reserved by the model, so usable text capacity is `bucket - 2` tokens. Word
counts are approximate because phoneme tokens vary by language and word length.

| Bucket | Usable tokens | Approx words | Notes |
| ---: | ---: | ---: | --- |
| 16 | 14 | 1-2 | Lowest TTFC; adaptive first chunk targets 1 word |
| 24 | 22 | 2-4 | Recommended default; adaptive first chunk targets 2 words |
| 32 | 30 | 4-5 | Larger chunks with moderate latency; adaptive first chunk targets 4 words |
| 48 | 46 | 6-8 | Fewer model calls for longer text; adaptive first chunk targets 6 words |
| 64 | 62 | 8-11 | Candidate larger export for paragraph-style streaming |
| 128 | 126 | 16-23 | Candidate maximum-continuity export with higher TTFC |

`FASTKOKORO_JIT` is enabled by default for PCM encoding and trim. The first call
compiles the kernels, so keep startup warmup enabled to absorb this cost before
serving requests. Set `FASTKOKORO_JIT=false` to force the NumPy path.

Enable built-in profiling with `FASTKOKORO_PROFILE=true` to write `cProfile` artifacts for
startup warmup and speech requests under `FASTKOKORO_PROFILE_DIR`. Each run produces a
raw `.prof` file plus a `.txt` summary sorted by cumulative time. Use
`FASTKOKORO_PROFILE_WARMUP` and `FASTKOKORO_PROFILE_REQUESTS` to narrow profiling to
startup or request handling when debugging TTFC regressions.

`FASTKOKORO_STREAM_STRATEGY=adaptive` is the default. It keeps short sentences
intact for more natural prosody, and uses scheduled word-boundary chunks for
longer sentences to keep TTFC bounded. With default settings, the first scheduled
chunk scales with the loaded model bucket; explicit segment env vars override
that schedule. Set `FASTKOKORO_STREAM_STRATEGY=chunk`
for minimum TTFC; this enforces `FASTKOKORO_STREAM_MAX_SEGMENT_WORDS` and
`FASTKOKORO_STREAM_MAX_SEGMENT_CHARS` from the first segment, trading continuity
for lower first audio latency. `phrase` splits only on phrase punctuation such
as commas, semicolons, and question marks. `sentence` synthesizes one sentence
at a time. For `response_format=pcm`, the server also slices each generated
segment into smaller audio frames controlled by `FASTKOKORO_STREAM_AUDIO_FRAME_MS`.
Set `FASTKOKORO_STREAM_STRATEGY=kokoro` to keep the legacy strategy name; it now
uses the local fastkokoro synthesis path instead of the upstream engine.

Inline pause tokens can be embedded in input text. `[pause:1.5s]` inserts 1.5
seconds of silence without running the model for that segment. The form is
strict: colon, numeric seconds, trailing `s`, and square brackets. Other forms
such as `[pause=1.5]` or SSML `<break/>` are treated as normal text.

The default ONNX Runtime thread settings prioritize low CPU latency. Set
`FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` or
`FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` to an empty value to use ONNX Runtime's
own defaults.

Set `FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS=4` or
`FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS=8` to generate a MatMul weight-only
quantized ONNX model on startup. The generated model is cached under
`FASTKOKORO_CACHE_DIR/quantized` and reused on later starts with the same
settings.

Set `FASTKOKORO_ONNX_ADAIN_FUSION=true` to use the experimental CPU-only AdaIN
custom op optimization. This preserves ONNX Runtime's normal `Conv` kernels and
rewrites generator AdaIN subgraphs into a native custom op. It requires
`FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider` and
`FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY` pointing to a compiled
`libfastkokoro_adain.so`. If `FASTKOKORO_ONNX_ADAIN_MODEL_PATH` is unset,
`fastkokoro` generates and caches an AdaIN-fused ONNX model under
`FASTKOKORO_CACHE_DIR/onnx`.

Build the custom op on the target machine with:

```bash
uv run python scripts/build_adain_op.py --print-env
```

The script writes the native library under `FASTKOKORO_CACHE_DIR/native` by
default and prints the `FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY` export line.
Set `FASTKOKORO_ONNX_ADAIN_FUSION=true` and
`FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider` when starting the server if you
want to enable the CPU custom op.

Restrict CORS by setting one or more allowed origins:

```bash
FASTKOKORO_CORS_ALLOW_ORIGINS=http://localhost:3000 fastkokoro
```

## ONNX Runtime Providers

`fastkokoro` creates the ONNX Runtime session directly, so provider selection is
explicit and predictable.

CPU:

```bash
FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider uv run fastkokoro
```

CUDA with CPU fallback:

```bash
FASTKOKORO_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider uv run fastkokoro
```

TensorRT with CUDA and CPU fallback:

```bash
FASTKOKORO_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider uv run fastkokoro
```

Recommended TensorRT provider options:

```bash
FASTKOKORO_ONNX_PROVIDER_OPTIONS='{"TensorrtExecutionProvider":{"trt_engine_cache_enable":"True","trt_engine_cache_path":"/models/trt-cache","trt_timing_cache_enable":"True","trt_timing_cache_path":"/models/trt-cache"}}'
```

Provider options can be passed as JSON keyed by provider name. For example,
CUDA GPU device selection:

```bash
FASTKOKORO_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
FASTKOKORO_ONNX_PROVIDER_OPTIONS='{"CUDAExecutionProvider":{"device_id":"0"}}' \
uv run fastkokoro
```

Set `FASTKOKORO_ONNX_AUTO_PROVIDERS=true` to pass every provider available in the
installed ONNX Runtime build to the session. Use this mostly for quick local
experiments; production deployments should pin an explicit provider order.

For latency tuning, run:

```bash
uv run python scripts/benchmark_latency.py --text short --iterations 5 --warmup
```

## API

Health:

```bash
curl http://localhost:8880/health
```

Models:

```bash
curl http://localhost:8880/v1/models
```

Metrics:

```bash
curl http://localhost:8880/metrics
```

The metrics endpoint returns JSON counters and latency summaries for HTTP
requests and speech generation, including streaming chunk counts, total bytes,
time to first speech chunk, and active ONNX Runtime providers. For streaming
speech responses, use the `speech` latency fields; HTTP middleware latency only
tracks response setup.

Run benchmarks with `FASTKOKORO_WARMUP=true`, which is the default. Compare
requests after startup so model/session initialization does not pollute latency
measurements.

The server exposes the local Kokoro model as `kokoro`. For client compatibility,
`/v1/audio/speech` also accepts `tts-1` and `gpt-4o-mini-tts` as aliases, but
they are not listed by `/v1/models` because the server is not running OpenAI TTS
models.

## Voices and Languages

The official Kokoro voice list maps voices to language codes. `fastkokoro`
accepts the Kokoro language code and common locale aliases, then validates that
the requested voice belongs to the resolved language.

| Language | Request `lang` values | Voices |
| --- | --- | --- |
| American English | `a`, `en-us`, `american` | `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| British English | `b`, `en-gb`, `british` | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` |
| Japanese | `j`, `ja`, `ja-jp` | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo` |
| Mandarin Chinese | `z`, `zh`, `zh-cn`, `mandarin` | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` |
| Spanish | `e`, `es`, `es-es` | `ef_dora`, `em_alex`, `em_santa` |
| French | `f`, `fr`, `fr-fr` | `ff_siwis` |
| Hindi | `h`, `hi`, `hi-in` | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |
| Italian | `i`, `it`, `it-it` | `if_sara`, `im_nicola` |
| Brazilian Portuguese | `p`, `pt`, `pt-br` | `pf_dora`, `pm_alex`, `pm_santa` |

Speech:

```bash
curl http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Hello from fastkokoro.",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Streaming PCM:

```bash
curl http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Streaming from fastkokoro.",
    "voice": "af_heart",
    "response_format": "pcm",
    "stream": true
  }' \
  --output speech.pcm
```

## OpenAI SDK Examples

The examples use inline script dependencies, so they can run directly with `uv`
without adding the OpenAI SDK to the project environment.

Start `fastkokoro` first:

```bash
uv run fastkokoro
```

Save synthesized audio to a file:

```bash
uv run examples/tts_save_file.py
```

Consume streamed audio chunks:

```bash
uv run examples/tts_stream_chunks.py
```

Useful environment variables:

| Variable | Default |
| --- | --- |
| `FASTKOKORO_BASE_URL` | `http://localhost:8880/v1` |
| `FASTKOKORO_API_KEY` | `fastkokoro` |
| `FASTKOKORO_VOICE` | `pf_dora` |
| `FASTKOKORO_TEXT` | `Ola, tudo bem?` |
| `FASTKOKORO_TTS_OUTPUT` | `speech.wav` |

## Python

```python
from fastkokoro import FastKokoro

engine = FastKokoro()
audio = engine.create(
    "Hello from fastkokoro.",
    voice="af_heart",
    response_format="wav",
)
```
