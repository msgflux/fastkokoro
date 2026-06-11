# fastkokoro

Lightweight OpenAI-compatible Kokoro TTS server powered by ONNX Runtime.

`fastkokoro` runs the 82M-parameter Kokoro text-to-speech model with low startup
overhead, fast local inference, and a small dependency footprint. It supports CPU
and GPU execution through ONNX Runtime providers, including CUDA, TensorRT, and
other providers when the matching runtime package is installed. The default
model is NVIDIA's optimized ONNX export: `nvidia/kokoro-82M-onnx-opt`.

The NVIDIA repo's `voices.bin` uses a raw float32 layout. `fastkokoro` converts it
once into the internal `.npz` voice format used by `fastkokoro`, so the default model
and voices both come from `nvidia/kokoro-82M-onnx-opt`.

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

Enable experimental TTFC multi-shape warmup from CLI:

```bash
uv run fastkokoro --warmup-multi-shape
```

Or provide custom buckets:

```bash
uv run fastkokoro --warmup-multi-shape-buckets 6,8,9,10,11,12,16,24
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

Published tags:

| Tag | Description |
| --- | --- |
| `cpu`, `latest-cpu` | Latest CPU image |
| `gpu`, `latest-gpu` | Alias for the latest CUDA 12.6/cuDNN9 GPU image |
| `gpu-cuda12.6-cudnn9`, `latest-gpu-cuda12.6-cudnn9` | Latest CUDA 12.6/cuDNN9 GPU image |
| `gpu-legacy`, `latest-gpu-legacy` | Alias for the CUDA 11.8/cuDNN8 GPU image |
| `gpu-cuda11.8-cudnn8`, `latest-gpu-cuda11.8-cudnn8` | Latest CUDA 11.8/cuDNN8 GPU image |
| `0.2.0-cpu`, `0.2-cpu` | Versioned CPU image |
| `0.2.0-gpu`, `0.2-gpu` | Versioned CUDA 12.6/cuDNN9 GPU image alias |
| `0.2.0-gpu-cuda12.6-cudnn9`, `0.2-gpu-cuda12.6-cudnn9` | Versioned CUDA 12.6/cuDNN9 GPU image |
| `0.2.0-gpu-legacy`, `0.2-gpu-legacy` | Versioned CUDA 11.8/cuDNN8 GPU image alias |
| `0.2.0-gpu-cuda11.8-cudnn8`, `0.2-gpu-cuda11.8-cudnn8` | Versioned CUDA 11.8/cuDNN8 GPU image |

Build and run the CPU image locally:

```bash
docker build -f Dockerfile.cpu -t fastkokoro:cpu .
docker run -p 8880:8880 fastkokoro:cpu
```

Build and run the GPU image locally:

```bash
docker build -f Dockerfile.gpu -t fastkokoro:gpu .
docker run --gpus all -p 8880:8880 fastkokoro:gpu
```

For older NVIDIA drivers or GPUs that do not work with the current CUDA 12
image, build the CUDA 11.8/cuDNN8 legacy image:

```bash
docker build -f Dockerfile.gpu-legacy -t fastkokoro:gpu-legacy .
docker run --gpus all -p 8880:8880 fastkokoro:gpu-legacy
```

Environment variables:

| Variable | Default |
| --- | --- |
| `FASTKOKORO_HOST` | `0.0.0.0` |
| `FASTKOKORO_PORT` | `8880` |
| `FASTKOKORO_MODEL_REPO` | `nvidia/kokoro-82M-onnx-opt` |
| `FASTKOKORO_MODEL_FILE` | `kokoro-82m-v1.0.onnx` |
| `FASTKOKORO_MODEL_PATH` | unset; downloads from Hugging Face |
| `FASTKOKORO_VOICES_FILE` | `voices.bin` |
| `FASTKOKORO_VOICES_INDEX_FILE` | `voices.txt` |
| `FASTKOKORO_VOICES_PATH` | unset; downloads and converts NVIDIA voices |
| `FASTKOKORO_DEFAULT_VOICE` | `af_heart` |
| `FASTKOKORO_DEFAULT_LANG` | `en-us` |
| `FASTKOKORO_WARMUP` | `true` |
| `FASTKOKORO_WARMUP_TEXT` | `hello` |
| `FASTKOKORO_STREAM_STRATEGY` | `adaptive` |
| `FASTKOKORO_STREAM_AUDIO_FRAME_MS` | `200` |
| `FASTKOKORO_STREAM_MAX_SEGMENT_CHARS` | `32` |
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
| `FASTKOKORO_WARMUP_MULTI_SHAPE` | `false` |
| `FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS` | `6,8,9,10,11,12,16,24` |
| `FASTKOKORO_JIT` | `true` |
| `FASTKOKORO_ONNX_ADAIN_FUSION` | `false` |
| `FASTKOKORO_ONNX_ADAIN_MODEL_PATH` | unset; generated under cache |
| `FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY` | unset |
| `FASTKOKORO_ONNX_CONV_ADAIN_FUSION` | `false` |
| `FASTKOKORO_ONNX_CONV_ADAIN_MODEL_PATH` | unset; generated under cache |
| `FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY` | unset |
| `FASTKOKORO_CORS_ALLOW_ORIGINS` | `*` |
| `FASTKOKORO_CORS_ALLOW_METHODS` | `GET,POST,OPTIONS` |
| `FASTKOKORO_CORS_ALLOW_HEADERS` | `*` |
| `FASTKOKORO_CORS_ALLOW_CREDENTIALS` | `false` |

`FASTKOKORO_WARMUP=true` runs a short synthesis during startup. This makes the
server take a little longer to become ready, but avoids paying most of the first
request latency on the first user request.

Set `FASTKOKORO_WARMUP_MULTI_SHAPE=true` to enable experimental multi-shape ONNX
warmup focused on first chunk latency. The server runs one pass per bucket from
`FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS` without changing request shapes at
runtime.

`FASTKOKORO_JIT` is enabled by default for PCM encoding and trim. The first call
compiles the kernels, so keep startup warmup enabled to absorb this cost before
serving requests. Set `FASTKOKORO_JIT=false` to force the NumPy path.

`FASTKOKORO_STREAM_STRATEGY=chunk` streams by splitting on punctuation when
possible while also enforcing `FASTKOKORO_STREAM_MAX_SEGMENT_WORDS` and
`FASTKOKORO_STREAM_MAX_SEGMENT_CHARS`. The default is intentionally small, up to
2 words or 32 characters per model call, to favor low TTFC for interactive
clients. `phrase` splits only on phrase punctuation such as commas, semicolons,
and question marks. `sentence` synthesizes one sentence at a time. For
`response_format=pcm`, the server also slices each generated segment into
smaller audio frames controlled by `FASTKOKORO_STREAM_AUDIO_FRAME_MS`. Set
`FASTKOKORO_STREAM_STRATEGY=kokoro` to keep the legacy strategy name; it now
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

Build and enable the custom op on the target machine with the server flag:

```bash
FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider uv run fastkokoro --build-custom-op
```

The flag writes the native library under `FASTKOKORO_CACHE_DIR/native` by
default, enables AdaIN fusion for the process, and points
`FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY` at the compiled library. Use
`--custom-op-output /path/libfastkokoro_adain.so` to choose a specific path.
For manual builds without starting the server, run
`uv run fastkokoro-build-adain-op --print-env`.

Set `FASTKOKORO_ONNX_CONV_ADAIN_FUSION=true` to use the experimental CPU-only
`Conv1dAdaIn` custom op optimization. This rewrites generator `Conv -> AdaIN`
subgraphs into a fused native custom op and can reduce CPU latency in the
decoder path. It requires `FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider` and
`FASTKOKORO_ONNX_CONV_ADAIN_CUSTOM_OP_LIBRARY` pointing to a compiled
`libfastkokoro_conv_adain.so`. If `FASTKOKORO_ONNX_CONV_ADAIN_MODEL_PATH` is
unset, `fastkokoro` generates and caches a ConvAdaIN-fused ONNX model under
`FASTKOKORO_CACHE_DIR/onnx`.

Build and enable the ConvAdaIN custom op on the target machine with:

```bash
FASTKOKORO_ONNX_PROVIDERS=CPUExecutionProvider uv run fastkokoro --build-conv-custom-op
```

Or build it manually without starting the server:

```bash
uv run fastkokoro-build-conv-adain-op --print-env
```

This path is highly hardware-dependent and may regress latency on some CPUs.
Always benchmark against the baseline before enabling it in production.

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
