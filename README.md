# fastkokoro

Lightweight OpenAI-compatible Kokoro TTS server powered by ONNX Runtime.

`fastkokoro` runs the 82M-parameter Kokoro text-to-speech model with low startup
overhead, fast local inference, and a small dependency footprint. It supports CPU
and GPU execution through ONNX Runtime providers, including CUDA, TensorRT, and
other providers when the matching runtime package is installed. The default
model is NVIDIA's optimized ONNX export: `nvidia/kokoro-82M-onnx-opt`.

The NVIDIA repo's `voices.bin` uses a raw float32 layout. `fastkokoro` converts it
once into the `.npz` voice format expected by `kokoro-onnx`, so the default model
and voices both come from `nvidia/kokoro-82M-onnx-opt`.

## Demo

Watch a short demo running a Brazilian Portuguese TTS request with `fastkokoro`:

https://github.com/user-attachments/assets/9bb0e108-cd89-4e40-a9a7-57f4c5964d52

## Install

With uv:

```bash
uv add fastkokoro
```

With pip:

```bash
pip install fastkokoro
```

For GPU builds on platforms supported by `onnxruntime-gpu`:

```bash
uv add 'fastkokoro[gpu]'
```

## Run

```bash
fastkokoro
```

The server starts on `http://0.0.0.0:8880` by default.

## From Source

Clone the repository and install the local development environment:

```bash
git clone https://github.com/msgflux/fastkokoro.git
cd fastkokoro
uv sync
```

Run the server from source:

```bash
uv run fastkokoro
```

For GPU development environments:

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
| `gpu`, `latest-gpu` | Latest GPU image |
| `0.2.0-cpu`, `0.2-cpu` | Versioned CPU image |
| `0.2.0-gpu`, `0.2-gpu` | Versioned GPU image |

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
| `FASTKOKORO_STREAM_STRATEGY` | `phrase` |
| `FASTKOKORO_STREAM_AUDIO_FRAME_MS` | `200` |
| `FASTKOKORO_ONNX_PROVIDERS` | `CPUExecutionProvider` |
| `FASTKOKORO_ONNX_AUTO_PROVIDERS` | `false` |
| `FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` | `min(4, CPU count)` |
| `FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` | `1` |
| `FASTKOKORO_ONNX_GRAPH_OPTIMIZATION_LEVEL` | `all` |
| `FASTKOKORO_ONNX_IO_BINDING` | `true` |
| `FASTKOKORO_ONNX_IO_BINDING_DEVICE` | `auto` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS` | unset; disabled |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_BLOCK_SIZE` | `128` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_ACCURACY_LEVEL` | `4` |
| `FASTKOKORO_ONNX_WEIGHT_ONLY_SYMMETRIC` | `true` |

`FASTKOKORO_WARMUP=true` runs a short synthesis during startup. This makes the
server take a little longer to become ready, but avoids paying most of the first
request latency on the first user request.

`FASTKOKORO_STREAM_STRATEGY=phrase` streams by splitting on phrase punctuation
such as commas, semicolons, and question marks. Set
`FASTKOKORO_STREAM_STRATEGY=sentence` to synthesize one sentence at a time. For
`response_format=pcm`, the server also slices each generated segment into
smaller audio frames controlled by `FASTKOKORO_STREAM_AUDIO_FRAME_MS`. Set
`FASTKOKORO_STREAM_STRATEGY=kokoro` to use the upstream `kokoro-onnx` streaming
path directly.

The default ONNX Runtime thread settings prioritize low CPU latency. Set
`FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` or
`FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` to an empty value to use ONNX Runtime's
own defaults.

Set `FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS=4` or
`FASTKOKORO_ONNX_WEIGHT_ONLY_NBITS=8` to generate a MatMul weight-only
quantized ONNX model on startup. The generated model is cached under
`FASTKOKORO_CACHE_DIR/quantized` and reused on later starts with the same
settings.

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
