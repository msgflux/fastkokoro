# fastkokoro

Lightweight OpenAI-compatible Kokoro TTS server powered by ONNX Runtime.

`fastkokoro` is focused on running Kokoro without a Torch server. It uses
`kokoro-onnx` for tokenization, phonemization, ONNX Runtime inference, and voice
handling. The default model is NVIDIA's optimized ONNX export:
`nvidia/kokoro-82M-onnx-opt`.

The NVIDIA repo's `voices.bin` uses a raw float32 layout. `fastkokoro` converts it
once into the `.npz` voice format expected by `kokoro-onnx`, so the default model
and voices both come from `nvidia/kokoro-82M-onnx-opt`.

## Install

```bash
uv sync
```

From PyPI:

```bash
pip install fastkokoro
```

For GPU builds on platforms supported by `onnxruntime-gpu`:

```bash
uv sync --extra gpu
```

## Run

```bash
uv run fastkokoro
```

The server starts on `http://0.0.0.0:8880` by default.

Docker CPU:

```bash
docker build -f Dockerfile.cpu -t fastkokoro:cpu .
docker run -p 8880:8880 fastkokoro:cpu
```

Docker Hub CPU:

```bash
docker run -p 8880:8880 msgflux/fastkokoro:cpu
```

Docker GPU:

```bash
docker build -f Dockerfile.gpu -t fastkokoro:gpu .
docker run --gpus all -p 8880:8880 fastkokoro:gpu
```

Docker Hub GPU:

```bash
docker run --gpus all -p 8880:8880 msgflux/fastkokoro:gpu
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
| `FASTKOKORO_ONNX_PROVIDERS` | `CPUExecutionProvider` |
| `FASTKOKORO_ONNX_AUTO_PROVIDERS` | `false` |
| `FASTKOKORO_ONNX_INTRA_OP_NUM_THREADS` | unset |
| `FASTKOKORO_ONNX_INTER_OP_NUM_THREADS` | unset |

`FASTKOKORO_WARMUP=true` runs a short synthesis during startup. This makes the
server take a little longer to become ready, but avoids paying most of the first
request latency on the first user request.

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

Intel/OpenVINO builds can use:

```bash
FASTKOKORO_ONNX_PROVIDERS=OpenVINOExecutionProvider,CPUExecutionProvider uv run fastkokoro
```

Set `FASTKOKORO_ONNX_AUTO_PROVIDERS=true` to pass every provider available in the
installed ONNX Runtime build to the session. Use this mostly for quick local
experiments; production deployments should pin an explicit provider order.

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

Brazilian Portuguese works best when using one of the PT-BR voices:
`pf_dora`, `pm_alex`, or `pm_santa`.

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
