# fastkokoro

Lightweight OpenAI-compatible Kokoro TTS server powered by ONNX Runtime.

`fastkokoro` runs the 82M-parameter Kokoro text-to-speech model with low startup
overhead, fast local inference, and a small dependency footprint. It supports CPU
and GPU execution through ONNX Runtime providers, including CUDA, TensorRT, and
other providers when the matching runtime package is installed. The default
model is the fixed-bucket streaming export:
`msgflux/Kokoro-82M-streaming-onnx`.

The default checkpoint is `onnx/kokoro-82m-streaming-b96-fp16.onnx`.

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
TensorRT engine caching:

```bash
docker run --gpus all -p 8880:8880 -v fastkokoro-models:/models msgflux/fastkokoro:tensorrt
```

## API

Generate speech:

```bash
curl http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Hello from fastkokoro.",
    "voice": "af_heart",
    "lang": "en-us",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Stream raw PCM audio:

```bash
curl http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "input": "Streaming from fastkokoro.",
    "voice": "af_heart",
    "lang": "en-us",
    "response_format": "pcm",
    "stream": true
  }' \
  --output speech.pcm
```

Service endpoints:

```bash
curl http://localhost:8880/health
curl http://localhost:8880/v1/models
curl http://localhost:8880/metrics
```

The metrics endpoint reports request latency, speech latency, streaming chunks,
total bytes, time to first speech chunk, and active ONNX Runtime providers.

The server exposes the local model as `kokoro`. For client compatibility,
`/v1/audio/speech` also accepts `tts-1` and `gpt-4o-mini-tts` as aliases.

## OpenAI SDK

Point the OpenAI Python SDK at the local server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="fastkokoro",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_heart",
    input="Hello from fastkokoro.",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

The repository also includes directly runnable examples with inline
dependencies:

```bash
uv run examples/tts_save_file.py
uv run examples/tts_stream_chunks.py
```

They accept `FASTKOKORO_BASE_URL`, `FASTKOKORO_API_KEY`,
`FASTKOKORO_VOICE`, `FASTKOKORO_TEXT`, and `FASTKOKORO_TTS_OUTPUT`.

## Python API

```python
from fastkokoro import FastKokoro

engine = FastKokoro()
audio = engine.create(
    "Hello from fastkokoro.",
    voice="af_heart",
    response_format="wav",
)
```

## Docker Details

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

TensorRT builds an engine the first time it sees a model, bucket, GPU
architecture, and provider option set. Keep `/models` mounted so
`/models/trt-cache` persists; otherwise startup will pay the TensorRT engine
build cost again.

The CUDA 11.8 legacy image is kept for older hosts, but TensorRT EP support is
published only through the TensorRT 25.06 image. Current Python 3.11 ONNX
Runtime GPU wheels expect TensorRT 10 libraries for TensorRT EP.

## Advanced Configuration

### Performance

Final checkpoint measurements on a GTX 1650 (SM75) with CUDA I/O Binding:

| Bucket | ORT | Provider | p50 | p90 | First engine build |
| ---: | ---: | --- | ---: | ---: | ---: |
| 96 | 1.18.1 | CUDA | 169.04 ms | 169.83 ms | - |
| 96 | 1.22.0 | TensorRT 10.11 | 118.16 ms | 132.63 ms | 206.96 s |

The TensorRT measurement is a cache-hit model call after five warmups. The b96
checkpoint compiled into one TensorRT engine with no CUDA or CPU node fallback.
Persist `/models/trt-cache` because engines are specific to the model, bucket,
runtime, and GPU architecture.

### Environment Variables

| Variable | Default |
| --- | --- |
| `FASTKOKORO_HOST` | `0.0.0.0` |
| `FASTKOKORO_PORT` | `8880` |
| `FASTKOKORO_MODEL_REPO` | `msgflux/Kokoro-82M-streaming-onnx` |
| `FASTKOKORO_MODEL_FILE` | `onnx/kokoro-82m-streaming-b96-fp16.onnx` |
| `FASTKOKORO_MODEL_PATH` | unset; downloads from Hugging Face |
| `FASTKOKORO_VOICES_FILE` | `voices.npz` |
| `FASTKOKORO_VOICES_INDEX_FILE` | `voices.txt` |
| `FASTKOKORO_VOICES_PATH` | unset; downloads from Hugging Face |
| `FASTKOKORO_DEFAULT_VOICE` | `af_heart` |
| `FASTKOKORO_DEFAULT_LANG` | `en-us` |
| `FASTKOKORO_WARMUP` | `true` |
| `FASTKOKORO_WARMUP_TEXT` | `Hello there. This is a warmup request for streaming speech generation.` |
| `FASTKOKORO_WARMUP_REQUEST` | `false` |
| `FASTKOKORO_STREAM_STRATEGY` | `sentence` |
| `FASTKOKORO_STREAM_AUDIO_FRAME_MS` | `200` |
| `FASTKOKORO_STREAM_BOUNDARY_SILENCE_MS` | `0` |
| `FASTKOKORO_STREAM_MAX_SEGMENT_CHARS` | unset; scheduled strategies choose from bucket |
| `FASTKOKORO_STREAM_MAX_SEGMENT_WORDS` | unset; scheduled strategies choose from bucket |
| `FASTKOKORO_RUNTIME_TAIL_TRIM_MS` | `150`; b48+ defaults to `220` unless explicitly set |
| `FASTKOKORO_RUNTIME_TAIL_FADE_MS` | `72`; b48+ defaults to `96` unless explicitly set |
| `FASTKOKORO_RUNTIME_PART_TRIM_PADDING_MS` | `80` |
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

### Model Geometry and Export

Long inputs are split automatically using the loaded model's token and duration
limits.

```bash
FASTKOKORO_MODEL_FILE=onnx/kokoro-82m-streaming-b96-fp16.onnx
```

Bucket size controls both token width and the fixed alignment window. Two token
positions are reserved by the model, so usable text capacity is `bucket - 2`
phoneme tokens. Practical word capacity is lower and depends on language,
punctuation, voice, and speed because the model also predicts duration. The
table below is the observed safe expectation from English/Portuguese probes at
speed `1.0`. The supported synthesis speed range is `1.0` through `2.0`.
The default `sentence` strategy uses this conservative capacity before falling
back to the model's real phonemized token width, so long sentences are split
before late words land at the tail of a near-full fixed-output window.

| Bucket | Usable tokens | Alignment frames | Output samples | Expected words | TensorRT p50 | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 96 | 94 | 200 | 104,400 | 14 | 118 ms | Adaptive margin 4,320/8,400 |

The server splits fixed-width ONNX requests by phonemized token count before
running inference, so a text segment is not sent to a checkpoint with more valid
tokens than its input width supports. Corrected exports also publish their
alignment and tail geometry as ONNX metadata. When predicted duration exceeds
the safe alignment capacity, the server retries smaller phoneme batches instead
of returning a truncated waveform. The b96 graph masks waveform output with a
4,320-sample margin for inputs up to 32 valid tokens and an 8,400-sample margin
above that threshold. This avoids short-utterance vocoder noise while preserving
longer endings. The exporter implements each margin by shifting the fixed
mask-position vector into negative indices, not by adding the margin to the
predicted active-sample count. Set
`FASTKOKORO_RUNTIME_TAIL_TRIM_MS` and `FASTKOKORO_RUNTIME_TAIL_FADE_MS` only if
you need to override the default final cleanup.

Direct PyTorch measurement shows that Kokoro's decoder tensor contains exactly
600 audio samples per alignment frame. Listening tests show that the useful
speech boundary is better modeled by 480 samples per predicted-duration frame;
keeping all 600 preserves stochastic vocoder output that is heard as tail noise.
The b96 recipe below uses the listening-tested 480 scale and selects the shifted
4,320/8,400-sample margin inside the ONNX graph from `input_lengths`.

```bash
B=96
ALIGN=$((2 * B + 8))
TAIL_MARGIN=8400
SAMPLES=$((ALIGN * 480 + TAIL_MARGIN))
SNAPSHOT="$HOME/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"

uv run \
  --with torch==2.5.1 \
  --with transformers==4.48.3 \
  --with onnx==1.21.0 \
  --with numpy==1.26.4 \
  --with huggingface-hub==0.36.2 \
  --with loguru==0.7.3 \
  --with 'misaki[en]==0.9.4' \
  python scripts/export_kokoro_torch_ttfc.py \
    --kokoro-repo demo-output/reexport/hexgrad-kokoro \
    --config "$SNAPSHOT/config.json" \
    --checkpoint "$SNAPSHOT/kokoro-v1_0.pth" \
    --output "demo-output/reexport/candidates/kokoro-82m-streaming-b96-align200-decoder-fp16-frame480-margin8400-short4320t32-foldrecip.onnx" \
    --bucket "$B" \
    --fixed-alignment-frames "$ALIGN" \
    --fixed-output-samples "$SAMPLES" \
    --output-samples-per-frame 480 \
    --output-tail-margin-samples "$TAIL_MARGIN" \
    --output-short-tail-margin-samples 4320 \
    --output-short-tail-margin-max-tokens 32 \
    --precision decoder-fp16 \
    --opset 17 \
    --legacy-export \
    --length-aware \
    --patch-fixed-lstm \
    --patch-scatterless-sine-source \
    --patch-split-adain \
    --patch-albert-sdpa-bool-mask-scale \
    --fold-constant-reciprocals \
    --device cuda
```

Post-process the export with standard ONNX operators only:

```bash
uv run --with onnxsim==0.6.5 --with onnx==1.21.0 --with numpy==1.26.4 \
  python scripts/optimize_kokoro_onnx.py \
    --input "$EXPORTED_MODEL" \
    --output "$FINAL_MODEL" \
    --simplify \
    --atan2 portable
```

The published graph uses opset 17, standard ALBERT attention subgraphs, and a
portable FP32 polynomial replacement for the vocoder's `atan2`. It requires no
external custom-op library and loads in ONNX Runtime 1.16.3, 1.17.3, and 1.18.1.
An experimental fusion to `com.microsoft.Attention` was discarded: it improved
CUDA latency by only 2-3%, while TensorRT 10.11 rejected the 12 fused nodes and
fragmented execution across providers on SM75. The release optimizer therefore
does not expose or apply that transformation.

Published artifact checksums:

| File | Nodes | SHA-256 |
| --- | ---: | --- |
| `onnx/kokoro-82m-streaming-b96-fp16.onnx` | 1,696 | `7a77a144601a7a37cee060cd26cdab3c6125f61cbcf87ddd0fc0f929c9d67ad8` |

### Runtime Behavior

`FASTKOKORO_JIT` is enabled by default for PCM encoding and trim. The first call
compiles the kernels, so keep startup warmup enabled to absorb this cost before
serving requests. Set `FASTKOKORO_JIT=false` to force the NumPy path.

Enable built-in profiling with `FASTKOKORO_PROFILE=true` to write `cProfile` artifacts for
startup warmup and speech requests under `FASTKOKORO_PROFILE_DIR`. Each run produces a
raw `.prof` file plus a `.txt` summary sorted by cumulative time. Use
`FASTKOKORO_PROFILE_WARMUP` and `FASTKOKORO_PROFILE_REQUESTS` to narrow profiling to
startup or request handling when debugging TTFC regressions.

`FASTKOKORO_STREAM_STRATEGY=sentence` is the default. It synthesizes one sentence
at a time, then applies the loaded ONNX bucket's real phonemized token width if a
sentence is too long. This favors natural continuity over aggressively small
text chunks. `phrase` splits on phrase punctuation such as commas, semicolons,
and question marks. `adaptive` and `chunk` use scheduled word-boundary chunks to
reduce first-audio latency, but can sound less natural on some voices/languages.
`FASTKOKORO_STREAM_MAX_SEGMENT_WORDS` and
`FASTKOKORO_STREAM_MAX_SEGMENT_CHARS` are unset and only act as explicit user
overrides for those scheduled strategies when configured. Static ONNX buckets
still apply their safe token-width cap, so overrides cannot send more text than
the checkpoint should handle. For `response_format=pcm`, the server also slices
each generated segment into smaller audio frames controlled by
`FASTKOKORO_STREAM_AUDIO_FRAME_MS`. `FASTKOKORO_RUNTIME_PART_TRIM_PADDING_MS`
keeps a small margin around the non-silent audio detected in each generated
part so syllable tails are not clipped when segments are concatenated.
`FASTKOKORO_STREAM_BOUNDARY_SILENCE_MS` can add silence between adjacent
generated text segments, but defaults to `0`. Explicit `[pause:...]` segments
control their own silence and do not receive extra boundary silence. Set
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
