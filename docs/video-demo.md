# Streaming Video Demo

Use this workflow to generate demo audio from the GPU Docker image while the
client receives PCM chunks in streaming mode.

## Prerequisites

- Docker with NVIDIA Container Toolkit.
- `ffmpeg` for WAV and MP4 rendering.
- `ffplay` only if live playback is enabled.

## Generate The Demo

```bash
FASTKOKORO_IMAGE=msgflux/fastkokoro:gpu \
FASTKOKORO_DEMO_PLAY=true \
scripts/render_streaming_video_demo.sh
```

The script starts a GPU container, waits for `/health`, streams
`/v1/audio/speech` with `response_format=pcm`, writes the raw PCM file as chunks
arrive, converts it to WAV, and renders a simple MP4 placeholder.

Outputs are written under `demo-output/`:

- `demo-stream.pcm`
- `demo-stream.wav`
- `demo-stream.mp4`

## Useful Options

```bash
# Use the CUDA 11.8/cuDNN8 legacy image.
FASTKOKORO_IMAGE=msgflux/fastkokoro:gpu-legacy scripts/render_streaming_video_demo.sh

# Use an already running server instead of starting Docker.
FASTKOKORO_DEMO_START_SERVER=false scripts/render_streaming_video_demo.sh

# Change voice, language, output directory, or script text.
FASTKOKORO_VOICE=pf_dora \
FASTKOKORO_LANG=pt-br \
FASTKOKORO_DEMO_TEXT_FILE=/path/to/text.txt \
FASTKOKORO_DEMO_OUT=/tmp/fastkokoro-demo \
scripts/render_streaming_video_demo.sh
```

The generated MP4 is intentionally minimal. For the final presentation video,
use `demo-stream.wav` as the authoritative audio track and replace the
placeholder visual layer with the edited screen recording or motion graphics.
