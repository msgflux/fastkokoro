#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${FASTKOKORO_IMAGE:-msgflux/fastkokoro:gpu}"
CONTAINER_NAME="${FASTKOKORO_CONTAINER_NAME:-fastkokoro-video-demo}"
PORT="${FASTKOKORO_PORT:-8880}"
VOICE="${FASTKOKORO_VOICE:-af_heart}"
LANG="${FASTKOKORO_LANG:-en-us}"
SPEED="${FASTKOKORO_SPEED:-1.0}"
TEXT_FILE="${FASTKOKORO_DEMO_TEXT_FILE:-$ROOT_DIR/examples/video_demo_script.txt}"
OUT_DIR="${FASTKOKORO_DEMO_OUT:-$ROOT_DIR/demo-output}"
PCM_FILE="$OUT_DIR/demo-stream.pcm"
WAV_FILE="$OUT_DIR/demo-stream.wav"
MP4_FILE="$OUT_DIR/demo-stream.mp4"
PLAY="${FASTKOKORO_DEMO_PLAY:-false}"
START_SERVER="${FASTKOKORO_DEMO_START_SERVER:-true}"
PYTHON_BIN="${FASTKOKORO_PYTHON:-python3}"
WARMUP="${FASTKOKORO_DEMO_WARMUP:-true}"

wait_for_server() {
  local url="http://localhost:$PORT/health"
  for _ in $(seq 1 180); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "FastKokoro did not become ready at $url" >&2
  return 1
}

start_server() {
  if docker ps --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null; then
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fx "$CONTAINER_NAME" >/dev/null; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi

  docker run \
    --detach \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --publish "$PORT:8880" \
    --env FASTKOKORO_STREAM_STRATEGY=adaptive \
    --env FASTKOKORO_ONNX_AUTO_PROVIDERS=true \
    "$IMAGE" >/dev/null
}

render_placeholder_video() {
  ffmpeg \
    -y \
    -loglevel warning \
    -f lavfi \
    -i "color=c=0x0b0f14:s=1920x1080:r=30" \
    -i "$WAV_FILE" \
    -vf "drawtext=text='FastKokoro GPU Streaming Demo':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2" \
    -shortest \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -c:a aac \
    "$MP4_FILE"
}

mkdir -p "$OUT_DIR"

if [[ "$START_SERVER" == "true" ]]; then
  start_server
fi

wait_for_server

PLAY_ARGS=()
if [[ "$PLAY" == "true" ]]; then
  PLAY_ARGS+=(--play)
fi

if [[ "$WARMUP" == "true" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/stream_speech_to_pcm.py" \
    --url "http://localhost:$PORT/v1/audio/speech" \
    --voice "$VOICE" \
    --lang "$LANG" \
    --speed "$SPEED" \
    --text "FastKokoro warmup." \
    --output "$OUT_DIR/warmup.pcm" >/dev/null
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/stream_speech_to_pcm.py" \
  --url "http://localhost:$PORT/v1/audio/speech" \
  --voice "$VOICE" \
  --lang "$LANG" \
  --speed "$SPEED" \
  --text-file "$TEXT_FILE" \
  --output "$PCM_FILE" \
  --wav-output "$WAV_FILE" \
  "${PLAY_ARGS[@]}"

render_placeholder_video

cat <<EOF
Generated:
  PCM: $PCM_FILE
  WAV: $WAV_FILE
  MP4: $MP4_FILE

Server:
  docker logs -f $CONTAINER_NAME
  docker rm -f $CONTAINER_NAME
EOF
