#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${FASTKOKORO_PORT:-8880}"
DISPLAY_NAME="${DISPLAY:-:0}"
VIDEO_SIZE="${FASTKOKORO_RECORD_SIZE:-1920x1080}"
FRAMERATE="${FASTKOKORO_RECORD_FPS:-30}"
OUT_DIR="${FASTKOKORO_RECORD_OUT:-$ROOT_DIR/demo-output/terminal-recording}"
SCREEN_VIDEO="$OUT_DIR/screen.mp4"
FINAL_VIDEO="$OUT_DIR/fastkokoro-streaming-terminal-demo.mp4"
MARKER="$OUT_DIR/.done"

mkdir -p "$OUT_DIR"
rm -f "$MARKER" "$SCREEN_VIDEO" "$FINAL_VIDEO"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal not found" >&2
  exit 1
fi

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null; then
  echo "FastKokoro is not healthy at http://localhost:$PORT/health" >&2
  exit 1
fi

ffmpeg \
  -y \
  -loglevel warning \
  -video_size "$VIDEO_SIZE" \
  -framerate "$FRAMERATE" \
  -f x11grab \
  -i "$DISPLAY_NAME" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  "$SCREEN_VIDEO" &
FFMPEG_PID=$!

cleanup() {
  if kill -0 "$FFMPEG_PID" >/dev/null 2>&1; then
    kill -INT "$FFMPEG_PID" >/dev/null 2>&1 || true
    wait "$FFMPEG_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

gnome-terminal \
  --title="FastKokoro GPU Streaming Demo" \
  --geometry=120x36 \
  -- bash -lc "
    cd '$ROOT_DIR'
    clear
    printf 'FastKokoro GPU Streaming Demo\n\n'
    printf 'Server: http://localhost:$PORT\n'
    printf 'Strategy: adaptive\n'
    printf 'Output: $OUT_DIR\n\n'
    FASTKOKORO_DEMO_START_SERVER=false \
    FASTKOKORO_DEMO_PLAY=false \
    FASTKOKORO_DEMO_OUT='$OUT_DIR' \
    scripts/render_streaming_video_demo.sh
    printf '\nRecording complete. Closing in 3 seconds...\n'
    sleep 3
    touch '$MARKER'
  "

for _ in $(seq 1 120); do
  if [[ -f "$MARKER" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "$MARKER" ]]; then
  echo "Terminal recording command did not finish before timeout" >&2
  exit 1
fi

cleanup
trap - EXIT

ffmpeg \
  -y \
  -loglevel warning \
  -i "$SCREEN_VIDEO" \
  -i "$OUT_DIR/demo-stream.wav" \
  -map 0:v:0 \
  -map 1:a:0 \
  -shortest \
  -c:v copy \
  -c:a aac \
  "$FINAL_VIDEO"

cat <<EOF
Generated:
  Screen: $SCREEN_VIDEO
  Audio: $OUT_DIR/demo-stream.wav
  Final: $FINAL_VIDEO
EOF
