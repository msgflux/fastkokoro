#!/bin/bash
set -e

# FastKokoro setup for Google Colab
# Installs all deps manually to avoid onnxruntime (CPU) <-> onnxruntime-gpu conflicts.

REPO_DIR="/content/fastkokoro"
BRANCH="perf/engine-warmup-cache-streaming"

echo "=== Cloning/updating fastkokoro ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/msgflux/fastkokoro "$REPO_DIR"
fi
cd "$REPO_DIR"
git checkout "$BRANCH" 2>/dev/null || true
git pull origin "$BRANCH" 2>/dev/null || true

echo "=== Installing all deps except onnxruntime* ==="
pip install \
    "fastapi>=0.115.0" \
    "huggingface-hub>=0.36.0" \
    "kokoro-onnx>=0.5.0" \
    "numba>=0.65.0" \
    "numpy>=2.0.0" \
    "onnx>=1.16.0" \
    "onnx-ir>=0.1.0" \
    "orjson>=3.10.0" \
    "pydantic>=2.0.0" \
    "soundfile>=0.13.0" \
    "sympy>=1.13.0" \
    "uvicorn>=0.32.0"

echo "=== Removing CPU onnxruntime (pulled by kokoro-onnx) ==="
pip uninstall onnxruntime onnxruntime-gpu -y 2>/dev/null || true

echo "=== Installing onnxruntime-gpu ==="
# Try latest first; if CUDA 13+, may need a very recent ORT.
pip install --upgrade onnxruntime-gpu || {
    echo "Trying onnxruntime-gpu 1.19.2..."
    pip install onnxruntime-gpu==1.19.2
}

echo "=== Installing fastkokoro (without re-resolving deps) ==="
pip install -e "$REPO_DIR" --no-deps

echo ""
echo "=== Done ==="
echo "Restart runtime (Runtime -> Restart runtime), then test:"
echo "  from fastkokoro.engine import FastKokoro"
echo "  e = FastKokoro()"
echo "  print(e.session.get_providers())"
