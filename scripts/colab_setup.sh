#!/bin/bash
set -e

# FastKokoro setup for Google Colab (T4 GPU, CUDA 12.2, cuDNN 8.x)
# onnxruntime-gpu >= 1.20 requires cuDNN 9.x which Colab doesn't have;
# we need 1.19.x which supports cuDNN 8.x.

REPO_DIR="/content/fastkokoro"
BRANCH="perf/engine-warmup-cache-streaming"

echo "=== Cleaning previous onnxruntime installs ==="
pip uninstall onnxruntime onnxruntime-gpu -y 2>/dev/null || true

echo "=== Installing onnxruntime-gpu 1.19.2 (compatible with Colab cuDNN 8.x) ==="
pip install onnxruntime-gpu==1.19.2

echo "=== Installing kokoro-onnx (without pulling onnxruntime CPU) ==="
# Install deps that kokoro-onnx needs, then install kokoro-onnx without its own deps
pip install espeakng-loader phonemizer-fork numpy 2>/dev/null || true
pip install kokoro-onnx==0.5.0 --no-deps

echo "=== Installing fastkokoro ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/msgflux/fastkokoro "$REPO_DIR"
fi
cd "$REPO_DIR"
git checkout "$BRANCH" 2>/dev/null || true
git pull origin "$BRANCH" 2>/dev/null || true
pip install -e .[gpu] --no-deps

echo ""
echo "=== Done ==="
echo "Now restart runtime: Runtime -> Restart runtime"
echo "Then test:"
echo "  from fastkokoro.engine import FastKokoro"
echo "  e = FastKokoro()"
echo "  print(e.session.get_providers())"
