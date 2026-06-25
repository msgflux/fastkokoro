# Changelog

## [Unreleased]

## [0.4.0] - 2026-06-25

## [0.3.0] - 2026-06-05

- Added low-latency streaming strategies, provider-aware scheduling, and
  benchmark/profile scripts for TTFC and end-to-end latency, including the
  adaptive strategy as the Docker default.
- Improved ONNX Runtime performance with configurable provider options,
  IOBinding, reusable buffers, graph optimization controls, multi-shape warmup,
  and weight-only quantization support.
- Removed the runtime dependency on `kokoro-onnx` by vendoring the minimal
  Kokoro voice/tokenization path used by fastkokoro.
- Split ONNX Runtime installation into explicit `cpu` and `gpu` extras so the
  base package no longer installs a CPU runtime by default.
- Added optional native/custom-op acceleration experiments for CPU AdaIN/STFT
  hotspots.
- Added explicit CUDA Docker tags, legacy GPU image publishing, Dockerfiles that
  install directly from project metadata, and CI/release workflows that generate
  a temporary lockfile instead of tracking `uv.lock`.

## [0.2.0] - 2026-06-02

- Initial OpenAI-compatible Kokoro TTS server release.
- Added NVIDIA optimized ONNX model support with automatic voice conversion.
- Added CPU/GPU ONNX Runtime provider configuration, streaming speech responses,
  voice/language validation, Docker images, and PyPI release workflows.
