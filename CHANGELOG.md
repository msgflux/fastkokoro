# Changelog

## [Unreleased]

## [0.3.0] - 2026-06-05

- Added low-latency streaming strategies, provider-aware scheduling, and
  benchmark/profile scripts for TTFC and end-to-end latency.
- Improved ONNX Runtime performance with configurable provider options,
  IOBinding, reusable buffers, graph optimization controls, multi-shape warmup,
  and weight-only quantization support.
- Added optional native/custom-op acceleration experiments for CPU AdaIN/STFT
  hotspots.
- Added explicit CUDA Docker tags, legacy GPU image publishing, and expanded
  install/source documentation.

## [0.2.0] - 2026-06-02

- Initial OpenAI-compatible Kokoro TTS server release.
- Added NVIDIA optimized ONNX model support with automatic voice conversion.
- Added CPU/GPU ONNX Runtime provider configuration, streaming speech responses,
  voice/language validation, Docker images, and PyPI release workflows.
