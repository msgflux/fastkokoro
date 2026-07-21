# Changelog

## [Unreleased]

## [0.6.1] - 2026-07-21

## [0.6.0] - 2026-07-21

## [0.5.0] - 2026-07-20

## [0.4.0] - 2026-06-25

- Switched the default model repository to
  `msgflux/Kokoro-82M-streaming-onnx` and the default checkpoint to the
  optimized fixed-bucket `onnx/kokoro-82m-streaming-b24-fp16.onnx` export.
- Added support for the streaming checkpoint family (`b16`, `b24`, `b32`,
  `b48`) so users can choose lower TTFC or larger chunks by changing
  `FASTKOKORO_MODEL_FILE`.
- Reworked the default streaming schedule around the optimized b24 model:
  `adaptive` remains the default, short sentences stay intact, and longer
  sentences are split on word boundaries to keep first-audio latency bounded.
- Made scheduled first chunks scale with the loaded model bucket by default:
  b16 targets 1 word, b24 targets 2 words, b32 targets 4 words, and b48 targets
  6 words, while explicit stream segment environment variables still override
  the automatic schedule.
- Added TensorRT EP support through a dedicated TensorRT Docker image based on
  NVIDIA TensorRT 25.06 and ONNX Runtime GPU 1.22.x, with TensorRT engine and
  timing cache persisted under `/models/trt-cache`.
- Documented local GTX 1650 streaming latency measurements for the b24 model:
  CUDA EP around sub-60 ms TTFC and TensorRT EP around sub-50 ms TTFC after the
  TensorRT engine cache is built.
- Removed the old multi-shape warmup and direct NVIDIA checkpoint assumptions
  from the default release path in favor of one fixed-bucket streaming model per
  selected checkpoint.
- Updated benchmark and profiling scripts to use the same bucket-aware
  streaming scheduler as the runtime.

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
