from __future__ import annotations

import logging
from pathlib import Path

from fastkokoro.config import Settings

logger = logging.getLogger("uvicorn.error")


def resolve_quantized_model_path(model_path: Path, settings: Settings) -> Path:
    if settings.onnx_weight_only_nbits is None:
        return model_path

    output_path = _quantized_model_path(model_path, settings)
    if output_path.exists():
        logger.info("Using cached weight-only ONNX model: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _quantize_matmul_nbits(model_path, output_path, settings)
    return output_path


def _quantized_model_path(model_path: Path, settings: Settings) -> Path:
    mode = "sym" if settings.onnx_weight_only_symmetric else "asym"
    filename = (
        f"{model_path.stem}-matmul-nbits{settings.onnx_weight_only_nbits}"
        f"-b{settings.onnx_weight_only_block_size}"
        f"-acc{settings.onnx_weight_only_accuracy_level}"
        f"-{mode}.onnx"
    )
    return settings.cache_dir / "quantized" / filename


def _quantize_matmul_nbits(
    model_path: Path,
    output_path: Path,
    settings: Settings,
) -> None:
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
    except ImportError as exc:
        raise RuntimeError(
            "Weight-only nbits quantization requires onnxruntime with "
            "matmul_nbits_quantizer plus onnx, onnx-ir, and sympy installed."
        ) from exc

    bits = settings.onnx_weight_only_nbits
    if bits is None:
        return

    op_types = ("MatMul",)
    quant_axes = (("MatMul", 0),)
    logger.info(
        "Quantizing ONNX model with MatMulNBits: model=%s output=%s bits=%s "
        "block_size=%s accuracy_level=%s symmetric=%s",
        model_path,
        output_path,
        bits,
        settings.onnx_weight_only_block_size,
        settings.onnx_weight_only_accuracy_level,
        settings.onnx_weight_only_symmetric,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
        logging.WARNING
    )
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=settings.onnx_weight_only_block_size,
        is_symmetric=settings.onnx_weight_only_symmetric,
        accuracy_level=settings.onnx_weight_only_accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=op_types,
        quant_axes=quant_axes,
    )
    config.bits = bits
    model = quant_utils.load_model_with_shape_infer(model_path)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=settings.onnx_weight_only_block_size,
        is_symmetric=settings.onnx_weight_only_symmetric,
        accuracy_level=settings.onnx_weight_only_accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=op_types,
        quant_axes=quant_axes,
        algo_config=config,
        nodes_to_exclude=None,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_path), True)
    logger.info("Weight-only ONNX model written: %s", output_path)
