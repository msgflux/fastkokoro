from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx


@dataclass(frozen=True)
class FixedShapeVariantSpec:
    name: str
    fixed_input_bucket: int | None = None
    fixed_output_length: int | None = None
    bert_attention_mask: bool = False
    bert_fixed_embedding_indices: bool = False
    bert_fixed_sequence_length: bool = False


EXPERIMENTAL_VARIANTS = (
    FixedShapeVariantSpec(name="base"),
    FixedShapeVariantSpec(name="slice-input", fixed_input_bucket=64),
    FixedShapeVariantSpec(
        name="attn-mask-bert",
        fixed_input_bucket=64,
        bert_attention_mask=True,
    ),
    FixedShapeVariantSpec(
        name="attn-mask-bert-emb",
        fixed_input_bucket=64,
        bert_attention_mask=True,
        bert_fixed_embedding_indices=True,
    ),
    FixedShapeVariantSpec(
        name="attn-mask-bert-emb-len",
        fixed_input_bucket=64,
        bert_attention_mask=True,
        bert_fixed_embedding_indices=True,
        bert_fixed_sequence_length=True,
    ),
    FixedShapeVariantSpec(name="output-pad", fixed_output_length=120000),
    FixedShapeVariantSpec(
        name="fixed-io",
        fixed_input_bucket=64,
        fixed_output_length=120000,
    ),
)


def write_fixed_shape_variant(
    model_path: Path,
    output_path: Path,
    *,
    fixed_input_bucket: int | None = None,
    fixed_output_length: int | None = None,
    bert_attention_mask: bool = False,
    bert_fixed_embedding_indices: bool = False,
    bert_fixed_sequence_length: bool = False,
) -> Path:
    model = onnx.load(model_path, load_external_data=False)
    if bert_attention_mask:
        if fixed_input_bucket is None:
            raise ValueError(
                "bert_attention_mask experiment requires fixed_input_bucket"
            )
        apply_fixed_input_with_attention_mask(model, fixed_input_bucket)
        if bert_fixed_embedding_indices:
            apply_fixed_bert_embedding_indices(model, fixed_input_bucket)
        if bert_fixed_sequence_length:
            apply_fixed_bert_sequence_length(model, fixed_input_bucket)
    elif fixed_input_bucket is not None:
        apply_fixed_token_slice(model, fixed_input_bucket)
    if fixed_output_length is not None:
        apply_output_pad(model, fixed_output_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def apply_fixed_input_with_attention_mask(
    model: onnx.ModelProto,
    bucket: int,
) -> None:
    if bucket <= 1:
        raise ValueError("fixed_input_bucket must be greater than 1")

    token_input = _find_token_input(model)
    token_shape = token_input.type.tensor_type.shape
    token_shape.dim[0].dim_value = 1
    token_shape.dim[1].ClearField("dim_param")
    token_shape.dim[1].dim_value = bucket

    attention_mask_name = "attention_mask"
    existing_inputs = {value.name for value in model.graph.input}
    if attention_mask_name not in existing_inputs:
        model.graph.input.append(
            onnx.helper.make_tensor_value_info(
                attention_mask_name,
                onnx.TensorProto.INT64,
                [1, bucket],
            )
        )

    zero_i64_name = "fastkokoro_attention_mask_zero_i64"
    zero_f16_name = "fastkokoro_attention_mask_zero_f16"
    neg_f16_name = "fastkokoro_attention_mask_neg_f16"
    unsqueeze_axes_name = "fastkokoro_attention_mask_unsqueeze_axes"
    mask_equal_name = "fastkokoro_attention_mask_is_padding"
    mask_unsqueeze_name = "fastkokoro_attention_mask_unsqueezed"
    mask_bias_name = "fastkokoro_attention_mask_bias"

    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(np.array(0, dtype=np.int64), zero_i64_name),
    )
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(np.array(0, dtype=np.float16), zero_f16_name),
    )
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(
            np.array(-10000.0, dtype=np.float16),
            neg_f16_name,
        ),
    )
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(
            np.array([1, 2], dtype=np.int64),
            unsqueeze_axes_name,
        ),
    )

    replacement_nodes = [
        onnx.helper.make_node(
            "Equal",
            inputs=[attention_mask_name, zero_i64_name],
            outputs=[mask_equal_name],
            name="FastKokoroAttentionMaskEqual",
        ),
        onnx.helper.make_node(
            "Unsqueeze",
            inputs=[mask_equal_name, unsqueeze_axes_name],
            outputs=[mask_unsqueeze_name],
            name="FastKokoroAttentionMaskUnsqueeze",
        ),
        onnx.helper.make_node(
            "Where",
            inputs=[mask_unsqueeze_name, neg_f16_name, zero_f16_name],
            outputs=[mask_bias_name],
            name="FastKokoroAttentionMaskWhere",
        ),
    ]
    for node in replacement_nodes:
        _replace_or_append_node(model, node)

    old_bias_name = "/encoder/bert/Where_2_output_0"
    for node in model.graph.node:
        if node.name.startswith("FastKokoroAttentionMask"):
            continue
        for index, input_name in enumerate(node.input):
            if input_name == old_bias_name:
                node.input[index] = mask_bias_name

    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            mask_bias_name,
            onnx.TensorProto.FLOAT16,
            [1, 1, 1, bucket],
        )
    )


def apply_fixed_bert_embedding_indices(
    model: onnx.ModelProto,
    bucket: int,
) -> None:
    if bucket <= 1:
        raise ValueError("fixed_input_bucket must be greater than 1")

    position_ids_name = "fastkokoro_bert_position_ids"
    token_type_ids_name = "fastkokoro_bert_token_type_ids"
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(
            np.arange(bucket, dtype=np.int64).reshape(1, bucket),
            position_ids_name,
        ),
    )
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(
            np.zeros((1, bucket), dtype=np.int64),
            token_type_ids_name,
        ),
    )

    for node in model.graph.node:
        if node.name == "/encoder/bert/embeddings/position_embeddings/Gather":
            node.input[1] = position_ids_name
        elif node.name == "/encoder/bert/embeddings/token_type_embeddings/Gather":
            node.input[1] = token_type_ids_name


def apply_fixed_bert_sequence_length(
    model: onnx.ModelProto,
    bucket: int,
) -> None:
    if bucket <= 1:
        raise ValueError("fixed_input_bucket must be greater than 1")

    shape_vector_name = "fastkokoro_bert_shape_vector"
    seq_len_name = "fastkokoro_bert_seq_len"
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(
            np.array([1, bucket], dtype=np.int64),
            shape_vector_name,
        ),
    )
    _upsert_initializer(
        model,
        onnx.numpy_helper.from_array(np.array(bucket, dtype=np.int64), seq_len_name),
    )

    for node in model.graph.node:
        if node.name == "/encoder/bert/Gather_1":
            node.input[0] = shape_vector_name
        elif node.name == "/encoder/bert/Unsqueeze_1":
            node.input[0] = seq_len_name


def apply_fixed_token_slice(model: onnx.ModelProto, bucket: int) -> None:
    if bucket <= 1:
        raise ValueError("fixed_input_bucket must be greater than 1")

    token_input_name = None
    for graph_input in model.graph.input:
        if graph_input.name not in {"tokens", "input_ids"}:
            continue
        token_input_name = graph_input.name
        shape = graph_input.type.tensor_type.shape
        if len(shape.dim) != 2:
            raise ValueError(
                f"{graph_input.name} rank must be 2, got {len(shape.dim)}"
            )
        shape.dim[0].dim_value = 1
        shape.dim[1].ClearField("dim_param")
        shape.dim[1].dim_value = bucket
        break

    if token_input_name is None:
        raise ValueError("Model has no tokens/input_ids graph input")

    sliced_name = f"fastkokoro_{token_input_name}_sliced"
    token_length_name = "token_length"
    starts_name = "fastkokoro_token_slice_starts"
    axes_name = "fastkokoro_token_slice_axes"
    steps_name = "fastkokoro_token_slice_steps"

    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == token_input_name:
                node.input[index] = sliced_name

    model.graph.input.append(
        onnx.helper.make_tensor_value_info(
            token_length_name,
            onnx.TensorProto.INT64,
            [1],
        )
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), starts_name),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), axes_name),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), steps_name),
        ]
    )
    model.graph.node.insert(
        0,
        onnx.helper.make_node(
            "Slice",
            inputs=[
                token_input_name,
                starts_name,
                token_length_name,
                axes_name,
                steps_name,
            ],
            outputs=[sliced_name],
            name="FastKokoroTokenLengthSlice",
        ),
    )
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            sliced_name,
            onnx.TensorProto.INT64,
            [1, "token_length_value"],
        )
    )


def apply_output_pad(model: onnx.ModelProto, output_length: int) -> None:
    if output_length <= 0:
        raise ValueError("fixed_output_length must be greater than 0")

    original_output = model.graph.output.pop()
    original_output_name = original_output.name
    raw_output_name = f"{original_output_name}_unpadded"
    original_output.name = raw_output_name
    for node in model.graph.node:
        for index, output_name in enumerate(node.output):
            if output_name == original_output_name:
                node.output[index] = raw_output_name

    target_name = "fastkokoro_output_pad_target"
    zero_i64_name = "fastkokoro_output_pad_zero_i64"
    axis_name = "fastkokoro_output_pad_axis"
    value_name = "fastkokoro_output_pad_value"
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(
                np.array(output_length, dtype=np.int64),
                target_name,
            ),
            onnx.numpy_helper.from_array(np.array(0, dtype=np.int64), zero_i64_name),
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), axis_name),
            onnx.numpy_helper.from_array(np.array(0, dtype=np.float32), value_name),
        ]
    )
    model.graph.node.extend(
        [
            onnx.helper.make_node(
                "Shape",
                inputs=[raw_output_name],
                outputs=["fastkokoro_output_pad_shape"],
                name="FastKokoroOutputPadShape",
            ),
            onnx.helper.make_node(
                "Gather",
                inputs=["fastkokoro_output_pad_shape", zero_i64_name],
                outputs=["fastkokoro_output_pad_length"],
                axis=0,
                name="FastKokoroOutputPadGatherLength",
            ),
            onnx.helper.make_node(
                "Sub",
                inputs=[target_name, "fastkokoro_output_pad_length"],
                outputs=["fastkokoro_output_pad_tail"],
                name="FastKokoroOutputPadTail",
            ),
            onnx.helper.make_node(
                "Unsqueeze",
                inputs=[zero_i64_name, axis_name],
                outputs=["fastkokoro_output_pad_head_1d"],
                name="FastKokoroOutputPadHeadUnsqueeze",
            ),
            onnx.helper.make_node(
                "Unsqueeze",
                inputs=["fastkokoro_output_pad_tail", axis_name],
                outputs=["fastkokoro_output_pad_tail_1d"],
                name="FastKokoroOutputPadTailUnsqueeze",
            ),
            onnx.helper.make_node(
                "Concat",
                inputs=[
                    "fastkokoro_output_pad_head_1d",
                    "fastkokoro_output_pad_tail_1d",
                ],
                outputs=["fastkokoro_output_pad_pads"],
                axis=0,
                name="FastKokoroOutputPadPads",
            ),
            onnx.helper.make_node(
                "Pad",
                inputs=[
                    raw_output_name,
                    "fastkokoro_output_pad_pads",
                    value_name,
                ],
                outputs=[original_output_name],
                mode="constant",
                name="FastKokoroOutputPad",
            ),
        ]
    )
    model.graph.value_info.append(original_output)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            original_output_name,
            onnx.TensorProto.FLOAT,
            [output_length],
        )
    )


def _find_token_input(model: onnx.ModelProto) -> onnx.ValueInfoProto:
    for graph_input in model.graph.input:
        if graph_input.name in {"tokens", "input_ids"}:
            shape = graph_input.type.tensor_type.shape
            if len(shape.dim) != 2:
                raise ValueError(
                    f"{graph_input.name} rank must be 2, got {len(shape.dim)}"
                )
            return graph_input
    raise ValueError("Model has no tokens/input_ids graph input")


def _upsert_initializer(model: onnx.ModelProto, initializer: onnx.TensorProto) -> None:
    for index, existing in enumerate(model.graph.initializer):
        if existing.name == initializer.name:
            model.graph.initializer[index].CopyFrom(initializer)
            return
    model.graph.initializer.append(initializer)


def _replace_or_append_node(
    model: onnx.ModelProto,
    replacement: onnx.NodeProto,
) -> None:
    for index, existing in enumerate(model.graph.node):
        if existing.name == replacement.name:
            model.graph.node[index].CopyFrom(replacement)
            return
    model.graph.node.insert(0, replacement)
