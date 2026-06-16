from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnx

SHAPE_DRIVER_OPS = frozenset(
    {
        "Cast",
        "Compress",
        "Concat",
        "ConstantOfShape",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "Identity",
        "Pad",
        "Range",
        "Reshape",
        "ScatterElements",
        "Shape",
        "Slice",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)


@dataclass(frozen=True)
class TensorShape:
    name: str
    dims: tuple[int | str | None, ...]
    source: str

    @property
    def is_dynamic(self) -> bool:
        return any(not isinstance(dim, int) for dim in self.dims)


@dataclass(frozen=True)
class DynamicBarrier:
    input_name: str
    input_dims: tuple[int | str | None, ...]
    node_name: str
    node_op_type: str
    output_name: str
    output_dims: tuple[int | str | None, ...]


@dataclass(frozen=True)
class DynamicNode:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dynamic_inputs: tuple[str, ...]
    dynamic_outputs: tuple[str, ...]


@dataclass(frozen=True)
class FixedShapeInspectionReport:
    model_path: Path
    inferred_shapes: bool
    graph_inputs: tuple[TensorShape, ...]
    graph_outputs: tuple[TensorShape, ...]
    dynamic_tensors: tuple[TensorShape, ...]
    seed_inputs: tuple[str, ...]
    reachable_dynamic_tensors: tuple[TensorShape, ...]
    reachable_dynamic_nodes: tuple[DynamicNode, ...]
    input_slice_barriers: tuple[DynamicBarrier, ...]
    shape_driver_counts: dict[str, int]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "inferred_shapes": self.inferred_shapes,
            "graph_inputs": [asdict(item) for item in self.graph_inputs],
            "graph_outputs": [asdict(item) for item in self.graph_outputs],
            "dynamic_tensors": [asdict(item) for item in self.dynamic_tensors],
            "seed_inputs": list(self.seed_inputs),
            "reachable_dynamic_tensors": [
                asdict(item) for item in self.reachable_dynamic_tensors
            ],
            "reachable_dynamic_nodes": [
                asdict(item) for item in self.reachable_dynamic_nodes
            ],
            "input_slice_barriers": [
                asdict(item) for item in self.input_slice_barriers
            ],
            "shape_driver_counts": self.shape_driver_counts,
            "notes": list(self.notes),
        }


def inspect_fixed_shape_readiness(
    model_path: Path,
    *,
    seed_inputs: tuple[str, ...] = ("tokens", "input_ids"),
    infer_shapes: bool = True,
) -> FixedShapeInspectionReport:
    model = onnx.load(model_path, load_external_data=False)
    inferred = False
    if infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
            inferred = True
        except Exception:
            inferred = False

    shape_map = _collect_tensor_shapes(model)
    _, consumer_map = _build_tensor_maps(model)
    graph_inputs = tuple(
        sorted(
            (
                shape_map.get(value.name, TensorShape(value.name, (), "graph.input"))
                for value in model.graph.input
            ),
            key=lambda item: item.name,
        )
    )
    graph_outputs = tuple(
        sorted(
            (
                shape_map.get(value.name, TensorShape(value.name, (), "graph.output"))
                for value in model.graph.output
            ),
            key=lambda item: item.name,
        )
    )
    dynamic_tensors = tuple(
        sorted(
            (
                tensor
                for tensor in shape_map.values()
                if tensor.dims and tensor.is_dynamic
            ),
            key=lambda item: item.name,
        )
    )

    present_seed_inputs = tuple(name for name in seed_inputs if name in consumer_map)
    reachable_tensor_names, reachable_nodes = _trace_reachable(
        present_seed_inputs, consumer_map
    )
    reachable_dynamic_tensors = tuple(
        sorted(
            (
                tensor
                for name, tensor in shape_map.items()
                if name in reachable_tensor_names and tensor.dims and tensor.is_dynamic
            ),
            key=lambda item: item.name,
        )
    )
    reachable_dynamic_nodes = tuple(
        _summarize_dynamic_nodes(reachable_nodes, shape_map)
    )
    input_slice_barriers = tuple(
        _find_input_slice_barriers(
            graph_inputs,
            consumer_map,
            shape_map,
            set(present_seed_inputs),
        )
    )
    shape_driver_counts = dict(
        sorted(
            Counter(
                node.op_type
                for node in reachable_nodes
                if node.op_type in SHAPE_DRIVER_OPS
            ).items()
        )
    )

    notes: list[str] = []
    if input_slice_barriers:
        notes.append(
            "Fixed-shape inputs are being converted back into dynamic tensors by an "
            "early graph op. This blocks whole-graph fixed-shape behavior."
        )
    if reachable_dynamic_nodes:
        notes.append(
            "Dynamic shapes remain reachable from token inputs. Bucketizing only the "
            "external input tensor is not enough for CUDA Graph or stable IOBinding."
        )
    if not inferred:
        notes.append(
            "ONNX shape inference could not complete. The report falls back to graph "
            "metadata and may miss some intermediate tensor shapes."
        )

    return FixedShapeInspectionReport(
        model_path=model_path,
        inferred_shapes=inferred,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        dynamic_tensors=dynamic_tensors,
        seed_inputs=present_seed_inputs,
        reachable_dynamic_tensors=reachable_dynamic_tensors,
        reachable_dynamic_nodes=reachable_dynamic_nodes,
        input_slice_barriers=input_slice_barriers,
        shape_driver_counts=shape_driver_counts,
        notes=tuple(notes),
    )


def render_fixed_shape_report(report: FixedShapeInspectionReport) -> str:
    lines = [
        f"model: {report.model_path}",
        f"shape_inference: {'ok' if report.inferred_shapes else 'partial'}",
        f"seed_inputs: {', '.join(report.seed_inputs) or '(none found)'}",
        f"dynamic_tensors: {len(report.dynamic_tensors)}",
        f"reachable_dynamic_tensors: {len(report.reachable_dynamic_tensors)}",
        f"reachable_dynamic_nodes: {len(report.reachable_dynamic_nodes)}",
    ]
    if report.shape_driver_counts:
        drivers = ", ".join(
            f"{op}={count}" for op, count in report.shape_driver_counts.items()
        )
        lines.append(f"shape_driver_ops: {drivers}")

    if report.input_slice_barriers:
        lines.append("input_barriers:")
        for barrier in report.input_slice_barriers:
            lines.append(
                "  - "
                f"{barrier.input_name}{_format_dims(barrier.input_dims)} -> "
                f"{barrier.node_op_type}({barrier.node_name}) -> "
                f"{barrier.output_name}{_format_dims(barrier.output_dims)}"
            )

    if report.reachable_dynamic_nodes:
        lines.append("first_dynamic_nodes:")
        for node in report.reachable_dynamic_nodes[:10]:
            lines.append(
                "  - "
                f"{node.op_type}({node.name}): "
                f"dynamic_inputs={','.join(node.dynamic_inputs) or '-'} "
                f"dynamic_outputs={','.join(node.dynamic_outputs) or '-'}"
            )

    if report.notes:
        lines.append("notes:")
        for note in report.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def _trace_reachable(seed_inputs: tuple[str, ...], consumer_map):
    reachable_tensors = set(seed_inputs)
    reachable_nodes = []
    queue = deque(seed_inputs)
    seen_nodes = set()
    while queue:
        tensor_name = queue.popleft()
        for node in consumer_map.get(tensor_name, []):
            identity = (node.name, tuple(node.output))
            if identity in seen_nodes:
                continue
            seen_nodes.add(identity)
            reachable_nodes.append(node)
            for output in node.output:
                if output and output not in reachable_tensors:
                    reachable_tensors.add(output)
                    queue.append(output)
    return reachable_tensors, reachable_nodes


def _summarize_dynamic_nodes(
    nodes,
    shape_map: dict[str, TensorShape],
) -> list[DynamicNode]:
    result = []
    for node in nodes:
        input_shapes = {
            name: shape_map.get(name, TensorShape(name, (), "")) for name in node.input
        }
        output_shapes = {
            name: shape_map.get(name, TensorShape(name, (), "")) for name in node.output
        }
        dynamic_inputs = tuple(
            name for name, shape in input_shapes.items() if shape.is_dynamic
        )
        dynamic_outputs = tuple(
            name for name, shape in output_shapes.items() if shape.is_dynamic
        )
        if not dynamic_inputs and not dynamic_outputs:
            continue
        result.append(
            DynamicNode(
                name=node.name or node.op_type,
                op_type=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                dynamic_inputs=dynamic_inputs,
                dynamic_outputs=dynamic_outputs,
            )
        )
    return result


def _find_input_slice_barriers(
    graph_inputs: tuple[TensorShape, ...],
    consumer_map,
    shape_map: dict[str, TensorShape],
    eligible_inputs: set[str],
) -> list[DynamicBarrier]:
    barriers = []
    for graph_input in graph_inputs:
        if graph_input.name not in eligible_inputs:
            continue
        if graph_input.is_dynamic or not graph_input.dims:
            continue
        for node in consumer_map.get(graph_input.name, []):
            for output in node.output:
                output_shape = shape_map.get(output)
                if output_shape is None or not output_shape.is_dynamic:
                    continue
                if node.op_type not in SHAPE_DRIVER_OPS:
                    continue
                barriers.append(
                    DynamicBarrier(
                        input_name=graph_input.name,
                        input_dims=graph_input.dims,
                        node_name=node.name or node.op_type,
                        node_op_type=node.op_type,
                        output_name=output,
                        output_dims=output_shape.dims,
                    )
                )
    return barriers


def _build_tensor_maps(model: onnx.ModelProto):
    producer_map = {}
    consumer_map = defaultdict(list)
    for node in model.graph.node:
        for output in node.output:
            if output:
                producer_map[output] = node
        for name in node.input:
            if name:
                consumer_map[name].append(node)
    for value in model.graph.input:
        consumer_map.setdefault(value.name, [])
    return producer_map, consumer_map


def _collect_tensor_shapes(model: onnx.ModelProto) -> dict[str, TensorShape]:
    items: dict[str, TensorShape] = {}
    for value in model.graph.input:
        shape = _tensor_shape(value)
        if shape is not None:
            items[value.name] = TensorShape(value.name, shape, "graph.input")
    for value in model.graph.output:
        shape = _tensor_shape(value)
        if shape is not None:
            items[value.name] = TensorShape(value.name, shape, "graph.output")
    for value in model.graph.value_info:
        shape = _tensor_shape(value)
        if shape is not None:
            items[value.name] = TensorShape(value.name, shape, "graph.value_info")
    return items


def _tensor_shape(value: onnx.ValueInfoProto) -> tuple[int | str | None, ...] | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return tuple(dims)


def _format_dims(dims: tuple[int | str | None, ...]) -> str:
    if not dims:
        return "[]"
    return "[" + ",".join(str(dim) for dim in dims) + "]"
