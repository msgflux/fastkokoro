import onnx
from onnx import TensorProto, helper

from fastkokoro.fixed_shape_analysis import inspect_fixed_shape_readiness


def test_inspect_fixed_shape_readiness_detects_input_slice_barrier(tmp_path):
    model_path = tmp_path / "slice.onnx"
    tokens = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8])
    token_length = helper.make_tensor_value_info("token_length", TensorProto.INT64, [1])
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    sliced_info = helper.make_tensor_value_info(
        "input_ids_sliced",
        TensorProto.INT64,
        [1, "token_length_value"],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.INT64,
        [1, "token_length_value"],
    )
    slice_node = helper.make_node(
        "Slice",
        inputs=["input_ids", "starts", "token_length", "axes", "steps"],
        outputs=["input_ids_sliced"],
        name="FastKokoroTokenLengthSlice",
    )
    passthrough = helper.make_node(
        "Identity",
        inputs=["input_ids_sliced"],
        outputs=["output"],
        name="OutputIdentity",
    )
    graph = helper.make_graph(
        [slice_node, passthrough],
        "slice_graph",
        [tokens, token_length],
        [output],
        initializer=[starts, axes, steps],
        value_info=[sliced_info],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, model_path)

    report = inspect_fixed_shape_readiness(model_path)

    assert report.seed_inputs == ("input_ids",)
    assert len(report.input_slice_barriers) == 1
    barrier = report.input_slice_barriers[0]
    assert barrier.node_op_type == "Slice"
    assert barrier.input_name == "input_ids"
    assert barrier.output_name == "input_ids_sliced"
    assert barrier.output_dims == (1, "token_length_value")
    assert report.shape_driver_counts["Slice"] == 1
    assert any(tensor.name == "input_ids_sliced" for tensor in report.dynamic_tensors)
    assert any(
        node.name == "FastKokoroTokenLengthSlice"
        for node in report.reachable_dynamic_nodes
    )


def test_inspect_fixed_shape_readiness_handles_fully_fixed_graph(tmp_path):
    model_path = tmp_path / "fixed.onnx"
    tokens = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [1, 8])
    node = helper.make_node(
        "Identity",
        inputs=["input_ids"],
        outputs=["output"],
        name="IdentityOut",
    )
    graph = helper.make_graph([node], "fixed_graph", [tokens], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path)

    report = inspect_fixed_shape_readiness(model_path)

    assert report.seed_inputs == ("input_ids",)
    assert report.input_slice_barriers == ()
    assert report.dynamic_tensors == ()
    assert report.reachable_dynamic_nodes == ()
