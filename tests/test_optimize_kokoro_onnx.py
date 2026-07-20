from onnx import TensorProto, helper

from scripts.optimize_kokoro_onnx import remove_unused_initializers


def test_remove_unused_initializers_keeps_referenced_values():
    used = helper.make_tensor("used", TensorProto.FLOAT, [1], [1.0])
    unused = helper.make_tensor("unused", TensorProto.FLOAT, [1], [2.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "used"], ["output"])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [used, unused],
    )
    model = helper.make_model(graph)

    removed = remove_unused_initializers(model)

    assert removed == 1
    assert [initializer.name for initializer in model.graph.initializer] == ["used"]
