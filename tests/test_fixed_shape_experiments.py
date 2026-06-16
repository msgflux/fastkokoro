from onnx import TensorProto, helper

from fastkokoro.fixed_shape_experiments import (
    apply_fixed_bert_attention_reshapes,
    apply_fixed_bert_embedding_indices,
    apply_fixed_bert_sequence_length,
    apply_fixed_input_with_attention_mask,
    apply_fixed_predictor_text_encoder_shapes,
    apply_fixed_text_encoder_lstm_reshapes,
    apply_fixed_token_slice,
    apply_output_pad,
)


def _model():
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast",
                inputs=["tokens"],
                outputs=["audio"],
                name="CastTokens",
                to=TensorProto.FLOAT,
            )
        ],
        "test",
        [
            helper.make_tensor_value_info(
                "tokens",
                TensorProto.INT64,
                [1, "sequence_length"],
            ),
            helper.make_tensor_value_info("style", TensorProto.FLOAT, [1, 256]),
            helper.make_tensor_value_info("speed", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info(
                "audio",
                TensorProto.FLOAT,
                ["audio_length"],
            )
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def _shape(value):
    return [
        dim.dim_value or dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def test_apply_fixed_token_slice_adds_fixed_input_and_length_slice():
    model = _model()

    apply_fixed_token_slice(model, 64)

    inputs = {value.name: value for value in model.graph.input}
    assert _shape(inputs["tokens"]) == [1, 64]
    assert _shape(inputs["token_length"]) == [1]
    assert model.graph.node[0].name == "FastKokoroTokenLengthSlice"
    assert list(model.graph.node[0].input) == [
        "tokens",
        "fastkokoro_token_slice_starts",
        "token_length",
        "fastkokoro_token_slice_axes",
        "fastkokoro_token_slice_steps",
    ]
    assert model.graph.node[1].input[0] == "fastkokoro_tokens_sliced"


def test_apply_output_pad_replaces_dynamic_audio_output():
    model = _model()

    apply_output_pad(model, 120000)

    outputs = {value.name: value for value in model.graph.output}
    assert _shape(outputs["audio"]) == [120000]
    assert model.graph.node[0].output == ["audio_unpadded"]
    assert model.graph.node[-1].name == "FastKokoroOutputPad"
    assert any(
        value.name == "audio_unpadded" for value in model.graph.value_info
    )


def test_apply_fixed_input_with_attention_mask_adds_mask_input_and_bias():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Add",
                inputs=["audio", "/encoder/bert/Where_2_output_0"],
                outputs=["audio_masked"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Add",
            )
        ]
    )

    apply_fixed_input_with_attention_mask(model, 64)

    inputs = {value.name: value for value in model.graph.input}
    assert _shape(inputs["tokens"]) == [1, 64]
    assert _shape(inputs["attention_mask"]) == [1, 64]
    consumers = [
        node
        for node in model.graph.node
        if node.name
        == "/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Add"
    ]
    assert consumers[0].input[1] == "fastkokoro_attention_mask_bias"


def test_apply_fixed_bert_embedding_indices_rewires_gather_inputs():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Gather",
                inputs=[
                    "encoder.bert.embeddings.position_embeddings.weight",
                    "old_pos",
                ],
                outputs=["pos_out"],
                name="/encoder/bert/embeddings/position_embeddings/Gather",
            ),
            helper.make_node(
                "Gather",
                inputs=[
                    "encoder.bert.embeddings.token_type_embeddings.weight",
                    "old_type",
                ],
                outputs=["type_out"],
                name="/encoder/bert/embeddings/token_type_embeddings/Gather",
            ),
        ]
    )

    apply_fixed_bert_embedding_indices(model, 64)

    nodes = {node.name: node for node in model.graph.node}
    assert (
        nodes["/encoder/bert/embeddings/position_embeddings/Gather"].input[1]
        == "fastkokoro_bert_position_ids"
    )
    assert (
        nodes["/encoder/bert/embeddings/token_type_embeddings/Gather"].input[1]
        == "fastkokoro_bert_token_type_ids"
    )


def test_apply_fixed_bert_sequence_length_rewires_shape_path():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Gather",
                inputs=["/encoder/bert/Shape_1_output_0", "axis_one"],
                outputs=["/encoder/bert/Gather_1_output_0"],
                name="/encoder/bert/Gather_1",
                axis=0,
            ),
            helper.make_node(
                "Unsqueeze",
                inputs=["/encoder/bert/Gather_1_output_0", "axes"],
                outputs=["/encoder/bert/Unsqueeze_1_output_0"],
                name="/encoder/bert/Unsqueeze_1",
            ),
        ]
    )

    apply_fixed_bert_sequence_length(model, 64)

    nodes = {node.name: node for node in model.graph.node}
    assert nodes["/encoder/bert/Gather_1"].input[0] == "fastkokoro_bert_shape_vector"
    assert nodes["/encoder/bert/Unsqueeze_1"].input[0] == "fastkokoro_bert_seq_len"


def test_apply_fixed_bert_attention_reshapes_reuses_constant_shapes():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Shape",
                inputs=["shape_in"],
                outputs=["shape_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Shape",
            ),
            helper.make_node(
                "Reshape",
                inputs=["q", "shape_q"],
                outputs=["q_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape",
            ),
            helper.make_node(
                "Reshape",
                inputs=["k", "shape_k"],
                outputs=["k_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape_1",
            ),
            helper.make_node(
                "Reshape",
                inputs=["v", "shape_v"],
                outputs=["v_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape_2",
            ),
            helper.make_node(
                "Reshape",
                inputs=["o", "shape_o"],
                outputs=["o_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape_3",
            ),
            helper.make_node(
                "Shape",
                inputs=["transpose_in"],
                outputs=["shape8_out"],
                name="/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Shape_8",
            ),
        ]
    )

    apply_fixed_bert_attention_reshapes(model, 64)

    nodes = {node.name: node for node in model.graph.node}
    assert (
        nodes["/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Shape"].input[0]
        == "fastkokoro_bert_attention_hidden_template"
    )
    assert (
        nodes["/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape"].input[1]
        == "fastkokoro_bert_attention_reshape_qkv"
    )
    assert (
        nodes["/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Shape_8"].input[0]
        == "fastkokoro_bert_attention_transposed_template"
    )
    assert (
        nodes["/encoder/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/Reshape_3"].input[1]
        == "fastkokoro_bert_attention_reshape_out"
    )


def test_apply_fixed_predictor_text_encoder_shapes_rewires_shape_helpers():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Shape",
                inputs=["hidden_in"],
                outputs=["shape_hidden"],
                name="/encoder/predictor/text_encoder/Shape",
            ),
            helper.make_node(
                "Expand",
                inputs=["speaker", "expand_shape"],
                outputs=["speaker_expanded"],
                name="/encoder/predictor/text_encoder/Expand",
            ),
            helper.make_node(
                "Shape",
                inputs=["concat_in"],
                outputs=["shape_concat"],
                name="/encoder/predictor/text_encoder/lstms.0/Shape",
            ),
            helper.make_node(
                "LSTM",
                inputs=["x", "w", "r", "b", "", "h0", "c0"],
                outputs=["y", "yh", "yc"],
                name="/encoder/predictor/text_encoder/lstms.0/LSTM",
                hidden_size=256,
            ),
        ]
    )

    apply_fixed_predictor_text_encoder_shapes(model, 64)

    nodes = {node.name: node for node in model.graph.node}
    assert (
        nodes["/encoder/predictor/text_encoder/Shape"].input[0]
        == "fastkokoro_predictor_hidden_template"
    )
    assert (
        nodes["/encoder/predictor/text_encoder/Expand"].input[1]
        == "fastkokoro_predictor_expand_shape"
    )
    assert (
        nodes["/encoder/predictor/text_encoder/lstms.0/Shape"].input[0]
        == "fastkokoro_predictor_concat_template"
    )
    assert (
        nodes["/encoder/predictor/text_encoder/lstms.0/LSTM"].input[5]
        == "fastkokoro_predictor_lstm_state"
    )


def test_apply_fixed_text_encoder_lstm_reshapes_reuses_constant_shape():
    model = _model()
    model.graph.node.extend(
        [
            helper.make_node(
                "Reshape",
                inputs=["x", "shape_a"],
                outputs=["y"],
                name="/encoder/text_encoder/lstm/Reshape",
            ),
            helper.make_node(
                "Reshape",
                inputs=["x0", "shape_b"],
                outputs=["y0"],
                name="/encoder/predictor/text_encoder/lstms.0/Reshape",
            ),
            helper.make_node(
                "Reshape",
                inputs=["x2", "shape_c"],
                outputs=["y2"],
                name="/encoder/predictor/text_encoder/lstms.2/Reshape",
            ),
            helper.make_node(
                "Reshape",
                inputs=["x4", "shape_d"],
                outputs=["y4"],
                name="/encoder/predictor/text_encoder/lstms.4/Reshape",
            ),
        ]
    )

    apply_fixed_text_encoder_lstm_reshapes(model, 64)

    nodes = {node.name: node for node in model.graph.node}
    assert nodes["/encoder/text_encoder/lstm/Reshape"].input[1] == (
        "fastkokoro_text_encoder_lstm_reshape"
    )
    assert nodes["/encoder/predictor/text_encoder/lstms.4/Reshape"].input[1] == (
        "fastkokoro_text_encoder_lstm_reshape"
    )
