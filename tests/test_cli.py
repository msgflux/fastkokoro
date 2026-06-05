from unittest.mock import patch

from fastkokoro import cli


def test_cli_build_custom_op_sets_adain_environment(monkeypatch, tmp_path):
    output = tmp_path / "libfastkokoro_adain.so"
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_FUSION", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY", raising=False)
    monkeypatch.delenv("FASTKOKORO_ONNX_PROVIDERS", raising=False)
    monkeypatch.delenv("CC", raising=False)

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch(
            "sys.argv",
            ["fastkokoro", "--build-custom-op", "--custom-op-output", str(output)],
        ),
        patch("fastkokoro.cli.build_adain_custom_op", return_value=output) as build,
        patch("fastkokoro.cli.uvicorn.run") as run,
    ):
        cli.main()
        assert cli.os.environ["FASTKOKORO_ONNX_ADAIN_FUSION"] == "true"
        assert cli.os.environ["FASTKOKORO_ONNX_ADAIN_CUSTOM_OP_LIBRARY"] == str(output)
        assert cli.os.environ["FASTKOKORO_ONNX_PROVIDERS"] == "CPUExecutionProvider"

    build.assert_called_once_with(output=output, cc="gcc", openmp=True)
    run.assert_called_once()


def test_cli_custom_op_preserves_existing_provider_env(monkeypatch, tmp_path):
    output = tmp_path / "libfastkokoro_adain.so"
    monkeypatch.setenv("FASTKOKORO_ONNX_PROVIDERS", "CUDAExecutionProvider")

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch(
            "sys.argv",
            ["fastkokoro", "--build-custom-op", "--custom-op-output", str(output)],
        ),
        patch("fastkokoro.cli.build_adain_custom_op", return_value=output),
        patch("fastkokoro.cli.uvicorn.run"),
    ):
        cli.main()
        assert cli.os.environ["FASTKOKORO_ONNX_PROVIDERS"] == "CUDAExecutionProvider"


def test_cli_warmup_multi_shape_sets_environment(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_WARMUP_MULTI_SHAPE", raising=False)

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch("sys.argv", ["fastkokoro", "--warmup-multi-shape"]),
        patch("fastkokoro.cli.uvicorn.run") as run,
    ):
        cli.main()
        assert cli.os.environ["FASTKOKORO_WARMUP_MULTI_SHAPE"] == "true"

    run.assert_called_once()


def test_cli_warmup_multi_shape_buckets_sets_environment(monkeypatch):
    monkeypatch.delenv("FASTKOKORO_WARMUP_MULTI_SHAPE", raising=False)
    monkeypatch.delenv("FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS", raising=False)

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch(
            "sys.argv",
            ["fastkokoro", "--warmup-multi-shape-buckets", "6,8,16"],
        ),
        patch("fastkokoro.cli.uvicorn.run") as run,
    ):
        cli.main()
        assert cli.os.environ["FASTKOKORO_WARMUP_MULTI_SHAPE"] == "true"
        assert cli.os.environ["FASTKOKORO_WARMUP_MULTI_SHAPE_BUCKETS"] == "6,8,16"

    run.assert_called_once()
