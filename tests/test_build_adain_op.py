import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def load_build_adain_op():
    path = Path(__file__).parents[1] / "scripts" / "build_adain_op.py"
    spec = importlib.util.spec_from_file_location("build_adain_op_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_build_cuda_adain_op():
    path = Path(__file__).parents[1] / "scripts" / "build_cuda_adain_op.py"
    spec = importlib.util.spec_from_file_location("build_cuda_adain_op_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_output_path_uses_platform_suffix(tmp_path):
    build_adain_op = load_build_adain_op()
    output = build_adain_op.default_output_path(tmp_path)

    assert output.parent == tmp_path / "native"
    assert output.name.startswith("libfastkokoro_adain")
    assert output.suffix in {".dll", ".dylib", ".so"}


def test_build_command_includes_packaged_native_sources(tmp_path):
    build_adain_op = load_build_adain_op()
    output = tmp_path / "libfastkokoro_adain.so"

    command = build_adain_op.build_command(output=output, cc="cc", openmp=True)

    assert command[0] == "cc"
    assert "-fopenmp" in command
    assert str(output) in command
    assert any(value.endswith("adain_custom_op.c") for value in command)
    assert any(value.startswith("-I") and value.endswith("native") for value in command)


def test_main_retries_without_openmp(tmp_path):
    build_adain_op = load_build_adain_op()
    output = tmp_path / "libfastkokoro_adain.so"
    calls = []

    def fake_run(command, check):
        calls.append(command)
        if len(calls) == 1:
            raise build_adain_op.subprocess.CalledProcessError(1, command)
        return Mock()

    with (
        patch.object(build_adain_op.subprocess, "run", fake_run),
        patch.object(
            build_adain_op.sys,
            "argv",
            ["build_adain_op.py", "--output", str(output)],
        ),
    ):
        build_adain_op.main()

    assert "-fopenmp" in calls[0]
    assert "-fopenmp" not in calls[1]


def test_main_keeps_no_openmp_failures(tmp_path):
    build_adain_op = load_build_adain_op()
    output = tmp_path / "libfastkokoro_adain.so"

    with (
        patch.object(
            build_adain_op.subprocess,
            "run",
            side_effect=build_adain_op.subprocess.CalledProcessError(1, "cc"),
        ),
        patch.object(
            build_adain_op.sys,
            "argv",
            ["build_adain_op.py", "--no-openmp", "--output", str(output)],
        ),
        pytest.raises(build_adain_op.subprocess.CalledProcessError),
    ):
        build_adain_op.main()


def test_cuda_builder_defaults_include_turing_and_ampere():
    build_cuda_adain_op = load_build_cuda_adain_op()

    assert build_cuda_adain_op.parse_archs(None) == ("sm_75", "sm_86")


def test_cuda_build_command_emits_each_requested_architecture(tmp_path):
    build_cuda_adain_op = load_build_cuda_adain_op()
    command = build_cuda_adain_op.build_command(
        output=tmp_path / "libfastkokoro_cuda_adain.so",
        nvcc="nvcc",
        source=tmp_path / "op.cu",
        include_dir=tmp_path,
        archs=("sm_75", "sm_86"),
        ort_api_version=18,
    )

    assert "arch=compute_75,code=sm_75" in command
    assert "arch=compute_86,code=sm_86" in command
