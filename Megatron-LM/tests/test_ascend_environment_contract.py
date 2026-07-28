from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "verify_ascend_env.py"
SPEC = importlib.util.spec_from_file_location("verify_ascend_env", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)


def _lock() -> dict:
    return json.loads(
        (PROJECT / "ASCEND_RUNTIME_LOCK.json").read_text(encoding="utf-8")
    )


def _write_version_cfg(
    root: Path,
    *,
    runtime: str = "[8.3.0.2.220:8.3.RC2]",
    hccl: str = "[8.3.0.2.220:8.3.RC2]",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "version.cfg").write_text(
        f"runtime_running_version={runtime}\n"
        f"hccl_running_version={hccl}\n",
        encoding="utf-8",
    )


def test_runtime_lock_records_real_cluster_stack_and_upstream_deviation() -> None:
    lock = _lock()
    assert lock["python"]["version"] == "3.10.20"
    assert lock["python_packages"]["torch"]["version"] == "2.6.0"
    assert lock["python_packages"]["torch_npu"]["version"] == "2.6.0"
    assert lock["ascend_components"]["runtime"] == {
        "version_key": "runtime_running_version",
        "build": "8.3.0.2.220",
        "release": "8.3.RC2",
    }
    assert lock["ascend_components"]["hccl"] == {
        "version_key": "hccl_running_version",
        "build": "8.3.0.2.220",
        "release": "8.3.RC2",
    }
    assert lock["upstream_declaration"]["pytorch"] == "2.7.1"
    assert lock["support_status"] == "project-validated-deviation"
    assert "2.6.0" in lock["validated_deviation"]["pytorch"]
    evidence = lock["validation_evidence"]
    assert evidence["kind"] == "isolated_end_to_end"
    assert evidence["result"] == "passed"
    assert evidence["upstream_commits"] == {
        "MindSpeed": "c94078bb90795dd8f3329a640eb5c0eac472f9d9",
        "MindSpeed-LLM": "630fe3c4cd1a5416c8b90986cd04e212ad654f06",
        "Megatron-LM": "a845aa7e12b3a117e24c2352b9e3e60bad2e3a17",
    }


def test_cann_and_hccl_versions_match_exactly(tmp_path: Path) -> None:
    _write_version_cfg(tmp_path)
    assert verify.verify_ascend_versions(_lock(), tmp_path) == {
        "runtime": "8.3.0.2.220/8.3.RC2",
        "hccl": "8.3.0.2.220/8.3.RC2",
    }


def test_absent_cann_can_defer_but_a_version_mismatch_cannot(
    tmp_path: Path,
) -> None:
    with pytest.raises(verify.RuntimeDeferred, match="version file is absent"):
        verify.verify_ascend_versions(_lock(), tmp_path)

    _write_version_cfg(tmp_path, hccl="[8.3.0.2.219:8.3.RC2]")
    with pytest.raises(verify.RuntimeContractError, match="hccl version mismatch"):
        verify.verify_ascend_versions(_lock(), tmp_path)


def test_static_version_mismatch_is_always_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify.platform, "python_version", lambda: "3.10.19")
    with pytest.raises(verify.RuntimeContractError, match="Python version mismatch"):
        verify.verify_static_versions(_lock())


def test_torch_version_mismatch_is_always_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify.platform, "python_version", lambda: "3.10.20")

    def version(distribution: str) -> str:
        return {"torch": "2.6.1", "torch-npu": "2.6.0"}[distribution]

    monkeypatch.setattr(verify.importlib.metadata, "version", version)
    with pytest.raises(verify.RuntimeContractError, match="torch version mismatch"):
        verify.verify_static_versions(_lock())


def test_only_known_missing_cann_import_errors_are_deferable() -> None:
    missing_acl = ModuleNotFoundError("No module named 'acl'", name="acl")
    missing_torch_npu = ModuleNotFoundError(
        "No module named 'torch_npu'", name="torch_npu"
    )
    assert verify._missing_cann_import(missing_acl)
    assert verify._missing_cann_import(
        ImportError("libascend_hal.so: cannot open shared object file")
    )
    assert not verify._missing_cann_import(missing_torch_npu)
    assert not verify._missing_cann_import(RuntimeError("unrelated runtime bug"))


def test_cli_defers_only_missing_cann_and_fails_mismatched_cann(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    absent = tmp_path / "absent"
    assert verify.main(["--ascend-home", str(absent)]) == 0
    assert "OLMO3_ASCEND_STACK_DEFERRED" in capsys.readouterr().out

    assert verify.main(["--strict", "--ascend-home", str(absent)]) == 1
    assert "OLMO3_ASCEND_STACK_FAILED" in capsys.readouterr().err

    mismatched = tmp_path / "mismatched"
    _write_version_cfg(mismatched, runtime="[8.3.0.2.219:8.3.RC2]")
    assert verify.main(["--ascend-home", str(mismatched)]) == 1
    error = capsys.readouterr()
    assert "OLMO3_ASCEND_STACK_FAILED" in error.err
    assert "runtime version mismatch" in error.err
    assert "DEFERRED" not in error.out


def test_host_only_checks_cann_before_python_runtime_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ascend_home = tmp_path / "ascend-toolkit" / "latest"
    _write_version_cfg(ascend_home)
    (ascend_home / "python" / "site-packages" / "acl").mkdir(
        parents=True
    )
    (ascend_home / "lib64").mkdir()
    env_script = tmp_path / "ascend-toolkit" / "set_env.sh"
    env_script.write_text("# test fixture\n", encoding="utf-8")

    assert (
        verify.main(
            [
                "--host-only",
                "--ascend-home",
                str(ascend_home),
                "--cann-env-script",
                str(env_script),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "OLMO3_ASCEND_HOST_PREREQUISITES_OK" in output
    assert "driver_npu_check=deferred-to-strict" in output


def test_host_only_rejects_incomplete_cann_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_version_cfg(tmp_path)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("# test fixture\n", encoding="utf-8")
    assert (
        verify.main(
            [
                "--host-only",
                "--ascend-home",
                str(tmp_path),
                "--cann-env-script",
                str(env_script),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "OLMO3_ASCEND_STACK_FAILED" in error
    assert "required host CANN path is missing" in error


def test_strict_probe_source_covers_device_tensor_and_hccl() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "torch.npu.set_device(device_index)" in text
    assert "torch.arange(8" in text
    assert 'backend="hccl"' in text
    assert "distributed.all_reduce(probe)" in text
    assert "verify_mindspeed_imports()" in text
    assert '"--strict",' in text
