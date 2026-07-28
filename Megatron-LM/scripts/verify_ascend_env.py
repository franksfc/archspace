#!/usr/bin/env python3
"""Verify the exact, cluster-tested Ascend runtime contract.

``--host-only`` validates the host-owned CANN/HCCL toolkit before a clean
Conda environment exists. ``--strict`` is the training-Pod gate. It checks the
locked Python/Torch versions, CANN and HCCL build, the pinned MindSpeed source
imports, ``set_device``, a real NPU tensor operation, and a one-rank HCCL
process group plus all-reduce.

The default non-strict mode is intended only for a control host. It may defer
when CANN libraries or a visible NPU are genuinely absent. Static version
mismatches, malformed CANN metadata, broken Python packages, wrong MindSpeed
imports, and failures on a visible NPU are hard errors in both modes.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = PROJECT_ROOT / "ASCEND_RUNTIME_LOCK.json"
DEFAULT_ASCEND_HOME = Path("/usr/local/Ascend/ascend-toolkit/latest")
DEFAULT_CANN_ENV_SCRIPT = Path(
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
)
VERSION_PATTERN = re.compile(r"^\[([^:\]]+):([^\]]+)\]$")

# Importing the pristine upstream packages must not leave ignored artifacts in
# them. Their location is later verified, so an unrelated site-package cannot
# satisfy this check.
sys.dont_write_bytecode = True


class RuntimeContractError(RuntimeError):
    """The installed runtime contradicts the immutable lock."""


class RuntimeDeferred(RuntimeError):
    """The control host genuinely lacks CANN libraries or a visible NPU."""


def load_runtime_lock(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot read runtime lock {path}: {exc}") from exc
    if (
        not isinstance(lock, dict)
        or lock.get("schema") != "olmo3.mindspeed.ascend-runtime-lock/v1"
    ):
        raise RuntimeContractError(f"invalid Ascend runtime lock schema: {path}")
    return lock


def _locked_value(lock: dict[str, Any], *keys: str) -> str:
    value: Any = lock
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeContractError(
                "runtime lock is missing " + ".".join(keys)
            )
        value = value[key]
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(
            "runtime lock value must be a non-empty string: " + ".".join(keys)
        )
    return value


def verify_static_versions(lock: dict[str, Any]) -> dict[str, str]:
    expected_python = _locked_value(lock, "python", "version")
    actual_python = platform.python_version()
    if actual_python != expected_python:
        raise RuntimeContractError(
            f"Python version mismatch: expected={expected_python} "
            f"actual={actual_python}"
        )

    versions = {"python": actual_python}
    packages = lock.get("python_packages")
    if not isinstance(packages, dict) or not packages:
        raise RuntimeContractError("runtime lock has no python_packages")
    for import_name, spec in packages.items():
        if not isinstance(spec, dict):
            raise RuntimeContractError(
                f"invalid python_packages entry: {import_name}"
            )
        distribution = spec.get("distribution")
        expected = spec.get("version")
        if not isinstance(distribution, str) or not isinstance(expected, str):
            raise RuntimeContractError(
                f"invalid locked package metadata: {import_name}"
            )
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeContractError(
                f"required distribution is missing: {distribution}"
            ) from exc
        if actual != expected:
            raise RuntimeContractError(
                f"{distribution} version mismatch: expected={expected} "
                f"actual={actual}"
            )
        versions[import_name] = actual
    return versions


def resolve_ascend_home(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return DEFAULT_ASCEND_HOME


def _parse_version_cfg(path: Path) -> dict[str, tuple[str, str]]:
    if not path.is_file():
        raise RuntimeDeferred(f"CANN version file is absent: {path}")
    values: dict[str, tuple[str, str]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeContractError(f"cannot read CANN version file {path}: {exc}") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        match = VERSION_PATTERN.fullmatch(value)
        if match:
            values[key] = (match.group(1), match.group(2))
    return values


def verify_ascend_versions(
    lock: dict[str, Any], ascend_home: Path
) -> dict[str, str]:
    values = _parse_version_cfg(ascend_home / "version.cfg")
    result: dict[str, str] = {}
    components = lock.get("ascend_components")
    if not isinstance(components, dict) or not components:
        raise RuntimeContractError("runtime lock has no ascend_components")
    for component, spec in components.items():
        if not isinstance(spec, dict):
            raise RuntimeContractError(
                f"invalid ascend_components entry: {component}"
            )
        key = spec.get("version_key")
        expected_build = spec.get("build")
        expected_release = spec.get("release")
        if not all(
            isinstance(value, str) and value
            for value in (key, expected_build, expected_release)
        ):
            raise RuntimeContractError(
                f"invalid locked Ascend component metadata: {component}"
            )
        if key not in values:
            raise RuntimeContractError(
                f"CANN version file is missing locked key: {key}"
            )
        actual_build, actual_release = values[key]
        if (actual_build, actual_release) != (
            expected_build,
            expected_release,
        ):
            raise RuntimeContractError(
                f"{component} version mismatch: "
                f"expected={expected_build}/{expected_release} "
                f"actual={actual_build}/{actual_release}"
            )
        result[component] = f"{actual_build}/{actual_release}"
    return result


def verify_cann_host_layout(
    ascend_home: Path, cann_env_script: Path
) -> dict[str, str]:
    """Validate host files that must exist independently of Conda."""

    required = {
        "cann_env_script": cann_env_script,
        "cann_python_acl": ascend_home / "python" / "site-packages" / "acl",
        "cann_lib64": ascend_home / "lib64",
    }
    result: dict[str, str] = {}
    for name, path in required.items():
        if not path.exists():
            raise RuntimeContractError(
                f"required host CANN path is missing: {name}={path}"
            )
        if not os.access(path, os.R_OK):
            raise RuntimeContractError(
                f"required host CANN path is not readable: {name}={path}"
            )
        result[name] = str(path.resolve())
    return result


def _missing_cann_import(exc: BaseException) -> bool:
    """Recognize only errors that mean the host lacks a CANN shared runtime."""

    library_markers = (
        "libascend",
        "libacl",
        "libhccl",
        "libge_runner",
        "libopapi",
    )
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name == "acl":
            return True
        text = str(current).lower()
        if isinstance(current, (ImportError, OSError)) and any(
            marker in text for marker in library_markers
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _verify_import_version(module: ModuleType, expected: str, name: str) -> None:
    actual = str(getattr(module, "__version__", "")).split("+", 1)[0]
    if actual != expected:
        raise RuntimeContractError(
            f"{name} import version mismatch: expected={expected} actual={actual!r}"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def verify_mindspeed_imports() -> dict[str, str]:
    roots = {
        "mindspeed": (
            PROJECT_ROOT / "third_party" / "MindSpeed"
        ).resolve(),
        "mindspeed_llm": (
            PROJECT_ROOT / "third_party" / "MindSpeed-LLM"
        ).resolve(),
    }
    for path in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT,
        roots["mindspeed"],
        roots["mindspeed_llm"],
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    imported: dict[str, str] = {}
    for name, expected_root in roots.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeContractError(
                f"cannot import pinned {name}: {type(exc).__name__}: {exc}"
            ) from exc
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeContractError(f"{name} import has no source path")
        actual_path = Path(module_file).resolve()
        if not _is_relative_to(actual_path, expected_root):
            raise RuntimeContractError(
                f"{name} was imported from an unpinned location: {actual_path}"
            )
        imported[name] = str(actual_path)
    return imported


def import_ascend_modules(lock: dict[str, Any]) -> tuple[Any, Any]:
    # PyTorch's entry-point auto-loader can import torch_npu while ``torch`` is
    # only half initialized, obscuring the actual missing-CANN exception.
    # Import the backend explicitly below so version/defer classification is
    # deterministic.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        import torch
    except Exception as exc:
        if _missing_cann_import(exc):
            raise RuntimeDeferred(
                f"CANN shared runtime is unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise RuntimeContractError(
            f"cannot import torch: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        import torch_npu
    except Exception as exc:
        if _missing_cann_import(exc):
            raise RuntimeDeferred(
                f"CANN shared runtime is unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise RuntimeContractError(
            f"cannot import torch_npu: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        import acl  # noqa: F401
    except Exception as exc:
        if _missing_cann_import(exc):
            raise RuntimeDeferred(
                f"CANN Python binding is unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise RuntimeContractError(
            f"cannot import acl: {type(exc).__name__}: {exc}"
        ) from exc

    _verify_import_version(
        torch, _locked_value(lock, "python_packages", "torch", "version"), "torch"
    )
    _verify_import_version(
        torch_npu,
        _locked_value(lock, "python_packages", "torch_npu", "version"),
        "torch_npu",
    )
    return torch, torch_npu


def verify_npu_and_hccl(torch: Any, device_index: int) -> int:
    try:
        count = int(torch.npu.device_count())
    except Exception as exc:
        raise RuntimeDeferred(
            f"cannot enumerate NPUs: {type(exc).__name__}: {exc}"
        ) from exc
    if count <= 0:
        raise RuntimeDeferred("torch.npu.device_count() returned 0")
    if device_index < 0 or device_index >= count:
        raise RuntimeContractError(
            f"verification NPU index is out of range: index={device_index} count={count}"
        )

    # Once an NPU is visible, failures are not treated as an absence: they
    # indicate a broken runtime and must fail even in non-strict mode.
    try:
        torch.npu.set_device(device_index)
        device = f"npu:{device_index}"
        tensor = torch.arange(8, dtype=torch.float32, device=device)
        observed = float(((tensor + 1.0) * 2.0).sum().cpu().item())
        if observed != 72.0:
            raise RuntimeError(f"unexpected NPU tensor result: {observed}")
        torch.npu.synchronize()
    except Exception as exc:
        raise RuntimeContractError(
            f"NPU set_device/tensor operation failed: {type(exc).__name__}: {exc}"
        ) from exc

    distributed = torch.distributed
    if distributed.is_initialized():
        raise RuntimeContractError(
            "torch.distributed is already initialized; strict verification "
            "requires an isolated process"
        )
    with tempfile.TemporaryDirectory(prefix="olmo3-hccl-check-") as directory:
        store_path = Path(directory) / "store"
        try:
            distributed.init_process_group(
                backend="hccl",
                init_method=f"file://{store_path}",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=60),
            )
            probe = torch.tensor([3.0], dtype=torch.float32, device=device)
            distributed.all_reduce(probe)
            torch.npu.synchronize()
            if float(probe.cpu().item()) != 3.0:
                raise RuntimeError("one-rank HCCL all-reduce changed the tensor")
        except Exception as exc:
            raise RuntimeContractError(
                f"one-rank HCCL init/all-reduce failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if distributed.is_initialized():
                distributed.destroy_process_group()
    return count


def _report_deferred(reason: RuntimeDeferred, strict: bool) -> int:
    marker = "OLMO3_ASCEND_STACK_FAILED" if strict else "OLMO3_ASCEND_STACK_DEFERRED"
    destination = sys.stderr if strict else sys.stdout
    print(f"{marker} reason={reason}", file=destination)
    if not strict:
        print(
            "Run scripts/verify_ascend_env.py --strict after sourcing "
            "/usr/local/Ascend/ascend-toolkit/set_env.sh inside every "
            "training Pod."
        )
    return 1 if strict else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-only",
        action="store_true",
        help=(
            "Validate only the external CANN/HCCL version and host layout. "
            "This mode does not require Python 3.10, torch, torch_npu, a "
            "driver, or a visible NPU and is intended to run before "
            "creating the clean Conda environment."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require the complete locked CANN/NPU runtime, MindSpeed imports, "
            "a real NPU operation, and a one-rank HCCL collective."
        ),
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ascend-home",
        type=Path,
        help="Override the Ascend toolkit root containing version.cfg.",
    )
    parser.add_argument(
        "--cann-env-script",
        type=Path,
        default=Path(
            os.environ.get(
                "OLMO3_CANN_ENV_SCRIPT", str(DEFAULT_CANN_ENV_SCRIPT)
            )
        ),
        help="Host CANN environment setup script.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("OLMO3_VERIFY_NPU_DEVICE", "0")),
        help="Local NPU index used by the strict tensor/HCCL probe (default: 0).",
    )
    args = parser.parse_args(argv)
    if args.host_only and args.strict:
        parser.error("--host-only and --strict are mutually exclusive")

    try:
        lock = load_runtime_lock(args.runtime_lock)
        ascend_home = resolve_ascend_home(args.ascend_home)
        ascend_versions = verify_ascend_versions(lock, ascend_home)
        host_layout = verify_cann_host_layout(
            ascend_home, args.cann_env_script
        )
        if args.host_only:
            print(
                "OLMO3_ASCEND_HOST_PREREQUISITES_OK "
                f"cann={ascend_versions['runtime']} "
                f"hccl={ascend_versions['hccl']} "
                f"ascend_home={ascend_home.resolve()} "
                f"cann_env_script={host_layout['cann_env_script']} "
                "driver_npu_check=deferred-to-strict"
            )
            return 0
        versions = verify_static_versions(lock)
        torch, _torch_npu = import_ascend_modules(lock)
        mindspeed = verify_mindspeed_imports()
        count = verify_npu_and_hccl(torch, args.device)
    except RuntimeDeferred as exc:
        return _report_deferred(exc, args.strict)
    except RuntimeContractError as exc:
        print(f"OLMO3_ASCEND_STACK_FAILED reason={exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Fail closed; never turn unknown errors into defer.
        print(
            "OLMO3_ASCEND_STACK_FAILED "
            f"reason=unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "OLMO3_ASCEND_STACK_OK "
        f"python={versions['python']} "
        f"torch={versions['torch']} "
        f"torch_npu={versions['torch_npu']} "
        f"cann={ascend_versions['runtime']} "
        f"hccl={ascend_versions['hccl']} "
        f"mindspeed={mindspeed['mindspeed']} "
        f"mindspeed_llm={mindspeed['mindspeed_llm']} "
        f"npu_count={count} device={args.device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
