from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


class ConfigurationError(ValueError):
    """Raised when a composed training configuration is invalid."""


_PROJECT_ROOT_MARKERS = (
    Path("pyproject.toml"),
    Path("UPSTREAM_LOCK.json"),
    Path("configs"),
    Path("scripts"),
    Path("src"),
)


def _valid_project_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _PROJECT_ROOT_MARKERS)


def discover_project_root(
    *,
    module_file: Path | None = None,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> Path:
    """Locate the checkout that owns non-package runtime assets.

    A non-editable installation contains the Python packages but intentionally
    does not duplicate the repository's configs, scripts, Megatron overlay, or
    pinned third-party checkouts.  ``OLMO3_ROOT`` is therefore the authoritative
    location for installed console entry points.  Source-tree imports and
    commands launched from inside a checkout remain self-locating.
    """

    environ = os.environ if environment is None else environment
    configured = environ.get("OLMO3_ROOT", "")
    if configured:
        if configured != configured.strip():
            raise ConfigurationError(
                "OLMO3_ROOT must not contain leading or trailing whitespace"
            )
        candidate = Path(configured).expanduser().resolve()
        if not _valid_project_root(candidate):
            missing = [
                str(marker)
                for marker in _PROJECT_ROOT_MARKERS
                if not (candidate / marker).exists()
            ]
            raise ConfigurationError(
                f"OLMO3_ROOT is not a complete pipeline checkout: {candidate}; "
                f"missing={missing}"
            )
        return candidate

    source_file = Path(__file__) if module_file is None else Path(module_file)
    source_file = source_file.expanduser().resolve()
    if len(source_file.parents) >= 3:
        source_candidate = source_file.parents[2]
        if _valid_project_root(source_candidate):
            return source_candidate

    working_directory = Path.cwd() if cwd is None else Path(cwd)
    working_directory = working_directory.expanduser().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if _valid_project_root(candidate):
            return candidate

    raise ConfigurationError(
        "cannot locate the OLMo 3 pipeline checkout; set OLMO3_ROOT to the "
        "repository root before using an installed olmo3ctl, olmo3ckpt, or "
        "olmo3eval command"
    )


PROJECT_ROOT = discover_project_root()


def verify_control_plane_install(
    project_root: Path = PROJECT_ROOT,
    *,
    module_file: Path | None = None,
) -> None:
    """Require every installed project package to match the selected checkout.

    Console entry points are installed non-editably, while configs, launch
    scripts, and the Megatron overlay stay in the checkout. Refusing a mixed
    revision prevents an old ``olmo3ctl`` copy from rendering a new source tree.
    """

    checkout_source = project_root / "src"
    active_package = (
        Path(__file__).expanduser().resolve().parent
        if module_file is None
        else Path(module_file).expanduser().resolve().parent
    )
    active_source = active_package.parent
    if active_source == checkout_source.resolve():
        return

    for package_name in (
        "olmo3_pipeline",
        "modeling",
        "runtime",
        "llama_config",
    ):
        checkout_package = checkout_source / package_name
        installed_package = active_source / package_name
        expected_files = {
            path.relative_to(checkout_source).as_posix(): path
            for path in checkout_package.rglob("*.py")
            if path.is_file()
        }
        active_files = {
            path.relative_to(active_source).as_posix(): path
            for path in installed_package.rglob("*.py")
            if path.is_file()
        }
        if set(expected_files) != set(active_files):
            raise ConfigurationError(
                "installed olmo3-mindspeed-pipeline does not match OLMO3_ROOT; "
                f"package_file_set={package_name!r}; "
                "re-run scripts/bootstrap_conda_env.sh for this checkout"
            )
        for relative, expected_path in expected_files.items():
            expected = hashlib.sha256(expected_path.read_bytes()).digest()
            actual = hashlib.sha256(active_files[relative].read_bytes()).digest()
            if actual != expected:
                raise ConfigurationError(
                    "installed olmo3-mindspeed-pipeline does not match "
                    f"OLMO3_ROOT; changed_package_file={relative!r}; "
                    "rebuild the clean Conda environment"
                )


verify_control_plane_install()
CONFIG_ROOT = PROJECT_ROOT / "configs"


def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``incoming`` into ``base`` without sharing containers."""

    result = copy.deepcopy(base)
    for key, value in incoming.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ConfigurationError(f"unsafe configuration name: {value!r}")
    return value


def load_named(kind: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / _safe_name(kind) / f"{_safe_name(name)}.json"
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.json"))
        raise ConfigurationError(
            f"unknown {kind} configuration {name!r}; available={available}"
        )
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return value


def parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_override(config: dict[str, Any], expression: str) -> None:
    key, separator, raw = expression.partition("=")
    if not separator or not key:
        raise ConfigurationError(
            f"override must use dotted.path=value syntax: {expression!r}"
        )
    parts = key.split(".")
    if any(not part for part in parts):
        raise ConfigurationError(f"invalid override path: {key!r}")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ConfigurationError(
                f"cannot set {key!r}: {part!r} is not an object"
            )
        cursor = child
    cursor[parts[-1]] = parse_scalar(raw)


def _compose_performance_profiles(names: Iterable[str]) -> dict[str, Any]:
    """Combine orthogonal production profiles without argument-order semantics."""

    requested = list(names)
    if len(requested) != len(set(requested)):
        raise ConfigurationError(
            f"performance profiles must be unique, got {requested!r}"
        )

    merged: dict[str, Any] = {}
    for name in sorted(requested):
        profile = load_named("performance", name)
        if set(profile) != {"performance"} or not isinstance(
            profile["performance"], dict
        ):
            raise ConfigurationError(
                f"performance profile {name!r} must contain only a performance object"
            )
        for field, value in profile["performance"].items():
            if field == "profiles":
                continue
            if field not in merged or merged[field] == value:
                merged[field] = copy.deepcopy(value)
                continue
            if field == "scalar_reduce_backend" and {
                merged[field],
                value,
            } == {"hccl", "gloo"}:
                # Scalar reporting is independent of tensor collectives.  The
                # retained CP profile uses Gloo to avoid injecting tiny HCCL
                # collectives into the CP/HCCL critical path, so it wins
                # deterministically when TP2 and CP are composed.
                merged[field] = "gloo"
                continue
            raise ConfigurationError(
                f"performance profiles conflict on {field!r}: "
                f"{merged[field]!r} != {value!r}"
            )
    merged["profiles"] = sorted(requested)
    return {"performance": merged}


def compose(
    *,
    model: str,
    variant: str,
    stage: str,
    data: str,
    topology: str,
    performance: Iterable[str],
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind, name in (
        ("models", model),
        ("variants", variant),
        ("stages", stage),
        ("data", data),
        ("topology", topology),
    ):
        result = deep_merge(result, load_named(kind, name))

    result = deep_merge(result, _compose_performance_profiles(performance))

    for expression in overrides:
        apply_override(result, expression)
    return result
