from __future__ import annotations

from pathlib import Path

import pytest

from olmo3_pipeline.config import (
    ConfigurationError,
    discover_project_root,
    verify_control_plane_install,
)


def _checkout(root: Path) -> Path:
    for directory in ("configs", "scripts", "src"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for filename in ("pyproject.toml", "UPSTREAM_LOCK.json"):
        (root / filename).write_text("{}\n", encoding="utf-8")
    return root


def test_explicit_project_root_is_authoritative(tmp_path: Path) -> None:
    explicit = _checkout(tmp_path / "explicit")
    source = _checkout(tmp_path / "source")
    cwd = _checkout(tmp_path / "cwd")

    assert (
        discover_project_root(
            module_file=source / "src" / "olmo3_pipeline" / "config.py",
            environment={"OLMO3_ROOT": str(explicit)},
            cwd=cwd,
        )
        == explicit.resolve()
    )


def test_source_checkout_is_detected_without_environment(tmp_path: Path) -> None:
    source = _checkout(tmp_path / "source")

    assert (
        discover_project_root(
            module_file=source / "src" / "olmo3_pipeline" / "config.py",
            environment={},
            cwd=tmp_path / "elsewhere",
        )
        == source.resolve()
    )


def test_installed_entry_point_can_locate_checkout_from_cwd(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "checkout")
    installed_module = (
        tmp_path
        / "conda"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "olmo3_pipeline"
        / "config.py"
    )

    assert (
        discover_project_root(
            module_file=installed_module,
            environment={},
            cwd=checkout / "docs",
        )
        == checkout.resolve()
    )


def test_invalid_explicit_project_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="not a complete pipeline checkout",
    ):
        discover_project_root(
            module_file=tmp_path / "site-packages" / "config.py",
            environment={"OLMO3_ROOT": str(tmp_path / "missing")},
            cwd=tmp_path,
        )


def test_installed_entry_point_requires_checkout_context(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="set OLMO3_ROOT",
    ):
        discover_project_root(
            module_file=tmp_path / "site-packages" / "config.py",
            environment={},
            cwd=tmp_path / "unrelated",
        )


def test_installed_control_plane_must_match_checkout(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "checkout")
    checkout_package = checkout / "src" / "olmo3_pipeline"
    installed_package = tmp_path / "site-packages" / "olmo3_pipeline"
    checkout_package.mkdir(parents=True, exist_ok=True)
    installed_package.mkdir(parents=True)
    (checkout_package / "config.py").write_text("version = 1\n", encoding="utf-8")
    (installed_package / "config.py").write_text("version = 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="changed_package_file"):
        verify_control_plane_install(
            checkout,
            module_file=installed_package / "config.py",
        )

    (installed_package / "config.py").write_text("version = 1\n", encoding="utf-8")
    verify_control_plane_install(
        checkout,
        module_file=installed_package / "config.py",
    )

    checkout_runtime = checkout / "src" / "runtime"
    installed_runtime = tmp_path / "site-packages" / "runtime"
    checkout_runtime.mkdir()
    installed_runtime.mkdir()
    (checkout_runtime / "dataset.py").write_text("version = 1\n", encoding="utf-8")
    (installed_runtime / "dataset.py").write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="changed_package_file"):
        verify_control_plane_install(
            checkout,
            module_file=installed_package / "config.py",
        )
