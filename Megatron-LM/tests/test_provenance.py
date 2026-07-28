from __future__ import annotations

import json
from pathlib import Path

import pytest

import olmo3_pipeline.provenance as provenance
from olmo3_pipeline.config import ConfigurationError


def test_rendered_source_manifest_detects_code_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    for name in provenance.ROOT_SOURCE_CONTRACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    for name in (
        "requirements.txt",
        "requirements/dolma-bootstrap.txt",
        "requirements/train.lock.txt",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")

    monkeypatch.setattr(provenance, "PROJECT_ROOT", tmp_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(provenance.source_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance.verify_source_manifest(manifest_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="changed after this run was rendered"):
        provenance.verify_source_manifest(manifest_path)


def test_source_manifest_seals_environment_and_operating_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "src").mkdir()
    required = {
        *provenance.ROOT_SOURCE_CONTRACTS,
        "requirements.txt",
        "requirements/dolma-bootstrap.txt",
        "requirements/train.lock.txt",
    }
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")

    monkeypatch.setattr(provenance, "PROJECT_ROOT", tmp_path)
    files = provenance.source_manifest()["files"]
    assert required <= set(files)

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(provenance.source_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "ASCEND_RUNTIME_LOCK.json").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="changed after this run was rendered"):
        provenance.verify_source_manifest(manifest_path)
