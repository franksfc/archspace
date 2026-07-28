from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, ConfigurationError


SOURCE_MANIFEST_SCHEMA = "olmo3.mindspeed.source-manifest/v1"
ROOT_SOURCE_CONTRACTS = (
    "ASCEND_RUNTIME_LOCK.json",
    "docs/RUNBOOK.md",
    "UPSTREAM_LOCK.json",
    "environment.yml",
    "pyproject.toml",
    "README.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def upstream_state() -> dict[str, Any]:
    lock = json.loads((PROJECT_ROOT / "UPSTREAM_LOCK.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for name, expected in lock["repositories"].items():
        if expected.get("vendored_and_modified"):
            marker_path = PROJECT_ROOT / "megatron" / "OLMO3_VENDOR.json"
            marker = (
                json.loads(marker_path.read_text(encoding="utf-8"))
                if marker_path.is_file()
                else {}
            )
            base = marker.get("upstream_commit")
            result[name] = {
                "url": expected["url"],
                "ref": expected["ref"],
                "expected_commit": expected["commit"],
                "actual_commit": base,
                "matches_lock": base == expected["commit"],
                "dirty": False,
                "mode": "vendored-and-modified",
            }
            continue
        repo = PROJECT_ROOT / expected["path"]
        head = _git(repo, "rev-parse", "HEAD")
        origin = _git(repo, "remote", "get-url", "origin")
        dirty = bool(_git(repo, "status", "--porcelain"))
        shallow = _git(repo, "rev-parse", "--is-shallow-repository") == "true"
        fetch_refspecs = _git(
            repo, "config", "--get-all", "remote.origin.fetch"
        ).splitlines()
        full_heads_refspec = (
            "+refs/heads/*:refs/remotes/origin/*" in fetch_refspecs
        )
        generated_python_artifacts = sum(
            1
            for path in repo.rglob("*")
            if path.name == "__pycache__"
            or (path.is_file() and path.suffix in {".pyc", ".pyo"})
        )
        result[name] = {
            "url": expected["url"],
            "actual_origin": origin,
            "matches_url": origin == expected["url"],
            "ref": expected["ref"],
            "expected_commit": expected["commit"],
            "actual_commit": head,
            "matches_lock": head == expected["commit"],
            "dirty": dirty,
            "shallow": shallow,
            "full_heads_refspec": full_heads_refspec,
            "generated_python_artifacts": generated_python_artifacts,
            "full_history_required": bool(expected.get("require_full_history")),
            "mode": "pristine-third-party",
        }
    return result


def _source_file_hashes() -> dict[str, str]:
    roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "megatron",
        PROJECT_ROOT / "docs",
    )
    result: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            result[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    for name in ROOT_SOURCE_CONTRACTS:
        path = PROJECT_ROOT / name
        result[name] = sha256(path)
    requirement_paths = [
        PROJECT_ROOT / "requirements.txt",
        *sorted((PROJECT_ROOT / "requirements").glob("*.txt")),
    ]
    for path in requirement_paths:
        result[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    for name in (".gitignore", ".gitmodules"):
        path = PROJECT_ROOT / name
        if path.is_file():
            result[name] = sha256(path)
    return result


def _manifest_digest(files: dict[str, str]) -> str:
    encoded = json.dumps(
        files, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_manifest() -> dict[str, Any]:
    files = _source_file_hashes()
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "files": files,
        "manifest_sha256": _manifest_digest(files),
    }


def verify_source_manifest(path: Path) -> None:
    """Reject code/config drift after an immutable run was rendered."""

    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read source manifest {path}: {exc}") from exc
    if (
        not isinstance(expected, dict)
        or expected.get("schema") != SOURCE_MANIFEST_SCHEMA
        or not isinstance(expected.get("files"), dict)
    ):
        raise ConfigurationError(f"invalid rendered source manifest: {path}")
    files = expected["files"]
    if expected.get("manifest_sha256") != _manifest_digest(files):
        raise ConfigurationError(
            f"rendered source manifest digest mismatch: {path}"
        )
    actual = _source_file_hashes()
    if actual != files:
        changed = sorted(
            name
            for name in set(actual) | set(files)
            if actual.get(name) != files.get(name)
        )
        raise ConfigurationError(
            "project sources changed after this run was rendered; "
            f"first changed paths={changed[:16]}"
        )


def verify_upstream_snapshot(path: Path) -> None:
    """Verify pristine third-party state still matches the rendered snapshot."""

    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read upstream snapshot {path}: {exc}") from exc
    actual = upstream_state()
    if expected != actual:
        changed = sorted(
            name
            for name in set(actual) | set(expected if isinstance(expected, dict) else {})
            if not isinstance(expected, dict) or actual.get(name) != expected.get(name)
        )
        raise ConfigurationError(
            "third-party runtime changed after this run was rendered; "
            f"repositories={changed}"
        )
