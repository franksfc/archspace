"""Reproducible, path-agnostic data preparation helpers for OLMo 3.

This module deliberately separates planning from execution.  Every operation
that can download, tokenize, index, or pack data first emits an immutable JSON
contract.  Large external tools are imported only by the corresponding
execution command, so ``--dry-run`` remains useful on login nodes.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import fnmatch
import glob
import hashlib
import heapq
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .config import CONFIG_ROOT, PROJECT_ROOT, load_named


INVENTORY_SCHEMA = "olmo3-data-inventory-v1"
RUNTIME_MANIFEST_SCHEMA = "olmo3-runtime-data-manifest-v1"
PLAN_SCHEMA = "olmo3-data-plan-v1"
REPORT_SCHEMA = "olmo3-data-report-v1"
PRODUCTION_DATA_SCOPE = "production"
FIXED_TOKEN_COUNT_POLICY = "fixed"


class DataPipelineError(RuntimeError):
    """Raised when a data contract is unsafe, incomplete, or inconsistent."""


def sft_packed_work_dir(work_root: str | Path) -> Path:
    """Return the single cache parent shared by SFT preparation and training."""

    return Path(work_root).expanduser() / "packed"


@dataclass(frozen=True)
class FileEntry:
    """A source file frozen by relative path, byte size, and optional digest."""

    path: str
    size: int
    sha256: str | None = None

    def as_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path, "size": self.size}
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        return result


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"provenance", "contract_sha256"}
    }


def validate_contract_sha256(value: dict[str, Any], *, source: Path) -> None:
    expected = value.get("contract_sha256")
    actual = sha256_json(contract_payload(value))
    if not isinstance(expected, str) or expected != actual:
        raise DataPipelineError(
            f"contract SHA256 mismatch in {source}: expected={expected}, actual={actual}"
        )


def require_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise DataPipelineError(
            f"{label} SHA256 changed: expected={expected}, actual={actual}, path={path}"
        )


def validate_sha256_text(value: str | None, *, label: str) -> None:
    if value is not None and (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataPipelineError(f"{label} must be a lowercase 64-character SHA256")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create ``path`` once, accepting a byte-equivalent existing contract."""

    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise DataPipelineError(
            f"refusing to overwrite immutable data artifact: {path}"
        )
    _atomic_write(path, encoded)


def write_immutable_text(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise DataPipelineError(
            f"refusing to overwrite immutable data artifact: {path}"
        )
    _atomic_write(path, encoded)


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DataPipelineError(f"unsafe relative path in data manifest: {value!r}")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise DataPipelineError(f"empty relative path in data manifest: {value!r}")
    return normalized


def data_config(name: str) -> tuple[dict[str, Any], Path]:
    value = load_named("data", name)
    data = value.get("data")
    if not isinstance(data, dict):
        raise DataPipelineError(f"configs/data/{name}.json has no data object")
    return data, CONFIG_ROOT / "data" / f"{name}.json"


def stage_token_count_contract(data: dict[str, Any]) -> tuple[str, str]:
    """Validate the fixed Stage-2/3 production token-count contract."""

    scope = data.get("scope", PRODUCTION_DATA_SCOPE)
    policy = data.get("token_count_policy", FIXED_TOKEN_COUNT_POLICY)
    if scope != PRODUCTION_DATA_SCOPE:
        raise DataPipelineError("Stage-2/3 data.scope must be 'production'")
    if policy != FIXED_TOKEN_COUNT_POLICY:
        raise DataPipelineError(
            "production Stage-2/3 data must use token_count_policy='fixed'"
        )
    return str(scope), str(policy)


def source_from_config(
    name: str,
    *,
    repository: str | None,
    revision: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    data, path = data_config(name)
    configured = data.get("source")
    configured = configured if isinstance(configured, dict) else {}
    repo = repository or configured.get("repository")
    rev = revision if revision is not None else configured.get("revision")
    if not isinstance(repo, str) or not repo:
        raise DataPipelineError(
            f"data source repository is not configured for {name!r}; "
            "pass --repository"
        )
    return data, {"repository": repo, "revision": rev}, path


def provenance(
    *,
    action: str,
    config_name: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "action": action,
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pipeline_commit": _git_head(PROJECT_ROOT),
        "pipeline_dirty": _git_dirty(PROJECT_ROOT),
    }
    if config_name is not None:
        result["data_config"] = config_name
    if config_path is not None:
        try:
            rendered_config_path = config_path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        except ValueError:
            rendered_config_path = str(config_path)
        result["data_config_path"] = rendered_config_path
        result["data_config_sha256"] = sha256_file(config_path)
    return result


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(root: Path) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _matches(path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(
        fnmatch.fnmatch(path, pattern)
        or (
            pattern.startswith("**/")
            and fnmatch.fnmatch(path, pattern.removeprefix("**/"))
        )
        for pattern in patterns
    )


def discover_files(root: Path, patterns: Sequence[str]) -> tuple[Path, ...]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise DataPipelineError(f"data root is not a directory: {root}")
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DataPipelineError(
                f"data inventories reject symlinks to avoid path escape: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _matches(relative, patterns):
            result.append(path)
    result.sort(key=lambda item: item.relative_to(root).as_posix())
    if not result:
        raise DataPipelineError(
            f"no files under {root} matched patterns={list(patterns)!r}"
        )
    return tuple(result)


def create_inventory(
    *,
    root: Path,
    patterns: Sequence[str],
    checksums: bool,
    workers: int,
    source: dict[str, Any] | None = None,
    config_name: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise DataPipelineError("workers must be positive")
    root = root.expanduser().resolve()
    paths = discover_files(root, patterns)

    digests: dict[Path, str] = {}
    if checksums:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(sha256_file, path): path for path in paths}
            for future in concurrent.futures.as_completed(futures):
                digests[futures[future]] = future.result()

    entries = [
        FileEntry(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=digests.get(path),
        )
        for path in paths
    ]
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "root": str(root),
        "patterns": list(patterns),
        "checksums": checksums,
        "files": len(entries),
        "bytes": sum(entry.size for entry in entries),
        "entries": [entry.as_json() for entry in entries],
        "source": source,
        "provenance": provenance(
            action="inventory",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    inventory["contract_sha256"] = sha256_json(contract_payload(inventory))
    return inventory


def load_inventory(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPipelineError(f"cannot read inventory {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != INVENTORY_SCHEMA:
        raise DataPipelineError(f"unsupported data inventory schema: {path}")
    validate_contract_sha256(value, source=path)
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DataPipelineError(f"data inventory contains no entries: {path}")
    seen: set[str] = set()
    total_bytes = 0
    for raw in entries:
        if not isinstance(raw, dict):
            raise DataPipelineError(f"invalid entry in data inventory: {raw!r}")
        relative = validate_relative_path(str(raw.get("path", "")))
        if relative in seen:
            raise DataPipelineError(f"duplicate path in data inventory: {relative}")
        seen.add(relative)
        size = raw.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DataPipelineError(
                f"invalid size for {relative!r} in data inventory"
            )
        digest = raw.get("sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DataPipelineError(f"invalid SHA256 for {relative!r}")
        total_bytes += size
    if value.get("files") != len(entries) or value.get("bytes") != total_bytes:
        raise DataPipelineError(f"inventory aggregate mismatch: {path}")
    return value


def verify_inventory(
    manifest: Path,
    *,
    root: Path | None,
    checksums: bool,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise DataPipelineError("workers must be positive")
    inventory = load_inventory(manifest)
    actual_root = (
        root.expanduser().resolve()
        if root is not None
        else Path(inventory["root"]).expanduser().resolve()
    )
    problems: list[dict[str, Any]] = []
    checksum_jobs: list[tuple[Path, str, str]] = []
    for raw in inventory["entries"]:
        relative = validate_relative_path(raw["path"])
        path = actual_root / relative
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            problems.append({"path": relative, "problem": "missing"})
            continue
        if size != raw["size"]:
            problems.append(
                {
                    "path": relative,
                    "problem": "size",
                    "expected": raw["size"],
                    "actual": size,
                }
            )
            continue
        expected_digest = raw.get("sha256")
        if checksums:
            if expected_digest is None:
                problems.append(
                    {"path": relative, "problem": "manifest_has_no_sha256"}
                )
            else:
                checksum_jobs.append((path, relative, expected_digest))

    if checksum_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {
                pool.submit(sha256_file, path): (relative, expected)
                for path, relative, expected in checksum_jobs
            }
            for future in concurrent.futures.as_completed(jobs):
                relative, expected = jobs[future]
                actual = future.result()
                if actual != expected:
                    problems.append(
                        {
                            "path": relative,
                            "problem": "sha256",
                            "expected": expected,
                            "actual": actual,
                        }
                    )

    return {
        "schema": REPORT_SCHEMA,
        "action": "verify",
        "manifest": str(manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(manifest),
        "root": str(actual_root),
        "checksums": checksums,
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "state": "ok" if not problems else "failed",
        "problems": problems,
        "verified_at": utc_now(),
    }


def inventory_entries(
    manifest: Path,
    *,
    root: Path | None = None,
    patterns: Sequence[str] = (),
) -> tuple[Path, list[FileEntry], dict[str, Any]]:
    inventory = load_inventory(manifest)
    resolved_root = (
        root.expanduser().resolve()
        if root is not None
        else Path(inventory["root"]).expanduser().resolve()
    )
    entries = [
        FileEntry(raw["path"], int(raw["size"]), raw.get("sha256"))
        for raw in inventory["entries"]
        if _matches(raw["path"], patterns)
    ]
    if not entries:
        raise DataPipelineError(
            f"no inventory entries matched patterns={list(patterns)!r}"
        )
    return resolved_root, entries, inventory


def build_runtime_data_manifest(
    *,
    config_name: str,
    inventory_path: Path,
    runtime_root: Path | None,
    token_patterns: Sequence[str] = ("**/*.npy",),
    metadata_suffix: str = ".csv.gz",
) -> dict[str, Any]:
    """Build the self-contained Stage-2/3 manifest consumed by training.

    The input inventory may contain arbitrary ancillary files. This function
    selects raw uint32 token arrays, requires exactly one paired Dolma document
    metadata file per token source, and freezes the source order explicitly.
    No source-count assumption is made.
    """

    data, config_path = data_config(config_name)
    if data.get("stage") not in {"stage2", "stage3"}:
        raise DataPipelineError(
            "runtime token manifests are defined only for Stage 2 and Stage 3"
        )
    if data.get("backend") not in {"olmo3_numpy_fsl", "olmo3_numpy_packed"}:
        raise DataPipelineError(
            f"unsupported runtime token backend: {data.get('backend')!r}"
        )
    if not metadata_suffix.startswith("."):
        raise DataPipelineError("metadata suffix must begin with '.'")

    inventory_root, all_entries, inventory = inventory_entries(
        inventory_path, root=None, patterns=()
    )
    by_path = {entry.path: entry for entry in all_entries}
    token_entries = [
        entry for entry in all_entries if _matches(entry.path, token_patterns)
    ]
    token_entries.sort(key=lambda entry: entry.path)
    if not token_entries:
        raise DataPipelineError(
            f"inventory has no token arrays matching {list(token_patterns)!r}"
        )

    sources: list[dict[str, Any]] = []
    expected_metadata_paths: set[str] = set()
    total_token_bytes = 0
    total_metadata_bytes = 0
    for source_id, token_entry in enumerate(token_entries):
        if token_entry.size <= 0 or token_entry.size % 4:
            raise DataPipelineError(
                f"raw uint32 token source is empty or unaligned: "
                f"{token_entry.path}={token_entry.size}"
            )
        token_path = PurePosixPath(token_entry.path)
        if token_path.suffix != ".npy":
            raise DataPipelineError(
                f"runtime token source must end in .npy: {token_entry.path}"
            )
        metadata_path = token_path.with_suffix(metadata_suffix).as_posix()
        metadata_entry = by_path.get(metadata_path)
        if metadata_entry is None:
            raise DataPipelineError(
                f"missing paired document metadata for {token_entry.path}: "
                f"{metadata_path}"
            )
        if metadata_entry.size <= 0:
            raise DataPipelineError(f"empty document metadata: {metadata_path}")
        for digest, label in (
            (token_entry.sha256, token_entry.path),
            (metadata_entry.sha256, metadata_path),
        ):
            if digest is None:
                raise DataPipelineError(
                    "Stage-2/3 runtime manifests require content SHA256 for "
                    f"every token and metadata source; rebuild the inventory "
                    f"with --checksums: {label}"
                )
            validate_sha256_text(digest, label=f"runtime source {label}")
        expected_metadata_paths.add(metadata_path)
        total_token_bytes += token_entry.size
        total_metadata_bytes += metadata_entry.size
        source: dict[str, Any] = {
            "source_id": source_id,
            "token_path": token_entry.path,
            "metadata_path": metadata_path,
            "token_bytes": token_entry.size,
            "tokens": token_entry.size // 4,
            "metadata_bytes": metadata_entry.size,
        }
        source["token_sha256"] = token_entry.sha256
        source["metadata_sha256"] = metadata_entry.sha256
        sources.append(source)

    actual_metadata_paths = {
        entry.path
        for entry in all_entries
        if entry.path.endswith(metadata_suffix)
    }
    unpaired_metadata = sorted(actual_metadata_paths - expected_metadata_paths)
    if unpaired_metadata:
        raise DataPipelineError(
            "tokenized inventory contains metadata with no selected token array; "
            f"first entries={unpaired_metadata[:8]}"
        )

    actual_token_count = total_token_bytes // 4
    data_scope, token_count_policy = stage_token_count_contract(data)
    configured_token_count = data.get("known_token_count")
    if (
        isinstance(configured_token_count, bool)
        or not isinstance(configured_token_count, int)
        or configured_token_count < 1
    ):
        raise DataPipelineError(
            f"{config_name} must define a positive known_token_count"
        )
    if actual_token_count != configured_token_count:
        raise DataPipelineError(
            f"{config_name} token count changed: "
            f"expected={configured_token_count}, actual={actual_token_count}"
        )
    known_token_count = configured_token_count

    packing = data.get("packing")
    if not isinstance(packing, dict):
        raise DataPipelineError(f"{config_name} has no packing contract")
    tokenizer = data.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise DataPipelineError(f"{config_name} has no tokenizer contract")
    resolved_runtime_root = (
        runtime_root.expanduser().resolve()
        if runtime_root is not None
        else inventory_root
    )
    manifest: dict[str, Any] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "data_config": config_name,
        "data_config_sha256": sha256_file(config_path),
        "data_scope": data_scope,
        "token_count_policy": token_count_policy,
        "stage": data["stage"],
        "backend": data["backend"],
        "root": str(resolved_runtime_root),
        "inventory_root": str(inventory_root),
        "inventory_manifest": str(inventory_path.expanduser().resolve()),
        "inventory_manifest_sha256": sha256_file(inventory_path),
        "inventory_contract_sha256": inventory["contract_sha256"],
        "token_dtype": "uint32",
        "token_item_size": 4,
        "metadata_format": "dolma-document-offsets-csv-gzip-v1",
        "metadata_suffix": metadata_suffix,
        "token_patterns": list(token_patterns),
        "source_count": len(sources),
        "token_bytes": total_token_bytes,
        "token_count": actual_token_count,
        "known_token_count": known_token_count,
        "metadata_bytes": total_metadata_bytes,
        "tokenizer": {
            "eos_token_id": tokenizer.get("eos_token_id"),
            "pad_token_id": tokenizer.get("pad_token_id"),
            "bos_token_id": tokenizer.get("bos_token_id"),
        },
        "packing": packing,
        "sources": sources,
        "provenance": provenance(
            action="runtime-data-manifest",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    manifest["contract_sha256"] = sha256_json(contract_payload(manifest))
    return manifest


def load_runtime_data_manifest(
    path: Path,
    *,
    root: Path | None = None,
    verify_files: bool = False,
    checksums: bool = False,
    expected_stage: str | None = None,
    expected_backend: str | None = None,
    expected_token_count: int | None = None,
) -> dict[str, Any]:
    """Load and validate a Stage-2/3 training manifest.

    Training should consume ``resolved_token_paths`` in their existing order.
    ``root`` can remap the same immutable relative paths to a pod-visible mount.
    """

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPipelineError(f"cannot read runtime data manifest {path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA
    ):
        raise DataPipelineError(f"unsupported runtime data manifest schema: {path}")
    validate_contract_sha256(manifest, source=path)
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise DataPipelineError(
            f"runtime manifest stage mismatch: expected={expected_stage}, "
            f"actual={manifest.get('stage')}"
        )
    if expected_backend is not None and manifest.get("backend") != expected_backend:
        raise DataPipelineError(
            f"runtime manifest backend mismatch: expected={expected_backend}, "
            f"actual={manifest.get('backend')}"
        )
    if manifest.get("token_dtype") != "uint32" or manifest.get("token_item_size") != 4:
        raise DataPipelineError("runtime manifest token dtype must be raw uint32")
    if not isinstance(manifest.get("packing"), dict):
        raise DataPipelineError("runtime manifest has no packing contract")
    data_scope = manifest.get("data_scope", PRODUCTION_DATA_SCOPE)
    token_count_policy = manifest.get(
        "token_count_policy", FIXED_TOKEN_COUNT_POLICY
    )
    stage_token_count_contract(
        {
            "scope": data_scope,
            "token_count_policy": token_count_policy,
        }
    )

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DataPipelineError("runtime data manifest has no sources")
    seen_tokens: set[str] = set()
    seen_metadata: set[str] = set()
    token_bytes = 0
    metadata_bytes = 0
    resolved_root = (
        root.expanduser().resolve()
        if root is not None
        else Path(manifest["root"]).expanduser().resolve()
    )
    resolved_token_paths: list[str] = []
    resolved_metadata_paths: list[str] = []
    for source_id, source in enumerate(sources):
        if not isinstance(source, dict) or source.get("source_id") != source_id:
            raise DataPipelineError(
                "runtime data source IDs must be contiguous and ordered from zero"
            )
        token_relative = validate_relative_path(str(source.get("token_path", "")))
        metadata_relative = validate_relative_path(
            str(source.get("metadata_path", ""))
        )
        if token_relative in seen_tokens or metadata_relative in seen_metadata:
            raise DataPipelineError("runtime data manifest contains duplicate paths")
        seen_tokens.add(token_relative)
        seen_metadata.add(metadata_relative)
        source_token_bytes = source.get("token_bytes")
        source_tokens = source.get("tokens")
        source_metadata_bytes = source.get("metadata_bytes")
        if (
            isinstance(source_token_bytes, bool)
            or not isinstance(source_token_bytes, int)
            or source_token_bytes <= 0
            or source_token_bytes % 4
            or source_tokens != source_token_bytes // 4
        ):
            raise DataPipelineError(
                f"invalid uint32 token size for source {source_id}"
            )
        if (
            isinstance(source_metadata_bytes, bool)
            or not isinstance(source_metadata_bytes, int)
            or source_metadata_bytes <= 0
        ):
            raise DataPipelineError(
                f"invalid metadata size for source {source_id}"
            )
        token_path = resolved_root / token_relative
        metadata_path = resolved_root / metadata_relative
        for digest_key in ("token_sha256", "metadata_sha256"):
            digest = source.get(digest_key)
            if not isinstance(digest, str):
                raise DataPipelineError(
                    f"runtime data source {source_id} has no {digest_key}"
                )
            validate_sha256_text(
                digest,
                label=f"runtime data source {source_id} {digest_key}",
            )
        if verify_files:
            for candidate, expected_size, digest_key in (
                (token_path, source_token_bytes, "token_sha256"),
                (metadata_path, source_metadata_bytes, "metadata_sha256"),
            ):
                try:
                    actual_size = candidate.stat().st_size
                except FileNotFoundError as exc:
                    raise DataPipelineError(
                        f"runtime data source is missing: {candidate}"
                    ) from exc
                if actual_size != expected_size:
                    raise DataPipelineError(
                        f"runtime data source size changed: {candidate}: "
                        f"expected={expected_size}, actual={actual_size}"
                    )
                if checksums:
                    expected_digest = source.get(digest_key)
                    if expected_digest is None:
                        raise DataPipelineError(
                            f"runtime manifest has no checksum for {candidate}"
                        )
                    require_sha256(
                        candidate, expected_digest, label="runtime data source"
                    )
        token_bytes += source_token_bytes
        metadata_bytes += source_metadata_bytes
        resolved_token_paths.append(str(token_path))
        resolved_metadata_paths.append(str(metadata_path))

    token_count = token_bytes // 4
    aggregate_expected = {
        "source_count": len(sources),
        "token_bytes": token_bytes,
        "token_count": token_count,
        "metadata_bytes": metadata_bytes,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in aggregate_expected.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise DataPipelineError(
            f"runtime data manifest aggregate mismatch: {mismatches}"
        )
    if manifest.get("known_token_count") != token_count:
        raise DataPipelineError(
            "runtime manifest token_count differs from known_token_count"
        )
    if expected_token_count is not None and token_count != expected_token_count:
        raise DataPipelineError(
            f"runtime token count mismatch: expected={expected_token_count}, "
            f"actual={token_count}"
        )

    result = dict(manifest)
    result["manifest_path"] = str(path.expanduser().resolve())
    result["manifest_sha256"] = sha256_file(path)
    result["resolved_root"] = str(resolved_root)
    result["resolved_token_paths"] = resolved_token_paths
    result["resolved_metadata_paths"] = resolved_metadata_paths
    return result


def validate_runtime_data_profile(
    contract: dict[str, Any],
    *,
    config_name: str | None,
    config_sha256: str | None,
    pad_token_id: int,
) -> None:
    """Bind a sealed Stage-2/3 manifest to one frozen data profile.

    A manifest already seals its source ordering and aggregate token count.
    This additional check prevents a valid manifest for another stage, packing
    policy, or tokenizer contract from being paired with the current launch.
    It intentionally has no torch dependency so control-plane tests can audit
    the exact runtime contract.
    """

    if not isinstance(config_name, str) or not config_name:
        raise DataPipelineError(
            "data config name is required for a Stage-2/3 runtime manifest"
        )
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise DataPipelineError(
            "data config SHA256 must be a lowercase 64-character digest"
        )

    selected, config_path = data_config(config_name)
    actual_digest = sha256_file(config_path)
    if config_sha256 != actual_digest:
        raise DataPipelineError(
            "resolved data profile SHA256 no longer matches this checkout: "
            f"{config_sha256} != {actual_digest}"
        )
    tokenizer = selected.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise DataPipelineError(
            f"configs/data/{config_name}.json has no tokenizer contract"
        )
    expected = {
        "data_config": config_name,
        "data_config_sha256": actual_digest,
        "data_scope": selected.get("scope", PRODUCTION_DATA_SCOPE),
        "token_count_policy": selected.get(
            "token_count_policy", FIXED_TOKEN_COUNT_POLICY
        ),
        "stage": selected.get("stage"),
        "backend": selected.get("backend"),
        "tokenizer": {
            key: tokenizer.get(key)
            for key in ("eos_token_id", "pad_token_id", "bos_token_id")
        },
        "packing": selected.get("packing"),
    }
    data_scope, token_count_policy = stage_token_count_contract(selected)
    expected["data_scope"] = data_scope
    expected["token_count_policy"] = token_count_policy
    expected["known_token_count"] = selected.get("known_token_count")
    actual_contract = dict(contract)
    # Older runtime manifests omitted these fields; their only supported
    # interpretation is the production/fixed contract.
    actual_contract.setdefault("data_scope", PRODUCTION_DATA_SCOPE)
    actual_contract.setdefault("token_count_policy", FIXED_TOKEN_COUNT_POLICY)
    mismatches = {
        key: (actual_contract.get(key), value)
        for key, value in expected.items()
        if actual_contract.get(key) != value
    }
    if mismatches:
        raise DataPipelineError(
            "runtime data manifest does not match the selected frozen profile: "
            f"{mismatches}"
        )
    if int(pad_token_id) != int(expected["tokenizer"]["pad_token_id"]):
        raise DataPipelineError(
            "runtime pad-token ID differs from the frozen data profile"
        )


def token_source_entries(
    manifest: Path,
    *,
    root: Path | None,
    patterns: Sequence[str],
) -> tuple[Path, list[FileEntry], dict[str, Any]]:
    """Read either a generic inventory or a runtime training manifest."""

    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPipelineError(f"cannot read token source manifest {manifest}: {exc}") from exc
    if isinstance(raw, dict) and raw.get("schema") == RUNTIME_MANIFEST_SCHEMA:
        runtime = load_runtime_data_manifest(manifest, root=root)
        entries = [
            FileEntry(
                source["token_path"],
                int(source["token_bytes"]),
                source.get("token_sha256"),
            )
            for source in runtime["sources"]
            if _matches(source["token_path"], patterns)
        ]
        if len(entries) != runtime["source_count"]:
            raise DataPipelineError(
                "patterns excluded sources from an already frozen runtime manifest"
            )
        return Path(runtime["resolved_root"]), entries, runtime
    return inventory_entries(manifest, root=root, patterns=patterns)


def balanced_shards(
    entries: Sequence[FileEntry], shards: int
) -> tuple[tuple[FileEntry, ...], ...]:
    if shards < 1 or len(entries) < shards:
        raise DataPipelineError(
            f"invalid shards={shards} for {len(entries)} source files"
        )
    buckets: list[list[FileEntry]] = [[] for _ in range(shards)]
    heap: list[tuple[int, int]] = [(0, index) for index in range(shards)]
    heapq.heapify(heap)
    for entry in sorted(entries, key=lambda item: (-item.size, item.path)):
        total, index = heapq.heappop(heap)
        buckets[index].append(entry)
        heapq.heappush(heap, (total + entry.size, index))
    return tuple(
        tuple(sorted(bucket, key=lambda item: item.path)) for bucket in buckets
    )


def tokenizer_contract(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    validate_sha256_text(expected_sha256, label="tokenizer SHA256")
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        tokenizer_json = resolved / "tokenizer.json"
        snapshot_root = resolved
    else:
        tokenizer_json = resolved
        snapshot_root = resolved.parent
    actual = sha256_file(tokenizer_json) if tokenizer_json.is_file() else None
    if actual is None and expected_sha256 is None:
        raise DataPipelineError(
            "tokenizer.json is not visible and no --tokenizer-sha256 was supplied: "
            f"{tokenizer_json}"
        )
    if expected_sha256 is not None and actual is not None and actual != expected_sha256:
        raise DataPipelineError(
            f"tokenizer checksum mismatch: expected={expected_sha256}, actual={actual}"
        )
    snapshot_entries: list[dict[str, Any]] = []
    if resolved.is_dir():
        for item in sorted(resolved.rglob("*")):
            if item.is_symlink():
                raise DataPipelineError(
                    f"tokenizer snapshots reject symlinks: {item}"
                )
            if item.is_file():
                snapshot_entries.append(
                    {
                        "path": item.relative_to(snapshot_root).as_posix(),
                        "size": item.stat().st_size,
                        "sha256": sha256_file(item),
                    }
                )
    elif resolved.is_file():
        snapshot_entries.append(
            {
                "path": resolved.name,
                "size": resolved.stat().st_size,
                "sha256": actual,
            }
        )
    return {
        "path": str(resolved),
        "tokenizer_json": str(tokenizer_json),
        "sha256": expected_sha256 or actual,
        "verified": actual is not None and (expected_sha256 is None or actual == expected_sha256),
        "snapshot_root": str(snapshot_root),
        "snapshot_files": snapshot_entries,
        "snapshot_sha256": (
            sha256_json(snapshot_entries) if snapshot_entries else None
        ),
    }


def file_contract(
    path: Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> dict[str, Any]:
    validate_sha256_text(expected_sha256, label=f"{label} SHA256")
    resolved = path.expanduser().resolve()
    actual = sha256_file(resolved) if resolved.is_file() else None
    if actual is None and expected_sha256 is None:
        raise DataPipelineError(
            f"{label} is not visible and no expected SHA256 was supplied: {resolved}"
        )
    if expected_sha256 is not None and actual is not None and actual != expected_sha256:
        raise DataPipelineError(
            f"{label} SHA256 mismatch: expected={expected_sha256}, actual={actual}"
        )
    return {
        "path": str(resolved),
        "sha256": expected_sha256 or actual,
        "verified": actual is not None,
    }


def verify_file_contract(contract: dict[str, Any], *, label: str) -> None:
    path = Path(contract["path"])
    if not path.is_file():
        raise DataPipelineError(f"frozen {label} is missing: {path}")
    require_sha256(path, contract["sha256"], label=label)


def verify_tokenizer_contract(contract: dict[str, Any]) -> None:
    tokenizer_json = Path(contract["tokenizer_json"])
    if not tokenizer_json.is_file():
        raise DataPipelineError(f"frozen tokenizer is missing: {tokenizer_json}")
    actual = sha256_file(tokenizer_json)
    expected = contract.get("sha256")
    if expected is None or actual != expected:
        raise DataPipelineError(
            f"frozen tokenizer SHA256 mismatch: expected={expected}, actual={actual}"
        )
    snapshot_entries = contract.get("snapshot_files")
    if not isinstance(snapshot_entries, list):
        raise DataPipelineError("frozen tokenizer contract has no snapshot inventory")
    snapshot_root = Path(contract["snapshot_root"])
    for raw in snapshot_entries:
        path = snapshot_root / validate_relative_path(raw["path"])
        if (
            not path.is_file()
            or path.stat().st_size != raw["size"]
            or sha256_file(path) != raw["sha256"]
        ):
            raise DataPipelineError(f"frozen tokenizer snapshot changed: {path}")
    if snapshot_entries:
        actual_snapshot = sha256_json(snapshot_entries)
        if actual_snapshot != contract.get("snapshot_sha256"):
            raise DataPipelineError("frozen tokenizer snapshot contract is corrupt")


def build_tokenization_plan(
    *,
    config_name: str,
    source_manifest: Path,
    source_root: Path | None,
    output_root: Path,
    work_root: Path,
    tokenizer: Path,
    tokenizer_sha256: str | None,
    engine: str,
    shards: int,
    workers: int,
    patterns: Sequence[str],
    python: str,
    preprocess_script: Path | None,
    preprocess_script_sha256: str | None,
    sequence_length: int | None,
) -> dict[str, Any]:
    data, config_path = data_config(config_name)
    if engine not in {"dolma", "megatron"}:
        raise DataPipelineError("tokenization engine must be dolma or megatron")
    if workers < 1:
        raise DataPipelineError("workers must be positive")
    root, entries, inventory = inventory_entries(
        source_manifest, root=source_root, patterns=patterns
    )
    buckets = balanced_shards(entries, shards)
    output_root = output_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    tokenizer_meta = tokenizer_contract(tokenizer, tokenizer_sha256)
    tokenizer_path = tokenizer_meta["path"]
    dolma_tokenizer_path = tokenizer_meta["tokenizer_json"]
    preprocess_contract: dict[str, Any] | None = None
    if engine == "megatron":
        if preprocess_script is None:
            raise DataPipelineError(
                "--preprocess-script is required for the megatron engine"
            )
        preprocess_contract = file_contract(
            preprocess_script,
            preprocess_script_sha256,
            label="MindSpeed-LLM preprocess script",
        )

    configured_packing = data.get("packing")
    if sequence_length is None and isinstance(configured_packing, dict):
        configured_length = configured_packing.get("sequence_length")
        if isinstance(configured_length, int):
            sequence_length = configured_length
    if engine == "megatron" and sequence_length is not None:
        raise DataPipelineError(
            "Megatron pretraining indexing must not set --seq-length: current "
            "MindSpeed-LLM preprocess_data.py drops documents at or above that "
            "length. Stage-1 sequence_length belongs to the training config."
        )
    parts: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        part = f"part-{index:04d}"
        documents = [str(root / entry.path) for entry in bucket]
        bytes_in_part = sum(entry.size for entry in bucket)
        if engine == "dolma":
            config_file = work_root / "configs" / f"{part}.json"
            destination = output_root / part
            dolma_config = {
                "documents": documents,
                "destination": str(destination),
                "tokenizer": {
                    # Dolma's fast-tokenizer loader treats a directory as a
                    # Hugging Face Hub repo ID. A local snapshot must point at
                    # its frozen tokenizer.json so make_tokenizer() selects
                    # Tokenizer.from_file().
                    "name_or_path": dolma_tokenizer_path,
                    "bos_token_id": None,
                    "eos_token_id": int(data["tokenizer"]["eos_token_id"]),
                    "pad_token_id": int(data["tokenizer"]["pad_token_id"]),
                    "segment_before_tokenization": False,
                    "refresh": 0,
                    "fast": True,
                    "encode_special_tokens": False,
                },
                "processes": workers,
                "files_per_process": None,
                "batch_size": 10000,
                "ring_size": 8,
                "sample_ring_prop": False,
                "max_size": 1073741824,
                "dtype": "uint32",
                "debug": False,
                "seed": 3920,
                "work_dir": {
                    "input": str(work_root / "dolma-work" / part / "input"),
                    "output": str(work_root / "dolma-work" / part / "output"),
                },
                "dryrun": False,
            }
            command = [python, "-m", "dolma.cli.__main__", "-c", str(config_file), "tokens"]
            part_payload: dict[str, Any] = {
                "name": part,
                "files": len(bucket),
                "bytes": bytes_in_part,
                "source_paths": [entry.path for entry in bucket],
                "source_entries": [entry.as_json() for entry in bucket],
                "config_path": str(config_file),
                "config": dolma_config,
                "output": str(destination),
                "command": command,
            }
        else:
            assert preprocess_contract is not None
            hf_params_file = work_root / "configs" / f"{part}.hf-datasets.json"
            output_prefix = output_root / part
            hf_params = {
                "path": "json",
                "data_files": {"train": documents},
                "split": "train",
                "num_proc": workers,
            }
            command = [
                python,
                preprocess_contract["path"],
                "--input",
                str(root),
                "--hf-datasets-params",
                str(hf_params_file),
                "--tokenizer-type",
                "PretrainedFromHF",
                "--tokenizer-name-or-path",
                tokenizer_path,
                "--append-eod",
                "--json-keys",
                "text",
                "--dataset-impl",
                "mmap",
                "--workers",
                str(workers),
                "--output-prefix",
                str(output_prefix),
            ]
            part_payload = {
                "name": part,
                "files": len(bucket),
                "bytes": bytes_in_part,
                "source_paths": [entry.path for entry in bucket],
                "source_entries": [entry.as_json() for entry in bucket],
                "config_path": str(hf_params_file),
                "config": hf_params,
                "output_prefix": str(output_prefix),
                "expected_data_prefix": f"{output_prefix}_text_document",
                "command": command,
            }
        parts.append(part_payload)

    plan = {
        "schema": PLAN_SCHEMA,
        "action": "tokenize",
        "data_config": config_name,
        "stage": data["stage"],
        "backend": data["backend"],
        "engine": engine,
        "source_manifest": str(source_manifest.expanduser().resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_contract_sha256": inventory.get("contract_sha256"),
        "source_root": str(root),
        "output_root": str(output_root),
        "work_root": str(work_root),
        "tokenizer": tokenizer_meta,
        "preprocess_script": preprocess_contract,
        "patterns": list(patterns),
        "shards": shards,
        "workers_per_shard": workers,
        "sequence_length": sequence_length,
        "parts": parts,
        "provenance": provenance(
            action="tokenize-plan",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    plan["contract_sha256"] = sha256_json(contract_payload(plan))
    return plan


def materialize_tokenization_plan(plan: dict[str, Any], plan_path: Path) -> None:
    """Write per-part engine configs plus the immutable top-level plan."""

    if plan.get("schema") != PLAN_SCHEMA or plan.get("action") != "tokenize":
        raise DataPipelineError("not a tokenization plan")
    # Both pinned engines require their destination parent to exist.  The
    # MindSpeed-LLM preprocessor deliberately refuses a missing output-prefix
    # parent; Dolma likewise expects a materialized work root.  Directory
    # creation is not part of the immutable contract payload.
    Path(plan["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(plan["work_root"]).mkdir(parents=True, exist_ok=True)
    for part in plan["parts"]:
        write_immutable_json(Path(part["config_path"]), part["config"])
    write_immutable_json(plan_path, plan)


def load_plan(path: Path, action: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPipelineError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise DataPipelineError(f"unsupported data plan schema: {path}")
    validate_contract_sha256(value, source=path)
    if action is not None and value.get("action") != action:
        raise DataPipelineError(
            f"expected a {action!r} plan, got {value.get('action')!r}"
        )
    config_name = value.get("data_config")
    expected_config_sha = value.get("provenance", {}).get("data_config_sha256")
    if isinstance(config_name, str) and isinstance(expected_config_sha, str):
        require_sha256(
            CONFIG_ROOT / "data" / f"{config_name}.json",
            expected_config_sha,
            label="data config",
        )
    return value


def tokenization_part(plan: dict[str, Any], part: int) -> dict[str, Any]:
    parts = plan.get("parts")
    if not isinstance(parts, list) or not 0 <= part < len(parts):
        raise DataPipelineError(
            f"part index {part} is outside [0, {len(parts) if isinstance(parts, list) else 0})"
        )
    value = parts[part]
    if not isinstance(value, dict) or not isinstance(value.get("command"), list):
        raise DataPipelineError(f"invalid part {part} in tokenization plan")
    return value


def run_tokenization_part(
    plan_path: Path,
    *,
    part: int,
    dry_run: bool,
) -> dict[str, Any]:
    plan = load_plan(plan_path, action="tokenize")
    require_sha256(
        Path(plan["source_manifest"]),
        plan["source_manifest_sha256"],
        label="source manifest",
    )
    value = tokenization_part(plan, part)
    command = [str(item) for item in value["command"]]
    verify_tokenizer_contract(plan["tokenizer"])
    preprocess_contract = plan.get("preprocess_script")
    if preprocess_contract is not None:
        verify_file_contract(
            preprocess_contract, label="MindSpeed-LLM preprocess script"
        )
    source_root = Path(plan["source_root"])
    for raw in value["source_entries"]:
        path = source_root / validate_relative_path(raw["path"])
        try:
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise DataPipelineError(f"tokenizer source is missing: {path}") from exc
        if size != raw["size"]:
            raise DataPipelineError(
                f"tokenizer source size changed: {path}: "
                f"expected={raw['size']}, actual={size}"
            )
        expected_digest = raw.get("sha256")
        if expected_digest is not None and sha256_file(path) != expected_digest:
            raise DataPipelineError(f"tokenizer source SHA256 changed: {path}")
    result = {
        "schema": REPORT_SCHEMA,
        "action": "tokenize-part",
        "plan": str(plan_path.expanduser().resolve()),
        "plan_sha256": sha256_file(plan_path),
        "part": value["name"],
        "command": command,
        "dry_run": dry_run,
        "tool_versions": package_versions(
            ("dolma", "datasets", "tokenizers", "transformers")
        ),
    }
    if dry_run:
        result["state"] = "planned"
        return result
    started = utc_now()
    completed = subprocess.run(command, check=False)
    result.update(
        {
            "started_at": started,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "state": "ok" if completed.returncode == 0 else "failed",
        }
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if plan["engine"] == "megatron":
        prefix = Path(value["expected_data_prefix"])
        artifacts = [Path(f"{prefix}.bin"), Path(f"{prefix}.idx")]
    else:
        output = Path(value["output"])
        artifacts = sorted(output.glob("*.npy")) + sorted(output.glob("*.csv.gz"))
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing or not artifacts:
        raise DataPipelineError(
            f"tokenizer part produced no complete artifact set; missing={missing}"
        )
    result["artifacts"] = [
        {"path": str(path), "bytes": path.stat().st_size} for path in artifacts
    ]
    return result


def finalize_stage1_index(
    *,
    plan_path: Path,
    data_args_path: Path,
) -> dict[str, Any]:
    plan = load_plan(plan_path, action="tokenize")
    if plan.get("engine") != "megatron":
        raise DataPipelineError("Stage-1 indexing requires a megatron tokenization plan")
    prefixes: list[str] = []
    weighted_entries: list[tuple[int, str]] = []
    artifacts: list[dict[str, Any]] = []
    for part in plan["parts"]:
        prefix = Path(part["expected_data_prefix"])
        bin_path = Path(f"{prefix}.bin")
        idx_path = Path(f"{prefix}.idx")
        if not bin_path.is_file() or not idx_path.is_file():
            raise DataPipelineError(
                f"missing Megatron mmap pair for {part['name']}: "
                f"{bin_path}, {idx_path}"
            )
        bin_bytes = bin_path.stat().st_size
        if bin_bytes < 1:
            raise DataPipelineError(f"empty Megatron token stream: {bin_path}")
        prefixes.append(str(prefix))
        # Separate mmap prefixes must be sampled in proportion to their token
        # counts. They share one dtype, so bin bytes are an exact proportional
        # weight and avoid oversampling a smaller preprocessing shard.
        weighted_entries.append((bin_bytes, str(prefix)))
        artifacts.append(
            {
                "prefix": str(prefix),
                "bin_bytes": bin_bytes,
                "idx_bytes": idx_path.stat().st_size,
                # Hashing a multi-terabyte token stream during finalization
                # would add hours of I/O. Source contracts plus immutable byte
                # sizes freeze the payload; the compact structural index is
                # hashed in full.
                "bin_integrity": "source-contract-and-byte-size",
                "idx_sha256": sha256_file(idx_path),
            }
        )
    write_immutable_text(
        data_args_path,
        "".join(f"{weight} {prefix}\n" for weight, prefix in weighted_entries),
    )
    report = {
        "schema": REPORT_SCHEMA,
        "action": "stage1-index",
        "state": "ok",
        "plan": str(plan_path.expanduser().resolve()),
        "plan_sha256": sha256_file(plan_path),
        "data_args_path": str(data_args_path.expanduser().resolve()),
        "data_args_sha256": sha256_file(data_args_path),
        "prefixes": prefixes,
        "blend_weights": [weight for weight, _ in weighted_entries],
        "artifacts": artifacts,
        "completed_at": utc_now(),
    }
    return report


def build_cache_plan(
    *,
    config_name: str,
    source_manifest: Path,
    source_root: Path | None,
    work_root: Path,
    workers: int,
    patterns: Sequence[str],
    python: str,
) -> dict[str, Any]:
    data, config_path = data_config(config_name)
    stage = data.get("stage")
    if stage not in {"stage2", "stage3"}:
        raise DataPipelineError("cache preparation supports Stage 2 or Stage 3")
    packing = data.get("packing")
    if not isinstance(packing, dict):
        raise DataPipelineError(f"{config_name} has no packing contract")
    resolved_root, source_entries, source_contract = token_source_entries(
        source_manifest, root=source_root, patterns=patterns
    )
    if source_contract.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise DataPipelineError(
            "Stage-2/3 cache preparation requires the sealed runtime data "
            "manifest, not a generic inventory"
        )
    if any(entry.sha256 is None for entry in source_entries):
        raise DataPipelineError(
            "Stage-2/3 cache sources must have content SHA256 values"
        )
    validate_runtime_data_profile(
        source_contract,
        config_name=config_name,
        config_sha256=sha256_file(config_path),
        pad_token_id=int(data["tokenizer"]["pad_token_id"]),
    )
    paths = tuple(resolved_root / entry.path for entry in source_entries)
    source_bytes = sum(entry.size for entry in source_entries)
    if source_bytes % 4:
        raise DataPipelineError(
            f"raw uint32 token sources are not 4-byte aligned: bytes={source_bytes}"
        )
    known_tokens = data.get("known_token_count")
    if known_tokens is not None and source_bytes // 4 != known_tokens:
        raise DataPipelineError(
            f"{config_name} token count changed: expected={known_tokens}, "
            f"actual={source_bytes // 4}"
        )
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "data" / "olmo3_data.py"),
        "prepare-cache",
        "--data",
        config_name,
        "--source-manifest",
        str(source_manifest.expanduser().resolve()),
        "--work-root",
        str(work_root.expanduser().resolve()),
        "--workers",
        str(workers),
        "--execute",
    ]
    if source_root is not None:
        command.extend(["--source-root", str(source_root.expanduser().resolve())])
    for pattern in patterns:
        command.extend(["--pattern", pattern])
    plan = {
        "schema": PLAN_SCHEMA,
        "action": "prepare-cache",
        "data_config": config_name,
        "stage": stage,
        "backend": data["backend"],
        "source_manifest": str(source_manifest.expanduser().resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_manifest_schema": source_contract["schema"],
        "source_contract_sha256": source_contract["contract_sha256"],
        "source_root": str(resolved_root),
        "source_files": len(paths),
        "source_bytes": source_bytes,
        "source_tokens": source_bytes // 4,
        "known_token_count": known_tokens,
        "source_paths_sha256": sha256_json([str(path) for path in paths]),
        "work_root": str(work_root.expanduser().resolve()),
        "workers": workers,
        "patterns": list(patterns),
        "packing": packing,
        "command": command,
        "provenance": provenance(
            action="prepare-cache-plan",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    plan["contract_sha256"] = sha256_json(contract_payload(plan))
    return plan


def execute_cache_plan(
    *,
    config_name: str,
    source_manifest: Path,
    source_root: Path | None,
    work_root: Path,
    workers: int,
    patterns: Sequence[str],
) -> dict[str, Any]:
    data, config_path = data_config(config_name)
    packing = data["packing"]
    resolved_root, source_entries, source_contract = token_source_entries(
        source_manifest, root=source_root, patterns=patterns
    )
    if source_contract.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise DataPipelineError(
            "Stage-2/3 cache execution requires the sealed runtime data manifest"
        )
    if any(entry.sha256 is None for entry in source_entries):
        raise DataPipelineError(
            "Stage-2/3 cache sources must have content SHA256 values"
        )
    validate_runtime_data_profile(
        source_contract,
        config_name=config_name,
        config_sha256=sha256_file(config_path),
        pad_token_id=int(data["tokenizer"]["pad_token_id"]),
    )
    paths = tuple(resolved_root / entry.path for entry in source_entries)
    source_contract_sha256 = str(source_contract["contract_sha256"])
    if data["stage"] == "stage2":
        from runtime.olmo3_packed_dataset import prepare_olmo3_midtraining_cache

        cache_root = prepare_olmo3_midtraining_cache(
            paths,
            work_root,
            source_contract_sha256=source_contract_sha256,
            sequence_length=int(packing["sequence_length"]),
            dtype="uint32",
        )
    elif data["stage"] == "stage3":
        from runtime.olmo3_packed_dataset import prepare_olmo3_long_context_cache

        cache_root = prepare_olmo3_long_context_cache(
            paths,
            work_root,
            source_contract_sha256=source_contract_sha256,
            sequence_length=int(packing["sequence_length"]),
            eos_token_id=int(data["tokenizer"]["eos_token_id"]),
            pad_token_id=int(data["tokenizer"]["pad_token_id"]),
            source_group_size=int(packing["source_group_size"]),
            source_permutation_seed=int(packing["source_permutation_seed"]),
            dtype="uint32",
            workers=workers,
        )
    else:
        raise DataPipelineError("cache preparation supports Stage 2 or Stage 3")
    complete = cache_root / "complete.json"
    return {
        "schema": REPORT_SCHEMA,
        "action": "prepare-cache",
        "state": "ok",
        "data_config": config_name,
        "source_manifest": str(source_manifest.expanduser().resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_contract_sha256": source_contract_sha256,
        "cache_root": str(cache_root),
        "complete_manifest": str(complete),
        "complete_manifest_sha256": sha256_file(complete),
        "completed_at": utc_now(),
        "tool_versions": package_versions(("numpy", "torch")),
    }


def run_cache_plan(plan_path: Path, *, dry_run: bool) -> dict[str, Any]:
    plan = load_plan(plan_path, action="prepare-cache")
    require_sha256(
        Path(plan["source_manifest"]),
        plan["source_manifest_sha256"],
        label="source manifest",
    )
    if dry_run:
        return {
            "schema": REPORT_SCHEMA,
            "action": "prepare-cache",
            "state": "planned",
            "plan": str(plan_path.expanduser().resolve()),
            "plan_sha256": sha256_file(plan_path),
            "command": plan["command"],
            "dry_run": True,
        }
    report = execute_cache_plan(
        config_name=plan["data_config"],
        source_manifest=Path(plan["source_manifest"]),
        source_root=Path(plan["source_root"]),
        work_root=Path(plan["work_root"]),
        workers=int(plan["workers"]),
        patterns=tuple(plan["patterns"]),
    )
    report["plan"] = str(plan_path.expanduser().resolve())
    report["plan_sha256"] = sha256_file(plan_path)
    return report


def build_sft_plan(
    *,
    config_name: str,
    raw_manifest: Path,
    raw_root: Path | None,
    expected_raw_files: int | None,
    raw_glob: str,
    converted_root: Path,
    converted_manifest: Path,
    work_root: Path,
    tokenizer: Path,
    tokenizer_sha256: str | None,
    converter: Path,
    converter_sha256: str | None,
    open_instruct_root: Path,
    python: str,
    workers: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    data, config_path = data_config(config_name)
    if data.get("stage") not in {"sft_think", "sft_instruct"}:
        raise DataPipelineError("SFT planning requires an SFT data config")
    if workers < 1:
        raise DataPipelineError("workers must be positive")
    packing = data.get("packing")
    if not isinstance(packing, dict):
        raise DataPipelineError(f"{config_name} has no SFT packing contract")
    tokenizer_meta = tokenizer_contract(tokenizer, tokenizer_sha256)
    resolved_raw_root, raw_entries, raw_inventory = inventory_entries(
        raw_manifest,
        root=raw_root,
        patterns=("**/*.parquet",),
    )
    if expected_raw_files is not None:
        if expected_raw_files < 1:
            raise DataPipelineError("expected SFT raw file count must be positive")
        if len(raw_entries) != expected_raw_files:
            raise DataPipelineError(
                f"SFT raw file count changed: expected={expected_raw_files}, "
                f"actual={len(raw_entries)}"
            )
    converted_root = converted_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    converter_meta = file_contract(
        converter,
        converter_sha256,
        label="Open-Instruct converter",
    )
    open_instruct_root = open_instruct_root.expanduser().resolve()
    open_instruct_commit = _git_head(open_instruct_root)
    open_instruct_dirty = _git_dirty(open_instruct_root)
    if config_name in {"dolci_think", "dolci_instruct"}:
        if (
            not isinstance(open_instruct_commit, str)
            or len(open_instruct_commit) != 40
        ):
            raise DataPipelineError(
                "production SFT requires a Git-pinned Open-Instruct checkout"
            )
        if open_instruct_dirty is not False:
            raise DataPipelineError(
                "production SFT requires a pristine Open-Instruct checkout"
            )
    sequence_length = int(packing["sequence_length"])
    frozen_raw_paths = [
        str(resolved_raw_root / validate_relative_path(entry.path))
        for entry in raw_entries
    ]
    globbed_raw_paths = sorted(
        str(Path(path).expanduser().resolve()) for path in glob.glob(raw_glob)
    )
    if globbed_raw_paths != frozen_raw_paths:
        raise DataPipelineError(
            "SFT raw glob does not resolve to the exact frozen parquet inventory"
        )
    dataset_mixer: list[str] = []
    for path in frozen_raw_paths:
        dataset_mixer.extend((path, "1.0"))
    converter_command = [
        python,
        converter_meta["path"],
        "--output_dir",
        str(converted_root),
        "--dataset_mixer_list",
        *dataset_mixer,
        "--dataset_mixer_list_splits",
        "train",
        "--dataset_transform_fn",
        "sft_tulu_tokenize_and_truncate_v1",
        "sft_tulu_filter_v1",
        "--dataset_cache_mode",
        "local",
        "--dataset_local_cache_dir",
        str(work_root / "hf-cache"),
        "--dataset_skip_cache",
        "True",
        "--max_seq_length",
        str(sequence_length),
        "--resume",
        "True",
        "--shuffle_seed",
        str(shuffle_seed),
        "--tokenizer_name_or_path",
        tokenizer_meta["path"],
        "--get_tokenizer_fn",
        "get_tokenizer_tulu_v2_2",
        "--use_fast",
        "True",
    ]
    packing_command = [
        python,
        "-m",
        "runtime.olmo3_sft_dataset",
        "prepare",
        "--data-dir",
        str(converted_root),
        "--work-dir",
        str(sft_packed_work_dir(work_root)),
        "--sequence-length",
        str(sequence_length),
        "--eos-token-id",
        str(data["tokenizer"]["eos_token_id"]),
        "--pad-token-id",
        str(data["tokenizer"]["pad_token_id"]),
        "--dtype",
        "uint32",
        "--workers",
        str(workers),
    ]
    plan = {
        "schema": PLAN_SCHEMA,
        "action": "prepare-sft",
        "data_config": config_name,
        "stage": data["stage"],
        "raw_glob": raw_glob,
        "raw_manifest": str(raw_manifest.expanduser().resolve()),
        "raw_manifest_sha256": sha256_file(raw_manifest),
        "raw_root": str(resolved_raw_root),
        "raw_files": len(raw_entries),
        "expected_raw_files": expected_raw_files,
        "raw_bytes": sum(entry.size for entry in raw_entries),
        "raw_contract_sha256": raw_inventory["contract_sha256"],
        "dataset_mixer": dataset_mixer,
        "converted_root": str(converted_root),
        "converted_manifest": str(converted_manifest.expanduser().resolve()),
        "work_root": str(work_root),
        "tokenizer": tokenizer_meta,
        "converter": converter_meta,
        "open_instruct": {
            "root": str(open_instruct_root),
            "git_commit": open_instruct_commit,
            "git_dirty": open_instruct_dirty,
        },
        "sequence_length": sequence_length,
        "shuffle_seed": shuffle_seed,
        "workers": workers,
        "assistant_only_loss": bool(packing["assistant_only_loss"]),
        "commands": [converter_command, packing_command],
        "environment": {
            "PYTHONPATH_PREPEND": [
                str(PROJECT_ROOT / "src" / "runtime" / "cpu_compat"),
                str(open_instruct_root),
                str(PROJECT_ROOT / "src"),
            ],
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        "provenance": provenance(
            action="prepare-sft-plan",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    plan["contract_sha256"] = sha256_json(contract_payload(plan))
    return plan


def _sft_packed_completion(
    *,
    config_name: str,
    converted_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Read and validate the exact packed ``complete.json`` for an SFT run."""

    data, _ = data_config(config_name)
    packing = data.get("packing")
    tokenizer = data.get("tokenizer")
    if not isinstance(packing, dict) or not isinstance(tokenizer, dict):
        raise DataPipelineError(f"{config_name} has no SFT packing contract")
    from runtime.olmo3_sft_dataset import describe_olmo3_sft_cache

    description = describe_olmo3_sft_cache(
        converted_root,
        sft_packed_work_dir(work_root),
        sequence_length=int(packing["sequence_length"]),
        eos_token_id=int(tokenizer["eos_token_id"]),
        pad_token_id=int(tokenizer["pad_token_id"]),
        dtype="uint32",
    )
    complete_path = Path(description["cache_root"]) / "complete.json"
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPipelineError(
            f"cannot read packed SFT completion manifest {complete_path}: {exc}"
        ) from exc
    instances = complete.get("instances")
    fingerprint = complete.get("fingerprint")
    if (
        isinstance(instances, bool)
        or not isinstance(instances, int)
        or instances < 1
        or not isinstance(fingerprint, str)
        or not fingerprint
    ):
        raise DataPipelineError(
            "packed SFT complete.json must contain positive instances and a fingerprint"
        )
    expected = {
        "instances": int(description["base_instances"]),
        "fingerprint": str(description["fingerprint"]),
        "sequence_length": int(packing["sequence_length"]),
        "assistant_only_loss": True,
    }
    mismatches = {
        key: (complete.get(key), value)
        for key, value in expected.items()
        if complete.get(key) != value
    }
    if mismatches:
        raise DataPipelineError(
            f"packed SFT completion contract mismatch: {mismatches}"
        )
    return {
        "packed_complete_manifest": str(complete_path),
        "packed_complete_manifest_sha256": sha256_file(complete_path),
        "instances": instances,
        "fingerprint": fingerprint,
    }


def run_sft_plan(plan_path: Path, *, dry_run: bool) -> dict[str, Any]:
    plan = load_plan(plan_path, action="prepare-sft")
    require_sha256(
        Path(plan["raw_manifest"]),
        plan["raw_manifest_sha256"],
        label="raw SFT manifest",
    )
    commands = [[str(item) for item in command] for command in plan["commands"]]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "action": "prepare-sft",
        "plan": str(plan_path.expanduser().resolve()),
        "plan_sha256": sha256_file(plan_path),
        "commands": commands,
        "dry_run": dry_run,
        "tool_versions": package_versions(
            ("datasets", "tokenizers", "transformers", "torch", "numpy")
        ),
    }
    if dry_run:
        report["state"] = "planned"
        return report
    verify_tokenizer_contract(plan["tokenizer"])
    verify_file_contract(plan["converter"], label="Open-Instruct converter")
    open_instruct = plan["open_instruct"]
    expected_commit = open_instruct.get("git_commit")
    if expected_commit is not None:
        actual_commit = _git_head(Path(open_instruct["root"]))
        if actual_commit != expected_commit:
            raise DataPipelineError(
                "Open-Instruct checkout changed after planning: "
                f"expected={expected_commit}, actual={actual_commit}"
            )
    if open_instruct.get("git_dirty") is False:
        actual_dirty = _git_dirty(Path(open_instruct["root"]))
        if actual_dirty is not False:
            raise DataPipelineError(
                "Open-Instruct checkout became dirty after SFT planning"
            )
    raw_report = verify_inventory(
        Path(plan["raw_manifest"]),
        root=Path(plan["raw_root"]),
        checksums=False,
        workers=int(plan["workers"]),
    )
    if raw_report["state"] != "ok":
        raise DataPipelineError(
            f"SFT raw-data inventory failed verification: {raw_report['problems'][:8]}"
        )
    environment = os.environ.copy()
    prepend = plan.get("environment", {}).get("PYTHONPATH_PREPEND", [])
    environment["PYTHONPATH"] = os.pathsep.join(
        [*prepend, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for key, value in plan.get("environment", {}).items():
        if key != "PYTHONPATH_PREPEND":
            environment[str(key)] = str(value)
    started = utc_now()
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    converted_root = Path(plan["converted_root"])
    statistics = converted_root / "dataset_statistics.json"
    if not statistics.is_file():
        raise DataPipelineError(
            f"SFT converter did not produce completion metadata: {statistics}"
        )
    converted_inventory = create_inventory(
        root=converted_root,
        patterns=(
            "token_ids_part_*.npy",
            "labels_mask_part_*.npy",
            "token_ids_part_*.csv.gz",
            "dataset_statistics.json",
            "tokenizer/*",
        ),
        checksums=False,
        workers=int(plan["workers"]),
        config_name=plan["data_config"],
        config_path=CONFIG_ROOT / "data" / f"{plan['data_config']}.json",
    )
    converted_manifest = Path(plan["converted_manifest"])
    write_immutable_json(converted_manifest, converted_inventory)
    packed_completion = _sft_packed_completion(
        config_name=str(plan["data_config"]),
        converted_root=converted_root,
        work_root=Path(plan["work_root"]),
    )
    report.update(
        {
            "state": "ok",
            "started_at": started,
            "finished_at": utc_now(),
            "dataset_statistics": str(statistics),
            "dataset_statistics_sha256": sha256_file(statistics),
            "converted_manifest": str(converted_manifest),
            "converted_manifest_sha256": sha256_file(converted_manifest),
            "converted_files": converted_inventory["files"],
            "converted_bytes": converted_inventory["bytes"],
            **packed_completion,
        }
    )
    return report


def download_plan(
    *,
    config_name: str,
    root: Path,
    manifest: Path | None,
    repository: str | None,
    revision: str | None,
    endpoint: str | None,
    patterns: Sequence[str],
    workers: int,
    checksums: bool,
) -> dict[str, Any]:
    _, source, config_path = source_from_config(
        config_name, repository=repository, revision=revision
    )
    if source["revision"] is None:
        raise DataPipelineError(
            "download revision must be frozen; pass --revision or set it in config"
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "action": "download",
        "data_config": config_name,
        "source": source,
        "endpoint": endpoint,
        "root": str(root.expanduser().resolve()),
        "manifest": (
            str(manifest.expanduser().resolve()) if manifest is not None else None
        ),
        "patterns": list(patterns),
        "workers": workers,
        "checksums": checksums,
        "provenance": provenance(
            action="download-plan",
            config_name=config_name,
            config_path=config_path,
        ),
    }
    plan["contract_sha256"] = sha256_json(contract_payload(plan))
    return plan


def execute_download(plan: dict[str, Any]) -> dict[str, Any]:
    """Download one frozen Hub dataset snapshot and create its inventory."""

    if plan.get("schema") != PLAN_SCHEMA or plan.get("action") != "download":
        raise DataPipelineError("not a download plan")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DataPipelineError(
            "download execution requires huggingface_hub; dry-run does not"
        ) from exc
    if plan.get("endpoint"):
        os.environ["HF_ENDPOINT"] = str(plan["endpoint"])
    source = plan["source"]
    root = Path(plan["root"])
    root_claim = root / ".olmo3-download-root.json"
    claim_payload = {
        "schema": "olmo3-download-root-v1",
        "contract_sha256": plan["contract_sha256"],
        "source": source,
        "patterns": plan["patterns"],
    }
    if root.exists():
        if not root.is_dir():
            raise DataPipelineError(f"download root is not a directory: {root}")
        entries = tuple(root.iterdir())
        if entries and not root_claim.is_file():
            raise DataPipelineError(
                "download root is non-empty and has no matching immutable "
                f"claim: {root}; use a new empty root"
            )
    else:
        root.mkdir(parents=True)
    write_immutable_json(root_claim, claim_payload)
    manifest_value = plan.get("manifest")
    if manifest_value is None:
        raise DataPipelineError("executed download plan has no output manifest")
    manifest = Path(manifest_value)
    if manifest.exists():
        inventory = load_inventory(manifest)
        if inventory.get("source") != source:
            raise DataPipelineError(
                "existing download inventory source differs from the frozen plan"
            )
        verification = verify_inventory(
            manifest,
            root=root,
            checksums=bool(plan["checksums"]),
            workers=int(plan["workers"]),
        )
        if verification["state"] != "ok":
            raise DataPipelineError(
                "existing download inventory failed verification; "
                f"problems={verification['problems']!r}"
            )
        return {
            "schema": REPORT_SCHEMA,
            "action": "download",
            "state": "ok",
            "source": source,
            "root": str(root),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "root_claim": str(root_claim),
            "root_claim_sha256": sha256_file(root_claim),
            "files": inventory["files"],
            "bytes": inventory["bytes"],
            "resumed_complete": True,
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "tool_versions": package_versions(("huggingface-hub",)),
        }
    started = utc_now()
    snapshot_download(
        repo_id=source["repository"],
        repo_type="dataset",
        revision=source["revision"],
        local_dir=root,
        allow_patterns=plan["patterns"] or None,
        max_workers=int(plan["workers"]),
    )
    inventory = create_inventory(
        root=root,
        patterns=plan["patterns"],
        checksums=bool(plan["checksums"]),
        workers=int(plan["workers"]),
        source=source,
        config_name=plan["data_config"],
        config_path=CONFIG_ROOT / "data" / f"{plan['data_config']}.json",
    )
    write_immutable_json(manifest, inventory)
    return {
        "schema": REPORT_SCHEMA,
        "action": "download",
        "state": "ok",
        "source": source,
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "root_claim": str(root_claim),
        "root_claim_sha256": sha256_file(root_claim),
        "files": inventory["files"],
        "bytes": inventory["bytes"],
        "started_at": started,
        "finished_at": utc_now(),
        "tool_versions": package_versions(("huggingface-hub",)),
    }


def run_download_plan(plan_path: Path, *, dry_run: bool) -> dict[str, Any]:
    plan = load_plan(plan_path, action="download")
    if dry_run:
        return {
            "schema": REPORT_SCHEMA,
            "action": "download",
            "state": "planned",
            "plan": str(plan_path.expanduser().resolve()),
            "plan_sha256": sha256_file(plan_path),
            "source": plan["source"],
            "root": plan["root"],
            "dry_run": True,
        }
    report = execute_download(plan)
    report["plan"] = str(plan_path.expanduser().resolve())
    report["plan_sha256"] = sha256_file(plan_path)
    return report


def shell_command(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)
