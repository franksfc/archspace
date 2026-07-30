#!/usr/bin/env python3
"""Clone or verify the immutable third-party repositories.

MindSpeed, MindSpeed-LLM, OLMo-core, and OLMES are pristine Git checkouts.
Project-specific runtime changes live only in the vendored ``megatron/`` tree.
This script never patches or rewrites an existing third-party checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "UPSTREAM_LOCK.json"


def output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        command, cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def load_lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_pristine(name: str, spec: dict[str, Any]) -> None:
    target = PROJECT_ROOT / spec["path"]
    if not (target / ".git").exists():
        raise SystemExit(f"{name}: missing Git checkout: {target}")
    try:
        origin = output("git", "remote", "get-url", "origin", cwd=target)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{name}: missing official origin remote") from exc
    if origin != spec["url"]:
        raise SystemExit(
            f"{name}: origin mismatch: expected={spec['url']} actual={origin}"
        )
    actual = output("git", "rev-parse", "HEAD", cwd=target)
    if actual != spec["commit"]:
        raise SystemExit(
            f"{name}: commit mismatch: expected={spec['commit']} actual={actual}"
        )
    dirty = output("git", "status", "--porcelain", cwd=target)
    if dirty:
        raise SystemExit(
            f"{name}: third-party checkout is modified; it must remain pristine:\n"
            f"{dirty}"
        )
    generated_python = [
        path
        for path in target.rglob("*")
        if path.name == "__pycache__"
        or (path.is_file() and path.suffix in {".pyc", ".pyo"})
    ]
    if spec.get("require_full_history") and generated_python:
        preview = "\n".join(
            str(path.relative_to(target)) for path in generated_python[:16]
        )
        raise SystemExit(
            f"{name}: generated Python artifacts were written into the "
            f"pristine checkout:\n{preview}"
        )
    if spec.get("require_full_history"):
        shallow = output(
            "git", "rev-parse", "--is-shallow-repository", cwd=target
        )
        if shallow != "false":
            raise SystemExit(
                f"{name}: shallow checkout is not allowed; fetch the complete "
                "official Git history"
            )
        fetch_refspecs = output(
            "git", "config", "--get-all", "remote.origin.fetch", cwd=target
        ).splitlines()
        full_heads_refspec = "+refs/heads/*:refs/remotes/origin/*"
        if full_heads_refspec not in fetch_refspecs:
            raise SystemExit(
                f"{name}: single-branch checkout is not allowed; expected "
                f"remote.origin.fetch={full_heads_refspec!r}"
            )


def clone_missing(name: str, spec: dict[str, Any]) -> None:
    target = PROJECT_ROOT / spec["path"]
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", spec["url"], str(target))
    run("git", "switch", "--detach", spec["commit"], cwd=target)


def verify_megatron(spec: dict[str, Any]) -> None:
    marker_path = PROJECT_ROOT / "megatron" / "OLMO3_VENDOR.json"
    if not marker_path.is_file():
        raise SystemExit(f"missing vendored Megatron marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("upstream_commit") != spec["commit"]:
        raise SystemExit(
            "vendored Megatron base does not match UPSTREAM_LOCK.json: "
            f"{marker.get('upstream_commit')} != {spec['commit']}"
        )
    if not (PROJECT_ROOT / "megatron" / "core").is_dir():
        raise SystemExit("vendored Megatron package is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clone-missing",
        action="store_true",
        help="Clone only absent pristine repositories before verification.",
    )
    args = parser.parse_args()
    lock = load_lock()

    for name, spec in lock["repositories"].items():
        if spec.get("must_be_pristine"):
            if args.clone_missing:
                clone_missing(name, spec)
            verify_pristine(name, spec)
        elif spec.get("vendored_and_modified"):
            verify_megatron(spec)
        else:
            raise SystemExit(f"{name}: unsupported repository policy")

    print("OLMO3_RUNTIME_STACK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
