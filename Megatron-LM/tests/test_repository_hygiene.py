from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "check_repository_hygiene.py"
SPEC = importlib.util.spec_from_file_location("check_repository_hygiene", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hygiene = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hygiene
SPEC.loader.exec_module(hygiene)


def _rules(findings: list[object]) -> set[str]:
    return {finding.rule for finding in findings}


def test_scanner_detects_secret_private_ip_and_operator_marker(
    tmp_path: Path,
) -> None:
    secret = "sk-" + ("A" * 32)
    private_ip = ".".join(("10", "23", "45", "67"))
    marker = "private" + "-storage-root"
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(
        f"token={secret}\nendpoint={private_ip}\npath={marker}\n",
        encoding="utf-8",
    )
    findings = hygiene.scan_file(
        candidate,
        root=tmp_path,
        deny_strings=[marker],
    )
    assert _rules(findings) == {
        "openai-token",
        "operator-deny-string",
        "rfc1918-address",
    }


def test_scanner_accepts_placeholders_and_public_addresses(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "example.env.example"
    candidate.write_text(
        "WANDB_API_KEY=${WANDB_API_KEY}\n"
        "OPENAI_API_KEY=<replace-me>\n"
        "MASTER_ADDR=127.0.0.1\n",
        encoding="utf-8",
    )
    assert hygiene.scan_file(candidate, root=tmp_path) == []


def test_scanner_blocks_generated_or_private_artifact_names(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / ("model" + ".so")
    candidate.write_bytes(b"\x7fELF")
    findings = hygiene.scan_file(candidate, root=tmp_path)
    assert _rules(findings) == {"blocked-artifact-name"}


def test_current_repository_passes_public_hygiene_scan() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(PROJECT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "repository hygiene passed"
