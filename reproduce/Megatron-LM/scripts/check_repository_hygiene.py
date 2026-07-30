#!/usr/bin/env python3
"""Fail closed when public Git candidates contain secrets or private metadata."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MAX_TEXT_BYTES = 10 * 1024 * 1024
RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        ".".join(("10", "0", "0", "0")) + "/8",
        ".".join(("172", "16", "0", "0")) + "/12",
        ".".join(("192", "168", "0", "0")) + "/16",
    )
)
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")

# These patterns intentionally require a provider-specific shape or an
# assignment with a non-placeholder value. Findings report only the location
# and rule name; the suspected value is never printed.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            r"-{5}BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-{5}"
        ),
    ),
    (
        "openai-token",
        re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "huggingface-token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "github-token",
        re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b"),
    ),
    (
        "aws-access-key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "google-api-key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    ),
    (
        "slack-token",
        re.compile(r"\bxox(?:a|b|p|r|s)-[A-Za-z0-9-]{20,}\b"),
    ),
    (
        "stripe-live-secret",
        re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "kubeconfig-inline-credential",
        re.compile(
            r"(?im)^\s*(?:client-key-data|client-certificate-data|"
            r"certificate-authority-data)\s*:\s*[A-Za-z0-9+/=]{32,}\s*$"
        ),
    ),
    (
        "credential-assignment",
        re.compile(
            r"""(?imx)
            ^\s*(?:export\s+)?
            (?:OPENAI_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|
               WANDB_API_KEY|GITHUB_TOKEN|GH_TOKEN|
               AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)
            \s*(?:=|:)\s*
            (?!["']?(?:|none|null|placeholder|changeme|replace[_-]?me|
                       example|dummy|redacted|<[^>]+>|\$\{?[\w]+\}?)["']?\s*$)
            ["']?[^\s"'#]{8,}["']?\s*$
            """
        ),
    ),
)

BLOCKED_BASENAMES = {
    ".env",
    ".netrc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "kubeconfig",
}
BLOCKED_SUFFIXES = {
    ".der",
    ".key",
    ".log",
    ".out",
    ".p12",
    ".pem",
    ".pfx",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".so",
}
IGNORED_GITLINK_PREFIXES = ("third_party/",)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str


def _git_candidates(root: Path) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    paths: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="strict")
        if relative.startswith(IGNORED_GITLINK_PREFIXES):
            continue
        path = root / relative
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return paths


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _looks_binary(payload: bytes) -> bool:
    return b"\0" in payload[:8192]


def _rfc1918_findings(text: str, relative: str) -> Iterable[Finding]:
    for match in IPV4_RE.finditer(text):
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if any(address in network for network in RFC1918_NETWORKS):
            yield Finding(
                relative,
                _line_number(text, match.start()),
                "rfc1918-address",
            )


def scan_file(
    path: Path,
    *,
    root: Path,
    deny_strings: Sequence[str] = (),
) -> list[Finding]:
    relative = path.relative_to(root).as_posix()
    findings: list[Finding] = []
    lower_name = path.name.lower()
    lower_suffix = path.suffix.lower()

    if (
        lower_name in BLOCKED_BASENAMES
        or lower_name.startswith("kubeconfig.")
        or lower_suffix in BLOCKED_SUFFIXES
    ):
        findings.append(Finding(relative, 0, "blocked-artifact-name"))

    try:
        if path.is_symlink():
            target = os.readlink(path)
            payload = target.encode("utf-8", errors="replace")
        else:
            if path.stat().st_size > MAX_TEXT_BYTES:
                return findings
            payload = path.read_bytes()
    except OSError:
        findings.append(Finding(relative, 0, "unreadable-candidate"))
        return findings

    if _looks_binary(payload):
        return findings
    text = payload.decode("utf-8", errors="replace")

    for rule, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                Finding(relative, _line_number(text, match.start()), rule)
            )
    findings.extend(_rfc1918_findings(text, relative))

    for marker in deny_strings:
        start = 0
        while marker and (offset := text.find(marker, start)) >= 0:
            findings.append(
                Finding(
                    relative,
                    _line_number(text, offset),
                    "operator-deny-string",
                )
            )
            start = offset + len(marker)

    return findings


def scan_repository(
    root: Path,
    *,
    deny_strings: Sequence[str] = (),
) -> list[Finding]:
    findings: list[Finding] = []
    for path in _git_candidates(root):
        findings.extend(
            scan_file(path, root=root, deny_strings=deny_strings)
        )
    return sorted(set(findings))


def _environment_deny_strings() -> list[str]:
    value = os.environ.get("OLMO3_HYGIENE_DENY_STRINGS", "")
    return [marker.strip() for marker in value.split(",") if marker.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tracked and unignored Git candidates without printing "
            "suspected secret values."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git worktree root (default: repository containing this script)",
    )
    parser.add_argument(
        "--deny-string",
        action="append",
        default=[],
        help=(
            "Reject an operator-specific private marker. Repeat as needed; "
            "the marker itself is never printed."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    deny_strings = [*args.deny_string, *_environment_deny_strings()]
    try:
        findings = scan_repository(root, deny_strings=deny_strings)
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        print(
            f"repository hygiene scan could not enumerate Git candidates: "
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    if findings:
        for finding in findings:
            location = (
                f"{finding.path}:{finding.line}"
                if finding.line
                else finding.path
            )
            print(f"{location}: {finding.rule}", file=sys.stderr)
        print(
            f"repository hygiene failed with {len(findings)} finding(s)",
            file=sys.stderr,
        )
        return 1

    print("repository hygiene passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
