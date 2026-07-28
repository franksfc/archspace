#!/usr/bin/env python3
"""Verify the exact Python lock and its one documented metadata override.

The production environment intentionally combines Transformers 4.46.1
(``tokenizers>=0.20,<0.21``) with Dolma 1.1.2, whose published metadata caps
tokenizers at 0.19.1 even though the exercised Dolma tokenization path works
with 0.20.3.  A raw ``pip check`` therefore cannot be the bootstrap gate.
This verifier checks every locked distribution and every active dependency
edge, while allowing only that exact, version-pinned incompatibility.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


class DependencyContractError(RuntimeError):
    """The installed Python environment contradicts the project lock."""


@dataclass(frozen=True)
class ApprovedOverride:
    owner: str
    owner_version: str
    dependency: str
    dependency_version: str
    required_specifier: str


APPROVED_OVERRIDES = frozenset(
    {
        ApprovedOverride(
            owner="dolma",
            owner_version="1.1.2",
            dependency="tokenizers",
            dependency_version="0.20.3",
            required_specifier="<=0.19.1,>=0.15.0",
        )
    }
)


def _load_exact_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise DependencyContractError(
                f"{path}:{line_number} is not an exact name==version lock"
            )
        name = canonicalize_name(match.group(1))
        version = match.group(2)
        previous = result.setdefault(name, version)
        if previous != version:
            raise DependencyContractError(
                f"conflicting locked versions for {name}: "
                f"{previous!r} != {version!r}"
            )
    return result


def _load_expected(project_root: Path) -> dict[str, str]:
    expected = _load_exact_requirements(
        project_root / "requirements" / "train.lock.txt"
    )
    ascend = _load_exact_requirements(
        project_root / "requirements" / "ascend-runtime.txt"
    )
    for name, version in ascend.items():
        previous = expected.setdefault(name, version)
        if previous != version:
            raise DependencyContractError(
                f"conflicting locked versions for {name}: "
                f"{previous!r} != {version!r}"
            )
    try:
        project_metadata = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        project_name = canonicalize_name(str(project_metadata["name"]))
        project_version = str(project_metadata["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DependencyContractError(
            f"invalid project metadata in {project_root / 'pyproject.toml'}"
        ) from exc
    expected[project_name] = project_version
    return expected


def _active_requirements(
    distribution: importlib.metadata.Distribution,
) -> Iterable[Requirement]:
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    for raw in distribution.requires or ():
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            raise DependencyContractError(
                f"{distribution.metadata['Name']} has invalid requirement "
                f"metadata: {raw!r}"
            ) from exc
        if requirement.marker is not None and not requirement.marker.evaluate(
            marker_environment
        ):
            continue
        yield requirement


def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyContractError(
            f"required distribution is missing: {name}"
        ) from exc


def verify(project_root: Path = PROJECT_ROOT) -> tuple[int, int]:
    expected = _load_expected(project_root)
    for name, version in sorted(expected.items()):
        actual = _installed_version(name)
        if actual != version:
            raise DependencyContractError(
                f"locked version mismatch for {name}: "
                f"expected={version} actual={actual}"
            )

    observed_overrides: set[ApprovedOverride] = set()
    checked_edges = 0
    for owner, owner_version in sorted(expected.items()):
        distribution = importlib.metadata.distribution(owner)
        for requirement in _active_requirements(distribution):
            dependency = canonicalize_name(requirement.name)
            dependency_version = _installed_version(dependency)
            checked_edges += 1
            try:
                compatible = requirement.specifier.contains(
                    Version(dependency_version),
                    prereleases=True,
                )
            except InvalidVersion as exc:
                raise DependencyContractError(
                    f"invalid installed version for {dependency}: "
                    f"{dependency_version!r}"
                ) from exc
            if compatible:
                continue
            candidate = ApprovedOverride(
                owner=owner,
                owner_version=owner_version,
                dependency=dependency,
                dependency_version=dependency_version,
                required_specifier=str(requirement.specifier),
            )
            if candidate not in APPROVED_OVERRIDES:
                raise DependencyContractError(
                    f"dependency mismatch: {owner}=={owner_version} requires "
                    f"{dependency}{requirement.specifier}, installed "
                    f"{dependency_version}"
                )
            observed_overrides.add(candidate)

    missing_overrides = APPROVED_OVERRIDES - observed_overrides
    if missing_overrides:
        raise DependencyContractError(
            "approved metadata override was not observed exactly; update or "
            f"remove the stale contract: {sorted(map(str, missing_overrides))}"
        )
    return len(expected), checked_edges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        packages, edges = verify(args.project_root.expanduser().resolve())
    except (DependencyContractError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        "OLMO3_PYTHON_DEPENDENCY_CONTRACT_OK "
        f"packages={packages} dependency_edges={edges} "
        f"approved_metadata_overrides={len(APPROVED_OVERRIDES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
