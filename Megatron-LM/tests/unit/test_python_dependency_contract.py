from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "verify_python_dependencies.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_python_dependencies",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_current_environment_satisfies_locked_dependency_contract() -> None:
    packages, edges = module.verify(PROJECT_ROOT)

    assert packages >= 100
    assert edges >= 100


def test_exact_lock_parser_rejects_ranges(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.txt"
    lock.write_text("numpy>=1.24\n", encoding="utf-8")

    with pytest.raises(module.DependencyContractError, match="not an exact"):
        module._load_exact_requirements(lock)


def test_expected_contract_uses_source_project_version() -> None:
    expected = module._load_expected(PROJECT_ROOT)

    assert expected["olmo3-mindspeed-pipeline"] == "0.1.0"


def test_only_documented_dolma_override_is_approved() -> None:
    assert module.APPROVED_OVERRIDES == {
        module.ApprovedOverride(
            owner="dolma",
            owner_version="1.1.2",
            dependency="tokenizers",
            dependency_version="0.20.3",
            required_specifier="<=0.19.1,>=0.15.0",
        )
    }
