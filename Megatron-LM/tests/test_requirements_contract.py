from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

# Direct imports supplied by the image/runtime rather than public PyPI.
IMAGE_OR_VENDOR_IMPORTS = {
    "acl",
    "megatron",
    "mindspeed",
    "mindspeed_llm",
    "torch",
    "torch_npu",
}
LOCAL_IMPORTS = {
    "llama_config",
    "modeling",
    "oe_eval",
    "olmo3_pipeline",
    "pretrain_olmo3_mindspeed",
    "runtime",
}
DIRECT_IMPORT_REQUIREMENTS = {
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "packaging": "packaging",
    "pytest": "pytest",
    "tomli": "tomli",
    "tomllib": "tomli",
    "transformers": "transformers",
}
PINNED_RUNTIME_REQUIREMENTS = {
    "datasets",
    "einops",
    "huggingface-hub",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "ninja",
    "numpy",
    "pip",
    "psutil",
    "pyarrow",
    "pytest",
    "scipy",
    "setuptools",
    "sympy",
    "tokenizers",
    "tomli",
    "transformers",
    "wandb",
    "wheel",
    "zstandard",
}


def _direct_imports() -> set[str]:
    result: set[str] = set()
    for directory in ("src", "scripts", "tests"):
        for path in (PROJECT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    result.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    result.add(node.module.split(".", 1)[0])
    return result


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", ".")):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", line)
        assert match, line
        names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_every_direct_external_import_has_an_owner() -> None:
    imports = _direct_imports()
    known = (
        set(sys.stdlib_module_names)
        | {"__future__"}
        | IMAGE_OR_VENDOR_IMPORTS
        | LOCAL_IMPORTS
        | set(DIRECT_IMPORT_REQUIREMENTS)
    )
    assert imports - known == set()

    declared = _requirement_names(PROJECT / "requirements.txt")
    assert set(DIRECT_IMPORT_REQUIREMENTS.values()) <= declared


def test_runtime_and_build_requirements_are_pinned_and_portable() -> None:
    requirements = PROJECT / "requirements.txt"
    text = requirements.read_text(encoding="utf-8")
    declared = _requirement_names(requirements)
    assert PINNED_RUNTIME_REQUIREMENTS <= declared
    assert _requirement_names(PROJECT / "requirements/dolma-bootstrap.txt") == {
        "dolma"
    }
    assert not re.search(r"(?m)^\s*-e(?:\s|$)", text)
    assert "@ file:" not in text
    assert "/data/" not in text
    assert "/" + "afs-" not in text

    resolved = PROJECT / "requirements/train.lock.txt"
    resolved_text = resolved.read_text(encoding="utf-8")
    resolved_names = _requirement_names(resolved)
    assert declared <= resolved_names
    assert "dolma" in resolved_names
    install_lines: list[str] = []
    for raw in resolved_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            install_lines.append(line)
            assert re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", line), line
    install_text = "\n".join(install_lines)
    assert "@ file:" not in install_text
    assert "/root/selfgz" not in install_text
    assert "/data/" not in install_text
    assert "/" + "afs-" not in install_text


def test_ascend_python_runtime_is_pinned_in_an_external_wheel_contract() -> None:
    requirements = PROJECT / "requirements/ascend-runtime.txt"
    assert _requirement_names(requirements) == {"torch", "torch-npu"}
    install_lines = [
        raw.split("#", 1)[0].strip()
        for raw in requirements.read_text(encoding="utf-8").splitlines()
        if raw.split("#", 1)[0].strip()
    ]
    assert install_lines == ["torch==2.6.0", "torch-npu==2.6.0"]


def test_environment_contract_uses_the_validated_exact_python_toolchain() -> None:
    text = (PROJECT / "environment.yml").read_text(encoding="utf-8")
    assert "python=3.10.20" in text
    assert re.search(r"(?m)^  - pip$", text)
    assert "ASCEND_RUNTIME_LOCK.json" in text
    assert "never inherits or clones" in text


def test_bootstrap_builds_cleanly_and_rejects_editable_installations() -> None:
    text = (PROJECT / "scripts" / "bootstrap_conda_env.sh").read_text(
        encoding="utf-8"
    )
    assert "--source-env" not in text
    assert '--clone "$source_env"' not in text
    assert "pip uninstall --yes" not in text
    assert '"$conda_exe" create' in text
    for requirement in (
        "python=3.10.20",
        "pip==26.0.1",
        "setuptools==80.9.0",
        "wheel==0.46.3",
    ):
        assert requirement in text
    assert "--ascend-wheelhouse" in text
    assert "requirements/ascend-runtime.txt" in text
    assert 'lock["python_packages"]["torch"]["version"]' in text
    assert 'lock["python_packages"]["torch_npu"]["version"]' in text
    assert "OLMO3_LOCKED_TORCH_VERSION" in text
    assert "OLMO3_LOCKED_TORCH_NPU_VERSION" in text
    assert "--no-index" in text
    assert '--find-links "$ascend_wheelhouse"' in text
    assert "--no-build-isolation" in text
    assert "--no-deps" in text
    assert 'get("editable", False)' in text
    assert 'glob("__editable__*")' in text
    assert 'glob("*.egg-link")' in text
    for command in ("olmo3ctl --help", "olmo3ckpt --help", "olmo3eval --help"):
        assert command in text
    assert 'OLMO3_ROOT="$project_root"' in text
    assert "verify_python_dependencies.py" in text


def test_bootstrap_clones_first_and_promotes_only_a_verified_staging_env() -> None:
    text = (PROJECT / "scripts" / "bootstrap_conda_env.sh").read_text(
        encoding="utf-8"
    )
    clone = text.index('"$project_root/scripts/bootstrap_runtime.py" --clone-missing')
    create = text.index('"$conda_exe" create')
    assert clone < create
    assert text.index("--host-only") < clone
    assert 'build_env="${target_env}.__olmo3_build_$$"' in text
    assert "cleanup_failed_build" in text
    assert 'env remove --yes --name "$candidate"' in text
    assert '"$conda_exe" rename --yes --name "$build_env" "$target_env"' in text
    assert text.index('"$conda_exe" rename') > text.index(
        "OLMO3_CLEAN_NON_EDITABLE_INSTALL_OK"
    )
