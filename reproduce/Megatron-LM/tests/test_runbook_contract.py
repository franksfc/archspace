from __future__ import annotations

import re
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_DOCUMENTS = (
    PROJECT / "README.md",
    PROJECT / "docs" / "RUNBOOK.md",
    PROJECT / "docs" / "INFERENCE_EVALUATION.md",
)


def _bash_blocks(path: Path) -> list[str]:
    return re.findall(
        r"```bash\n(.*?)```",
        path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )


def test_documented_bash_blocks_are_shell_syntax_valid() -> None:
    checked = 0
    for path in PRIMARY_DOCUMENTS:
        for index, block in enumerate(_bash_blocks(path)):
            result = subprocess.run(
                ["bash", "-n"],
                input=block,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, (
                f"{path.relative_to(PROJECT)} bash block {index}: "
                f"{result.stderr}"
            )
            checked += 1
    assert checked >= 50


def test_runbook_covers_the_complete_production_chain() -> None:
    text = (PROJECT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    required_fragments = (
        "scripts/bootstrap_conda_env.sh",
        "dolma3_6t",
        "dolmino_100b",
        "longmino_50b",
        "dolci_think",
        "dolci_instruct",
        "finalize-stage1-index",
        "runtime-manifest",
        "prepare-cache",
        "run-sft",
        "--lifecycle fresh",
        "--lifecycle transition",
        "--lifecycle resume",
        "transition_checkpoint",
        "olmo3_native.py",
        "olmo3_ppl.py",
        "olmes_freeze_score.py",
    )
    for fragment in required_fragments:
        assert fragment in text
    assert "data_args.json" not in text


def test_primary_pipeline_text_has_no_user_directed_voice() -> None:
    for path in PRIMARY_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for phrase in ("请你", "你需要", "我让", "用户要求", "as you requested"):
            assert phrase not in text, f"{path.relative_to(PROJECT)}: {phrase}"


def test_production_input_template_exposes_all_unresolved_decisions() -> None:
    text = (
        PROJECT / "configs" / "pipeline" / "production.env.example"
    ).read_text(encoding="utf-8")
    required_keys = (
        "OLMO3_ASCEND_WHEELHOUSE",
        "RAW_ROOT",
        "TOKENIZED_ROOT",
        "DATA_WORK_ROOT",
        "CHECKPOINT_ROOT",
        "MODEL",
        "VARIANT",
        "STAGE1_PEAK_LR",
        "STAGE1_MIN_LR",
        "STAGE1_WARMUP_TOKENS",
        "STAGE2_PEAK_LR",
        "STAGE3_PEAK_LR",
        "THINK_PEAK_LR",
        "INSTRUCT_PEAK_LR",
        "STAGE1_RUN_ID",
        "STAGE2_RUN_ID",
        "STAGE3_RUN_ID",
        "THINK_RUN_ID",
        "INSTRUCT_RUN_ID",
    )
    for key in required_keys:
        assert re.search(rf"(?m)^{key}=", text), key


def test_conda_documentation_describes_empty_prefix_creation() -> None:
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    runbook = (PROJECT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    bootstrap = (
        PROJECT / "scripts" / "bootstrap_conda_env.sh"
    ).read_text(encoding="utf-8")

    assert "Conda 环境从空前缀创建" in readme
    assert "conda create" in runbook
    assert "--source-env" not in bootstrap
    assert re.search(r'"\$conda_exe" create\b', bootstrap)
    assert re.search(r'"\$conda_exe" rename\b', bootstrap)
