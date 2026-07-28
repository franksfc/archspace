from __future__ import annotations

import subprocess
from pathlib import Path


def test_path_free_training_shell_entrypoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", str(project_root / "tests/shell/test_train_entrypoint.sh")],
        cwd=project_root,
        check=True,
    )
