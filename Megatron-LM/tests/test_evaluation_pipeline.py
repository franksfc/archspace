from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from olmo3_pipeline.checkpoint import build_contract, write_contract
from olmo3_pipeline.config import ConfigurationError, PROJECT_ROOT, compose
from olmo3_pipeline.evaluation import (
    _load_source_config,
    _manifest_paths,
    _validate_parallel,
    build_parser,
)
from olmo3_pipeline.resolve import resolve


PINNED_OLMES_COMMIT = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"


def _resolved_config(
    tmp_path: Path,
    *,
    model: str = "1b",
    variant: str = "base",
) -> dict:
    config = compose(
        model=model,
        variant=variant,
        stage="stage1",
        data="dolma3_6t",
        topology="stage1_tp1_4096npu",
        performance=["ordinary_hsdp"],
    )
    config["data"]["root"] = str(tmp_path / "training-data-must-not-leak")
    config["data"]["tokenizer"]["path"] = str(tmp_path / "unused-tokenizer")
    config["data"]["data_args_path"] = str(tmp_path / "data-args.json")
    config["training"]["train_tokens"] = 6_000_000_000_000
    config["optimization"]["peak_lr"] = 1.0e-3
    config["optimization"]["min_lr"] = 1.0e-4
    config["optimization"]["warmup_tokens"] = 8_388_608_000
    config["runtime"] = {
        "run_id": f"evaluation-{model}-{variant}",
        "lifecycle": "fresh",
        "load": None,
        "save": str(tmp_path / "training-save"),
        "output": str(tmp_path / "training-output"),
        "python": sys.executable,
    }
    return resolve(config)


def _write_dcp_metadata(iteration_dir: Path) -> None:
    metadata = SimpleNamespace(
        storage_data={
            "model.weight": SimpleNamespace(
                relative_path="__0_0.distcp",
                offset=2,
                length=5,
            )
        }
    )
    with (iteration_dir / ".metadata").open("wb") as stream:
        pickle.dump(metadata, stream)


def _complete_checkpoint(
    tmp_path: Path,
    config: dict,
    *,
    iteration: int = 37,
) -> Path:
    root = tmp_path / "checkpoint"
    iteration_dir = root / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        '{"sharded_backend":"torch_dist","sharded_backend_version":1}\n',
        encoding="utf-8",
    )
    (iteration_dir / "common.pt").write_bytes(b"common-state")
    (iteration_dir / "__0_0.distcp").write_bytes(b"0123456789")
    _write_dcp_metadata(iteration_dir)
    (root / "latest_checkpointed_iteration.txt").write_text(
        f"{iteration}\n",
        encoding="utf-8",
    )
    write_contract(root, build_contract(config))
    return root


def _write_tokenizer(tmp_path: Path) -> Path:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    return tokenizer


def _write_source_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "resolved.json"
    path.write_text(
        json.dumps(config, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _plan_args(
    *,
    action: str,
    source: Path,
    checkpoint: Path,
    tokenizer: Path,
    plan_dir: Path,
    tail: list[str],
) -> SimpleNamespace:
    parser = build_parser()
    arguments = [
        action,
        "--resolved",
        str(source),
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer),
        "--plan-dir",
        str(plan_dir),
        "--python",
        sys.executable,
        "--nnodes",
        "1",
        "--nproc-per-node",
        "2",
        "--tensor-parallel",
        "2",
        "--master-port",
        "29871",
        *tail,
    ]
    return parser.parse_args(arguments)


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_frozen_plan_files(plan_dir: Path) -> None:
    expected = {
        "plan.json",
        "source_resolved.json",
        "model_config.json",
        "command.json",
        "command.txt",
        "environment.json",
        "environment.sh",
        "launch.sh",
    }
    assert {path.name for path in plan_dir.iterdir()} == expected
    assert os.access(plan_dir / "launch.sh", os.X_OK)
    assert os.access(plan_dir / "environment.sh", os.X_OK)
    subprocess.run(
        ["bash", "-n", str(plan_dir / "launch.sh")],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("model", ("1b", "3b", "7b"))
@pytest.mark.parametrize("tensor_parallel", (1, 2, 4, 8, 16))
def test_native_tp_contract_accepts_all_supported_model_sizes(
    tmp_path: Path,
    model: str,
    tensor_parallel: int,
) -> None:
    config = _resolved_config(tmp_path, model=model)
    topology = _validate_parallel(
        config,
        nnodes=1,
        nproc_per_node=16,
        tensor_parallel=tensor_parallel,
    )
    assert topology == {
        "world_size": 16,
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": 1,
        "context_parallel": 1,
        "data_parallel": 16 // tensor_parallel,
        "nnodes": 1,
        "nproc_per_node": 16,
    }


def test_native_tp_contract_rejects_world_and_model_divisibility(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path, model="1b")
    with pytest.raises(ConfigurationError, match="must divide world_size=6"):
        _validate_parallel(
            config,
            nnodes=1,
            nproc_per_node=6,
            tensor_parallel=4,
        )
    with pytest.raises(
        ConfigurationError,
        match=r"num_attention_heads=16 is not divisible by TP=32",
    ):
        _validate_parallel(
            config,
            nnodes=2,
            nproc_per_node=16,
            tensor_parallel=32,
        )
    with pytest.raises(ConfigurationError, match="nnodes must be a positive"):
        _validate_parallel(
            config,
            nnodes=True,
            nproc_per_node=16,
            tensor_parallel=1,
        )


def test_evaluation_rejects_numerically_tampered_resolved_config(
    tmp_path: Path,
) -> None:
    config = _resolved_config(tmp_path)
    config["model"]["rope_theta"] = 10_000
    source = _write_source_config(tmp_path, config)

    with pytest.raises(ConfigurationError, match="official OLMo 3"):
        _load_source_config(source)


def test_inference_plan_is_complete_relocatable_and_immutable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _resolved_config(tmp_path, variant="siamese_depth")
    source = _write_source_config(tmp_path, config)
    checkpoint = _complete_checkpoint(tmp_path, config)
    tokenizer = _write_tokenizer(tmp_path)
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        '{"request_id":"r0","request_type":"generate_until",'
        '"request":{"context":"test"}}\n',
        encoding="utf-8",
    )
    plan_dir = tmp_path / "plans" / "inference"
    output = tmp_path / "results" / "responses"
    args = _plan_args(
        action="inference-plan",
        source=source,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        plan_dir=plan_dir,
        tail=[
            "--output",
            str(output),
            "--mode",
            "run",
            "--frozen-requests",
            str(requests),
            "--run-cache-smoke-first",
        ],
    )

    assert args.handler(args) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["iteration"] == 37
    assert rendered["variant"] == "siamese_depth"
    _assert_frozen_plan_files(plan_dir)

    plan = _read_json(plan_dir / "plan.json")
    command = _read_json(plan_dir / "command.json")
    environment = _read_json(plan_dir / "environment.json")
    assert plan["schema"] == "olmo3.mindspeed.evaluation-plan/v1"
    assert plan["kind"] == "native-inference"
    assert plan["checkpoint_load"] == {
        "format": "torch_dist",
        "optimizer": False,
        "rng": False,
        "strict": "raise_all",
    }
    assert plan["checkpoint"]["iteration"] == 37
    assert plan["model"]["variant"] == "siamese_depth"
    assert plan["frozen_requests"]["path"] == str(requests.resolve())
    assert plan["frozen_requests"]["sha256"]
    assert plan["topology"]["tensor_parallel"] == 2
    assert "--no-load-optim" in command
    assert "--no-load-rng" in command
    assert "--dist-ckpt-strictness" in command
    assert "raise_all" in command
    assert "--run-cache-smoke-first" in command
    assert "__OLMO3_NODE_RANK__" in command
    assert str(config["data"]["root"]) not in command
    assert environment["WANDB_MODE"] == "disabled"
    assert "${worker_node_rank}" in (plan_dir / "launch.sh").read_text(
        encoding="utf-8"
    )
    assert not output.exists()

    with pytest.raises(
        ConfigurationError,
        match="immutable evaluation plan already exists",
    ):
        args.handler(args)
    assert _read_json(plan_dir / "plan.json") == plan


def test_ppl_plan_is_document_only_and_manifest_confined(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _resolved_config(tmp_path)
    source = _write_source_config(tmp_path, config)
    checkpoint = _complete_checkpoint(tmp_path, config)
    tokenizer = _write_tokenizer(tmp_path)
    validation_root = tmp_path / "validation"
    validation_root.mkdir()
    tokens = validation_root / "valid.tokens"
    tokens.write_bytes(b"\x01\x02\x03\x04")
    manifest = validation_root / "ppl.csv"
    manifest.write_text("validation-fixture,valid.tokens\n", encoding="utf-8")
    plan_dir = tmp_path / "plans" / "ppl"
    output = tmp_path / "results" / "ppl.json"
    args = _plan_args(
        action="ppl-plan",
        source=source,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        plan_dir=plan_dir,
        tail=[
            "--manifest",
            str(manifest),
            "--cache-dir",
            str(tmp_path / "ppl-cache"),
            "--output",
            str(output),
            "--sequence-length",
            "2048",
            "--expected-documents",
            "11",
            "--expected-valid-targets",
            "15000",
            "--expected-truncated-tokens",
            "0",
        ],
    )

    assert args.handler(args) == 0
    capsys.readouterr()
    _assert_frozen_plan_files(plan_dir)
    plan = _read_json(plan_dir / "plan.json")
    command = _read_json(plan_dir / "command.json")
    assert plan["kind"] == "document-ppl"
    assert plan["model"]["eval_sequence_length"] == 2048
    assert plan["manifest"]["path"] == str(manifest.resolve())
    assert plan["manifest"]["sha256"]
    assert plan["manifest"]["token_streams"] == [
        {"path": str(tokens.resolve()), "size": 4}
    ]
    assert "--skip-train" in command
    assert "--calculate-per-token-loss" in command
    assert "--valid-ppl-manifest" in command
    assert "--valid-ppl-expected-documents" in command
    assert "--valid-ppl-expected-valid-targets" in command
    assert "--valid-ppl-expected-truncated-tokens" in command
    assert str(config["data"]["root"]) not in command
    assert not output.exists()

    escape = tmp_path / "escape.tokens"
    escape.write_bytes(b"x")
    escaped_manifest = validation_root / "escaped.csv"
    escaped_manifest.write_text("escaped,../escape.tokens\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="relative and confined"):
        _manifest_paths(escaped_manifest)


def _load_olmes_control_module() -> ModuleType:
    module_path = PROJECT_ROOT / "scripts" / "eval" / "olmes_freeze_score.py"
    spec = importlib.util.spec_from_file_location(
        "olmo3_test_olmes_freeze_score",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_olmes_defaults_and_upstream_lock_are_project_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_olmes_control_module()
    expected_root = PROJECT_ROOT / "third_party" / "olmes"
    expected_stubs = PROJECT_ROOT / "scripts" / "eval" / "stubs"
    assert module.REPO_ROOT == PROJECT_ROOT
    assert module.DEFAULT_OLMES_ROOT == expected_root
    assert module.DEFAULT_OLMES_STUBS == expected_stubs
    assert module.PINNED_OLMES_COMMIT == PINNED_OLMES_COMMIT

    parsed = module._build_parser().parse_args(
        ["freeze", "--output-dir", "/tmp/olmes-test-output"]
    )
    assert parsed.olmes_root == expected_root
    assert parsed.olmes_stubs == expected_stubs

    lock = _read_json(PROJECT_ROOT / "UPSTREAM_LOCK.json")
    locked = lock["repositories"]["OLMES"]
    assert locked["url"] == "https://github.com/allenai/olmes.git"
    assert locked["path"] == "third_party/olmes"
    assert locked["commit"] == PINNED_OLMES_COMMIT
    assert locked["ref"] == PINNED_OLMES_COMMIT
    assert locked["must_be_pristine"] is True
    assert "evaluation" in locked["role"].lower()

    monkeypatch.setattr(
        module,
        "_git_info",
        lambda path: {
            "root": str(path),
            "commit": "0" * 40,
            "dirty": False,
        },
    )
    with pytest.raises(module.ContractError, match="repository lock"):
        module._require_pinned_olmes(expected_root)


def test_olmes_online_mode_clears_inherited_offline_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_olmes_control_module()
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        monkeypatch.setenv(name, "1")
    args = SimpleNamespace(
        hf_home=tmp_path / "hf",
        nltk_data=tmp_path / "nltk",
        offline=False,
    )
    module._configure_huggingface(args)
    assert os.environ["HF_HOME"] == str((tmp_path / "hf").resolve())
    assert os.environ["NLTK_DATA"] == str((tmp_path / "nltk").resolve())
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        assert name not in os.environ


def test_checked_out_olmes_is_clean_and_exactly_pinned() -> None:
    root = PROJECT_ROOT / "third_party" / "olmes"
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert head == PINNED_OLMES_COMMIT
    assert dirty == ""


def test_olmes_combined_request_freezes_both_contexts() -> None:
    module = _load_olmes_control_module()
    messages = {
        "messages": [{"role": "user", "content": "Question"}],
        "assistant_prefix": "",
    }
    request = SimpleNamespace(
        context=messages,
        perplexity_context=messages,
        continuation=" answer",
        stop_sequences=[],
        generation_kwargs={
            "max_gen_toks": 4,
            "do_sample": False,
        },
    )
    instance = SimpleNamespace(
        request=request,
        request_type="generate_until_and_loglikelihood",
        task_name="combined",
        doc_id=0,
        idx=0,
        native_id="n0",
        native_id_description=None,
        label=None,
        doc={"question": "Question"},
    )
    record = module._freeze_instance(
        instance,
        suite_alias="combined-suite",
        task_alias="combined-task",
        task_index=0,
        request_index=0,
        request_index_in_task=0,
        task_hash="task-hash",
        task_metadata={},
        olmes_commit=PINNED_OLMES_COMMIT,
    )
    module._validate_frozen_record(record)
    assert isinstance(record["request"]["context"], str)
    assert isinstance(record["request"]["perplexity_context"], str)
    assert record["request"]["continuation"] == "answer"


def test_native_inference_runner_self_test_is_dependency_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "inference" / "olmo3_native.py"),
            "--self-test",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.stdout.strip() == "FROZEN_EVAL_SELF_TEST_OK"
    assert result.stderr == ""


def test_olmes_environment_is_fresh_and_isolated_from_ascend() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "eval" / "bootstrap_eval_env.sh"
    ).read_text(encoding="utf-8")
    environment = (
        PROJECT_ROOT / "environment-eval.yml"
    ).read_text(encoding="utf-8")
    constraints = (
        PROJECT_ROOT / "requirements" / "eval-constraints.txt"
    ).read_text(encoding="utf-8")

    assert '"$conda_exe" create' in script
    assert "--clone " not in script
    assert "https://download.pytorch.org/whl/cpu" in script
    assert '"$project_root/third_party/olmes"' in script
    assert "python=3.10.20" in script
    assert "torch==2.8.0" in constraints
    assert "transformers==4.57.6" in constraints
    assert "torch_npu=" not in environment
    assert "Ascend training environment" in environment
