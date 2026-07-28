from __future__ import annotations

import ast
import json
import os
import sys
import types
from functools import cache
from pathlib import Path

import pytest

from olmo3_pipeline import cli
from olmo3_pipeline.command import (
    build_environment,
    build_megatron_args,
    build_torchrun_command,
)
from olmo3_pipeline.config import PROJECT_ROOT, ConfigurationError, compose
from olmo3_pipeline.resolve import resolve
from olmo3_pipeline.topology import validate_local_sio_pairs


STAGES = {
    "stage1": ("dolma3_6t", "stage1_tp1_4096npu", ("ordinary_hsdp",)),
    "stage2": ("dolmino_100b", "stage2_cp2_512npu", ("cp_single_halo",)),
    "stage3": ("longmino_50b", "stage3_cp16_1024npu", ("cp_single_halo",)),
    "sft_think": ("dolci_think", "sft_cp8_256npu", ("cp_single_halo",)),
    "sft_instruct": (
        "dolci_instruct",
        "sft_cp8_256npu",
        ("cp_single_halo",),
    ),
}


def _resolved(model: str, variant: str, stage: str) -> dict:
    data, topology, performance = STAGES[stage]
    config = compose(
        model=model,
        variant=variant,
        stage=stage,
        data=data,
        topology=topology,
        performance=performance,
    )
    config["data"]["root"] = "/runtime/data"
    config["data"]["tokenizer"]["path"] = "/runtime/tokenizer"
    config["optimization"]["peak_lr"] = 2.5e-4
    if stage == "stage1":
        config["data"]["data_args_path"] = "/runtime/data-args.txt"
        config["training"]["train_tokens"] = 6_000_000_000_000
        config["optimization"]["min_lr"] = 2.5e-5
        config["optimization"]["warmup_tokens"] = 8_388_608_000
        lifecycle = "fresh"
    elif stage in {"stage2", "stage3"}:
        config["data"]["manifest"] = "/runtime/data-manifest.json"
        config["data"]["work_dir"] = "/runtime/data-work"
        if stage == "stage3":
            config["optimization"]["warmup_tokens"] = 838_860_800
        lifecycle = "transition"
    else:
        config["data"]["work_dir"] = "/runtime/sft-work"
        config["data"]["expected_instances"] = 1024
        config["data"]["expected_fingerprint"] = "f" * 64
        lifecycle = "transition"
    config["runtime"] = {
        "run_id": f"integration-{model}-{variant}-{stage}",
        "lifecycle": lifecycle,
        "load": "/runtime/source-checkpoint" if lifecycle != "fresh" else None,
        "save": "/runtime/target-checkpoint",
        "output": "/runtime/output",
        "python": "/runtime/python3",
        "master_addr": "rendezvous",
        "master_port": 29500,
    }
    return resolve(config)


@cache
def _effective_argument_flags() -> frozenset[str]:
    """Statically inspect the exact pinned argument providers used at runtime."""

    sources = [
        PROJECT_ROOT / "megatron" / "training" / "arguments.py",
        PROJECT_ROOT
        / "third_party"
        / "MindSpeed"
        / "mindspeed"
        / "arguments.py",
        PROJECT_ROOT / "src" / "pretrain_olmo3_mindspeed.py",
    ]
    sources.extend(
        (
            PROJECT_ROOT
            / "third_party"
            / "MindSpeed"
            / "mindspeed"
            / "features_manager"
        ).rglob("*.py")
    )
    sources.extend(
        (
            PROJECT_ROOT
            / "third_party"
            / "MindSpeed-LLM"
            / "mindspeed_llm"
            / "features_manager"
        ).rglob("*.py")
    )
    flags: set[str] = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            literals = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ]
            flags.update(literals)
            boolean_optional = any(
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "BooleanOptionalAction"
                for keyword in node.keywords
            )
            if boolean_optional:
                flags.update(f"--no-{flag[2:]}" for flag in literals)
    return frozenset(flags)


@pytest.mark.parametrize("model", ("1b", "3b", "7b"))
@pytest.mark.parametrize("variant", ("base", "siamese_depth"))
@pytest.mark.parametrize("stage", tuple(STAGES))
def test_every_rendered_megatron_flag_exists_in_pinned_runtime(
    model: str, variant: str, stage: str
) -> None:
    config = _resolved(model, variant, stage)
    config["runtime"]["wandb"] = {"enabled": True, "project": "cpu-audit"}
    emitted = build_megatron_args(config, Path("/runtime/model-config.json"))
    emitted_flags = [part for part in emitted if part.startswith("--")]
    missing = sorted(set(emitted_flags) - _effective_argument_flags())
    assert missing == []
    assert len(emitted_flags) == len(set(emitted_flags))


def test_exit_interval_stops_early_without_shortening_training_schedule() -> None:
    config = _resolved("1b", "siamese_depth", "stage1")
    original_train_iters = config["training"]["train_iters"]
    config["training"]["exit_interval"] = 1
    args = build_megatron_args(config, Path("/runtime/model-config.json"))
    assert args[args.index("--exit-interval") + 1] == "1"
    assert args[args.index("--train-iters") + 1] == str(original_train_iters)
    assert args[args.index("--lr-decay-iters") + 1] == str(original_train_iters)


def test_dynamic_stage2_and_stage3_manifest_arguments_are_not_profile_sized() -> None:
    for stage, prefix in (
        ("stage2", "--olmo3-fsl"),
        ("stage3", "--olmo3-packed"),
    ):
        config = _resolved("1b", "siamese_depth", stage)
        config["data"]["root"] = "/mounted/arbitrary-layout"
        config["data"]["manifest"] = "/contracts/runtime-v1.json"
        config["data"]["work_dir"] = "/cache/runtime-v1"
        args = build_megatron_args(config, Path("/contracts/model.json"))
        assert args[args.index(f"{prefix}-data-root") + 1] == config["data"]["root"]
        assert (
            args[args.index(f"{prefix}-data-manifest") + 1]
            == config["data"]["manifest"]
        )
        assert args[args.index(f"{prefix}-work-dir") + 1] == config["data"]["work_dir"]
        assert (
            int(args[args.index("--olmo3-expected-data-tokens") + 1])
            == config["data"]["known_token_count"]
        )
        assert args[args.index("--olmo3-data-config-name") + 1] == config["data"]["name"]
        config_digest = args[args.index("--olmo3-data-config-sha256") + 1]
        assert len(config_digest) == 64
        assert set(config_digest) <= set("0123456789abcdef")


def test_sft_training_uses_the_cache_parent_created_by_preparation() -> None:
    config = _resolved("1b", "siamese_depth", "sft_think")
    config["data"]["work_dir"] = "/cache/dolci-think-sft"
    args = build_megatron_args(config, Path("/contracts/model.json"))
    assert args[args.index("--olmo3-sft-work-dir") + 1] == (
        "/cache/dolci-think-sft/packed"
    )


def test_stage1_data_cache_path_is_shareable_with_run_local_fallback() -> None:
    config = _resolved("1b", "base", "stage1")
    args = build_megatron_args(config, Path("/contracts/model.json"))
    assert args[args.index("--data-cache-path") + 1] == "/runtime/output/data-cache"

    config["runtime"]["data_cache_path"] = "/cache/shared-stage1-indices"
    config = resolve(config)
    args = build_megatron_args(config, Path("/contracts/model.json"))
    assert args[args.index("--data-cache-path") + 1] == (
        "/cache/shared-stage1-indices"
    )


def test_cli_maps_explicit_data_cache_path_into_the_resolved_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        cli,
        "_print_summary",
        lambda config, run_dir=None: captured.append(config),
    )
    assert (
        cli.main(
            [
                "validate",
                "--model",
                "1b",
                "--variant",
                "base",
                "--stage",
                "stage1",
                "--data",
                "dolma3_6t",
                "--topology",
                "stage1_tp1_4096npu",
                "--performance",
                "ordinary_hsdp",
                "--run-id",
                "shared-data-cache-cli",
                "--lifecycle",
                "fresh",
                "--data-root",
                "/runtime/data",
                "--data-args-path",
                "/runtime/data-args",
                "--data-cache-path",
                "/cache/shared-stage1-indices",
                "--tokenizer",
                "/runtime/tokenizer",
                "--save",
                "/runtime/save",
                "--output",
                "/runtime/output",
                "--train-tokens",
                "6000000000000",
                "--peak-lr",
                "0.001",
                "--min-lr",
                "0.0001",
                "--warmup-tokens",
                "8388608000",
            ]
        )
        == 0
    )
    assert captured[0]["runtime"]["data_cache_path"] == (
        "/cache/shared-stage1-indices"
    )


def test_combined_tp2_cp2_ga_and_hsdp_math_matches_megatron_domains() -> None:
    config = compose(
        model="3b",
        variant="siamese_depth",
        stage="stage2",
        data="dolmino_100b",
        topology="tp2_sio_2048npu",
        performance=("tp2_sio_mc2", "cp_single_halo"),
        overrides=(
            "topology.world_size=1024",
            "topology.context_parallel=2",
            "topology.global_batch_size=512",
            "training.global_batch_size=512",
            "topology.hsdp.shard_size=8",
            "runtime.run_id=tp2-cp2-audit",
            "runtime.lifecycle=transition",
            "runtime.load=/runtime/stage1",
            "runtime.save=/runtime/stage2",
            "runtime.output=/runtime/output",
            "runtime.master_addr=rendezvous",
            "data.root=/runtime/data",
            "data.manifest=/runtime/manifest.json",
            "data.work_dir=/runtime/work",
            "data.tokenizer.path=/runtime/tokenizer",
            "optimization.peak_lr=0.00025",
        ),
    )
    config = resolve(config)
    args = build_megatron_args(config, Path("/runtime/model.json"))
    assert config["topology"]["data_parallel"] == 256
    assert config["training"]["gradient_accumulation"] == 2
    assert config["derived"]["local_sequence_length"] == 4096
    assert config["topology"]["hsdp"]["dp_cp_size"] == 512
    assert config["topology"]["hsdp"]["num_instances"] == 64
    assert args[args.index("--num-distributed-optimizer-instances") + 1] == "64"
    assert "--sequence-parallel" in args
    environment = build_environment(config)
    assert environment["OLMO3_MC2_ALL_GATHER_RECOMPUTATION"] == "1"
    assert environment["OLMO3_SWA_CP_MODE"] == "single_halo"


def test_runtime_node_count_and_rank_are_part_of_fail_closed_topology() -> None:
    config = _resolved("1b", "base", "stage1")
    assert config["runtime"]["nnodes"] == 256
    assert config["derived"]["nnodes"] == 256
    command = build_torchrun_command(config, Path("/runtime/model.json"))
    assert command[command.index("--nnodes") + 1] == "256"

    broken = compose(
        model="1b",
        variant="base",
        stage="stage1",
        data="dolma3_6t",
        topology="stage1_tp1_4096npu",
        performance=("ordinary_hsdp",),
        overrides=(
            "runtime.run_id=bad-nnodes",
            "runtime.lifecycle=fresh",
            "runtime.save=/runtime/save",
            "runtime.output=/runtime/output",
            "runtime.nnodes=255",
            "runtime.node_rank=255",
            "data.root=/runtime/data",
            "data.data_args_path=/runtime/data-args",
            "data.tokenizer.path=/runtime/tokenizer",
            "training.train_tokens=6000000000000",
            "optimization.peak_lr=0.001",
            "optimization.min_lr=0.0001",
            "optimization.warmup_tokens=8388608000",
        ),
    )
    with pytest.raises(ConfigurationError, match="does not equal world_size"):
        resolve(broken)


def test_hccl_if_base_port_is_optional_but_frozen_when_configured() -> None:
    config = _resolved("1b", "base", "stage1")
    assert "HCCL_IF_BASE_PORT" not in build_environment(config)

    config["runtime"]["hccl_if_base_port"] = 41_000
    config = resolve(config)
    assert build_environment(config)["HCCL_IF_BASE_PORT"] == "41000"

    for invalid in (True, "41000", 1_023, 65_521, None):
        broken = json.loads(json.dumps(config))
        broken["runtime"]["hccl_if_base_port"] = invalid
        with pytest.raises(
            ConfigurationError,
            match=r"runtime\.hccl_if_base_port must be an integer in "
            r"\[1024, 65520\]",
        ):
            resolve(broken)


def test_optional_data_cache_path_must_be_a_nonempty_string() -> None:
    config = _resolved("1b", "base", "stage1")
    for invalid in ("", 1, True):
        broken = json.loads(json.dumps(config))
        broken["runtime"]["data_cache_path"] = invalid
        with pytest.raises(
            ConfigurationError,
            match=r"runtime\.data_cache_path must be a non-empty string",
        ):
            resolve(broken)


def test_cli_sft_token_warmup_explicitly_replaces_fraction_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokens_per_step = 32 * 32768
    rc = cli.main(
        [
            "validate",
            "--model",
            "1b",
            "--variant",
            "siamese_depth",
            "--stage",
            "sft_think",
            "--data",
            "dolci_think",
            "--topology",
            "sft_cp8_256npu",
            "--performance",
            "cp_single_halo",
            "--run-id",
            "sft-token-warmup",
            "--load",
            "/runtime/stage3",
            "--save",
            "/runtime/sft",
            "--output",
            "/runtime/output",
            "--data-root",
            "/runtime/data",
            "--data-work-dir",
            "/runtime/work",
            "--tokenizer",
            "/runtime/tokenizer",
            "--expected-instances",
            "1024",
            "--expected-fingerprint",
            "f" * 64,
            "--peak-lr",
            "0.0001",
            "--min-lr",
            "0",
            "--warmup-tokens",
            str(tokens_per_step * 3),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert summary["warmup_steps"] == 3


def test_cli_launch_activates_checkpoint_once_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The control-plane launch path must never delegate sidecar writes to ranks."""

    config = _resolved("1b", "base", "stage1")
    config["topology"]["world_size"] = 16
    config["topology"]["hsdp"]["shard_size"] = 8
    config["runtime"]["nnodes"] = 1
    config = resolve(config)
    events: list[object] = []
    command = [
        "/runtime/python3",
        "-m",
        "torch.distributed.run",
        "--node_rank",
        "0",
    ]
    environment = {"PIPELINE_AUDIT": "1"}
    run_dir = tmp_path / "runs" / "launch-order"

    monkeypatch.setattr(cli, "_compose_args", lambda _args: config)
    monkeypatch.setattr(
        cli,
        "prepare_lifecycle",
        lambda _config, *, write: events.append(("checkpoint", write)),
    )
    monkeypatch.setattr(
        cli,
        "validate_runtime_stack",
        lambda _root: events.append("runtime"),
    )
    monkeypatch.setattr(
        cli,
        "validate_local_sio_pairs",
        lambda _config: events.append("sio"),
    )
    monkeypatch.setattr(
        cli,
        "_render",
        lambda _config: (
            events.append("render") or (run_dir, command, environment)
        ),
    )
    monkeypatch.setattr(cli, "_print_summary", lambda *_args: None)

    class ExecCalled(RuntimeError):
        pass

    def fake_execvpe(
        executable: str,
        argv: list[str],
        env: dict[str, str],
    ) -> None:
        events.append(("exec", executable, argv, env["PIPELINE_AUDIT"]))
        raise ExecCalled

    monkeypatch.setattr(os, "execvpe", fake_execvpe)
    with pytest.raises(ExecCalled):
        cli.main(
            [
                "launch",
                "--model",
                "1b",
                "--variant",
                "base",
                "--stage",
                "stage1",
                "--data",
                "dolma3_6t",
                "--topology",
                "stage1_tp1_4096npu",
                "--run-id",
                "launch-order",
                "--save",
                "/runtime/checkpoint",
                "--output",
                "/runtime/output",
            ]
        )

    assert events == [
        ("checkpoint", False),
        "runtime",
        "sio",
        "render",
        ("checkpoint", True),
        ("exec", command[0], command, "1"),
    ]


def test_cli_launch_rejects_multi_node_control_plane_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-node ranks must share one pre-rendered immutable run directory."""

    config = _resolved("1b", "base", "stage1")
    monkeypatch.setattr(cli, "_compose_args", lambda _args: config)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "launch",
                "--model",
                "1b",
                "--variant",
                "base",
                "--stage",
                "stage1",
                "--data",
                "dolma3_6t",
                "--topology",
                "stage1_tp1_4096npu",
                "--run-id",
                "multi-node-shortcut",
                "--save",
                "/runtime/checkpoint",
                "--output",
                "/runtime/output",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("environment", "expected"),
    (
        ({"OLMO3_NODE_RANK": "3", "NODE_RANK": "2"}, 3),
        ({"NODE_RANK": "2"}, 2),
        ({}, 0),
    ),
)
def test_worker_node_rank_is_deployment_only_and_range_checked(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: int,
) -> None:
    config = _resolved("1b", "base", "stage1")
    monkeypatch.delenv("OLMO3_NODE_RANK", raising=False)
    monkeypatch.delenv("NODE_RANK", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert cli._deployment_node_rank(config) == expected

    monkeypatch.setenv("OLMO3_NODE_RANK", str(config["runtime"]["nnodes"]))
    with pytest.raises(ConfigurationError, match=r"must be in \[0,"):
        cli._deployment_node_rank(config)


def test_worker_preflight_revalidates_config_and_checks_local_sio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _resolved("3b", "siamese_depth", "stage1")
    config["topology"]["world_size"] = 16
    config["topology"]["nproc_per_node"] = 16
    config["topology"]["tensor_parallel"] = 2
    config["topology"]["context_parallel"] = 1
    config["topology"]["sequence_parallel"] = True
    config["topology"]["global_batch_size"] = 8
    config["topology"]["micro_batch_size"] = 1
    config["training"]["global_batch_size"] = 8
    config["training"]["micro_batch_size"] = 1
    config["topology"]["hsdp"]["shard_size"] = 8
    config["performance"] = compose(
        model="3b",
        variant="siamese_depth",
        stage="stage1",
        data="dolma3_6t",
        topology="tp2_sio_2048npu",
        performance=["tp2_sio_mc2"],
    )["performance"]
    config["performance"].update(
        {
            "swa_cp_mode": "none",
        }
    )
    config["runtime"]["nnodes"] = 1
    config["optimization"].pop("warmup_steps", None)
    config = resolve(config)
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(config), encoding="utf-8")

    checked: list[dict] = []
    monkeypatch.setattr(
        cli,
        "validate_local_sio_pairs",
        lambda value: checked.append(value),
    )
    monkeypatch.setattr(cli, "verify_source_manifest", lambda _path: None)
    monkeypatch.setattr(cli, "verify_upstream_snapshot", lambda _path: None)
    assert (
        cli.main(
            [
                "worker-preflight",
                "--resolved",
                str(resolved_path),
            ]
        )
        == 0
    )
    assert len(checked) == 1
    assert checked[0]["topology"]["tensor_parallel"] == 2


def test_worker_data_mount_preflight_is_single_process_size_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolved("1b", "siamese_depth", "stage2")
    calls: list[dict] = []
    cache_calls: list[tuple[tuple[Path, ...], Path, dict]] = []

    def fake_load(_path: Path, **kwargs):
        calls.append(kwargs)
        return {
            "data_config": config["data"]["name"],
            "data_config_sha256": "0" * 64,
            "contract_sha256": "a" * 64,
            "resolved_token_paths": ["/runtime/data/part-0000.npy"],
        }

    def fake_describe(
        source_paths: tuple[Path, ...],
        work_dir: Path,
        **kwargs: object,
    ) -> dict:
        cache_calls.append((source_paths, work_dir, kwargs))
        return {"sequence_length": 8192, "source_count": 1}

    fake_module = types.ModuleType("runtime.olmo3_packed_dataset")
    fake_module.describe_olmo3_midtraining_cache = fake_describe
    fake_module.describe_olmo3_long_context_cache = lambda *_a, **_k: {}
    monkeypatch.setitem(
        sys.modules,
        "runtime.olmo3_packed_dataset",
        fake_module,
    )
    monkeypatch.setattr(cli, "load_runtime_data_manifest", fake_load)
    monkeypatch.setattr(cli, "validate_runtime_data_profile", lambda *_a, **_k: None)
    cli._validate_worker_data_mount(config)
    assert calls == [
        {
            "root": Path(config["data"]["root"]),
            "verify_files": True,
            "checksums": False,
            "expected_stage": "stage2",
            "expected_backend": "olmo3_numpy_fsl",
            "expected_token_count": config["data"]["known_token_count"],
        }
    ]
    assert cache_calls == [
        (
            (Path("/runtime/data/part-0000.npy"),),
            Path(config["data"]["work_dir"]),
            {
                "source_contract_sha256": "a" * 64,
                "global_batch_size": config["training"]["global_batch_size"],
                "data_seed": config["training"]["data_seed"],
                "sequence_length": 8192,
                "pad_token_id": 100277,
                "dtype": "uint32",
            },
        )
    ]


def test_worker_sft_preflight_uses_packed_subdirectory_and_checks_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolved("1b", "siamese_depth", "sft_think")
    config["data"]["expected_instances"] = 19
    config["data"]["expected_fingerprint"] = "f" * 64
    calls: list[tuple[Path, Path, dict]] = []

    def fake_describe(data_dir: Path, work_dir: Path, **kwargs: object) -> dict:
        calls.append((data_dir, work_dir, kwargs))
        return {"base_instances": 19, "fingerprint": "f" * 64}

    fake_module = types.ModuleType("runtime.olmo3_sft_dataset")
    fake_module.describe_olmo3_sft_cache = fake_describe
    monkeypatch.setitem(
        sys.modules,
        "runtime.olmo3_sft_dataset",
        fake_module,
    )
    cli._validate_worker_data_mount(config)
    assert calls == [
        (
            Path(config["data"]["root"]),
            Path(config["data"]["work_dir"]) / "packed",
            {
                "sequence_length": 32768,
                "eos_token_id": 100257,
                    "pad_token_id": 100277,
                    "dtype": "uint32",
                    "expected_fingerprint": "f" * 64,
                },
        )
    ]


def test_worker_stage3_preflight_fails_before_torchrun_when_cache_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolved("1b", "siamese_depth", "stage3")
    contract = {
        "data_config": config["data"]["name"],
        "data_config_sha256": "0" * 64,
        "contract_sha256": "a" * 64,
        "resolved_token_paths": ["/runtime/data/part-0000.npy"],
    }
    monkeypatch.setattr(
        cli,
        "load_runtime_data_manifest",
        lambda *_a, **_k: contract,
    )
    monkeypatch.setattr(
        cli,
        "validate_runtime_data_profile",
        lambda *_a, **_k: None,
    )
    fake_module = types.ModuleType("runtime.olmo3_packed_dataset")
    fake_module.describe_olmo3_midtraining_cache = lambda *_a, **_k: {}

    def missing_cache(*_args: object, **_kwargs: object) -> dict:
        raise FileNotFoundError("prepared cache is missing")

    fake_module.describe_olmo3_long_context_cache = missing_cache
    monkeypatch.setitem(
        sys.modules,
        "runtime.olmo3_packed_dataset",
        fake_module,
    )

    with pytest.raises(
        ConfigurationError,
        match="worker data-mount preflight failed: prepared cache is missing",
    ):
        cli._validate_worker_data_mount(config)


def test_worker_sft_preflight_fails_before_torchrun_on_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolved("1b", "siamese_depth", "sft_instruct")

    fake_module = types.ModuleType("runtime.olmo3_sft_dataset")
    fake_module.describe_olmo3_sft_cache = lambda *_args, **_kwargs: {
        "base_instances": config["data"]["expected_instances"] + 1,
        "fingerprint": config["data"]["expected_fingerprint"],
    }
    monkeypatch.setitem(
        sys.modules,
        "runtime.olmo3_sft_dataset",
        fake_module,
    )
    with pytest.raises(ConfigurationError, match="SFT cache contract mismatch"):
        cli._validate_worker_data_mount(config)


def _npu_smi_sio_fixture() -> tuple[str, str]:
    mapping = "\n".join(
        f"{card} {die} {2 * card + die} {2 * card + die}"
        for card in range(8)
        for die in range(2)
    )
    rows = []
    for row in range(16):
        peer = row + 1 if row % 2 == 0 else row - 1
        relations = [
            "X" if column == row else "SIO" if column == peer else "HCCS"
            for column in range(16)
        ]
        rows.append(" ".join([f"Phy-ID{row}", *relations]))
    return mapping, "\n".join(rows)


def test_tp2_sio_parser_matches_production_npu_smi_layout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mapping, topology = _npu_smi_sio_fixture()
    calls: list[tuple[str, ...]] = []

    def fake_check_output(command: list[str], **_kwargs: object) -> str:
        calls.append(tuple(command))
        return mapping if command[-1] == "-m" else topology

    monkeypatch.setattr(
        "olmo3_pipeline.topology.subprocess.check_output",
        fake_check_output,
    )
    validate_local_sio_pairs(
        {
            "performance": {"require_tp_same_physical_card": True},
            "topology": {"tensor_parallel": 2, "nproc_per_node": 16},
        }
    )
    assert calls == [
        ("npu-smi", "info", "-m"),
        ("npu-smi", "info", "-t", "topo"),
    ]
    assert capsys.readouterr().out == (
        "OLMO3_RUNTIME_SIO_PREFLIGHT_OK local_npus=16 tp_pairs=8\n"
    )


def test_tp2_sio_parser_rejects_non_sio_adjacent_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping, topology = _npu_smi_sio_fixture()
    topology = topology.replace("Phy-ID0 X SIO", "Phy-ID0 X HCCS", 1)

    def fake_check_output(command: list[str], **_kwargs: object) -> str:
        return mapping if command[-1] == "-m" else topology

    monkeypatch.setattr(
        "olmo3_pipeline.topology.subprocess.check_output",
        fake_check_output,
    )
    with pytest.raises(ConfigurationError, match="not connected by SIO"):
        validate_local_sio_pairs(
            {
                "performance": {"require_tp_same_physical_card": True},
                "topology": {"tensor_parallel": 2, "nproc_per_node": 16},
            }
        )


def test_resolver_rejects_runtime_values_that_would_crash_late() -> None:
    invalid_nnodes = _resolved("1b", "base", "stage1")
    invalid_nnodes["runtime"]["nnodes"] = "not-an-integer"
    with pytest.raises(ConfigurationError, match="runtime.nnodes"):
        resolve(invalid_nnodes)

    unsafe_run_id = _resolved("1b", "base", "stage1")
    unsafe_run_id["runtime"]["run_id"] = "../escape"
    with pytest.raises(ConfigurationError, match="run_id is unsafe"):
        resolve(unsafe_run_id)

    nonfinite_lr = _resolved("1b", "base", "stage1")
    nonfinite_lr["optimization"]["peak_lr"] = float("nan")
    with pytest.raises(ConfigurationError, match="peak_lr must be positive"):
        resolve(nonfinite_lr)


@pytest.mark.parametrize(
    "run_id",
    (
        "../escape",
        "nested/path",
        "line\nbreak",
        "-leading-dash",
        "a" * 256,
    ),
)
def test_resolver_rejects_unsafe_run_ids(run_id: str) -> None:
    config = _resolved("1b", "base", "stage1")
    config["runtime"]["run_id"] = run_id

    with pytest.raises(ConfigurationError, match="runtime.run_id is unsafe"):
        resolve(config)


@pytest.mark.parametrize(
    ("section", "value"),
    (
        ("model", None),
        ("variant", []),
        ("stage", "stage1"),
        ("training", 1),
        ("optimization", False),
        ("data", []),
        ("topology", None),
        ("performance", "ordinary_hsdp"),
        ("runtime", []),
    ),
)
def test_resolver_rejects_malformed_sections_without_internal_errors(
    section: str,
    value: object,
) -> None:
    config = _resolved("1b", "base", "stage1")
    config[section] = value

    with pytest.raises(ConfigurationError, match=rf"{section} must be an object"):
        resolve(config)


def test_resolver_checks_query_groups_for_ulysses() -> None:
    config = _resolved("1b", "siamese_depth", "stage3")
    config["model"]["num_query_groups"] = 8
    with pytest.raises(
        ConfigurationError,
        match="attention and query-group heads",
    ):
        resolve(config)


def test_tp2_mc2_all_gather_recomputation_is_fail_closed() -> None:
    config = compose(
        model="3b",
        variant="siamese_depth",
        stage="stage1",
        data="dolma3_6t",
        topology="tp2_sio_2048npu",
        performance=("tp2_sio_mc2",),
        overrides=(
            "runtime.run_id=mc2-recompute-required",
            "runtime.lifecycle=fresh",
            "runtime.save=/runtime/save",
            "runtime.output=/runtime/output",
            "data.root=/runtime/data",
            "data.data_args_path=/runtime/data-args",
            "data.tokenizer.path=/runtime/tokenizer",
            "training.train_tokens=6000000000000",
            "optimization.peak_lr=0.0007",
            "optimization.min_lr=0.00007",
            "optimization.warmup_tokens=8388608000",
            "performance.mc2_all_gather_recomputation=false",
        ),
    )
    with pytest.raises(
        ConfigurationError,
        match="requires all-gather recomputation",
    ):
        resolve(config)
