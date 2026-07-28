from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from olmo3_pipeline import cli
from olmo3_pipeline.command import (
    build_environment,
    build_megatron_args,
    build_torchrun_command,
)
from olmo3_pipeline.config import ConfigurationError, compose
from olmo3_pipeline.model_config import build_model_config
from olmo3_pipeline.resolve import resolve


STAGE_CASES = {
    "stage1": {
        "data": "dolma3_6t",
        "topology": "stage1_tp1_4096npu",
        "performance": "ordinary_hsdp",
        "backend": "megatron_indexed",
        "dp": 4096,
        "ga": 1,
        "packed": False,
        "yarn": False,
    },
    "stage2": {
        "data": "dolmino_100b",
        "topology": "stage2_cp2_512npu",
        "performance": "cp_single_halo",
        "backend": "olmo3_numpy_fsl",
        "dp": 256,
        "ga": 1,
        "packed": False,
        "yarn": False,
    },
    "stage3": {
        "data": "longmino_50b",
        "topology": "stage3_cp16_1024npu",
        "performance": "cp_single_halo",
        "backend": "olmo3_numpy_packed",
        "dp": 64,
        "ga": 1,
        "packed": True,
        "yarn": True,
    },
    "sft_think": {
        "data": "dolci_think",
        "topology": "sft_cp8_256npu",
        "performance": "cp_single_halo",
        "backend": "olmo3_sft_numpy",
        "dp": 32,
        "ga": 1,
        "packed": True,
        "yarn": True,
    },
    "sft_instruct": {
        "data": "dolci_instruct",
        "topology": "sft_cp8_256npu",
        "performance": "cp_single_halo",
        "backend": "olmo3_sft_numpy",
        "dp": 32,
        "ga": 1,
        "packed": True,
        "yarn": True,
    },
}

MODEL_SHAPES = {
    "1b": (16, 2048, 8192, 16, 128),
    "3b": (16, 3328, 13312, 16, 208),
    "7b": (32, 4096, 11008, 32, 128),
}


def _overrides(stage: str, run_id: str = "cpu-test") -> list[str]:
    values = [
        f"runtime.run_id={run_id}",
        "runtime.save=/artifacts/save",
        "runtime.output=/artifacts/output",
        "runtime.python=/usr/bin/python3",
        "data.root=/fixtures/data",
        "data.tokenizer.path=/fixtures/tokenizer",
        "optimization.peak_lr=0.00025",
    ]
    if stage == "stage1":
        values.extend(
            [
                "runtime.lifecycle=fresh",
                "data.data_args_path=/fixtures/data_args.json",
                "training.train_tokens=6000000000000",
                "optimization.min_lr=0.0001",
                "optimization.warmup_tokens=8388608000",
            ]
        )
    else:
        values.extend(
            [
                "runtime.lifecycle=transition",
                "runtime.load=/artifacts/load",
            ]
        )
        if stage in {"stage2", "stage3"}:
            values.extend(
                [
                    "data.manifest=/fixtures/runtime.data-manifest.json",
                    "data.work_dir=/work/data",
                ]
            )
        else:
            values.extend(
                [
                    "data.work_dir=/work/sft",
                    "data.expected_instances=1001",
                    f"data.expected_fingerprint={'f' * 64}",
                ]
            )
        if stage == "stage3":
            values.append("optimization.warmup_tokens=838860800")
    return values


def _resolved(
    model: str,
    variant: str,
    stage: str,
    *,
    topology: str | None = None,
    performance: str | None = None,
    extra: tuple[str, ...] = (),
) -> dict:
    case = STAGE_CASES[stage]
    config = compose(
        model=model,
        variant=variant,
        stage=stage,
        data=case["data"],
        topology=topology or case["topology"],
        performance=[performance or case["performance"]],
        overrides=[*_overrides(stage), *extra],
    )
    return resolve(config)


@pytest.mark.parametrize("model", ("1b", "3b", "7b"))
@pytest.mark.parametrize("variant", ("base", "siamese_depth"))
@pytest.mark.parametrize("stage", tuple(STAGE_CASES))
def test_all_model_variant_stage_compositions_validate(
    model: str, variant: str, stage: str
) -> None:
    """Every advertised model × variant × stage composition is CPU-valid."""

    case = STAGE_CASES[stage]
    config = _resolved(model, variant, stage)
    generated = build_model_config(config)

    layers, hidden, ffn, heads, head_dim = MODEL_SHAPES[model]
    assert config["model"]["num_layers"] == layers
    assert config["model"]["hidden_size"] == hidden
    assert config["model"]["ffn_hidden_size"] == ffn
    assert config["model"]["num_attention_heads"] == heads
    assert config["model"]["kv_channels"] == head_dim
    assert config["data"]["backend"] == case["backend"]
    assert config["topology"]["data_parallel"] == case["dp"]
    assert config["training"]["gradient_accumulation"] == case["ga"]
    assert config["stage"]["packed"] is case["packed"]
    assert (config["stage"]["full_attention_yarn"] is not None) is case["yarn"]
    assert generated["rope_theta"] == 500000.0
    assert generated["rope_full_precision"] is True
    assert generated["sliding_window"] == 4096

    if variant == "base":
        assert generated["architectures"] == ["OLMo3ForCausalLM"]
        assert generated["use_siamese_norm"] is False
        assert generated["use_depth_attention"] is False
        assert config["variant"]["checkpoint_keyspace"] == "olmo3_base_v1"
    else:
        assert generated["architectures"] == [
            "OLMo3SiameseDepthForCausalLM"
        ]
        assert generated["use_siamese_norm"] is True
        assert generated["siamese_norm_variant"] == "hybrid_pre"
        assert generated["use_depth_attention"] is True
        assert generated["depth_attention_stride"] == 8
        assert (
            config["variant"]["checkpoint_keyspace"]
            == "olmo3_siamese_depth_v1"
        )


def test_stage1_token_and_hsdp_contract() -> None:
    config = _resolved("1b", "base", "stage1")

    assert config["topology"]["data_parallel"] == 4096
    assert config["training"]["micro_batch_size"] == 2
    assert config["training"]["global_batch_size"] == 8192
    assert config["training"]["gradient_accumulation"] == 1
    assert config["training"]["tokens_per_step"] == 67_108_864
    assert config["training"]["train_iters"] == 89_407
    assert config["optimization"]["warmup_steps"] == 125
    assert config["topology"]["hsdp"]["num_instances"] == 2
    assert config["topology"]["hsdp"]["ddp_num_buckets"] == 8


def test_stage2_fsl_cp_and_iteration_contract() -> None:
    config = _resolved("1b", "siamese_depth", "stage2")
    args = build_megatron_args(config, Path("/contract/model_config.json"))

    assert config["data"]["backend"] == "olmo3_numpy_fsl"
    assert config["stage"]["full_attention_yarn"] is None
    assert config["topology"]["context_parallel"] == 2
    assert config["topology"]["data_parallel"] == 256
    assert config["derived"]["local_sequence_length"] == 4096
    assert config["training"]["tokens_per_step"] == 2_097_152
    assert config["training"]["train_iters"] == 47_684
    assert config["optimization"]["warmup_steps"] == 0
    assert config["training"]["seed"] == 1337
    assert config["training"]["data_seed"] == 1337
    assert config["topology"]["hsdp"]["num_instances"] == 32
    assert config["stage"]["activation_checkpoint"] == "none"
    assert "--recompute-granularity" not in args
    assert "--olmo3-fsl-data-root" in args
    assert "--olmo3-fsl-data-manifest" in args
    assert "--olmo3-expected-data-tokens" in args
    assert "--olmo3-packed-documents" not in args


def test_stage3_yarn_packed_halo_and_iteration_contract() -> None:
    config = _resolved("7b", "siamese_depth", "stage3")
    env = build_environment(config)
    args = build_megatron_args(config, Path("/contract/model_config.json"))
    model = build_model_config(config)

    assert config["topology"]["context_parallel"] == 16
    assert config["topology"]["data_parallel"] == 64
    assert config["derived"]["local_sequence_length"] == 4096
    assert config["training"]["tokens_per_step"] == 4_194_304
    assert config["training"]["train_iters"] == 11_921
    assert config["optimization"]["warmup_steps"] == 200
    assert config["training"]["seed"] == 4123
    assert config["training"]["data_seed"] == 4123
    assert config["data"]["packing"]["training_shuffle_seed"] == 4123
    assert config["topology"]["hsdp"]["num_instances"] == 64
    assert model["rope_scaling"] == {
        "type": "yarn",
        "factor": 8.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 8192,
    }
    assert env["OLMO3_SWA_CP_MODE"] == "single_halo"
    assert env["OLMO3_SWA_HALO_OVERLAP"] == "1"
    assert env["OLMO3_SWA_HALO_BACKWARD_OVERLAP"] == "1"
    assert env["OLMO3_FUSED_QKV_A2A_PACKING"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["OLMO3_WANDB_LOG_STYLE"] == "llamafactory"
    assert "--olmo3-packed-documents" in args
    assert "--olmo3-packed-data-root" in args
    assert "--olmo3-packed-data-manifest" in args
    assert "--olmo3-expected-data-tokens" in args


def test_production_data_cannot_enable_manifest_count_policy() -> None:
    config = compose(
        model="1b",
        variant="base",
        stage="stage2",
        data="dolmino_100b",
        topology="stage2_cp2_512npu",
        performance=["cp_single_halo"],
        overrides=[
            *_overrides("stage2"),
            "data.token_count_policy=manifest",
        ],
    )
    with pytest.raises(ConfigurationError, match="must use"):
        resolve(config)

    count_override = compose(
        model="1b",
        variant="base",
        stage="stage2",
        data="dolmino_100b",
        topology="stage2_cp2_512npu",
        performance=["cp_single_halo"],
        overrides=[
            *_overrides("stage2"),
            "data.known_token_count=17",
        ],
    )
    with pytest.raises(ConfigurationError, match="fixed production token count"):
        resolve(count_override)


def test_tp2_sio_mc2_contract() -> None:
    config = _resolved(
        "3b",
        "siamese_depth",
        "stage1",
        topology="tp2_sio_2048npu",
        performance="tp2_sio_mc2",
    )
    env = build_environment(config)
    command = build_torchrun_command(
        config, Path("/contract/model_config.json")
    )

    assert config["topology"]["tensor_parallel"] == 2
    assert config["topology"]["sequence_parallel"] is True
    assert config["topology"]["data_parallel"] == 1024
    assert config["training"]["global_batch_size"] == 1024
    assert config["training"]["gradient_accumulation"] == 1
    assert config["training"]["tokens_per_step"] == 8_388_608
    assert config["training"]["train_iters"] == 715_256
    assert config["optimization"]["warmup_steps"] == 1000
    assert config["topology"]["hsdp"]["num_instances"] == 128
    assert env["OLMO3_MC2_ALL_GATHER_RECOMPUTATION"] == "1"
    assert env["OLMO3_HSDP_INTER_OVERLAP_BACKWARD"] == "1"
    assert env["OLMO3_HSDP_INTER_TP_LANE_PACK"] == "2"
    assert any(part.endswith("pretrain_olmo3_mindspeed_mc2.py") for part in command)
    assert "--sequence-parallel" in command


def test_tp2_rejects_disabling_mc2_all_gather_recomputation() -> None:
    with pytest.raises(ConfigurationError, match="all-gather recomputation"):
        _resolved(
            "3b",
            "siamese_depth",
            "stage1",
            topology="tp2_sio_2048npu",
            performance="tp2_sio_mc2",
            extra=("performance.mc2_all_gather_recomputation=false",),
        )


def test_boolean_configuration_fields_reject_truthy_strings() -> None:
    topology = _resolved("1b", "base", "stage1")
    topology["topology"]["sequence_parallel"] = "false"
    with pytest.raises(
        ConfigurationError,
        match=r"topology\.sequence_parallel must be a boolean",
    ):
        resolve(topology)

    performance = _resolved("1b", "base", "stage1")
    performance["performance"]["manual_gc"] = "false"
    with pytest.raises(
        ConfigurationError,
        match=r"performance\.manual_gc must be a boolean",
    ):
        resolve(performance)

    wandb = _resolved("1b", "base", "stage1")
    wandb["runtime"]["wandb"] = {"enabled": "false"}
    with pytest.raises(
        ConfigurationError,
        match=r"runtime\.wandb\.enabled must be a boolean",
    ):
        resolve(wandb)


@pytest.mark.parametrize("model", ("1b", "3b", "7b"))
@pytest.mark.parametrize("variant", ("base", "siamese_depth"))
@pytest.mark.parametrize(
    ("stage", "data", "topology", "tp", "cp", "shard_size"),
    (
        ("stage3", "longmino_50b", "stage3_cp16_1024npu", 2, 8, 16),
        ("sft_think", "dolci_think", "sft_cp8_256npu", 2, 4, 8),
        (
            "sft_instruct",
            "dolci_instruct",
            "sft_cp8_256npu",
            2,
            4,
            8,
        ),
    ),
)
def test_combined_tp2_cp_profiles_support_all_models_and_variants(
    model: str,
    variant: str,
    stage: str,
    data: str,
    topology: str,
    tp: int,
    cp: int,
    shard_size: int,
) -> None:
    config = compose(
        model=model,
        variant=variant,
        stage=stage,
        data=data,
        topology=topology,
        performance=("tp2_sio_mc2", "cp_single_halo"),
        overrides=(
            *_overrides(stage),
            f"topology.tensor_parallel={tp}",
            f"topology.context_parallel={cp}",
            "topology.sequence_parallel=true",
            f"topology.hsdp.shard_size={shard_size}",
        ),
    )
    config = resolve(config)
    args = build_megatron_args(config, Path("/contract/model_config.json"))

    assert config["topology"]["tensor_parallel"] == tp
    assert config["topology"]["context_parallel"] == cp
    assert config["topology"]["sequence_parallel"] is True
    assert config["derived"]["local_sequence_length"] >= 4_095
    assert "--sequence-parallel" in args


@pytest.mark.parametrize("stage", ("sft_think", "sft_instruct"))
def test_sft_epoch_and_warmup_arithmetic(stage: str) -> None:
    config = _resolved("1b", "siamese_depth", stage)

    assert config["training"]["epochs"] == 2
    assert config["training"]["instances_per_epoch"] == 992
    assert config["training"]["train_iters"] == 62
    assert config["optimization"]["warmup_steps"] == 2
    assert config["optimization"]["warmup_tokens"] == 2_097_152
    assert config["stage"]["assistant_only_loss"] is True
    assert config["stage"]["activation_checkpoint"] == "mlp"
    assert config["training"]["seed"] == (
        53_184 if stage == "sft_think" else 543_210
    )


@pytest.mark.parametrize(
    "fingerprint",
    ("sha256:" + "f" * 64, "F" * 64, "f" * 63, "not-a-digest"),
)
def test_sft_rejects_noncanonical_cache_fingerprint(fingerprint: str) -> None:
    with pytest.raises(ConfigurationError, match="64-character lowercase"):
        _resolved(
            "1b",
            "siamese_depth",
            "sft_think",
            extra=(f"data.expected_fingerprint={fingerprint}",),
        )


@pytest.mark.parametrize("stage", ("sft_think", "sft_instruct"))
def test_sft_fraction_warmup_resolved_contract_is_idempotent(stage: str) -> None:
    config = _resolved("1b", "siamese_depth", stage)

    revalidated = resolve(config)

    assert revalidated == config
    assert revalidated["optimization"]["warmup_fraction"] == 0.03
    assert revalidated["optimization"]["warmup_steps"] == 2
    assert revalidated["optimization"]["warmup_tokens"] == 2_097_152


def test_sft_rejects_two_unresolved_warmup_authorities() -> None:
    config = compose(
        model="1b",
        variant="siamese_depth",
        stage="sft_instruct",
        data="dolci_instruct",
        topology="sft_cp8_256npu",
        performance=["cp_single_halo"],
        overrides=[
            *_overrides("sft_instruct"),
            "optimization.warmup_tokens=2097152",
        ],
    )

    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        resolve(config)


def test_stage_recipe_fields_fail_closed() -> None:
    for stage, override, match in (
        ("stage1", "training.seed=42", "training.seed"),
        ("stage2", "stage.activation_checkpoint=mlp", "activation_checkpoint"),
        ("stage3", "stage.full_attention_yarn=null", "full_attention_yarn"),
        ("stage3", "optimization.z_loss=0", "z_loss"),
        ("sft_think", "stage.assistant_only_loss=false", "assistant_only_loss"),
        ("sft_instruct", "optimization.weight_decay=0.1", "weight_decay"),
    ):
        with pytest.raises(ConfigurationError, match=match):
            _resolved(
                "1b",
                "siamese_depth",
                stage,
                extra=(override,),
            )


def test_stage1_transition_lifecycle_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="stage1 lifecycle"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=(
                "runtime.lifecycle=transition",
                "runtime.load=/artifacts/load",
            ),
        )


def test_tp_and_cp_profile_composition_is_order_independent() -> None:
    common = dict(
        model="3b",
        variant="siamese_depth",
        stage="stage3",
        data="longmino_50b",
        topology="stage2_cp2_512npu",
        overrides=[
            *_overrides("stage3"),
            "topology.tensor_parallel=2",
            "topology.context_parallel=2",
            "topology.sequence_parallel=true",
            "performance.swa_cp_mode=single_halo",
        ],
    )
    first = resolve(
        compose(
            **common,
            performance=["tp2_sio_mc2", "cp_single_halo"],
        )
    )
    second = resolve(
        compose(
            **common,
            performance=["cp_single_halo", "tp2_sio_mc2"],
        )
    )
    assert first == second
    assert first["performance"]["scalar_reduce_backend"] == "gloo"
    assert first["performance"]["profiles"] == [
        "cp_single_halo",
        "tp2_sio_mc2",
    ]


def test_render_is_cpu_only_and_writes_immutable_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _resolved("1b", "base", "stage3", extra=("runtime.run_id=render-test",))

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("olmo3_pipeline.command.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "upstream_state",
        lambda: {
            "MindSpeed": {
                "matches_lock": True,
                "matches_url": True,
                "dirty": False,
            },
            "MindSpeed-LLM": {
                "matches_lock": True,
                "matches_url": True,
                "dirty": False,
            },
            "Megatron-LM": {"matches_lock": True, "dirty": False},
            "OLMo-core": {
                "matches_lock": True,
                "matches_url": True,
                "dirty": False,
            },
        },
    )
    monkeypatch.setattr(
        cli,
        "source_manifest",
        lambda: {"kind": "cpu-test", "files": []},
    )

    run_dir, command, environment = cli._render(config)
    assert run_dir == tmp_path / "runs" / "render-test"
    for name in (
        "resolved.json",
        "model_config.json",
        "upstream.json",
        "source_manifest.json",
        "environment.json",
        "command.json",
        "command.txt",
        "environment.sh",
        "checkpoint-activate.sh",
        "launch.sh",
    ):
        assert (run_dir / name).is_file()
    assert json.loads((run_dir / "resolved.json").read_text())[
        "training"
    ]["train_iters"] == 11_921
    assert "--olmo3-stage-transition" in command
    assert environment["OLMO3_SWA_CP_MODE"] == "single_halo"
    activation = (run_dir / "checkpoint-activate.sh").read_text()
    launcher = (run_dir / "launch.sh").read_text()
    environment_script = (run_dir / "environment.sh").read_text()
    assert (
        f"export PATH={Path(command[0]).parent}"
        '${PATH:+:${PATH}}'
    ) in environment_script
    assert "checkpoint_cli preflight" in activation
    assert "--write" in activation
    assert "checkpoint_cli verify-prepared" in launcher
    assert "olmo3_pipeline.cli worker-preflight" in launcher
    assert "--write" not in launcher
    assert "OLMO3_NODE_RANK" in launcher
    assert "NODE_RANK" in launcher
    assert '"${worker_node_rank}"' in launcher
    subprocess.run(
        ["bash", "-n", str(run_dir / "checkpoint-activate.sh")],
        check=True,
    )
    subprocess.run(["bash", "-n", str(run_dir / "launch.sh")], check=True)

    with pytest.raises(ConfigurationError, match="immutable run directory"):
        cli._render(config)


def test_render_rejects_wrong_pristine_upstream_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _resolved(
        "1b",
        "base",
        "stage1",
        extra=("runtime.run_id=wrong-upstream-origin",),
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "upstream_state",
        lambda: {
            "MindSpeed": {
                "matches_lock": True,
                "matches_url": False,
                "dirty": False,
            }
        },
    )
    with pytest.raises(ConfigurationError, match="do not match the lock"):
        cli._render(config)
    assert not (tmp_path / "runs").exists()


def test_render_rejects_single_branch_third_party_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _resolved(
        "1b",
        "base",
        "stage1",
        extra=("runtime.run_id=single-branch-upstream",),
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "upstream_state",
        lambda: {
            "MindSpeed": {
                "matches_lock": True,
                "matches_url": True,
                "dirty": False,
                "shallow": False,
                "full_history_required": True,
                "full_heads_refspec": False,
                "generated_python_artifacts": 0,
            }
        },
    )
    with pytest.raises(ConfigurationError, match="do not match the lock"):
        cli._render(config)
    assert not (tmp_path / "runs").exists()


def test_cli_validate_does_not_render_or_touch_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(
        [
            "validate",
            "--model",
            "1b",
            "--variant",
            "base",
            "--stage",
            "stage3",
            "--data",
            "longmino_50b",
            "--topology",
            "stage3_cp16_1024npu",
            "--performance",
            "cp_single_halo",
            "--run-id",
            "validate-only",
            "--data-root",
            "/fixtures/longmino",
            "--data-manifest",
            "/fixtures/longmino-runtime.data-manifest.json",
            "--data-work-dir",
            "/work/longmino",
            "--tokenizer",
            "/fixtures/tokenizer",
            "--load",
            "/artifacts/stage2",
            "--save",
            "/artifacts/stage3",
            "--output",
            "/artifacts/output",
            "--peak-lr",
            "0.00025",
            "--min-lr",
            "0",
            "--warmup-tokens",
            "838860800",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert summary["dp"] == 64
    assert summary["train_iters"] == 11_921
    assert summary["warmup_steps"] == 200
    assert "run_dir" not in summary


def test_removed_multihop_configuration_fails_closed() -> None:
    with pytest.raises(ConfigurationError, match="single-hop halo requires"):
        _resolved(
            "7b",
            "base",
            "stage3",
            extra=(
                "topology.context_parallel=32",
                "topology.world_size=1024",
                "topology.hsdp.shard_size=16",
            ),
        )


def test_cp_profile_cannot_silently_disable_retained_optimizations() -> None:
    with pytest.raises(ConfigurationError, match="swa_halo_backward_overlap"):
        _resolved(
            "1b",
            "base",
            "stage3",
            extra=("performance.swa_halo_backward_overlap=false",),
        )
    with pytest.raises(ConfigurationError, match="CP1 requires"):
        _resolved(
            "1b",
            "base",
            "stage1",
            performance="cp_single_halo",
        )


def test_base_variant_cannot_enable_siamese_norm() -> None:
    with pytest.raises(ConfigurationError, match="base variant cannot enable"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("variant.use_siamese_norm=true",),
        )


def test_variant_cannot_claim_another_checkpoint_keyspace() -> None:
    with pytest.raises(ConfigurationError, match="checkpoint keyspace"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("variant.checkpoint_keyspace=olmo3_siamese_depth_v1",),
        )


@pytest.mark.parametrize(
    ("model", "override", "message"),
    (
        ("1b", "model.hidden_size=3328", "official OLMo 3 1b"),
        ("3b", "model.ffn_hidden_size=11008", "official OLMo 3 3b"),
        ("7b", "model.ffn_hidden_size=16384", "official OLMo 3 7b"),
        ("1b", "model.swa_window=8192", "official OLMo 3 1b"),
    ),
)
def test_model_size_is_bound_to_exact_official_geometry(
    model: str,
    override: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _resolved(model, "base", "stage1", extra=(override,))


def test_stage_sequence_contract_cannot_drift_silently() -> None:
    with pytest.raises(ConfigurationError, match="stage3 sequence contract"):
        _resolved(
            "1b",
            "base",
            "stage3",
            extra=(
                "stage.sequence_length=32768",
                "topology.context_parallel=8",
            ),
        )


def test_rank_order_keeps_tp_cp_groups_node_contiguous() -> None:
    with pytest.raises(ConfigurationError, match="rank_order"):
        _resolved(
            "1b",
            "base",
            "stage3",
            extra=("topology.rank_order=dp-cp-tp-pp",),
        )


def test_tokenizer_ids_are_locked_to_olmo3() -> None:
    with pytest.raises(ConfigurationError, match="OLMo 3 tokenizer contract"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("data.tokenizer.eos_token_id=2",),
        )


def test_validation_manifest_and_eval_iterations_must_be_enabled_together() -> None:
    with pytest.raises(ConfigurationError, match="requires training.eval_iters"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("runtime.validation.manifest=/validation/manifest.txt",),
        )
    with pytest.raises(ConfigurationError, match="requires runtime.validation.manifest"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("training.eval_iters=19",),
        )
    config = _resolved(
        "1b",
        "base",
        "stage1",
        extra=(
            "runtime.validation.manifest=/validation/manifest.txt",
            "training.eval_iters=19",
        ),
    )
    assert config["training"]["eval_iters"] == 19
    with pytest.raises(ConfigurationError, match="only for Stage 1"):
        _resolved(
            "1b",
            "base",
            "stage2",
            extra=(
                "runtime.validation.manifest=/validation/manifest.txt",
                "training.eval_iters=19",
            ),
        )


def test_optimizer_and_unknown_fields_fail_closed() -> None:
    with pytest.raises(ConfigurationError, match="must be adamw"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("optimization.optimizer=sgd",),
        )
    with pytest.raises(ConfigurationError, match="unsupported performance fields"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("performance.unsupported_mode=true",),
        )
    with pytest.raises(ConfigurationError, match="unsupported optimization fields"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("optimization.unsupported_mode=true",),
        )
    with pytest.raises(ConfigurationError, match="frozen token schedule"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("optimization.warmup_steps=999",),
        )
    with pytest.raises(ConfigurationError, match="gradients must remain FP32"):
        _resolved(
            "1b",
            "base",
            "stage1",
            extra=("performance.grad_reduce_dtype=bf16",),
        )
