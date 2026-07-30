from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .config import CONFIG_ROOT, PROJECT_ROOT
from .data_pipeline import sft_packed_work_dir
from .provenance import sha256


def build_environment(config: dict[str, Any]) -> dict[str, str]:
    perf = config["performance"]
    runtime = config["runtime"]
    pythonpath = [
        str(PROJECT_ROOT / "third_party" / "MindSpeed"),
        str(PROJECT_ROOT / "third_party" / "MindSpeed-LLM"),
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
    ]
    env = {
        "PYTHONPATH": ":".join(pythonpath),
        # Preserve the third-party checkouts physically as well as at the Git
        # level: imports must never create ignored __pycache__ files in them.
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRAINING_BACKEND": "mindspeed",
        "MODEL_USE_NULL_ATTENTION_MASK": "1",
        "MCORE_DIST_CKPT_NO_FORK": "1",
        "LM_LOSS_UPCAST": "0",
        "MEGATRON_OLMO3_WEIGHT_DECAY": "1",
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
        "OLMO3_DEFER_DP_LOSS_REPORTING": (
            "1" if perf.get("defer_dp_loss_reporting", True) else "0"
        ),
        "OLMO3_SCALAR_REDUCE_BACKEND": str(
            perf.get("scalar_reduce_backend", "hccl")
        ),
        "OLMO3_SWA_CP_MODE": str(perf.get("swa_cp_mode", "none")),
        "OLMO3_SWA_HALO_OVERLAP": (
            "1" if perf.get("swa_halo_overlap", False) else "0"
        ),
        "OLMO3_SWA_HALO_BACKWARD_OVERLAP": (
            "1" if perf.get("swa_halo_backward_overlap", False) else "0"
        ),
        "OLMO3_FUSED_QKV_A2A_PACKING": (
            "1" if perf.get("fused_qkv_a2a_packing", False) else "0"
        ),
        "OLMO3_HSDP_INTER_OVERLAP_BACKWARD": (
            "1" if perf.get("hsdp_inter_backward_overlap", False) else "0"
        ),
        "OLMO3_HSDP_INTER_TP_LANE_PACK": str(
            perf.get("hsdp_inter_tp_lane_pack", 1)
        ),
        "OLMO3_MC2_ALL_GATHER_RECOMPUTATION": (
            "1" if perf.get("mc2_all_gather_recomputation", False) else "0"
        ),
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "WANDB_RUN_ID": str(runtime["run_id"]),
        "WANDB_RESUME": "never" if runtime["lifecycle"] != "resume" else "allow",
    }
    if "hccl_if_base_port" in runtime:
        env["HCCL_IF_BASE_PORT"] = str(runtime["hccl_if_base_port"])
    if config["topology"]["tensor_parallel"] * config["topology"]["context_parallel"] > 1:
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    wandb = runtime.get("wandb", {})
    env["OLMO3_WANDB_LOG_STYLE"] = str(
        wandb.get("log_style", "llamafactory")
    )
    for source, target in (
        ("project", "WANDB_PROJECT"),
        ("entity", "WANDB_ENTITY"),
        ("base_url", "WANDB_BASE_URL"),
        ("save_dir", "WANDB_SAVE_DIR"),
    ):
        value = wandb.get(source)
        if value:
            env[target] = str(value)
    return env


def build_megatron_args(
    config: dict[str, Any], model_config_path: Path
) -> list[str]:
    model = config["model"]
    variant = config["variant"]
    stage = config["stage"]
    training = config["training"]
    optim = config["optimization"]
    data = config["data"]
    topology = config["topology"]
    perf = config["performance"]
    runtime = config["runtime"]
    hsdp = topology["hsdp"]

    args = [
        "--use-mcore-models",
        "--model-impl",
        variant["model_impl"],
        "--model-config",
        str(model_config_path),
        "--num-layers",
        str(model["num_layers"]),
        "--hidden-size",
        str(model["hidden_size"]),
        "--ffn-hidden-size",
        str(model["ffn_hidden_size"]),
        "--num-attention-heads",
        str(model["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups",
        str(model["num_query_groups"]),
        "--kv-channels",
        str(model["kv_channels"]),
        "--seq-length",
        str(stage["sequence_length"]),
        "--max-position-embeddings",
        str(stage["model_max_sequence_length"]),
        "--model-max-position-embeddings",
        str(stage["model_max_sequence_length"]),
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        str(model["rope_theta"]),
        "--rotary-percent",
        "1.0",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(model["rms_norm_eps"]),
        "--qk-layernorm",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--attention-dropout",
        str(model["attention_dropout"]),
        "--hidden-dropout",
        str(model["hidden_dropout"]),
        "--init-method-std",
        str(model["initializer_range"]),
        "--vocab-size",
        str(model["true_vocab_size"]),
        "--make-vocab-size-divisible-by",
        "128",
        "--micro-batch-size",
        str(training["micro_batch_size"]),
        "--global-batch-size",
        str(training["global_batch_size"]),
        "--train-iters",
        str(training["train_iters"]),
        "--lr-decay-iters",
        str(training["train_iters"]),
        "--lr-warmup-iters",
        str(optim["warmup_steps"]),
        "--optimizer",
        "adam",
        "--lr",
        str(optim["peak_lr"]),
        "--min-lr",
        str(optim["min_lr"]),
        "--lr-decay-style",
        str(optim["scheduler"]),
        "--weight-decay",
        str(optim["weight_decay"]),
        "--adam-beta1",
        str(optim["adam_beta1"]),
        "--adam-beta2",
        str(optim["adam_beta2"]),
        "--adam-eps",
        str(optim["adam_eps"]),
        "--clip-grad",
        str(optim["grad_clip"]),
        "--vocab-z-loss-coeff",
        str(optim["z_loss"]),
        "--bf16",
        "--accumulate-allreduce-grads-in-fp32",
        "--use-distributed-optimizer",
        "--calculate-per-token-loss",
        "--num-distributed-optimizer-instances",
        str(hsdp["num_instances"]),
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        "flash",
        "--use-flash-attn",
        "--use-fused-rmsnorm",
        "--use-fused-swiglu",
        "--use-fused-rotary-pos-emb",
        "--tensor-model-parallel-size",
        str(topology["tensor_parallel"]),
        "--pipeline-model-parallel-size",
        str(topology["pipeline_parallel"]),
        "--context-parallel-size",
        str(topology["context_parallel"]),
        "--context-parallel-algo",
        str(perf.get("context_parallel_algorithm", "ulysses_cp_algo")),
        "--dataset-backend",
        str(data["backend"]),
        "--data-cache-path",
        str(runtime.get("data_cache_path", Path(runtime["output"]) / "data-cache")),
        "--num-dataset-builder-threads",
        "4",
        "--no-create-attention-mask-in-dataloader",
        "--pad-token-id",
        str(data["tokenizer"]["pad_token_id"]),
        "--sampler-seed-mode",
        "megatron",
        "--sampler-data-seed",
        str(training["data_seed"]),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(data["tokenizer"]["path"]),
        "--dataloader-type",
        "single" if stage["name"] != "stage1" else "cyclic",
        "--num-workers",
        "4",
        "--attn-implementation",
        "flash_attention_2",
        "--no-model-bf16-autocast",
        "--mcore-native",
        "--seed",
        str(training["seed"]),
        "--log-interval",
        str(training.get("log_interval", 10)),
        "--eval-iters",
        str(training.get("eval_iters") or 0),
        "--eval-interval",
        str(training["eval_interval"]),
        "--save-interval",
        str(training["save_interval"]),
        "--save",
        str(runtime["save"]),
        "--ckpt-format",
        "torch_dist",
        "--dist-ckpt-strictness",
        "raise_all",
        "--experiment-output-dir",
        str(runtime["output"]),
    ]

    if training.get("exit_interval") is not None:
        args.extend(["--exit-interval", str(training["exit_interval"])])

    if perf.get("manual_gc", True):
        args.extend(
            ["--manual-gc", "--manual-gc-interval", str(perf.get("manual_gc_interval", 0))]
        )
    if topology.get("sequence_parallel"):
        args.append("--sequence-parallel")
    if perf.get("overlap_grad_reduce"):
        args.append("--overlap-grad-reduce")
    if perf.get("overlap_param_gather"):
        args.append("--overlap-param-gather")
    if perf.get("overlap_param_gather_with_optimizer_step"):
        args.append("--overlap-param-gather-with-optimizer-step")
    if hsdp.get("ddp_num_buckets", 0) > 0:
        args.extend(["--ddp-num-buckets", str(hsdp["ddp_num_buckets"])])
    if perf.get("scalar_reduce_backend") != "gloo":
        args.append("--disable-gloo-process-groups")

    if stage["activation_checkpoint"] == "mlp":
        args.extend(
            ["--recompute-granularity", "selective", "--recompute-modules", "mlp"]
        )
    if stage["packed"]:
        args.append("--olmo3-packed-documents")

    if data["backend"] == "megatron_indexed":
        args.extend(
            [
                "--data-args-path",
                str(data["data_args_path"]),
                "--split",
                str(data.get("split", "100,0,0")),
                "--no-mmap-bin-files",
            ]
        )
    elif data["backend"] == "olmo3_numpy_fsl":
        data_config_path = CONFIG_ROOT / "data" / f"{data['name']}.json"
        args.extend(
            [
                "--olmo3-data-config-name",
                str(data["name"]),
                "--olmo3-data-config-sha256",
                sha256(data_config_path),
                "--olmo3-fsl-data-root",
                str(data["root"]),
                "--olmo3-fsl-data-manifest",
                str(data["manifest"]),
                "--olmo3-fsl-work-dir",
                str(data["work_dir"]),
                "--olmo3-expected-data-tokens",
                str(data["known_token_count"]),
            ]
        )
    elif data["backend"] == "olmo3_numpy_packed":
        data_config_path = CONFIG_ROOT / "data" / f"{data['name']}.json"
        args.extend(
            [
                "--olmo3-data-config-name",
                str(data["name"]),
                "--olmo3-data-config-sha256",
                sha256(data_config_path),
                "--olmo3-packed-data-root",
                str(data["root"]),
                "--olmo3-packed-data-manifest",
                str(data["manifest"]),
                "--olmo3-packed-work-dir",
                str(data["work_dir"]),
                "--olmo3-expected-data-tokens",
                str(data["known_token_count"]),
            ]
        )
    elif data["backend"] == "olmo3_sft_numpy":
        args.extend(
            [
                "--olmo3-sft-profile",
                "think" if stage["name"] == "sft_think" else "instruct",
                "--olmo3-sft-data-dir",
                str(data["root"]),
                "--olmo3-sft-work-dir",
                str(sft_packed_work_dir(data["work_dir"])),
                "--olmo3-sft-expected-instances",
                str(data["expected_instances"]),
                "--olmo3-sft-expected-fingerprint",
                str(data["expected_fingerprint"]),
            ]
        )

    lifecycle = runtime["lifecycle"]
    if lifecycle in {"resume", "transition"}:
        args.extend(["--load", str(runtime["load"]), "--exit-on-missing-checkpoint"])
    if lifecycle == "transition":
        args.append("--olmo3-stage-transition")

    valid = runtime.get("validation", {})
    if valid.get("manifest"):
        args.extend(
            [
                "--valid-ppl-manifest",
                str(valid["manifest"]),
                "--valid-ppl-cache-dir",
                str(valid.get("cache_dir", Path(runtime["output"]) / "ppl-cache")),
            ]
        )
    wandb = runtime.get("wandb", {})
    if wandb.get("enabled"):
        args.extend(
            [
                "--use-wandb",
                "--wandb-project",
                str(wandb["project"]),
                "--wandb-exp-name",
                str(runtime["run_id"]),
                "--wandb-save-dir",
                str(wandb.get("save_dir", Path(runtime["output"]) / "wandb")),
            ]
        )
    return args


def build_torchrun_command(
    config: dict[str, Any], model_config_path: Path
) -> list[str]:
    runtime = config["runtime"]
    topology = config["topology"]
    perf = config["performance"]
    entry = (
        PROJECT_ROOT / "src" / "pretrain_olmo3_mindspeed_mc2.py"
        if perf.get("tensor_parallel_kernel") == "ascend_mc2"
        else PROJECT_ROOT / "src" / "pretrain_olmo3_mindspeed.py"
    )
    python = str(runtime.get("python", "python3"))
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(runtime.get("nnodes", topology["world_size"] // topology["nproc_per_node"])),
        "--node_rank",
        str(runtime.get("node_rank", 0)),
        "--nproc_per_node",
        str(topology["nproc_per_node"]),
        "--master_addr",
        str(runtime.get("master_addr", "127.0.0.1")),
        "--master_port",
        str(runtime.get("master_port", 29500)),
        "--rdzv_backend",
        "static",
        str(entry),
        *build_megatron_args(config, model_config_path),
    ]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)
