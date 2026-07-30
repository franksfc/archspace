from __future__ import annotations

import copy
import math
import re
from typing import Any

from .config import ConfigurationError


HCCL_IF_BASE_PORT_MIN = 1_024
# HCCL may bind one interface port per local device.  Reserving the final
# sixteen ports keeps a valid base usable on the project's 16-NPU nodes.
HCCL_IF_BASE_PORT_MAX = 65_520


def validate_hccl_if_base_port(value: Any) -> int:
    """Return a valid explicitly configured HCCL listener base port."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not HCCL_IF_BASE_PORT_MIN <= value <= HCCL_IF_BASE_PORT_MAX
    ):
        raise ConfigurationError(
            "runtime.hccl_if_base_port must be an integer in "
            f"[{HCCL_IF_BASE_PORT_MIN}, {HCCL_IF_BASE_PORT_MAX}], got {value!r}"
        )
    return value


_OFFICIAL_MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "1b": {
        "num_layers": 16,
        "hidden_size": 2_048,
        "ffn_hidden_size": 8_192,
        "num_attention_heads": 16,
        "num_query_groups": 16,
        "kv_channels": 128,
    },
    "3b": {
        "num_layers": 16,
        "hidden_size": 3_328,
        "ffn_hidden_size": 13_312,
        "num_attention_heads": 16,
        "num_query_groups": 16,
        "kv_channels": 208,
    },
    "7b": {
        "num_layers": 32,
        "hidden_size": 4_096,
        "ffn_hidden_size": 11_008,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "kv_channels": 128,
    },
}

_OFFICIAL_COMMON_MODEL_CONTRACT: dict[str, Any] = {
    "true_vocab_size": 100_278,
    "padded_vocab_size": 100_352,
    "tie_word_embeddings": False,
    "qk_norm": True,
    "qk_norm_mode": "full_projection",
    "rms_norm_eps": 1.0e-6,
    "rope_theta": 500_000,
    "rope_full_precision": True,
    "swa_window": 4_096,
    "swa_pattern": [4_096, 4_096, 4_096, -1],
    "initializer_range": 0.02,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
}

_STAGE_SEQUENCE_CONTRACTS: dict[str, tuple[int, int, bool]] = {
    "stage1": (8_192, 8_192, False),
    "stage2": (8_192, 8_192, False),
    "stage3": (65_536, 65_536, True),
    "sft_think": (32_768, 65_536, True),
    "sft_instruct": (32_768, 65_536, True),
}

_FULL_ATTENTION_YARN: dict[str, Any] = {
    "type": "yarn",
    "factor": 8.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 8_192,
}

_PRODUCTION_STAGE_TOKEN_COUNTS: dict[str, int] = {
    "dolmino_100b": 102_421_858_230,
    "longmino_50b": 55_279_928_852,
}

_STAGE_RECIPE_CONTRACTS: dict[str, dict[str, dict[str, Any]]] = {
    "stage1": {
        "stage": {
            "assistant_only_loss": False,
            "full_attention_yarn": None,
            "activation_checkpoint": "none",
            "checkpoint_lifecycle": "fresh_or_resume",
        },
        "training": {"seed": 12_536, "data_seed": 34_521},
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "z_loss": 1.0e-5,
            "grad_clip": 1.0,
        },
    },
    "stage2": {
        "stage": {
            "assistant_only_loss": False,
            "full_attention_yarn": None,
            "activation_checkpoint": "none",
            "checkpoint_lifecycle": "transition_or_resume",
        },
        "training": {"seed": 1_337, "data_seed": 1_337},
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "linear",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "z_loss": 1.0e-5,
            "grad_clip": 1.0,
        },
    },
    "stage3": {
        "stage": {
            "assistant_only_loss": False,
            "full_attention_yarn": _FULL_ATTENTION_YARN,
            "activation_checkpoint": "none",
            "checkpoint_lifecycle": "transition_or_resume",
        },
        "training": {"seed": 4_123, "data_seed": 4_123},
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "linear",
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "z_loss": 1.0e-5,
            "grad_clip": 1.0,
        },
    },
    "sft_think": {
        "stage": {
            "assistant_only_loss": True,
            "full_attention_yarn": _FULL_ATTENTION_YARN,
            "activation_checkpoint": "mlp",
            "checkpoint_lifecycle": "transition_or_resume",
        },
        "training": {"seed": 53_184, "data_seed": 34_521},
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "linear",
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "z_loss": 0.0,
            "grad_clip": 1.0,
        },
    },
    "sft_instruct": {
        "stage": {
            "assistant_only_loss": True,
            "full_attention_yarn": _FULL_ATTENTION_YARN,
            "activation_checkpoint": "mlp",
            "checkpoint_lifecycle": "transition_or_resume",
        },
        "training": {"seed": 543_210, "data_seed": 34_521},
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "linear",
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "z_loss": 0.0,
            "grad_clip": 1.0,
        },
    },
}

_PERFORMANCE_PROFILE_NAMES = frozenset(
    {"ordinary_hsdp", "tp2_sio_mc2", "cp_single_halo"}
)

_PERFORMANCE_FIELDS = frozenset(
    {
        "profiles",
        "tensor_parallel_kernel",
        "mc2_all_gather_recomputation",
        "require_tp_same_physical_card",
        "require_tp_link",
        "hsdp_inter_backward_overlap",
        "hsdp_inter_tp_lane_pack",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
        "grad_reduce_dtype",
        "manual_gc",
        "manual_gc_interval",
        "defer_dp_loss_reporting",
        "scalar_reduce_backend",
        "context_parallel_algorithm",
        "swa_cp_mode",
        "swa_halo_overlap",
        "swa_halo_backward_overlap",
        "fused_qkv_a2a_packing",
        "require_cp_within_node",
    }
)

_PERFORMANCE_BOOLEAN_FIELDS = frozenset(
    {
        "mc2_all_gather_recomputation",
        "require_tp_same_physical_card",
        "hsdp_inter_backward_overlap",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
        "manual_gc",
        "defer_dp_loss_reporting",
        "swa_halo_overlap",
        "swa_halo_backward_overlap",
        "fused_qkv_a2a_packing",
        "require_cp_within_node",
    }
)

_OPTIMIZATION_FIELDS = frozenset(
    {
        "optimizer",
        "scheduler",
        "peak_lr",
        "min_lr",
        "warmup_tokens",
        "warmup_fraction",
        "warmup_steps",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "z_loss",
        "grad_clip",
    }
)

_SAFE_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _positive_int(value: Any, name: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} must be a positive integer, got {value!r}")
        return 1
    return value


def _required(value: Any, name: str, errors: list[str]) -> Any:
    if value is None or value == "":
        errors.append(f"{name} is required")
    return value


def _required_string(value: Any, name: str, errors: list[str]) -> str:
    _required(value, name, errors)
    if not isinstance(value, str):
        errors.append(f"{name} must be a string, got {value!r}")
        return ""
    return value


def resolve(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    errors: list[str] = []

    if not isinstance(result, dict):
        raise ConfigurationError("configuration must be a JSON object")
    for section_name in (
        "model",
        "variant",
        "stage",
        "training",
        "optimization",
        "data",
        "topology",
    ):
        if not isinstance(result.get(section_name), dict):
            errors.append(f"{section_name} must be an object")
    for section_name in ("performance", "runtime"):
        value = result.get(section_name)
        if value is None:
            result[section_name] = {}
        elif not isinstance(value, dict):
            errors.append(f"{section_name} must be an object")
    if errors:
        raise ConfigurationError(
            "configuration is invalid:\n- " + "\n- ".join(errors)
        )

    model = result["model"]
    variant = result["variant"]
    stage = result["stage"]
    training = result["training"]
    optim = result["optimization"]
    data = result["data"]
    topology = result["topology"]
    perf = result.setdefault("performance", {})
    runtime = result.setdefault("runtime", {})

    profiles = perf.get("profiles")
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(name, str) for name in profiles)
        or profiles != sorted(set(profiles))
    ):
        errors.append(
            "performance.profiles must be a non-empty, sorted list of unique names"
        )
        profile_names: set[str] = set()
    else:
        profile_names = set(profiles)
        unknown_profiles = sorted(profile_names - _PERFORMANCE_PROFILE_NAMES)
        if unknown_profiles:
            errors.append(f"unsupported performance profiles: {unknown_profiles}")

    unknown_performance_fields = sorted(set(perf) - _PERFORMANCE_FIELDS)
    if unknown_performance_fields:
        errors.append(
            f"unsupported performance fields: {unknown_performance_fields}"
        )
    unknown_optimization_fields = sorted(set(optim) - _OPTIMIZATION_FIELDS)
    if unknown_optimization_fields:
        errors.append(
            f"unsupported optimization fields: {unknown_optimization_fields}"
        )
    for field in sorted(_PERFORMANCE_BOOLEAN_FIELDS):
        if field in perf and not isinstance(perf[field], bool):
            errors.append(
                f"performance.{field} must be a boolean, got {perf[field]!r}"
            )
    if perf.get("tensor_parallel_kernel") not in {None, "ascend_mc2"}:
        errors.append(
            "performance.tensor_parallel_kernel must be null or ascend_mc2"
        )
    if perf.get("swa_cp_mode", "none") not in {"none", "single_halo"}:
        errors.append(
            "performance.swa_cp_mode must be none or single_halo"
        )
    if perf.get("scalar_reduce_backend", "hccl") not in {"hccl", "gloo"}:
        errors.append(
            "performance.scalar_reduce_backend must be hccl or gloo"
        )
    if perf.get("grad_reduce_dtype", "fp32") != "fp32":
        errors.append("OLMo 3 production gradients must remain FP32")
    if (
        perf.get("tensor_parallel_kernel") != "ascend_mc2"
        and perf.get("hsdp_inter_tp_lane_pack", 1) != 1
    ):
        errors.append("TP-lane pack is supported only by the TP2 MC2 profile")
    if (
        perf.get("tensor_parallel_kernel") != "ascend_mc2"
        and perf.get("hsdp_inter_backward_overlap", False)
    ):
        errors.append(
            "HSDP inter-backward overlap is supported only by the TP2 MC2 profile"
        )

    model_size = model.get("size")
    model_contract = _OFFICIAL_MODEL_CONTRACTS.get(model_size)
    if model_contract is None:
        errors.append(
            "model.size must be one of "
            f"{sorted(_OFFICIAL_MODEL_CONTRACTS)}, got {model_size!r}"
        )
    else:
        for field, expected in {
            **model_contract,
            **_OFFICIAL_COMMON_MODEL_CONTRACT,
        }.items():
            if model.get(field) != expected:
                errors.append(
                    f"model.{field} does not match official OLMo 3 {model_size}: "
                    f"expected={expected!r}, got={model.get(field)!r}"
                )

    stage_name = stage.get("name")
    stage_sequence_contract = _STAGE_SEQUENCE_CONTRACTS.get(stage_name)
    if stage_sequence_contract is not None:
        expected_sequence, expected_max_sequence, expected_packed = (
            stage_sequence_contract
        )
        actual_stage_contract = (
            stage.get("sequence_length"),
            stage.get("model_max_sequence_length"),
            stage.get("packed"),
        )
        if actual_stage_contract != stage_sequence_contract:
            errors.append(
                f"{stage_name} sequence contract must be "
                f"(sequence={expected_sequence}, max={expected_max_sequence}, "
                f"packed={expected_packed}), got {actual_stage_contract!r}"
            )
    recipe_contract = _STAGE_RECIPE_CONTRACTS.get(stage_name)
    if recipe_contract is not None:
        for section_name, expected_fields in recipe_contract.items():
            actual_section = result.get(section_name)
            if not isinstance(actual_section, dict):
                errors.append(f"{section_name} must be an object")
                continue
            for field, expected in expected_fields.items():
                actual = actual_section.get(field)
                if actual != expected:
                    errors.append(
                        f"{stage_name} requires {section_name}.{field}="
                        f"{expected!r}, got {actual!r}"
                    )

    world = _positive_int(topology.get("world_size"), "topology.world_size", errors)
    tp = _positive_int(
        topology.get("tensor_parallel"), "topology.tensor_parallel", errors
    )
    cp = _positive_int(
        topology.get("context_parallel"), "topology.context_parallel", errors
    )
    pp = _positive_int(
        topology.get("pipeline_parallel"), "topology.pipeline_parallel", errors
    )
    if pp != 1:
        errors.append(
            "pipeline parallelism is fail-closed at PP=1; cross-layer "
            "Siamese/Depth state has not been implemented across PP stages"
        )
    model_parallel = tp * cp * pp
    if world % model_parallel:
        errors.append(
            f"world_size={world} is not divisible by TP*CP*PP={model_parallel}"
        )
        dp = 1
    else:
        dp = world // model_parallel
    topology["data_parallel"] = dp

    nproc = _positive_int(
        topology.get("nproc_per_node"), "topology.nproc_per_node", errors
    )
    sequence_parallel = topology.get("sequence_parallel")
    if not isinstance(sequence_parallel, bool):
        errors.append(
            "topology.sequence_parallel must be a boolean, "
            f"got {sequence_parallel!r}"
        )
    if topology.get("rank_order") != "tp-cp-dp-pp":
        errors.append(
            "topology.rank_order must be tp-cp-dp-pp so TP*CP groups are "
            "contiguous inside each node"
        )
    if world % nproc:
        errors.append(
            f"world_size={world} must be divisible by nproc_per_node={nproc}"
        )
        derived_nnodes = 1
    else:
        derived_nnodes = world // nproc
    configured_nnodes = runtime.get("nnodes")
    if configured_nnodes is None:
        runtime["nnodes"] = derived_nnodes
    else:
        configured_nnodes = _positive_int(
            configured_nnodes, "runtime.nnodes", errors
        )
        runtime["nnodes"] = configured_nnodes
        if configured_nnodes * nproc != world:
            errors.append(
                f"runtime.nnodes*nproc_per_node={configured_nnodes}*{nproc} "
                f"does not equal world_size={world}"
            )
    node_rank = runtime.get("node_rank", 0)
    if isinstance(node_rank, bool) or not isinstance(node_rank, int) or node_rank < 0:
        errors.append(f"runtime.node_rank must be a non-negative integer, got {node_rank!r}")
    elif node_rank >= int(runtime["nnodes"]):
        errors.append(
            f"runtime.node_rank={node_rank} must be smaller than "
            f"runtime.nnodes={runtime['nnodes']}"
        )
    runtime["node_rank"] = node_rank
    master_port = runtime.get("master_port", 29500)
    if (
        isinstance(master_port, bool)
        or not isinstance(master_port, int)
        or not 1 <= master_port <= 65535
    ):
        errors.append(f"runtime.master_port must be in [1, 65535], got {master_port!r}")
    if "hccl_if_base_port" in runtime:
        try:
            runtime["hccl_if_base_port"] = validate_hccl_if_base_port(
                runtime["hccl_if_base_port"]
            )
        except ConfigurationError as exc:
            errors.append(str(exc))
    _required_string(
        runtime.get("master_addr", "127.0.0.1"),
        "runtime.master_addr",
        errors,
    )
    seq = _positive_int(stage.get("sequence_length"), "stage.sequence_length", errors)
    max_seq = _positive_int(
        stage.get("model_max_sequence_length"),
        "stage.model_max_sequence_length",
        errors,
    )
    if max_seq < seq:
        errors.append("model_max_sequence_length must be >= sequence_length")
    if seq % cp:
        errors.append(f"sequence_length={seq} must be divisible by CP={cp}")
    local_seq = seq // cp if seq % cp == 0 else 0

    gbs = training.get("global_batch_size")
    if gbs is None:
        gbs = topology.get("global_batch_size")
    mbs = training.get("micro_batch_size")
    if mbs is None:
        mbs = topology.get("micro_batch_size")
    gbs = _positive_int(gbs, "training.global_batch_size", errors)
    mbs = _positive_int(mbs, "training.micro_batch_size", errors)
    training["global_batch_size"] = gbs
    training["micro_batch_size"] = mbs
    topology["global_batch_size"] = gbs
    topology["micro_batch_size"] = mbs

    denominator = dp * mbs
    if gbs % denominator:
        errors.append(
            f"GBS={gbs} must be divisible by DP*MBS={dp}*{mbs}={denominator}; "
            "fractional gradient accumulation is forbidden"
        )
        ga = 0
    else:
        ga = gbs // denominator
    if ga < 1:
        errors.append(f"gradient accumulation must be >=1, got {ga}")
    training["gradient_accumulation"] = ga
    tokens_per_step = gbs * seq
    training["tokens_per_step"] = tokens_per_step

    train_tokens = training.get("train_tokens")
    epochs = training.get("epochs")
    if train_tokens is not None and epochs is not None:
        errors.append("training.train_tokens and training.epochs are mutually exclusive")
    if train_tokens is not None:
        train_tokens = _positive_int(
            train_tokens, "training.train_tokens", errors
        )
        train_iters = math.ceil(train_tokens / tokens_per_step)
        training["scheduled_tokens"] = train_iters * tokens_per_step
        training["token_overshoot"] = train_iters * tokens_per_step - train_tokens
    elif epochs is not None:
        instances = data.get("expected_instances")
        instances = _positive_int(
            instances, "data.expected_instances (required for epoch SFT)", errors
        )
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, (int, float))
            or not math.isfinite(float(epochs))
            or epochs <= 0
        ):
            errors.append(f"training.epochs must be positive, got {epochs!r}")
            epochs = 1
        instances_per_epoch = (instances // gbs) * gbs
        if instances_per_epoch < 1:
            errors.append(
                "SFT data must contain at least one complete global batch"
            )
            instances_per_epoch = gbs
        train_iters = math.ceil(instances_per_epoch * float(epochs) / gbs)
        training["instances_per_epoch"] = instances_per_epoch
        training["requested_instances"] = instances_per_epoch * float(epochs)
        training["scheduled_instances"] = train_iters * gbs
    else:
        errors.append("one of training.train_tokens or training.epochs is required")
        train_iters = 1
    training["train_iters"] = train_iters

    warmup_tokens = optim.get("warmup_tokens")
    warmup_fraction = optim.get("warmup_fraction")
    canonical_fraction_warmup = False
    if warmup_tokens is not None and warmup_fraction is not None:
        # ``resolve`` freezes the token-equivalent of a percentage warmup in
        # the resolved contract.  Worker and checkpoint preflights re-run this
        # function on that contract, so recognize only the exact canonical
        # triple produced below.  A user-supplied fraction plus token budget
        # still fails closed because an unresolved composition has no matching
        # derived ``warmup_steps``.
        fraction_is_valid = (
            not isinstance(warmup_fraction, bool)
            and isinstance(warmup_fraction, (int, float))
            and math.isfinite(float(warmup_fraction))
            and 0 <= warmup_fraction < 1
        )
        tokens_are_valid = (
            not isinstance(warmup_tokens, bool)
            and isinstance(warmup_tokens, int)
            and warmup_tokens >= 0
        )
        if fraction_is_valid and tokens_are_valid:
            fraction_steps = math.ceil(train_iters * float(warmup_fraction))
            canonical_fraction_warmup = (
                optim.get("warmup_steps") == fraction_steps
                and warmup_tokens == fraction_steps * tokens_per_step
            )
        if not canonical_fraction_warmup:
            errors.append(
                "optimization.warmup_tokens and warmup_fraction are mutually exclusive"
            )
    if canonical_fraction_warmup:
        warmup_steps = math.ceil(train_iters * float(warmup_fraction))
    elif warmup_tokens is not None:
        if not isinstance(warmup_tokens, int) or warmup_tokens < 0:
            errors.append("optimization.warmup_tokens must be a non-negative integer")
            warmup_steps = 0
        else:
            warmup_steps = math.ceil(warmup_tokens / tokens_per_step)
    elif warmup_fraction is not None:
        if (
            isinstance(warmup_fraction, bool)
            or not isinstance(warmup_fraction, (int, float))
            or not math.isfinite(float(warmup_fraction))
            or not 0 <= warmup_fraction < 1
        ):
            errors.append("optimization.warmup_fraction must be in [0, 1)")
            warmup_steps = 0
        else:
            warmup_steps = math.ceil(train_iters * float(warmup_fraction))
            optim["warmup_tokens"] = warmup_steps * tokens_per_step
    else:
        errors.append(
            "one of optimization.warmup_tokens or warmup_fraction is required"
        )
        warmup_steps = 0
    frozen_warmup_steps = optim.get("warmup_steps")
    if frozen_warmup_steps is not None and (
        isinstance(frozen_warmup_steps, bool)
        or not isinstance(frozen_warmup_steps, int)
        or frozen_warmup_steps != warmup_steps
    ):
        errors.append(
            "optimization.warmup_steps must equal the value derived from "
            "the frozen token schedule"
        )
    optim["warmup_steps"] = warmup_steps
    if optim.get("optimizer") != "adamw":
        errors.append(
            "optimization.optimizer must be adamw (Megatron --optimizer adam "
            "uses AdamW semantics)"
        )
    peak_lr = _required(optim.get("peak_lr"), "optimization.peak_lr", errors)
    min_lr = _required(optim.get("min_lr"), "optimization.min_lr", errors)
    if (
        not isinstance(peak_lr, (int, float))
        or isinstance(peak_lr, bool)
        or not math.isfinite(float(peak_lr))
        or peak_lr <= 0
    ):
        errors.append(f"optimization.peak_lr must be positive, got {peak_lr!r}")
    if (
        not isinstance(min_lr, (int, float))
        or isinstance(min_lr, bool)
        or not math.isfinite(float(min_lr))
        or min_lr < 0
    ):
        errors.append(f"optimization.min_lr must be non-negative, got {min_lr!r}")
    elif (
        isinstance(peak_lr, (int, float))
        and not isinstance(peak_lr, bool)
        and math.isfinite(float(peak_lr))
        and isinstance(min_lr, (int, float))
        and not isinstance(min_lr, bool)
        and math.isfinite(float(min_lr))
        and min_lr > peak_lr
    ):
        errors.append("optimization.min_lr must not exceed optimization.peak_lr")
    if warmup_steps > train_iters:
        errors.append(
            f"warmup_steps={warmup_steps} must not exceed train_iters={train_iters}"
        )
    if optim.get("scheduler") not in {
        "constant",
        "linear",
        "cosine",
        "inverse-square-root",
        "WSD",
    }:
        errors.append(f"unsupported optimization.scheduler={optim.get('scheduler')!r}")
    for beta_name in ("adam_beta1", "adam_beta2"):
        beta = optim.get(beta_name)
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not 0 <= beta < 1
        ):
            errors.append(f"optimization.{beta_name} must be in [0, 1)")
    for positive_name in ("adam_eps", "grad_clip"):
        value = optim.get(positive_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            errors.append(f"optimization.{positive_name} must be positive")
    for nonnegative_name in ("weight_decay", "z_loss"):
        value = optim.get(nonnegative_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            errors.append(f"optimization.{nonnegative_name} must be non-negative")
    hsdp = topology.setdefault("hsdp", {})
    if hsdp.get("mode", "ordinary") != "ordinary":
        errors.append("only ordinary HSDP is retained in the production pipeline")
    shard_size = _positive_int(hsdp.get("shard_size"), "hsdp.shard_size", errors)
    dp_cp = dp * cp
    if dp_cp % shard_size:
        errors.append(
            f"DP*CP={dp_cp} must be divisible by HSDP shard_size={shard_size}"
        )
        instances = 1
    else:
        instances = dp_cp // shard_size
    hsdp["num_instances"] = instances
    hsdp["replica_count"] = instances
    hsdp["dp_cp_size"] = dp_cp
    _positive_int(hsdp.get("ddp_num_buckets"), "hsdp.ddp_num_buckets", errors)

    heads = _positive_int(
        model.get("num_attention_heads"), "model.num_attention_heads", errors
    )
    if heads % tp:
        errors.append(f"attention heads={heads} must be divisible by TP={tp}")
    groups = _positive_int(
        model.get("num_query_groups"), "model.num_query_groups", errors
    )
    if groups % tp:
        errors.append(f"query groups={groups} must be divisible by TP={tp}")
    swa_window = _positive_int(
        model.get("swa_window"), "model.swa_window", errors
    )

    if cp > 1:
        if heads % (tp * cp) or groups % (tp * cp):
            errors.append(
                "Full-attention Ulysses requires attention and query-group "
                f"heads ({heads}/{groups}) divisible by TP*CP={tp * cp}"
            )
        if perf.get("swa_cp_mode") != "single_halo":
            errors.append(
                "CP>1 requires the retained cp_single_halo performance profile"
            )
        if "cp_single_halo" not in profile_names:
            errors.append("CP>1 requires profile cp_single_halo")
        for field, expected in {
            "context_parallel_algorithm": "ulysses_cp_algo",
            "swa_halo_overlap": True,
            "swa_halo_backward_overlap": True,
            "fused_qkv_a2a_packing": True,
            "require_cp_within_node": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "grad_reduce_dtype": "fp32",
        }.items():
            if perf.get(field) != expected:
                errors.append(
                    f"CP>1 production profile requires performance.{field}="
                    f"{expected!r}"
                )
        if perf.get("require_cp_within_node") and (
            tp * cp > nproc or nproc % (tp * cp)
        ):
            errors.append(
                f"node-local TP*CP groups require TP*CP={tp*cp} to tile "
                f"nproc_per_node={nproc}"
            )
        if local_seq < swa_window - 1:
            errors.append(
                f"single-hop halo requires local sequence {local_seq} >= "
                f"SWA window-1={swa_window - 1}"
            )
    elif perf.get("swa_cp_mode", "none") != "none":
        errors.append("CP1 requires performance.swa_cp_mode=none")

    if perf.get("tensor_parallel_kernel") == "ascend_mc2":
        if "tp2_sio_mc2" not in profile_names:
            errors.append("Ascend MC2 requires profile tp2_sio_mc2")
        if tp != 2:
            errors.append("tp2_sio_mc2 requires tensor_parallel=2")
        if not topology.get("sequence_parallel"):
            errors.append("Ascend MC2 requires sequence_parallel=true")
        if perf.get("mc2_all_gather_recomputation") is not True:
            errors.append(
                "Ascend MC2 requires all-gather recomputation during backward"
            )
        if perf.get("hsdp_inter_tp_lane_pack") != 2:
            errors.append("the TP2 MC2 profile requires TP-lane pack2")
        if not perf.get("overlap_grad_reduce"):
            errors.append("TP-lane pack2 requires overlap_grad_reduce=true")
        if perf.get("grad_reduce_dtype") != "fp32":
            errors.append("TP-lane pack2 was validated only with FP32 gradients")
        if perf.get("require_tp_same_physical_card") and nproc != 16:
            errors.append(
                "the TP2 SIO mapping requires nproc_per_node=16"
            )
        elif nproc % tp:
            errors.append(
                f"node-local TP groups require TP={tp} to tile nproc_per_node={nproc}"
            )
    if "tp2_sio_mc2" in profile_names:
        for field, expected in {
            "tensor_parallel_kernel": "ascend_mc2",
            "mc2_all_gather_recomputation": True,
            "require_tp_same_physical_card": True,
            "require_tp_link": "SIO",
            "hsdp_inter_backward_overlap": True,
            "hsdp_inter_tp_lane_pack": 2,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "overlap_param_gather_with_optimizer_step": False,
            "grad_reduce_dtype": "fp32",
        }.items():
            if perf.get(field) != expected:
                errors.append(
                    f"tp2_sio_mc2 requires performance.{field}="
                    f"{expected!r}, got {perf.get(field)!r}"
                )

    if data.get("stage") != stage.get("name"):
        errors.append(
            f"data profile stage={data.get('stage')!r} does not match "
            f"stage={stage.get('name')!r}"
        )
    attention_packed = data.get("backend") in {
        "olmo3_numpy_packed",
        "olmo3_sft_numpy",
    }
    if attention_packed != bool(stage.get("packed")):
        errors.append("stage.packed and data packing contract disagree")
    for identity_name, value, choices in (
        ("model.size", model.get("size"), {"1b", "3b", "7b"}),
        (
            "variant.name",
            variant.get("name"),
            {"base", "siamese_depth"},
        ),
        (
            "stage.name",
            stage.get("name"),
            {"stage1", "stage2", "stage3", "sft_think", "sft_instruct"},
        ),
    ):
        if value not in choices:
            errors.append(
                f"{identity_name} must be one of {sorted(choices)}, got {value!r}"
            )
    data_name = _required_string(data.get("name"), "data.name", errors)
    if (
        data_name in {".", ".."}
        or "/" in data_name
        or "\\" in data_name
    ):
        errors.append(f"data.name is unsafe: {data_name!r}")
    data_scope = data.get("scope", "production")
    if data_scope != "production":
        errors.append(
            f"data.scope must be production, got {data_scope!r}"
        )
    _required_string(data.get("root"), "data.root", errors)
    tokenizer = data.get("tokenizer", {})
    _required_string(tokenizer.get("path"), "data.tokenizer.path", errors)
    expected_tokenizer = {
        "eos_token_id": 100_257,
        "pad_token_id": 100_277,
        "bos_token_id": None,
    }
    for field, expected in expected_tokenizer.items():
        if tokenizer.get(field) != expected:
            errors.append(
                f"data.tokenizer.{field} must match the OLMo 3 tokenizer "
                f"contract: expected={expected!r}, got={tokenizer.get(field)!r}"
            )
    if data.get("backend") == "megatron_indexed":
        _required_string(
            data.get("data_args_path"),
            "data.data_args_path",
            errors,
        )
    elif data.get("backend") in {"olmo3_numpy_fsl", "olmo3_numpy_packed"}:
        _required_string(data.get("manifest"), "data.manifest", errors)
        _required_string(data.get("work_dir"), "data.work_dir", errors)
        token_count_policy = data.get("token_count_policy", "fixed")
        if token_count_policy != "fixed":
            errors.append(
                "production Stage 2/3 profiles must use "
                "data.token_count_policy='fixed'"
            )
        known_token_count = data.get("known_token_count")
        if (
            isinstance(known_token_count, bool)
            or not isinstance(known_token_count, int)
            or known_token_count <= 0
        ):
            errors.append(
                "Stage 2/3 data.known_token_count must be a positive integer"
            )
        production_token_count = _PRODUCTION_STAGE_TOKEN_COUNTS.get(data_name)
        if (
            production_token_count is not None
            and known_token_count != production_token_count
        ):
            errors.append(
                f"{data_name} must retain its fixed production token count "
                f"{production_token_count}, got {known_token_count!r}"
            )
    elif data.get("backend") == "olmo3_sft_numpy":
        _required_string(data.get("work_dir"), "data.work_dir", errors)
        expected_fingerprint = _required_string(
            data.get("expected_fingerprint"),
            "data.expected_fingerprint",
            errors,
        )
        if re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint) is None:
            errors.append(
                "data.expected_fingerprint must be a 64-character lowercase "
                "SHA-256 hexadecimal digest"
            )

    lifecycle = runtime.get("lifecycle", "fresh")
    if lifecycle not in {"fresh", "resume", "transition"}:
        errors.append("runtime.lifecycle must be fresh, resume, or transition")
    if lifecycle in {"resume", "transition"}:
        _required_string(runtime.get("load"), "runtime.load", errors)
    if lifecycle == "fresh" and stage.get("name") != "stage1":
        errors.append(
            f"{stage.get('name')} must use transition or resume, not fresh"
        )
    allowed_lifecycles = (
        {"fresh", "resume"} if stage_name == "stage1" else {"transition", "resume"}
    )
    if stage_name in _STAGE_RECIPE_CONTRACTS and lifecycle not in allowed_lifecycles:
        errors.append(
            f"{stage_name} lifecycle must be one of "
            f"{sorted(allowed_lifecycles)}, got {lifecycle!r}"
        )
    _required_string(runtime.get("save"), "runtime.save", errors)
    _required_string(runtime.get("output"), "runtime.output", errors)
    data_cache_path = runtime.get("data_cache_path")
    if data_cache_path is not None and (
        not isinstance(data_cache_path, str) or not data_cache_path
    ):
        errors.append(
            "runtime.data_cache_path must be a non-empty string when supplied"
        )
    runtime_python = runtime.get("python")
    if runtime_python is not None and (
        not isinstance(runtime_python, str) or not runtime_python
    ):
        errors.append(
            f"runtime.python must be a non-empty string, got {runtime_python!r}"
        )
    run_id = _required_string(runtime.get("run_id"), "runtime.run_id", errors)
    if (
        _SAFE_RUNTIME_ID.fullmatch(run_id) is None
        or len(run_id.encode("utf-8")) > 255
    ):
        errors.append(
            "runtime.run_id is unsafe; use 1-255 bytes of ASCII letters, "
            f"digits, '.', '_' or '-', starting with a letter or digit: {run_id!r}"
        )
    validation = runtime.get("validation", {})
    if not isinstance(validation, dict):
        errors.append("runtime.validation must be an object")
    else:
        valid_manifest = validation.get("manifest")
        valid_cache_dir = validation.get("cache_dir")
        if valid_manifest is not None and (
            not isinstance(valid_manifest, str) or not valid_manifest
        ):
            errors.append("runtime.validation.manifest must be a non-empty string")
        if valid_cache_dir is not None and (
            not isinstance(valid_cache_dir, str) or not valid_cache_dir
        ):
            errors.append("runtime.validation.cache_dir must be a non-empty string")
        if valid_cache_dir and not valid_manifest:
            errors.append(
                "runtime.validation.cache_dir requires runtime.validation.manifest"
            )
    wandb = runtime.get("wandb", {})
    if not isinstance(wandb, dict):
        errors.append("runtime.wandb must be an object")
    else:
        enabled = wandb.get("enabled", False)
        if not isinstance(enabled, bool):
            errors.append(
                f"runtime.wandb.enabled must be a boolean, got {enabled!r}"
            )
        log_style = wandb.get("log_style", "llamafactory")
        if log_style not in {"llamafactory", "native"}:
            errors.append(
                "runtime.wandb.log_style must be 'llamafactory' or 'native'"
            )
        if wandb.get("enabled"):
            _required_string(wandb.get("project"), "runtime.wandb.project", errors)
    for interval_name in ("save_interval", "eval_interval", "log_interval"):
        value = training.get(interval_name)
        if interval_name == "log_interval" and value is None:
            continue
        _positive_int(value, f"training.{interval_name}", errors)
    exit_interval = training.get("exit_interval")
    if exit_interval is not None:
        exit_interval = _positive_int(
            exit_interval, "training.exit_interval", errors
        )
        if (
            isinstance(exit_interval, int)
            and not isinstance(exit_interval, bool)
            and exit_interval > train_iters
        ):
            errors.append(
                "training.exit_interval must not exceed "
                f"train_iters={train_iters}, got {exit_interval}"
            )
    eval_iters = training.get("eval_iters")
    if (
        eval_iters is not None
        and (
            isinstance(eval_iters, bool)
            or not isinstance(eval_iters, int)
            or eval_iters < 0
        )
    ):
        errors.append("training.eval_iters must be a non-negative integer or null")
    valid_manifest = (
        validation.get("manifest") if isinstance(validation, dict) else None
    )
    if valid_manifest and (
        isinstance(eval_iters, bool)
        or not isinstance(eval_iters, int)
        or eval_iters <= 0
    ):
        errors.append(
            "runtime.validation.manifest requires training.eval_iters > 0"
        )
    if isinstance(eval_iters, int) and not isinstance(eval_iters, bool):
        if eval_iters > 0 and not valid_manifest:
            errors.append(
                "training.eval_iters > 0 requires runtime.validation.manifest"
            )
    if stage.get("name") != "stage1" and (
        valid_manifest
        or (
            isinstance(eval_iters, int)
            and not isinstance(eval_iters, bool)
            and eval_iters > 0
        )
    ):
        errors.append(
            "independent PPL validation is supported only for Stage 1; "
            "Stage 2, Stage 3, and SFT require training.eval_iters=0 "
            "without runtime.validation.manifest"
        )

    if stage.get("full_attention_yarn") is not None:
        yarn = stage["full_attention_yarn"]
        if yarn != _FULL_ATTENTION_YARN:
            errors.append("Stage3/SFT Full-attention YaRN must match OLMo 3")
    if model.get("rope_theta") != 500000:
        errors.append("OLMo 3 rope_theta must be exactly 500000")
    if not model.get("rope_full_precision"):
        errors.append("OLMo 3 requires full-precision RoPE application")

    if variant.get("name") == "base":
        expected_variant = {
            "model_impl": "olmo3",
            "use_siamese_norm": False,
            "siamese_norm_variant": None,
            "use_depth_attention": False,
            "depth_attention_stride": None,
            "depth_attention_recent_window": 0,
            "checkpoint_keyspace": "olmo3_base_v1",
        }
        mismatches = {
            field: (variant.get(field), expected)
            for field, expected in expected_variant.items()
            if variant.get(field) != expected
        }
        if mismatches:
            errors.append(
                "base variant cannot enable Siamese Norm or Depth Attention "
                f"and must keep its checkpoint keyspace: {mismatches}"
            )
    elif variant.get("name") == "siamese_depth":
        expected_variant = {
            "model_impl": "olmo3_siamese_depth",
            "use_siamese_norm": True,
            "siamese_norm_variant": "hybrid_pre",
            "use_depth_attention": True,
            "depth_attention_stride": 8,
            "depth_attention_recent_window": 0,
            "checkpoint_keyspace": "olmo3_siamese_depth_v1",
        }
        mismatches = {
            field: (variant.get(field), expected)
            for field, expected in expected_variant.items()
            if variant.get(field) != expected
        }
        if mismatches:
            errors.append(
                "siamese_depth variant requires the complete Siamese Norm + "
                f"Depth Attention contract: {mismatches}"
            )
    else:
        errors.append(f"unknown model variant {variant.get('name')!r}")

    result["derived"] = {
        "model_parallel_size": model_parallel,
        "data_parallel_size": dp,
        "gradient_accumulation": ga,
        "local_sequence_length": local_seq,
        "tokens_per_step": tokens_per_step,
        "train_iters": train_iters,
        "warmup_steps": warmup_steps,
        "num_distributed_optimizer_instances": instances,
        "nnodes": derived_nnodes,
    }
    if errors:
        raise ConfigurationError("configuration is invalid:\n- " + "\n- ".join(errors))
    return result
