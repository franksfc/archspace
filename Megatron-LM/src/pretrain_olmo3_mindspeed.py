#!/usr/bin/env python3
# isort: skip_file
# ruff: noqa: E402
"""MindSpeed-LLM pretrain entry for the supported OLMo3 implementations."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "FALSE")

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from torch.utils.data import Dataset
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from llama_config.megatron_export import (
    attach_mindspeed_runtime_config,
    build_model_config_for_megatron,
)
from modeling.lm_loss_utils import dataset_labels_are_shifted
from modeling.registry import build_model
from olmo3_pipeline.data_pipeline import (
    load_runtime_data_manifest,
    validate_runtime_data_profile,
)
from runtime.mindspeed_runtime import (
    install_mindspeed_cross_entropy_patches,
    load_mindspeed_runtime,
)
from runtime.olmo3_parallel_topology import (
    validate_stage2_parallel_topology,
    validate_stage3_parallel_topology,
)
from runtime.ppl_validation import (
    PPLManifestDataset,
    build_local_source_statistics,
    load_ppl_manifest,
    source_statistics_to_loss_dict,
)


get_batch_on_this_cp_rank, pretrain = load_mindspeed_runtime()

# MindSpeed patches torch.compile, Transformer Engine, Apex, and NPU helpers.
# Import Megatron only after those patches are installed.
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.datasets.utils import Split
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.olmo3_runtime_markers import (
    emit_rank0_runtime_marker_once,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_blend_and_blend_per_split,
)
from runtime.olmo3_packed_dataset import (
    OLMO3_LONGMINO_DATA_SEED,
    OLMO3_MIDTRAINING_DATA_SEED,
    Olmo3NumpyFSLDataset,
    Olmo3NumpyPackedFSLDataset,
    Olmo3PackedGPTDataset,
    build_packed_seq_params,
    build_packed_seq_params_from_cu_seqlens,
    cumulative_document_lengths,
)
from runtime.olmo3_sft_dataset import (
    OLMO3_SFT_DATA_SEED,
    OLMO3_SFT_SEQUENCE_LENGTH,
    Olmo3SFTNumpyPackedDataset,
)


def _deferred_dp_loss_reporting_enabled() -> bool:
    value = os.getenv("OLMO3_DEFER_DP_LOSS_REPORTING", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(
            "OLMO3_DEFER_DP_LOSS_REPORTING must be exactly 0 or 1."
        )
    return value == "1"


def _scalar_reduce_backend() -> str:
    value = os.getenv("OLMO3_SCALAR_REDUCE_BACKEND", "hccl").strip().lower()
    if value not in ("hccl", "gloo"):
        raise ValueError(
            "OLMO3_SCALAR_REDUCE_BACKEND must be exactly 'hccl' or 'gloo'."
        )
    return value


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("olmo")
    group.add_argument(
        "--model-config",
        type=_path_arg,
        required=True,
        help="Resolved OLMo 3 model configuration emitted for this immutable run.",
    )
    group.add_argument(
        "--vocab-z-loss-coeff",
        type=float,
        default=None,
        help="Override vocab z-loss coefficient from the frozen model config.",
    )
    group.add_argument(
        "--olmo3-packed-documents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable official OLMo3 Stage-3 document packing, TND fused attention, "
            "and document-relative RoPE."
        ),
    )
    group.add_argument(
        "--olmo3-stage-transition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load model and optimizer state from the previous OLMo3 stage while "
            "starting trainer/scheduler/sample counters at zero."
        ),
    )
    group.add_argument(
        "--dataset-backend",
        choices=(
            "megatron_indexed",
            "olmo3_numpy_fsl",
            "olmo3_numpy_packed",
            "olmo3_sft_numpy",
        ),
        default="megatron_indexed",
        help=(
            "Use Megatron .bin/.idx data or the official raw-Numpy "
            "Stage-2/Stage-3/SFT data contracts."
        ),
    )
    group.add_argument(
        "--olmo3-data-config-name",
        default=None,
        help="Frozen configs/data profile name for a Stage-2/3 runtime manifest.",
    )
    group.add_argument(
        "--olmo3-data-config-sha256",
        default=None,
        help="SHA256 of the exact configs/data profile used to build the manifest.",
    )
    group.add_argument(
        "--olmo3-fsl-data-root",
        type=_path_arg,
        default=None,
        help="Root containing official Dolmino 100B Stage-2 raw uint32 sources.",
    )
    group.add_argument(
        "--olmo3-fsl-data-manifest",
        type=_path_arg,
        default=None,
        help=(
            "Self-contained olmo3-runtime-data-manifest-v1 for Stage 2. "
            "Source order and the complete token contract come only from this file."
        ),
    )
    group.add_argument(
        "--olmo3-fsl-work-dir",
        type=_path_arg,
        default=None,
        help="Shared directory containing the prepared Stage-2 FSL index.",
    )
    group.add_argument(
        "--olmo3-packed-data-root",
        type=_path_arg,
        default=None,
        help="Root containing official preprocessed/dolma3_longmino_0625 data.",
    )
    group.add_argument(
        "--olmo3-packed-data-manifest",
        type=_path_arg,
        default=None,
        help=(
            "Self-contained olmo3-runtime-data-manifest-v1 for Stage 3. "
            "No source-count or directory-layout profile is inferred."
        ),
    )
    group.add_argument(
        "--olmo3-packed-work-dir",
        type=_path_arg,
        default=None,
        help="Shared directory containing the pre-built official OBFD cache.",
    )
    group.add_argument(
        "--olmo3-expected-data-tokens",
        type=int,
        default=None,
        help=(
            "Expected immutable Stage-2/3 corpus token count. The runtime "
            "manifest must match this value exactly."
        ),
    )
    group.add_argument(
        "--olmo3-sft-data-dir",
        type=_path_arg,
        default=None,
        help=(
            "Open-Instruct converted SFT directory containing paired "
            "token_ids_part_*.npy and labels_mask_part_*.npy files."
        ),
    )
    group.add_argument(
        "--olmo3-sft-work-dir",
        type=_path_arg,
        default=None,
        help="Shared directory containing the prepared SFT OBFD cache.",
    )
    group.add_argument(
        "--olmo3-sft-profile",
        choices=("think", "instruct"),
        default=None,
        help="Traceable Dolci SFT branch identity.",
    )
    group.add_argument(
        "--olmo3-sft-expected-instances",
        type=int,
        default=None,
        help="Optional immutable assertion for the prepared packed-instance count.",
    )
    group.add_argument(
        "--olmo3-sft-expected-fingerprint",
        default=None,
        help="Optional immutable assertion for the prepared SFT cache fingerprint.",
    )
    group.add_argument(
        "--valid-ppl-manifest",
        type=_path_arg,
        default=None,
        help=(
            "Optional independent OLMo-style name,relative/path PPL manifest. "
            "With megatron_indexed training, this replaces only the validation "
            "dataset and leaves the training mix at split=100,0,0."
        ),
    )
    group.add_argument(
        "--valid-ppl-cache-dir",
        type=_path_arg,
        default=None,
        help="Shared cache for the EOS-delimited PPL document index.",
    )
    group.add_argument(
        "--valid-ppl-eos-token-id",
        type=int,
        default=None,
        help="EOS ID delimiting independent PPL documents; defaults to tokenizer.eod.",
    )
    group.add_argument(
        "--valid-ppl-pad-token-id",
        type=int,
        default=None,
        help="PPL padding ID; defaults to --pad-token-id.",
    )
    group.add_argument("--valid-ppl-expected-documents", type=int, default=None)
    group.add_argument("--valid-ppl-expected-valid-targets", type=int, default=None)
    group.add_argument("--valid-ppl-expected-truncated-tokens", type=int, default=None)
    group.add_argument("--attn-implementation", default="flash_attention_2")
    group.add_argument("--model-max-position-embeddings", type=int, default=8192)
    group.add_argument("--pad-token-id", type=int, default=1)
    group.add_argument(
        "--sampler-seed-mode",
        choices=("megatron",),
        default="megatron",
        help="Frozen production Megatron sampler semantics.",
    )
    group.add_argument(
        "--sampler-data-seed",
        type=int,
        default=None,
        help=(
            "Explicit deterministic data-sampler seed."
        ),
    )
    group.add_argument("--model-bf16-autocast", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--mcore-native", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--experiment-output-dir", type=_path_arg, default=None)
    group.add_argument(
        "--model-impl",
        choices=(
            "olmo3",
            "olmo3_siamese_depth",
        ),
        default="olmo3_siamese_depth",
    )
    return parser


def _validate_node_local_context_parallel(
    args: Any,
    errors: list[str],
    *,
    check_topology: bool,
) -> None:
    """Validate the common Ulysses/head and node-local placement contract."""

    context_parallel_size = int(args.context_parallel_size)
    if context_parallel_size <= 1:
        return
    if getattr(args, "context_parallel_algo", None) != "ulysses_cp_algo":
        errors.append("CP>1 requires context_parallel_algo=ulysses_cp_algo")

    tensor_parallel_size = int(args.tensor_model_parallel_size)
    parallel_head_shards = tensor_parallel_size * context_parallel_size
    attention_heads = int(args.num_attention_heads)
    query_groups = int(args.num_query_groups)
    if (
        attention_heads % parallel_head_shards
        or query_groups % parallel_head_shards
    ):
        errors.append(
            "attention heads and KV heads must be divisible by TP*CP "
            f"(heads={attention_heads}, kv_heads={query_groups}, "
            f"TP*CP={parallel_head_shards})"
        )

    if not check_topology:
        return
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0") or "0")
    if local_world_size <= 0:
        errors.append("LOCAL_WORLD_SIZE must be set for CP>1")
        return
    if parallel_head_shards > local_world_size or (
        local_world_size % parallel_head_shards
    ):
        errors.append(
            "TP*CP groups must tile a node exactly "
            f"(LOCAL_WORLD_SIZE={local_world_size}, TP*CP={parallel_head_shards})"
        )
        return

    cp_ranks = tuple(int(rank) for rank in mpu.get_context_parallel_global_ranks())
    cp_nodes = {rank // local_world_size for rank in cp_ranks}
    if len(cp_ranks) != context_parallel_size or len(cp_nodes) != 1:
        errors.append(
            "context-parallel group crosses a node: "
            f"ranks={cp_ranks}, node_ids={sorted(cp_nodes)}, "
            f"LOCAL_WORLD_SIZE={local_world_size}"
        )


def _validate_midtraining_runtime(
    args: Any,
    *,
    check_topology: bool = True,
) -> None:
    """Validate Stage-2 raw-FSL data, model, and derived parallel invariants."""

    if args.dataset_backend != "olmo3_numpy_fsl":
        return
    errors: list[str] = []
    if args.olmo3_fsl_data_root is None:
        errors.append("olmo3_fsl_data_root is required")
    if args.olmo3_fsl_data_manifest is None:
        errors.append("olmo3_fsl_data_manifest is required")
    if (
        args.olmo3_expected_data_tokens is None
        or int(args.olmo3_expected_data_tokens) <= 0
    ):
        errors.append("olmo3_expected_data_tokens must be positive")
    if args.olmo3_fsl_work_dir is None:
        errors.append("olmo3_fsl_work_dir is required")
    if args.mock_data:
        errors.append("mock_data is unsupported")
    if args.olmo3_packed_documents:
        errors.append("packed documents must be disabled")
    if int(args.seq_length) != 8192:
        errors.append("seq_length must be 8192")
    if int(args.model_max_position_embeddings) != 8192:
        errors.append("model_max_position_embeddings must be 8192")
    try:
        validate_stage2_parallel_topology(
            world_size=int(args.world_size),
            tensor_parallel_size=int(args.tensor_model_parallel_size),
            pipeline_parallel_size=int(args.pipeline_model_parallel_size),
            context_parallel_size=int(args.context_parallel_size),
            micro_batch_size=int(args.micro_batch_size),
            global_batch_size=int(args.global_batch_size),
            sequence_length=int(args.seq_length),
            sequence_parallel=bool(args.sequence_parallel),
            context_parallel_algo=str(args.context_parallel_algo),
            num_attention_heads=int(args.num_attention_heads),
            num_query_groups=int(args.num_query_groups),
        )
    except ValueError as error:
        errors.append(str(error))
    if not bool(getattr(args, "calculate_per_token_loss", False)):
        errors.append("calculate_per_token_loss must be enabled")
    if args.dataloader_type != "single":
        errors.append("dataloader_type must be single (dataset owns official PCG64 order)")
    if args.sampler_data_seed != OLMO3_MIDTRAINING_DATA_SEED:
        errors.append(f"sampler_data_seed must be {OLMO3_MIDTRAINING_DATA_SEED}")
    if bool(args.reset_position_ids) or bool(args.reset_attention_mask):
        errors.append("reset_position_ids/reset_attention_mask must be disabled")
    if bool(args.eod_mask_loss):
        errors.append("eod_mask_loss must be disabled")
    if bool(args.create_attention_mask_in_dataloader):
        errors.append("create_attention_mask_in_dataloader must be disabled")
    if int(getattr(args, "eval_iters", 0) or 0) != 0:
        errors.append("official Stage-2 training must use eval_iters=0")
    if args.valid_ppl_manifest is not None:
        errors.append("official Stage-2 training does not attach the Stage-1 PPL evaluator")
    _validate_node_local_context_parallel(
        args,
        errors,
        check_topology=check_topology,
    )
    if errors:
        raise ValueError("Invalid OLMo3 raw-FSL Stage-2 runtime: " + "; ".join(errors))


def _validate_packed_long_context_runtime(args: Any, *, check_topology: bool) -> None:
    """Fail closed on modes not equivalent to the official Stage-3 recipe."""

    if not args.olmo3_packed_documents:
        return
    # SFT deliberately reuses the packed/TND/halo machinery at 32K, but owns
    # a separate data and schedule contract validated below.
    if args.dataset_backend == "olmo3_sft_numpy":
        return
    errors: list[str] = []
    if args.dataset_backend != "olmo3_numpy_packed":
        errors.append("dataset_backend must be olmo3_numpy_packed")
    if args.olmo3_packed_data_root is None:
        errors.append("olmo3_packed_data_root is required")
    if args.olmo3_packed_work_dir is None:
        errors.append("olmo3_packed_work_dir is required")
    if args.olmo3_packed_data_manifest is None:
        errors.append("olmo3_packed_data_manifest is required")
    if (
        args.olmo3_expected_data_tokens is None
        or int(args.olmo3_expected_data_tokens) <= 0
    ):
        errors.append("olmo3_expected_data_tokens must be positive")
    if args.mock_data:
        errors.append("mock_data is unsupported")
    if int(args.seq_length) != 65536:
        errors.append("seq_length must be 65536")
    if int(args.model_max_position_embeddings) != 65536:
        errors.append("model_max_position_embeddings must be 65536")
    try:
        validate_stage3_parallel_topology(
            world_size=int(args.world_size),
            tensor_parallel_size=int(args.tensor_model_parallel_size),
            pipeline_parallel_size=int(args.pipeline_model_parallel_size),
            context_parallel_size=int(args.context_parallel_size),
            micro_batch_size=int(args.micro_batch_size),
            global_batch_size=int(args.global_batch_size),
            sequence_length=int(args.seq_length),
            sequence_parallel=bool(args.sequence_parallel),
            context_parallel_algo=str(args.context_parallel_algo),
            num_attention_heads=int(args.num_attention_heads),
            num_query_groups=int(args.num_query_groups),
        )
    except ValueError as error:
        errors.append(str(error))
    if not bool(getattr(args, "calculate_per_token_loss", False)):
        errors.append("calculate_per_token_loss must be enabled")
    if args.dataloader_type != "single":
        errors.append("dataloader_type must be single (dataset owns official PCG64 order)")
    if args.sampler_data_seed != OLMO3_LONGMINO_DATA_SEED:
        errors.append(
            f"sampler_data_seed must be {OLMO3_LONGMINO_DATA_SEED}"
        )
    if bool(args.reset_position_ids) or bool(args.reset_attention_mask):
        errors.append("Megatron reset_position_ids/reset_attention_mask must be disabled")
    if bool(args.eod_mask_loss):
        errors.append("eod_mask_loss must be disabled to match OLMo-core labels")
    if bool(args.create_attention_mask_in_dataloader):
        errors.append("create_attention_mask_in_dataloader must be disabled")
    if getattr(args, "attention_mask_type", "causal") != "causal":
        errors.append("attention_mask_type must be causal")
    if int(getattr(args, "eval_iters", 0) or 0) != 0:
        errors.append("Stage-3 packed training must use eval_iters=0")
    if args.valid_ppl_manifest is not None:
        errors.append("Stage-3 packed training does not run PPL validation")

    _validate_node_local_context_parallel(
        args,
        errors,
        check_topology=check_topology,
    )
    if errors:
        raise ValueError("Invalid OLMo3 packed Stage-3 runtime: " + "; ".join(errors))


def _validate_sft_runtime(args: Any, *, check_topology: bool) -> None:
    """Validate the official 32K packed OLMo3 SFT execution contract."""

    if args.dataset_backend != "olmo3_sft_numpy":
        return
    errors: list[str] = []
    if not args.olmo3_packed_documents:
        errors.append("olmo3_packed_documents must be enabled")
    if args.olmo3_sft_data_dir is None:
        errors.append("olmo3_sft_data_dir is required")
    if args.olmo3_sft_work_dir is None:
        errors.append("olmo3_sft_work_dir is required")
    if args.olmo3_sft_profile not in ("think", "instruct"):
        errors.append("olmo3_sft_profile must be think or instruct")
    if args.mock_data:
        errors.append("mock_data is unsupported")
    if int(args.seq_length) != OLMO3_SFT_SEQUENCE_LENGTH:
        errors.append(f"seq_length must be {OLMO3_SFT_SEQUENCE_LENGTH}")
    # The Stage-3 checkpoint keeps its 65K Full/SWA rotary split. SFT consumes
    # only the first 32K positions but must not mutate the checkpoint model
    # contract or its rotary buffers.
    if int(args.model_max_position_embeddings) != 65536:
        errors.append("model_max_position_embeddings must remain 65536")
    try:
        validate_stage3_parallel_topology(
            world_size=int(args.world_size),
            tensor_parallel_size=int(args.tensor_model_parallel_size),
            pipeline_parallel_size=int(args.pipeline_model_parallel_size),
            context_parallel_size=int(args.context_parallel_size),
            micro_batch_size=int(args.micro_batch_size),
            global_batch_size=int(args.global_batch_size),
            sequence_length=int(args.seq_length),
            sequence_parallel=bool(args.sequence_parallel),
            context_parallel_algo=str(args.context_parallel_algo),
            num_attention_heads=int(args.num_attention_heads),
            num_query_groups=int(args.num_query_groups),
        )
    except ValueError as error:
        errors.append(str(error))
    if not bool(getattr(args, "calculate_per_token_loss", False)):
        errors.append("calculate_per_token_loss must be enabled")
    if args.dataloader_type != "single":
        errors.append("dataloader_type must be single (dataset owns PCG64 order)")
    if args.sampler_data_seed != OLMO3_SFT_DATA_SEED:
        errors.append(f"sampler_data_seed must be {OLMO3_SFT_DATA_SEED}")
    if bool(args.reset_position_ids) or bool(args.reset_attention_mask):
        errors.append("Megatron reset_position_ids/reset_attention_mask must be disabled")
    if bool(args.eod_mask_loss):
        errors.append("eod_mask_loss must be disabled; the assistant mask owns loss")
    if bool(args.create_attention_mask_in_dataloader):
        errors.append("create_attention_mask_in_dataloader must be disabled")
    if getattr(args, "attention_mask_type", "causal") != "causal":
        errors.append("attention_mask_type must be causal")
    if int(getattr(args, "eval_iters", 0) or 0) != 0:
        errors.append("official SFT training must use eval_iters=0")
    if args.valid_ppl_manifest is not None:
        errors.append("SFT does not attach the Stage-1 PPL evaluator")

    _validate_node_local_context_parallel(
        args,
        errors,
        check_topology=check_topology,
    )
    if errors:
        raise ValueError("Invalid OLMo3 packed SFT runtime: " + "; ".join(errors))


def model_provider(pre_process: bool = True, post_process: bool = True, **_: Any) -> nn.Module:
    del pre_process, post_process
    args = get_args()
    _validate_midtraining_runtime(args)
    _validate_packed_long_context_runtime(args, check_topology=True)
    _validate_sft_runtime(args, check_topology=True)
    if args.pipeline_model_parallel_size != 1:
        raise ValueError("The OLMo MindSpeed backend currently supports PP=1 only.")
    if args.context_parallel_size > 1 and (
        args.model_impl not in ("olmo3", "olmo3_siamese_depth")
        or getattr(args, "context_parallel_algo", None) != "ulysses_cp_algo"
    ):
        raise ValueError(
            "CP>1 is supported only by OLMo3 variants with "
            "context_parallel_algo='ulysses_cp_algo'."
        )
    megatron_config = core_transformer_config_from_args(args)
    attach_mindspeed_runtime_config(megatron_config, args)
    # ``loss_func`` returns the same CP-wide token count on every CP rank.
    # Mark that contract so finalization can use one DP reduction per CP group
    # followed by a CP broadcast instead of CP redundant cross-node reductions.
    megatron_config.olmo3_cp_aggregated_num_tokens = True
    # When detached reporting is enabled, its exact DP reduction starts before
    # backward. Finalization reuses the resulting global token count instead of
    # issuing a second scalar collective after backward.
    megatron_config.olmo3_finalize_deferred_num_tokens_func = (
        _finalize_deferred_global_num_tokens
    )
    if not args.mcore_native:
        raise ValueError("The OLMo MindSpeed backend only supports the MCore native model path.")
    model_config = build_model_config_for_megatron(args)
    vocab_size = getattr(args, "padded_vocab_size", None) or getattr(model_config, "vocab_size")
    model = build_model(
        args.model_impl,
        config=megatron_config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=args.model_max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        use_transformer_engine_spec=args.transformer_impl == "transformer_engine",
    )
    if bool(getattr(args, "use_ascend_mc2", False)):
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_TP2_SP_MC2_ACTIVE",
            mc2_all_gather_recomputation=os.environ.get(
                "OLMO3_MC2_ALL_GATHER_RECOMPUTATION",
                "0",
            ),
            sequence_parallel=int(bool(args.sequence_parallel)),
            tensor_parallel=int(args.tensor_model_parallel_size),
        )
    return model


def get_batch(
    data_iterator: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    Any,
]:
    args = get_args()
    cp_loader_owner_only = (
        args.dataset_backend in ("olmo3_numpy_packed", "olmo3_sft_numpy")
        and args.olmo3_packed_documents
        and mpu.get_context_parallel_world_size() > 1
    )
    if cp_loader_owner_only:
        data = _load_packed_batch_once_per_cp_group(data_iterator, args)
    else:
        data = next(data_iterator) if mpu.get_tensor_model_parallel_rank() == 0 else None
    if data is not None:
        data = dict(data)
        if "ppl_source_id" not in data:
            data["ppl_source_id"] = torch.full(
                (data["tokens"].shape[0],), -1, dtype=torch.long
            )
    integer_keys = ["tokens", "labels", "position_ids", "ppl_source_id"]
    if args.olmo3_packed_documents:
        integer_keys.append(
            "packed_cu_seqlens" if cp_loader_owner_only else "document_ids"
        )
    int_batch = tensor_parallel.broadcast_data(
        integer_keys,
        data,
        torch.int64,
    )
    float_batch = tensor_parallel.broadcast_data(["loss_mask"], data, torch.float32)
    batch: dict[str, torch.Tensor | None] = {
        "tokens": int_batch["tokens"],
        "labels": int_batch["labels"],
        "loss_mask": float_batch["loss_mask"],
        "attention_mask": int_batch.get("attention_mask"),
        "position_ids": int_batch["position_ids"],
    }
    ppl_source_ids = int_batch["ppl_source_id"]
    global_document_ids = int_batch.get("document_ids")
    packed_cu_seqlens = int_batch.get("packed_cu_seqlens")
    if not cp_loader_owner_only:
        batch = get_batch_on_this_cp_rank(batch)
    packed_seq_params = None
    if args.olmo3_packed_documents:
        if packed_cu_seqlens is not None:
            packed_seq_params = build_packed_seq_params_from_cu_seqlens(
                packed_cu_seqlens,
                batch["position_ids"],
                micro_batch_size=int(args.micro_batch_size),
                global_tokens_per_sample=int(args.seq_length),
            )
        else:
            if global_document_ids is None:
                raise RuntimeError(
                    "Packed OLMo3 batch is missing document IDs or compact endpoints."
                )
            packed_seq_params = build_packed_seq_params(
                global_document_ids,
                batch["position_ids"],
            )
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        ppl_source_ids,
        packed_seq_params,
    )


def _load_packed_batch_once_per_cp_group(
    data_iterator: Any,
    args: Any,
) -> dict[str, torch.Tensor] | None:
    """Load one 65K packed sample per CP group and distribute it node-locally.

    Only TP-rank 0 participates in the CP broadcast. It then provides the
    already-sliced local shard to the ordinary TP broadcast in ``get_batch``.
    This preserves TP/SP semantics while reducing Stage-3 DataLoader workers
    and AFS reads by exactly the CP factor.
    """

    if mpu.get_tensor_model_parallel_rank() != 0:
        return None

    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        raise RuntimeError("CP-owner packed loading requires context parallelism > 1.")
    if args.seq_length % cp_size != 0:
        raise ValueError(
            f"Packed sequence length {args.seq_length} is not divisible by CP={cp_size}."
        )

    batch_size = int(args.micro_batch_size)
    sequence_length = int(args.seq_length)
    expected_sequence_shape = (batch_size, sequence_length)
    cp_group = mpu.get_context_parallel_group()
    cp_global_ranks = mpu.get_context_parallel_global_ranks()
    cp_src_rank = int(cp_global_ranks[0])
    device = torch.cuda.current_device()

    if cp_rank == 0:
        if data_iterator is None:
            raise RuntimeError("The CP loader owner is missing its data iterator.")
        raw = dict(next(data_iterator))
        if "ppl_source_id" not in raw:
            raw["ppl_source_id"] = torch.full(
                (batch_size,),
                -1,
                dtype=torch.long,
            )
        for key in ("tokens", "labels", "position_ids", "document_ids", "loss_mask"):
            if key not in raw:
                raise RuntimeError(f"Packed OLMo3 batch is missing {key}.")
            if tuple(raw[key].shape) != expected_sequence_shape:
                raise RuntimeError(
                    f"Packed OLMo3 {key} has shape {tuple(raw[key].shape)}; "
                    f"expected {expected_sequence_shape}."
                )
        if tuple(raw["ppl_source_id"].shape) != (batch_size,):
            raise RuntimeError(
                "Packed OLMo3 ppl_source_id has shape "
                f"{tuple(raw['ppl_source_id'].shape)}; expected {(batch_size,)}."
            )
        packed_cu_seqlens = cumulative_document_lengths(
            raw["document_ids"]
        ).to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        integer_sequence_batch = torch.stack(
            [
                raw["tokens"],
                raw["labels"],
                raw["position_ids"],
            ],
            dim=0,
        ).to(device=device, dtype=torch.int64, non_blocking=True)
        loss_mask = raw["loss_mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        ppl_source_ids = raw["ppl_source_id"].to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        metadata_header = torch.cat(
            (
                torch.tensor(
                    [packed_cu_seqlens.numel()],
                    device=device,
                    dtype=torch.int64,
                ),
                ppl_source_ids,
            )
        )
    else:
        metadata_header = torch.empty(
            (batch_size + 1,),
            device=device,
            dtype=torch.int64,
        )

    # CP groups are constrained by the Stage-3 topology validator to remain
    # within a physical node. Scatter only the shard each rank consumes rather
    # than broadcasting four complete 65K rows and discarding CP-1 copies.
    local_sequence_length = sequence_length // cp_size
    local_integer_batch = torch.empty(
        (3, batch_size, local_sequence_length),
        device=device,
        dtype=torch.int64,
    )
    local_loss_mask = torch.empty(
        (batch_size, local_sequence_length),
        device=device,
        dtype=torch.float32,
    )
    if cp_rank == 0:
        # Materialize each CP destination layout with one full-buffer transform.
        # Building one ``narrow(...).contiguous()`` allocation per destination
        # caused 2*CP allocator/kernel launches in every batch-generator step.
        # The views below share two contiguous owner buffers and preserve the
        # identical sequence-major CP partition.
        integer_scatter_buffer = (
            integer_sequence_batch.reshape(
                3,
                batch_size,
                cp_size,
                local_sequence_length,
            )
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        loss_scatter_buffer = (
            loss_mask.reshape(
                batch_size,
                cp_size,
                local_sequence_length,
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        integer_scatter_list = list(integer_scatter_buffer.unbind(0))
        loss_scatter_list = list(loss_scatter_buffer.unbind(0))
    else:
        integer_scatter_list = None
        loss_scatter_list = None
    torch.distributed.scatter(
        local_integer_batch,
        scatter_list=integer_scatter_list,
        src=cp_src_rank,
        group=cp_group,
    )
    torch.distributed.scatter(
        local_loss_mask,
        scatter_list=loss_scatter_list,
        src=cp_src_rank,
        group=cp_group,
    )

    # CANN and the halo planner need only canonical endpoints, not one int64
    # document ID per token. One header collective carries both its dynamic
    # length and the per-sample PPL source IDs.
    torch.distributed.broadcast(
        metadata_header,
        src=cp_src_rank,
        group=cp_group,
    )
    endpoint_count_value = int(metadata_header[0].item())
    ppl_source_ids = metadata_header[1:]
    if endpoint_count_value < batch_size + 1:
        raise RuntimeError(
            "Packed OLMo3 endpoint count cannot cover every microbatch row: "
            f"count={endpoint_count_value}, batch={batch_size}."
        )
    if cp_rank != 0:
        packed_cu_seqlens = torch.empty(
            endpoint_count_value,
            device=device,
            dtype=torch.int64,
        )
    torch.distributed.broadcast(
        packed_cu_seqlens,
        src=cp_src_rank,
        group=cp_group,
    )
    return {
        "tokens": local_integer_batch[0],
        "labels": local_integer_batch[1],
        "position_ids": local_integer_batch[2],
        "packed_cu_seqlens": packed_cu_seqlens,
        "loss_mask": local_loss_mask,
        "ppl_source_id": ppl_source_ids,
    }


def _split_loss_components(
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return differentiable total plus detached LM/z and optional token count."""
    if output_tensor.numel() == 1:
        total_loss = output_tensor.reshape(()).float()
        lm_loss = total_loss.detach()
        z_loss = torch.zeros_like(lm_loss)
        return total_loss, lm_loss, z_loss, None
    if output_tensor.ndim != 1 or output_tensor.numel() not in (3, 4):
        raise ValueError(
            "Model loss output must be a scalar, [total, lm, z], or "
            "[total_sum, lm_sum, z_sum, valid_tokens] vector; "
            f"got shape {tuple(output_tensor.shape)}."
        )
    total_loss = output_tensor[0].float()
    lm_loss = output_tensor[1].detach().float()
    z_loss = output_tensor[2].detach().float()
    token_count = (
        output_tensor[3].detach().float() if output_tensor.numel() == 4 else None
    )
    return total_loss, lm_loss, z_loss, token_count


class _Olmo3DeferredLossReporting:
    """Keep a tiny exact DP loss reduction alive behind model backward."""

    metric_names = ("total loss", "lm loss", "z loss")

    def __init__(
        self,
        reporting: torch.Tensor,
        work: Any | None,
        *,
        cp_group: Any | None = None,
        cp_src_rank: int | None = None,
        cpu_reporting: torch.Tensor | None = None,
        cpu_num_tokens: torch.Tensor | None = None,
        cpu_reduction: bool = False,
    ) -> None:
        self.reporting = reporting
        self.work = work
        self.cp_group = cp_group
        self.cp_src_rank = cp_src_rank
        self.cpu_reporting = cpu_reporting
        self.cpu_num_tokens = cpu_num_tokens
        self.cpu_reduction = cpu_reduction
        self.finalized = False

    def finalize_loss_metrics(self) -> torch.Tensor:
        if not self.finalized:
            if isinstance(self.work, (tuple, list)):
                for work in self.work:
                    work.wait()
            elif self.work is not None:
                self.work.wait()
            if self.cpu_reduction:
                if self.cpu_reporting is not None:
                    reporting = self.cpu_reporting.to(
                        device=self.reporting.device,
                        dtype=self.reporting.dtype,
                    )
                else:
                    reporting = torch.empty(
                        self.reporting.shape,
                        dtype=self.reporting.dtype,
                        device=self.reporting.device,
                    )
                self.reporting = reporting
            if self.cp_group is not None:
                if self.cp_src_rank is None:
                    raise RuntimeError("Deferred CP reporting is missing its source rank.")
                torch.distributed.broadcast(
                    self.reporting,
                    src=self.cp_src_rank,
                    group=self.cp_group,
                )
            self.finalized = True
        return self.reporting


_OLMO3_DEFERRED_TOKEN_REPORTERS: list[_Olmo3DeferredLossReporting] = []


def _register_deferred_token_reporter(
    reporter: _Olmo3DeferredLossReporting,
) -> None:
    _OLMO3_DEFERRED_TOKEN_REPORTERS.append(reporter)


def _finalize_deferred_global_num_tokens(num_tokens: torch.Tensor) -> bool:
    """Reuse pre-backward loss reductions for exact global token scaling.

    One reporter exists per training microbatch. Each reporter's fourth scalar
    is the exact CP×DP token count after finalization. Summing those scalars
    therefore matches Megatron's former post-backward token all-reduce for any
    microbatch count, batch size, CP size, or loss-mask pattern.
    """

    if not _OLMO3_DEFERRED_TOKEN_REPORTERS:
        return False
    reporters = tuple(_OLMO3_DEFERRED_TOKEN_REPORTERS)
    _OLMO3_DEFERRED_TOKEN_REPORTERS.clear()
    if not all(reporter.cpu_reduction for reporter in reporters):
        # HCCL reporting stores its token field in FP32. Preserve exact token
        # scaling for large global batches by retaining the integer fallback
        # reduction instead of reusing a potentially rounded FP32 total.
        return False
    global_num_tokens_cpu = None
    for reporter in reporters:
        reporting = reporter.finalize_loss_metrics()
        if reporting.ndim != 1 or reporting.numel() != 4:
            raise RuntimeError(
                "Deferred token reporting must contain "
                "[total_sum, lm_sum, z_sum, valid_tokens]."
            )
        if reporter.cpu_reporting is not None:
            if reporter.cpu_num_tokens is None:
                raise RuntimeError("CP leader did not retain its integer token count.")
            rounded_token_count = int(reporter.cpu_num_tokens.item())
            global_num_tokens_cpu = (
                rounded_token_count
                if global_num_tokens_cpu is None
                else global_num_tokens_cpu + rounded_token_count
            )
    first_reporter = reporters[0]
    if first_reporter.cpu_reporting is not None:
        if global_num_tokens_cpu is None:
            raise RuntimeError("CP leader did not retain its CPU token count.")
        global_num_tokens = torch.tensor(
            global_num_tokens_cpu,
            dtype=torch.int64,
            device=num_tokens.device,
        )
    else:
        global_num_tokens = torch.empty(
            (),
            dtype=torch.int64,
            device=num_tokens.device,
        )
    if first_reporter.cp_group is not None:
        if first_reporter.cp_src_rank is None:
            raise RuntimeError("Deferred token reporting is missing its CP source rank.")
        torch.distributed.broadcast(
            global_num_tokens,
            src=first_reporter.cp_src_rank,
            group=first_reporter.cp_group,
        )
    num_tokens.copy_(global_num_tokens)
    return True


def loss_func(
    loss: torch.Tensor,
    *,
    defer_dp_reporting: bool = False,
) -> tuple[Any, ...]:
    total_loss, lm_loss, z_loss, token_count = _split_loss_components(loss)
    context_parallel_size = mpu.get_context_parallel_world_size()
    cp_token_count = None
    deferred_reporting = None
    if token_count is not None:
        if not bool(getattr(get_args(), "calculate_per_token_loss", False)):
            raise RuntimeError(
                "Summed OLMo3 losses require --calculate-per-token-loss."
            )
        reporting = torch.stack(
            (total_loss.detach(), lm_loss, z_loss, token_count)
        )
        if context_parallel_size > 1:
            torch.distributed.all_reduce(
                reporting, group=mpu.get_context_parallel_group()
            )
        # reporting[3] is now the exact CP-wide token count. Clone it before
        # the DP reduction instead of launching a duplicate scalar CP
        # all-reduce every microbatch.
        cp_token_count = reporting[3].clone()
        can_defer_reporting = (
            defer_dp_reporting
            and _deferred_dp_loss_reporting_enabled()
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and mpu.get_data_parallel_world_size() > 1
        )
        if can_defer_reporting:
            scalar_backend = _scalar_reduce_backend()
            cp_group = (
                mpu.get_context_parallel_group()
                if context_parallel_size > 1
                else None
            )
            cp_src_rank = (
                int(mpu.get_context_parallel_global_ranks()[0])
                if context_parallel_size > 1
                else None
            )
            if scalar_backend == "gloo":
                work = None
                cpu_reporting = None
                cpu_num_tokens = None
                if context_parallel_size == 1 or mpu.get_context_parallel_rank() == 0:
                    cpu_reporting = reporting.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    cpu_num_tokens = torch.round(cpu_reporting[3]).to(
                        dtype=torch.int64
                    )
                    try:
                        gloo_group = mpu.get_data_parallel_group_gloo()
                    except AssertionError as error:
                        raise RuntimeError(
                            "OLMO3_SCALAR_REDUCE_BACKEND=gloo requires "
                            "Megatron Gloo process groups to be enabled."
                        ) from error
                    work = (
                        torch.distributed.all_reduce(
                            cpu_reporting,
                            group=gloo_group,
                            async_op=True,
                        ),
                        torch.distributed.all_reduce(
                            cpu_num_tokens,
                            group=gloo_group,
                            async_op=True,
                        ),
                    )
                deferred_reporting = _Olmo3DeferredLossReporting(
                    reporting,
                    work,
                    cp_group=cp_group,
                    cp_src_rank=cp_src_rank,
                    cpu_reporting=cpu_reporting,
                    cpu_num_tokens=cpu_num_tokens,
                    cpu_reduction=True,
                )
            else:
                work = None
                if context_parallel_size == 1 or mpu.get_context_parallel_rank() == 0:
                    work = torch.distributed.all_reduce(
                        reporting,
                        group=mpu.get_data_parallel_group(),
                        async_op=True,
                    )
                deferred_reporting = _Olmo3DeferredLossReporting(
                    reporting,
                    work,
                    cp_group=cp_group,
                    cp_src_rank=cp_src_rank,
                )
            _register_deferred_token_reporter(deferred_reporting)
        else:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if context_parallel_size > 1:
                    if mpu.get_context_parallel_rank() == 0:
                        torch.distributed.all_reduce(
                            reporting,
                            group=mpu.get_data_parallel_group(),
                        )
                    torch.distributed.broadcast(
                        reporting,
                        src=int(mpu.get_context_parallel_global_ranks()[0]),
                        group=mpu.get_context_parallel_group(),
                    )
                else:
                    torch.distributed.all_reduce(
                        reporting, group=mpu.get_data_parallel_group()
                    )
            reporting_denominator = reporting[3].clamp_min(1.0)
            logged_total_loss = reporting[0] / reporting_denominator
            logged_lm_loss = reporting[1] / reporting_denominator
            logged_z_loss = reporting[2] / reporting_denominator
        backward_total_loss = total_loss
    else:
        backward_total_loss = total_loss / context_parallel_size
        logged_total_loss, logged_lm_loss, logged_z_loss = (
            average_losses_across_data_parallel_group(
                [total_loss.detach(), lm_loss, z_loss]
            )
        )
        if context_parallel_size > 1:
            logged_components = [
                logged_total_loss.clone(),
                logged_lm_loss.clone(),
                logged_z_loss.clone(),
            ]
            for component in logged_components:
                torch.distributed.all_reduce(
                    component, group=mpu.get_context_parallel_group()
                )
            logged_total_loss, logged_lm_loss, logged_z_loss = [
                component / context_parallel_size for component in logged_components
            ]
    if deferred_reporting is not None:
        metrics = {"_olmo3_deferred_loss_reporting": deferred_reporting}
    else:
        metrics = {
            "lm loss": logged_lm_loss,
            "z loss": logged_z_loss,
            "total loss": logged_total_loss,
        }
    if token_count is not None:
        return backward_total_loss, cp_token_count.to(dtype=torch.int), metrics
    return backward_total_loss, metrics


def ppl_eval_loss_func(
    per_sample_loss_stats: torch.Tensor,
    *,
    source_ids: torch.Tensor,
    dataset_names: tuple[str, ...],
) -> tuple[
    torch.Tensor,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    """Aggregate exact token-weighted PPL statistics across CP and DP ranks."""

    source_stats = build_local_source_statistics(
        per_sample_loss_stats,
        source_ids,
        len(dataset_names),
    )
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and mpu.get_data_parallel_world_size(with_context_parallel=True) > 1
    ):
        torch.distributed.all_reduce(
            source_stats,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
    loss_dict = source_statistics_to_loss_dict(source_stats, dataset_names)
    total_sum, token_count = loss_dict["total loss"]
    total_mean = total_sum / token_count.clamp_min(1.0)
    return total_mean, loss_dict


def forward_step(data_iterator: Any, model: nn.Module) -> tuple[torch.Tensor, Any]:
    args = get_args()
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        ppl_source_ids,
        packed_seq_params,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()
    # Labels are a private per-step batch buffer. Mask them in place to avoid
    # allocating and copying another local 65K/CP row before the LM head.
    labels.masked_fill_(loss_mask == 0, -100)
    ppl_eval = args.valid_ppl_manifest is not None and not model.training
    global_step_offset = int(os.getenv("WANDB_GLOBAL_STEP_OFFSET", os.getenv("GLOBAL_STEP_OFFSET", "0")))
    global_step = int(getattr(args, "curr_iteration", 0)) + global_step_offset
    loss = model(
        tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        global_step=global_step,
        labels_are_shifted=dataset_labels_are_shifted(args.dataset_backend),
        return_per_sample_loss_sums=ppl_eval,
        packed_seq_params=packed_seq_params,
    )
    if ppl_eval:
        dataset_names = tuple(getattr(args, "ppl_validation_dataset_names", ()))
        if not dataset_names:
            raise RuntimeError("PPL validation dataset names were not initialized.")
        return loss, partial(
            ppl_eval_loss_func,
            source_ids=ppl_source_ids,
            dataset_names=dataset_names,
        )
    return loss, partial(
        loss_func,
        defer_dp_reporting=bool(model.training),
    )


def _is_dataset_built_on_rank() -> bool:
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def megatron_indexed_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
    """Build full Dolma 3 training data plus an optional independent PPL set."""
    args = get_args()
    _validate_packed_long_context_runtime(args, check_topology=False)
    if args.dataset_backend != "megatron_indexed":
        raise ValueError(
            "megatron_indexed_datasets_provider requires "
            "--dataset-backend megatron_indexed."
        )
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )
    if args.olmo3_packed_documents:
        dataset_type = Olmo3PackedGPTDataset
    else:
        dataset_type = MockGPTDataset if args.mock_data else GPTDataset
    print_rank_0("> building Megatron-indexed datasets for OLMo MindSpeed training ...")
    indexed_num_samples = list(train_val_test_num_samples)
    if args.valid_ppl_manifest is not None:
        # The independent PPL data must not cause the Dolma 3 training blend to
        # reserve or synthesize validation/test samples.
        indexed_num_samples[1:] = [0, 0]
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        indexed_num_samples,
        _is_dataset_built_on_rank,
        config,
    ).build()
    args.dataset_train_len = int(
        args.train_samples
        or train_val_test_num_samples[0]
        or (len(train_ds) if train_ds is not None else 0)
    )

    if args.valid_ppl_manifest is not None:
        manifest_entries = load_ppl_manifest(args.valid_ppl_manifest)
        args.ppl_validation_dataset_names = tuple(
            entry.name for entry in manifest_entries
        )
        valid_ds = None
        runtime_metadata = torch.zeros(4, dtype=torch.long, device="cuda")
        if _is_dataset_built_on_rank():
            eos_token_id = (
                args.valid_ppl_eos_token_id
                if args.valid_ppl_eos_token_id is not None
                else get_tokenizer().eod
            )
            pad_token_id = (
                args.valid_ppl_pad_token_id
                if args.valid_ppl_pad_token_id is not None
                else args.pad_token_id
            )
            cache_dir = args.valid_ppl_cache_dir
            if cache_dir is None and args.data_cache_path is not None:
                cache_dir = Path(args.data_cache_path) / "ppl_validation"
            valid_ds = PPLManifestDataset(
                args.valid_ppl_manifest,
                args.seq_length,
                args.global_batch_size,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                split=Split.valid,
                cache_dir=cache_dir,
                sync_process_group=(
                    mpu.get_data_parallel_group(with_context_parallel=True)
                    if torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                    else None
                ),
            )
            valid_ds.validate_expected_counts(
                documents=args.valid_ppl_expected_documents,
                valid_targets=args.valid_ppl_expected_valid_targets,
                truncated_tokens=args.valid_ppl_expected_truncated_tokens,
            )
            runtime_metadata.copy_(
                torch.tensor(
                    [
                        valid_ds.eval_iters,
                        len(valid_ds),
                        valid_ds.real_sample_count,
                        valid_ds.dummy_sample_count,
                    ],
                    dtype=torch.long,
                    device=runtime_metadata.device,
                )
            )
            print_rank_0(
                "> independent PPL validation dataset: "
                + json.dumps(valid_ds.summary(), sort_keys=True)
            )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and mpu.get_tensor_model_parallel_world_size() > 1
        ):
            torch.distributed.broadcast(
                runtime_metadata,
                src=mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )
        elif valid_ds is None:
            raise RuntimeError("PPL validation dataset was not built.")

        args.eval_iters = int(runtime_metadata[0].item())
        args.dataset_valid_len = int(runtime_metadata[1].item())
        if args.eval_iters <= 0 or args.dataset_valid_len <= 0:
            raise RuntimeError(
                "PPL validation runtime accounting produced an empty dataset."
            )
        previous_consumed_valid = int(args.consumed_valid_samples)
        args.consumed_valid_samples = (
            previous_consumed_valid // args.dataset_valid_len
        ) * args.dataset_valid_len
        if previous_consumed_valid != args.consumed_valid_samples:
            print_rank_0(
                "> aligned consumed-valid-samples to a complete document-PPL "
                f"epoch: {previous_consumed_valid} -> "
                f"{args.consumed_valid_samples}"
            )
        print_rank_0(
            "> document-PPL runtime shape: "
            f"global_batch_size={args.global_batch_size}, "
            f"eval_iters={args.eval_iters}, "
            f"real_documents={int(runtime_metadata[2].item())}, "
            f"dummy_samples={int(runtime_metadata[3].item())}"
        )
        test_ds = None
    print_rank_0("> finished building Megatron-indexed datasets.")
    return train_ds, valid_ds, test_ds


def olmo3_numpy_packed_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[Dataset, None, None]:
    """Build the exact OLMo3 Longmino raw-Numpy + OBFD Stage-3 dataset."""

    args = get_args()
    _validate_packed_long_context_runtime(args, check_topology=False)
    contract = load_runtime_data_manifest(
        Path(args.olmo3_packed_data_manifest),
        root=Path(args.olmo3_packed_data_root),
        verify_files=False,
        expected_stage="stage3",
        expected_backend="olmo3_numpy_packed",
        expected_token_count=int(args.olmo3_expected_data_tokens),
    )
    validate_runtime_data_profile(
        contract,
        config_name=args.olmo3_data_config_name,
        config_sha256=args.olmo3_data_config_sha256,
        pad_token_id=args.pad_token_id,
    )
    source_paths = tuple(Path(path) for path in contract["resolved_token_paths"])
    data_contract = (
        f"{contract['schema']}:{contract['contract_sha256']}:"
        f"{contract['manifest_sha256']}"
    )
    requested_train_samples = int(train_val_test_num_samples[0] or 0)
    if requested_train_samples < 1:
        raise ValueError("Stage-3 training requested no packed samples.")
    train_ds = Olmo3NumpyPackedFSLDataset(
        source_paths,
        args.olmo3_packed_work_dir,
        num_samples=requested_train_samples,
        global_batch_size=args.global_batch_size,
        source_contract_sha256=contract["contract_sha256"],
        data_seed=args.sampler_data_seed,
        sequence_length=args.seq_length,
        eos_token_id=get_tokenizer().eod,
        pad_token_id=args.pad_token_id,
    )
    args.dataset_train_len = len(train_ds)
    args.dataset_valid_len = 0
    print_rank_0(
        "> OLMo3 Stage-3 packed dataset: "
        + json.dumps(
            {
                "base_instances": train_ds.base_instances,
                "cache_root": str(train_ds.cache_root),
                "data_contract": data_contract,
                "data_seed": args.sampler_data_seed,
                "instances_per_epoch": train_ds.instances_per_epoch,
                "requested_train_samples": requested_train_samples,
                "sequence_length": args.seq_length,
                "source_count": len(source_paths),
                "source_group_size": 8,
                "source_permutation_seed": 123,
            },
            sort_keys=True,
        )
    )
    return train_ds, None, None


def olmo3_numpy_fsl_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[Dataset, None, None]:
    """Build the exact OLMo3 Dolmino raw-Numpy FSL Stage-2 dataset."""

    args = get_args()
    _validate_midtraining_runtime(args)
    contract = load_runtime_data_manifest(
        Path(args.olmo3_fsl_data_manifest),
        root=Path(args.olmo3_fsl_data_root),
        verify_files=False,
        expected_stage="stage2",
        expected_backend="olmo3_numpy_fsl",
        expected_token_count=int(args.olmo3_expected_data_tokens),
    )
    validate_runtime_data_profile(
        contract,
        config_name=args.olmo3_data_config_name,
        config_sha256=args.olmo3_data_config_sha256,
        pad_token_id=args.pad_token_id,
    )
    source_paths = tuple(Path(path) for path in contract["resolved_token_paths"])
    actual_manifest_sha256 = contract["manifest_sha256"]
    data_contract = (
        f"{contract['schema']}:{contract['contract_sha256']}:"
        f"{actual_manifest_sha256}"
    )
    requested_train_samples = int(train_val_test_num_samples[0] or 0)
    if requested_train_samples < 1:
        raise ValueError("Stage-2 training requested no FSL samples.")
    train_ds = Olmo3NumpyFSLDataset(
        source_paths,
        args.olmo3_fsl_work_dir,
        num_samples=requested_train_samples,
        global_batch_size=args.global_batch_size,
        source_contract_sha256=contract["contract_sha256"],
        data_seed=args.sampler_data_seed,
        sequence_length=args.seq_length,
        pad_token_id=args.pad_token_id,
    )
    args.dataset_train_len = len(train_ds)
    args.dataset_valid_len = 0
    print_rank_0(
        "> OLMo3 Stage-2 raw-FSL dataset: "
        + json.dumps(
            {
                "base_instances": train_ds.base_instances,
                "cache_root": str(train_ds.cache_root),
                "data_contract": data_contract,
                "data_seed": args.sampler_data_seed,
                "instances_per_epoch": train_ds.instances_per_epoch,
                "manifest_sha256": actual_manifest_sha256,
                "requested_train_samples": requested_train_samples,
                "sequence_length": args.seq_length,
                "source_count": len(source_paths),
            },
            sort_keys=True,
        )
    )
    return train_ds, None, None


def olmo3_sft_numpy_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[Dataset, None, None]:
    """Build official-format packed Dolci SFT data with assistant-only loss."""

    args = get_args()
    _validate_sft_runtime(args, check_topology=False)
    requested_train_samples = int(train_val_test_num_samples[0] or 0)
    if requested_train_samples < 1:
        raise ValueError("OLMo3 SFT training requested no packed samples.")
    train_ds = Olmo3SFTNumpyPackedDataset(
        args.olmo3_sft_data_dir,
        args.olmo3_sft_work_dir,
        num_samples=requested_train_samples,
        global_batch_size=args.global_batch_size,
        expected_fingerprint=args.olmo3_sft_expected_fingerprint,
        data_seed=args.sampler_data_seed,
        sequence_length=args.seq_length,
        eos_token_id=get_tokenizer().eod,
        pad_token_id=args.pad_token_id,
    )
    if (
        args.olmo3_sft_expected_instances is not None
        and train_ds.base_instances != args.olmo3_sft_expected_instances
    ):
        raise RuntimeError(
            "Prepared SFT instance-count assertion failed: "
            f"{train_ds.base_instances} != {args.olmo3_sft_expected_instances}."
        )
    cache_fingerprint = train_ds.cache_root.name.removeprefix(
        "olmo3-sft-packed-"
    )
    if (
        args.olmo3_sft_expected_fingerprint is not None
        and cache_fingerprint != args.olmo3_sft_expected_fingerprint
    ):
        raise RuntimeError(
            "Prepared SFT fingerprint assertion failed: "
            f"{cache_fingerprint} != {args.olmo3_sft_expected_fingerprint}."
        )
    args.dataset_train_len = len(train_ds)
    args.dataset_valid_len = 0
    print_rank_0(
        "> OLMo3 packed SFT dataset: "
        + json.dumps(
            {
                "assistant_only_loss": True,
                "base_instances": train_ds.base_instances,
                "cache_root": str(train_ds.cache_root),
                "data_seed": args.sampler_data_seed,
                "instances_per_epoch": train_ds.instances_per_epoch,
                "profile": args.olmo3_sft_profile,
                "requested_train_samples": requested_train_samples,
                "sequence_length": args.seq_length,
                "source_count": len(train_ds.source_paths),
                "source_group_size": 1,
            },
            sort_keys=True,
        )
    )
    return train_ds, None, None


megatron_indexed_datasets_provider.is_distributed = True
olmo3_numpy_fsl_datasets_provider.is_distributed = True
olmo3_numpy_packed_datasets_provider.is_distributed = True
olmo3_sft_numpy_datasets_provider.is_distributed = True
# Stage 3 uses one dataset/DataLoader per (DP, CP) replica. The owning rank
# broadcasts the full packed sample inside its node-local CP group before TP
# fan-out; non-owner CP ranks must not construct workers or touch AFS.
olmo3_numpy_packed_datasets_provider.cp_loader_owner_only = True
olmo3_sft_numpy_datasets_provider.cp_loader_owner_only = True


def _requested_dataset_backend() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset-backend",
        choices=(
            "megatron_indexed",
            "olmo3_numpy_fsl",
            "olmo3_numpy_packed",
            "olmo3_sft_numpy",
        ),
        default="megatron_indexed",
    )
    known, _ = parser.parse_known_args()
    return known.dataset_backend


def main() -> None:
    install_mindspeed_cross_entropy_patches()
    dataset_backend = _requested_dataset_backend()
    if dataset_backend == "megatron_indexed":
        datasets_provider = megatron_indexed_datasets_provider
    elif dataset_backend == "olmo3_numpy_fsl":
        datasets_provider = olmo3_numpy_fsl_datasets_provider
    elif dataset_backend == "olmo3_numpy_packed":
        datasets_provider = olmo3_numpy_packed_datasets_provider
    elif dataset_backend == "olmo3_sft_numpy":
        datasets_provider = olmo3_sft_numpy_datasets_provider
    else:
        raise ValueError(f"Unsupported production dataset backend: {dataset_backend}")
    pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "dataloader_type": "cyclic",
        },
    )


if __name__ == "__main__":
    main()
