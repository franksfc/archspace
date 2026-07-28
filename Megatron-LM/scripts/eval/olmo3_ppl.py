#!/usr/bin/env python3
"""Run independent document-aware PPL from a native OLMo 3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Importing the project entry point first establishes the same MindSpeed/NPU
# runtime order used by production training.
from pretrain_olmo3_mindspeed import (
    _is_dataset_built_on_rank,
    extra_args_provider,
    forward_step,
    get_args,
    get_tokenizer,
    install_mindspeed_cross_entropy_patches,
    model_provider,
    mpu,
    print_rank_0,
)
import torch
from torch.utils.data import Dataset

from megatron.core.datasets.utils import Split
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import (
    build_train_valid_test_data_iterators,
    get_model,
)
from mindspeed_llm.training import training as mindspeed_training
from mindspeed_llm.training.initialize import set_jit_fusion_options
from runtime.ppl_validation import PPLManifestDataset, load_ppl_manifest


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def ppl_extra_args_provider(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = extra_args_provider(parser)
    group = parser.add_argument_group("olmo3 independent PPL")
    group.add_argument(
        "--ppl-output-json",
        type=_path_arg,
        required=True,
        help="New immutable JSON result path written by the global last rank.",
    )
    return parser


def ppl_only_datasets_provider(
    _: list[int],
) -> tuple[None, Dataset | None, None]:
    """Build only the independent EOS-delimited validation corpus."""

    args = get_args()
    if args.valid_ppl_manifest is None:
        raise ValueError("--valid-ppl-manifest is required")
    if args.dataset_backend != "megatron_indexed":
        raise ValueError(
            "independent PPL requires --dataset-backend megatron_indexed"
        )

    entries = load_ppl_manifest(args.valid_ppl_manifest)
    args.ppl_validation_dataset_names = tuple(
        entry.name for entry in entries
    )
    valid_ds: PPLManifestDataset | None = None
    # [eval iters, padded samples, real documents, dummy samples,
    #  valid targets, truncated tokens]
    runtime_metadata = torch.zeros(6, dtype=torch.long, device="cuda")
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
        valid_ds = PPLManifestDataset(
            args.valid_ppl_manifest,
            args.seq_length,
            args.global_batch_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            split=Split.valid,
            cache_dir=args.valid_ppl_cache_dir,
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
                    valid_ds.valid_target_count,
                    valid_ds.truncated_token_count,
                ],
                dtype=torch.long,
                device=runtime_metadata.device,
            )
        )
        print_rank_0(
            "> independent document-PPL dataset: "
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
        raise RuntimeError("document-PPL dataset was not built")

    args.eval_iters = int(runtime_metadata[0].item())
    args.dataset_valid_len = int(runtime_metadata[1].item())
    if args.eval_iters <= 0 or args.dataset_valid_len <= 0:
        raise RuntimeError("document-PPL dataset is empty")
    args.dataset_train_len = 0
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.ppl_validation_summary = {
        "datasets": list(args.ppl_validation_dataset_names),
        "dataset_count": len(entries),
        "total_tokens": sum(entry.token_count for entry in entries),
        "documents": int(runtime_metadata[2].item()),
        "padded_samples": int(runtime_metadata[1].item()),
        "dummy_samples": int(runtime_metadata[3].item()),
        "valid_targets": int(runtime_metadata[4].item()),
        "truncated_tokens": int(runtime_metadata[5].item()),
        "sequence_length": int(args.seq_length),
    }
    print_rank_0(
        "> document-PPL runtime shape: "
        f"global_batch_size={args.global_batch_size}, "
        f"eval_iters={args.eval_iters}, "
        f"documents={args.ppl_validation_summary['documents']}, "
        f"dummy_samples={args.ppl_validation_summary['dummy_samples']}"
    )
    return None, valid_ds, None


ppl_only_datasets_provider.is_distributed = True


def main() -> None:
    install_mindspeed_cross_entropy_patches()
    initialize_megatron(
        extra_args_provider=ppl_extra_args_provider,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "dataloader_type": "single",
        },
    )
    args = get_args()
    if not args.skip_train:
        raise ValueError("independent document PPL requires --skip-train")
    if not args.no_load_optim or not args.no_load_rng:
        raise ValueError(
            "independent document PPL requires --no-load-optim and "
            "--no-load-rng"
        )
    set_jit_fusion_options()

    # Forward-only PPL deliberately bypasses setup_model_and_optimizer().
    # No optimizer, FP32 master parameters, Adam moments, or scheduler exist.
    model = get_model(
        model_provider,
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    iteration, _ = load_checkpoint(model, None, None, strict=True)
    if int(iteration) != int(args.ckpt_step):
        raise RuntimeError(
            f"loaded checkpoint iteration {iteration}, "
            f"expected {args.ckpt_step}"
        )
    if len(model) != 1:
        raise RuntimeError(
            f"document PPL requires one local model chunk, got {len(model)}"
        )

    _, valid_iterator, _ = build_train_valid_test_data_iterators(
        ppl_only_datasets_provider
    )
    if valid_iterator is None:
        raise RuntimeError("document-PPL validation iterator is unavailable")
    config = get_model_config(model[0])
    mindspeed_training.evaluate_and_print_results(
        f"iteration {iteration} on independent document-PPL set",
        forward_step,
        valid_iterator,
        model,
        int(iteration),
        None,
        config,
        verbose=True,
        write_to_tensorboard=False,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if not Path(args.ppl_output_json).is_file():
        raise RuntimeError(
            f"document-PPL result was not written: {args.ppl_output_json}"
        )
    print_rank_0(
        "OLMO3_DOCUMENT_PPL_COMPLETE "
        f"checkpoint_step={iteration} output={args.ppl_output_json}"
    )


if __name__ == "__main__":
    main()
