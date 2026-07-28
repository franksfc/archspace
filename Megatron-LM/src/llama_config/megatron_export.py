"""Load a validated OLMo-family config into Megatron-compatible settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from transformers.models.llama.configuration_llama import LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def build_model_config_for_megatron(args: argparse.Namespace) -> LlamaConfig:
    """Load one of this repository's dedicated OLMo3 config directories."""

    add_repo_to_path()
    config = LlamaConfig.from_pretrained(str(args.model_config))
    config.max_position_embeddings = args.model_max_position_embeddings
    config._attn_implementation = args.attn_implementation
    vocab_z_loss_coeff = getattr(args, "vocab_z_loss_coeff", None)
    if vocab_z_loss_coeff is not None:
        config.vocab_z_loss_coeff = vocab_z_loss_coeff
    if args.attn_implementation == "flash_attention_2":
        setattr(config, "_attn_implementation_autoset", True)
    return config


def attach_mindspeed_runtime_config(config: Any, args: argparse.Namespace) -> None:
    """Attach MindSpeed runtime fields that are not Megatron Core dataclass fields."""

    runtime_fields = {
        # This function is called only by the MindSpeed entrypoint.  Carry the
        # backend capability explicitly instead of asking model code to infer
        # it from a monkey-patched class's ``__module__`` value.
        "olmo3_mindspeed_runtime": True,
        "transformer_impl": args.transformer_impl,
        "use_flash_attn": bool(getattr(args, "use_flash_attn", False)),
        "attention_mask_type": getattr(args, "attention_mask_type", "causal"),
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "pre_tockens": getattr(args, "pre_tockens", 1048576),
        "next_tockens": getattr(args, "next_tockens", 0),
        "sparse_mode": getattr(args, "sparse_mode", 0),
        "shape_order": getattr(args, "shape_order", "SBH"),
        "context_parallel_algo": getattr(args, "context_parallel_algo", "megatron_cp_algo"),
    }
    for key, value in runtime_fields.items():
        setattr(config, key, value)
