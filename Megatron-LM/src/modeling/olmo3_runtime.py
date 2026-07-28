"""Runtime validation shared by the OLMo3 model variants."""

from __future__ import annotations

from typing import Any

from megatron.core.transformer.transformer_config import TransformerConfig


def attach_olmo3_swa_config(config: TransformerConfig, settings: Any) -> None:
    """Attach per-layer OLMo3 fields without enabling global all-layer SWA."""
    if config.window_size is not None:
        raise ValueError(
            "Do not set TransformerConfig.window_size globally for OLMo3; "
            "the architecture selects SWA per layer."
        )
    config.olmo3_sliding_window = settings.sliding_window
    config.olmo3_layer_types = settings.layer_types
    config.olmo3_yarn = settings.yarn
    config.rope_full_precision = settings.rope_full_precision


def validate_olmo3_mindspeed_runtime_args(
    settings: Any,
    *,
    model_impl: str,
) -> None:
    """Reject MindSpeed features that would replace OLMo3's per-layer attention."""
    try:
        from megatron.training import get_args

        args = get_args()
    except (AssertionError, ImportError, RuntimeError):
        return

    incompatible_flags = {
        "share_kvstates": bool(getattr(args, "share_kvstates", False)),
        "input_embeds_norm": bool(getattr(args, "input_embeds_norm", False)),
        "enable_mhc": bool(getattr(args, "enable_mhc", False)),
        "recompute_norm": bool(getattr(args, "recompute_norm", False)),
        "noop_layers": bool(getattr(args, "noop_layers", False)),
        "full_attention_interval": bool(getattr(args, "full_attention_interval", False)),
        "global_sliding_window": getattr(args, "sliding_window", None) is not None,
        "n_hash_layers": int(getattr(args, "n_hash_layers", 0) or 0) >= 1,
        "tp_2d": bool(getattr(args, "tp_2d", False)),
        "num_experts": getattr(args, "num_experts", None) is not None,
    }
    enabled = [name for name, active in incompatible_flags.items() if active]
    if enabled:
        raise ValueError(
            "OLMo3 per-layer SWA is incompatible with MindSpeed features: "
            + ", ".join(enabled)
        )
    if getattr(args, "transformer_impl", "transformer_engine") != "transformer_engine":
        raise ValueError("OLMo3 requires transformer_impl=transformer_engine.")
    if not bool(getattr(args, "use_flash_attn", False)):
        raise ValueError("OLMo3 on MindSpeed requires --use-flash-attn.")
    if getattr(args, "attention_mask_type", "causal") != "causal":
        raise ValueError("OLMo3 MindSpeed SWA requires attention_mask_type='causal'.")
    if getattr(args, "shape_order", "SBH") != "SBH":
        raise ValueError("OLMo3 MindSpeed SWA currently requires shape_order='SBH'.")
    context_parallel_size = int(getattr(args, "context_parallel_size", 1))
    if context_parallel_size > 1:
        context_parallel_algo = getattr(args, "context_parallel_algo", None)
        if context_parallel_algo != "ulysses_cp_algo":
            raise ValueError(
                "OLMo3 SWA with CP>1 requires context_parallel_algo='ulysses_cp_algo'."
            )
        tensor_parallel_size = int(getattr(args, "tensor_model_parallel_size", 1))
        parallel_head_shards = context_parallel_size * tensor_parallel_size
        if (
            settings.num_attention_heads % parallel_head_shards
            or settings.num_key_value_heads % parallel_head_shards
        ):
            raise ValueError(
                "OLMo3 Ulysses CP requires attention heads and KV heads "
                "divisible by CP x TP."
            )

    expected_values = {
        "model_impl": model_impl,
        "vocab_size": settings.vocab_size,
        "pad_token_id": settings.pad_token_id,
        "max_position_embeddings": settings.max_position_embeddings,
        "model_max_position_embeddings": settings.max_position_embeddings,
        "rotary_base": settings.rope_theta,
    }
    mismatches = [
        f"{name}: expected {expected!r}, got {getattr(args, name, None)!r}"
        for name, expected in expected_values.items()
        if getattr(args, name, None) != expected
    ]
    runtime_sequence_length = getattr(args, "seq_length", None)
    if (
        isinstance(runtime_sequence_length, bool)
        or not isinstance(runtime_sequence_length, int)
        or runtime_sequence_length < 1
        or runtime_sequence_length > settings.max_position_embeddings
    ):
        mismatches.append(
            "seq_length: expected an integer in "
            f"[1, {settings.max_position_embeddings}], got "
            f"{runtime_sequence_length!r}"
        )
    if mismatches:
        raise ValueError(
            "OLMo3 model config and MindSpeed runtime arguments disagree:\n  "
            + "\n  ".join(mismatches)
        )
