from __future__ import annotations

from typing import Any


SLIDING = "sliding_attention"
FULL = "full_attention"


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    variant = config["variant"]
    stage = config["stage"]
    optim = config["optimization"]

    layer_types: list[str] = []
    pattern = model["swa_pattern"]
    for layer_index in range(model["num_layers"]):
        window = pattern[layer_index % len(pattern)]
        layer_types.append(FULL if window == -1 else SLIDING)
    layer_types[-1] = FULL

    architecture = (
        "OLMo3ForCausalLM"
        if variant["name"] == "base"
        else "OLMo3SiameseDepthForCausalLM"
    )
    return {
        "_name_or_path": (
            f"olmo3-{model['size']}-{variant['name']}-{stage['name']}"
        ),
        "architectures": [architecture],
        "model_type": "llama",
        "attention_bias": False,
        "attention_dropout": model["attention_dropout"],
        "hidden_dropout": model["hidden_dropout"],
        "hidden_act": "silu",
        "hidden_size": model["hidden_size"],
        "intermediate_size": model["ffn_hidden_size"],
        "num_hidden_layers": model["num_layers"],
        "num_attention_heads": model["num_attention_heads"],
        "num_key_value_heads": model["num_query_groups"],
        "initializer_range": model["initializer_range"],
        "truncated_normal_factor": 3.0,
        "max_position_embeddings": stage["model_max_sequence_length"],
        "max_sequence_length": stage["model_max_sequence_length"],
        "layer_types": layer_types,
        "sliding_window": model["swa_window"],
        "rope_theta": float(model["rope_theta"]),
        "rope_full_precision": model["rope_full_precision"],
        "rope_scaling": stage["full_attention_yarn"],
        "qk_norm": model["qk_norm"],
        "qk_norm_mode": model["qk_norm_mode"],
        "rms_norm_eps": model["rms_norm_eps"],
        "mlp_bias": False,
        "tie_word_embeddings": model["tie_word_embeddings"],
        "scale_embeds": False,
        "scale_output_layer_init": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": model["true_vocab_size"],
        "padded_vocab_size": model["padded_vocab_size"],
        "bos_token_id": None,
        "eos_token_id": 100257,
        "pad_token_id": 100277,
        "olmo3_weight_decay": True,
        "vocab_z_loss_coeff": optim["z_loss"],
        "use_siamese_norm": variant["use_siamese_norm"],
        "siamese_norm_variant": variant["siamese_norm_variant"],
        "use_depth_attention": variant["use_depth_attention"],
        "depth_attention_stride": variant["depth_attention_stride"],
        "depth_attention_recent_window": variant["depth_attention_recent_window"],
    }

