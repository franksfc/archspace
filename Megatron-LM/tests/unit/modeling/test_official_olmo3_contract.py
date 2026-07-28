"""Locked OLMo-core architecture and project-config regression tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeling.olmo3_base_config import (
    OLMO3_SUPPORTED_GEOMETRIES,
    Olmo3BaseSettings,
)
from modeling.olmo3_config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Olmo3Settings,
    default_olmo3_layer_types,
)
from modeling.olmo3_siamese_depth_config import Olmo3SiameseDepthSettings


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CONFIG_SOURCE = (
    ROOT / "third_party/OLMo-core/src/olmo_core/nn/transformer/config.py"
)
OFFICIAL_INIT_SOURCE = (
    ROOT / "third_party/OLMo-core/src/olmo_core/nn/transformer/init.py"
)
OFFICIAL_LONG_CONTEXT_SOURCE = (
    ROOT / "third_party/OLMo-core/src/scripts/train/OLMo3/OLMo3-7B-long-context.py"
)
PACKED_DATASET_SOURCE = ROOT / "src/runtime/olmo3_packed_dataset.py"
PRETRAIN_SOURCE = ROOT / "src/pretrain_olmo3_mindspeed.py"

OFFICIAL_MODELS = {
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


def _valid_config(size: str = "1b", *, siamese_depth: bool = False, **overrides):
    geometry = OFFICIAL_MODELS[size]
    values = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_dropout": 0.0,
        "hidden_size": geometry["hidden_size"],
        "intermediate_size": geometry["ffn_hidden_size"],
        "num_hidden_layers": geometry["num_layers"],
        "num_attention_heads": geometry["num_attention_heads"],
        "num_key_value_heads": geometry["num_query_groups"],
        "vocab_size": 100_278,
        "padded_vocab_size": 100_352,
        "pad_token_id": 100_277,
        "max_position_embeddings": 8_192,
        "max_sequence_length": 8_192,
        "mlp_bias": False,
        "initializer_range": 0.02,
        "truncated_normal_factor": 3.0,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 500_000.0,
        "rope_full_precision": True,
        "rope_scaling": None,
        "vocab_z_loss_coeff": 1.0e-5,
        "depth_attention_stride": 8 if siamese_depth else None,
        "depth_attention_recent_window": 0,
        "qk_norm": True,
        "qk_norm_mode": "full_projection",
        "use_depth_attention": siamese_depth,
        "use_siamese_norm": siamese_depth,
        "siamese_norm_variant": "hybrid_pre" if siamese_depth else None,
        "olmo3_weight_decay": True,
        "scale_embeds": False,
        "scale_output_layer_init": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "sliding_window": 4_096,
        "layer_types": list(default_olmo3_layer_types(geometry["num_layers"])),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _factory_body(source: str, factory: str) -> str:
    marker = f"    def {factory}("
    start = source.index(marker)
    next_method = source.find("\n    @classmethod", start + len(marker))
    return source[start:] if next_method < 0 else source[start:next_method]


def test_project_model_files_match_locked_olmo_core_factories() -> None:
    source = OFFICIAL_CONFIG_SOURCE.read_text(encoding="utf-8")
    assert "def olmo3_1B(" in source
    assert "def olmo3_3B(" in source
    assert "def olmo3_7B(" in source
    assert "pattern=[4096, 4096, 4096, -1]" in source

    # The locked 7B factory delegates to llama2_7B without a hidden-size
    # multiplier. llama_like therefore resolves ceil_256(8 * 4096 / 3)=11008.
    olmo2_7b = _factory_body(source, "olmo2_7B")
    assert "return cls.llama2_7B(" in olmo2_7b
    assert "hidden_size_multiplier" not in olmo2_7b
    assert 256 * math.ceil(int(8 * 4_096 / 3) / 256) == 11_008

    actual_geometries = set()
    for size, expected in OFFICIAL_MODELS.items():
        with (ROOT / f"configs/models/{size}.json").open(encoding="utf-8") as handle:
            model = json.load(handle)["model"]
        for field, expected_value in expected.items():
            assert model[field] == expected_value
        assert model["swa_pattern"] == [4096, 4096, 4096, -1]
        assert model["swa_window"] == 4096
        assert model["rope_theta"] == 500_000
        assert model["rope_full_precision"] is True
        assert model["qk_norm_mode"] == "full_projection"
        assert model["rms_norm_eps"] == 1.0e-6
        assert model["initializer_range"] == 0.02
        assert model["tie_word_embeddings"] is False
        actual_geometries.add(
            (
                model["num_layers"],
                model["hidden_size"],
                model["ffn_hidden_size"],
                model["num_attention_heads"],
                model["num_query_groups"],
            )
        )
    assert actual_geometries == set(OLMO3_SUPPORTED_GEOMETRIES)


def test_official_truncated_normal_recipe_is_locked() -> None:
    source = OFFICIAL_INIT_SOURCE.read_text(encoding="utf-8")
    assert "std: float = 0.02" in source
    assert "a=-3 * std" in source
    assert "b=3 * std" in source


def test_stage3_runtime_uses_the_official_long_context_seed() -> None:
    official = OFFICIAL_LONG_CONTEXT_SOURCE.read_text(encoding="utf-8")
    packed_dataset = PACKED_DATASET_SOURCE.read_text(encoding="utf-8")
    pretrain = PRETRAIN_SOURCE.read_text(encoding="utf-8")

    assert "SEED = 4123" in official
    assert "OLMO3_LONGMINO_DATA_SEED = 4123" in packed_dataset
    assert "args.sampler_data_seed != OLMO3_LONGMINO_DATA_SEED" in pretrain


@pytest.mark.parametrize("size", ("1b", "3b", "7b"))
def test_base_and_siamese_depth_accept_each_official_geometry(size: str) -> None:
    base = Olmo3Settings.from_model_config(_valid_config(size))
    experimental = Olmo3SiameseDepthSettings.from_model_config(
        _valid_config(size, siamese_depth=True)
    )
    expected = OFFICIAL_MODELS[size]
    for settings in (base, experimental):
        assert settings.hidden_size == expected["hidden_size"]
        assert settings.intermediate_size == expected["ffn_hidden_size"]
        assert settings.num_hidden_layers == expected["num_layers"]
        assert settings.sliding_window == 4_096
        assert settings.rope_theta == 500_000.0
        assert settings.rope_full_precision
    assert experimental.depth_attention_stride == 8


def test_non_official_7b_ffn_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="locked official 1B, 3B, or 7B"):
        Olmo3BaseSettings.from_model_config(
            _valid_config("7b", intermediate_size=16_384)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sliding_window", 2_048, "4096-token sliding window"),
        ("initializer_range", 0.01, "initializer_range"),
        ("truncated_normal_factor", 2.0, "truncated_normal_factor"),
        ("rms_norm_eps", 1.0e-5, "rms_norm_eps"),
        ("attention_dropout", 0.1, "attention_dropout"),
        ("vocab_z_loss_coeff", 2.0e-5, "vocab_z_loss_coeff"),
    ],
)
def test_strict_contract_rejects_recipe_drift(field: str, value, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Olmo3Settings.from_model_config(_valid_config(**{field: value}))


def test_layer_pattern_is_exact_three_swa_then_full() -> None:
    pattern = default_olmo3_layer_types(32)
    assert pattern[:4] == (
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        FULL_ATTENTION,
    )
    assert pattern.count(SLIDING_ATTENTION) == 24
    assert pattern.count(FULL_ATTENTION) == 8
