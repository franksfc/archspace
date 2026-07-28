"""Stage 1 -> 2 -> 3 keyspace and Full-only YaRN regression tests."""

from __future__ import annotations
# ruff: noqa: E402

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from modeling.olmo3_config import Olmo3Settings
from modeling.olmo3_rotary import (
    Olmo3LayerwiseRotaryEmbedding,
    select_olmo3_layer_rotary,
)
from modeling.olmo3_siamese_depth_config import Olmo3SiameseDepthSettings
from olmo3_pipeline.model_config import build_model_config


ROOT = Path(__file__).resolve().parents[3]


class _CpuRotary(torch.nn.Module):
    def __init__(self, dim: int, theta: float = 500_000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.rotary_interleaved = False
        self.seq_len_interpolation_factor = None

    def forward(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False
    ) -> torch.Tensor:
        del packed_seq
        positions = torch.arange(max_seq_len, dtype=torch.float32) + offset
        frequencies = torch.outer(positions, self.inv_freq)
        return torch.cat((frequencies, frequencies), dim=-1)[:, None, None, :]

    def get_rotary_seq_len(self, *args, **kwargs) -> int:
        del args, kwargs
        return 0


def _load(section: str, name: str) -> dict:
    with (ROOT / f"configs/{section}/{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolved_model_config(size: str, variant: str, stage: str) -> dict:
    resolved = {}
    for section, name in (
        ("models", size),
        ("variants", variant),
        ("stages", stage),
    ):
        resolved.update(_load(section, name))
    return build_model_config(resolved)


KEYSPACE_FIELDS = {
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "padded_vocab_size",
    "qk_norm",
    "qk_norm_mode",
    "rms_norm_eps",
    "mlp_bias",
    "attention_bias",
    "tie_word_embeddings",
    "use_siamese_norm",
    "siamese_norm_variant",
    "use_depth_attention",
    "depth_attention_stride",
    "depth_attention_recent_window",
}


@pytest.mark.parametrize("size", ("1b", "3b", "7b"))
@pytest.mark.parametrize("variant", ("base", "siamese_depth"))
def test_stage_transition_keeps_every_parameter_shaping_field(
    size: str, variant: str
) -> None:
    configs = {
        stage: _resolved_model_config(size, variant, stage)
        for stage in ("stage1", "stage2", "stage3")
    }
    reference = {field: configs["stage1"][field] for field in KEYSPACE_FIELDS}
    for stage in ("stage2", "stage3"):
        assert {field: configs[stage][field] for field in KEYSPACE_FIELDS} == reference

    settings_type = (
        Olmo3Settings if variant == "base" else Olmo3SiameseDepthSettings
    )
    stage1 = settings_type.from_model_config(type("Config", (), configs["stage1"])())
    stage2 = settings_type.from_model_config(type("Config", (), configs["stage2"])())
    stage3 = settings_type.from_model_config(type("Config", (), configs["stage3"])())
    assert stage1.yarn is None
    assert stage2.yarn is None
    assert stage3.yarn is not None
    assert stage1.max_position_embeddings == stage2.max_position_embeddings == 8_192
    assert stage3.max_position_embeddings == 65_536


@pytest.mark.parametrize("head_dim", (128, 208))
def test_yarn_matches_locked_olmo_core_formula_and_adds_no_keys(
    head_dim: int,
) -> None:
    stage3_config = _resolved_model_config(
        "3b" if head_dim == 208 else "1b", "base", "stage3"
    )
    settings = Olmo3Settings.from_model_config(type("Config", (), stage3_config)())
    assert settings.yarn is not None
    base = _CpuRotary(head_dim)
    keys_before = tuple(base.state_dict())
    rotary = Olmo3LayerwiseRotaryEmbedding(base, settings.yarn)
    assert tuple(rotary.state_dict()) == keys_before == ()

    actual = rotary._scaled_inv_freq()
    extrapolation = base.inv_freq
    interpolation = extrapolation / 8.0
    indices = torch.arange(extrapolation.numel(), dtype=torch.float32)

    def dimension_from_rotations(rotations: int) -> float:
        return (
            head_dim
            * math.log(8_192 / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(500_000.0))
        )

    low = max(int(math.floor(dimension_from_rotations(32))), 0)
    high = min(
        int(math.ceil(dimension_from_rotations(1))),
        extrapolation.numel() - 1,
    )
    ramp = ((indices - low) / max(high - low, 1.0e-3)).clamp(0.0, 1.0)
    expected = interpolation * ramp + extrapolation * (1.0 - ramp)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_yarn_is_selected_only_for_full_attention_layers() -> None:
    config_dict = _resolved_model_config("1b", "siamese_depth", "stage3")
    settings = Olmo3SiameseDepthSettings.from_model_config(
        type("Config", (), config_dict)()
    )
    assert settings.yarn is not None
    rotary = Olmo3LayerwiseRotaryEmbedding(_CpuRotary(128), settings.yarn)
    outputs = rotary(128, packed_seq=True)
    runtime_config = type(
        "Runtime",
        (),
        {
            "num_layers": 16,
            "olmo3_layer_types": settings.layer_types,
        },
    )()
    sliding = select_olmo3_layer_rotary(
        outputs, config=runtime_config, layer_number=1
    )
    full = select_olmo3_layer_rotary(
        outputs, config=runtime_config, layer_number=4
    )
    torch.testing.assert_close(
        sliding.freqs,
        rotary.base_rotary(128, packed_seq=True),
    )
    assert sliding.mscale == 1.0
    assert full.mscale == pytest.approx(0.1 * math.log(8.0) + 1.0)
    assert not torch.equal(full.freqs, sliding.freqs)
