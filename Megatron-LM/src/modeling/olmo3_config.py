"""Strict configuration contract shared by the OLMo3 implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from modeling.olmo3_base_config import OLMO3_ROPE_THETA, Olmo3BaseSettings


SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class Olmo3YarnSettings:
    """The exact Full-attention-only YaRN recipe used by OLMo3 Stage 3."""

    factor: float
    beta_fast: int
    beta_slow: int
    old_context_len: int
    theta: float

    @property
    def attention_rescale_factor(self) -> float:
        """Return YaRN's per-Q/K m-scale from section 3.4 of the paper."""

        import math

        return 0.1 * math.log(self.factor) + 1.0


def default_olmo3_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
    """Return Ai2's repeating ``SWA, SWA, SWA, Full`` layer pattern."""
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}.")
    return tuple(
        FULL_ATTENTION if (layer_index + 1) % 4 == 0 else SLIDING_ATTENTION
        for layer_index in range(num_hidden_layers)
    )


def parse_olmo3_swa_fields(
    config: Any,
    *,
    num_hidden_layers: int,
    max_position_embeddings: int,
) -> tuple[int, tuple[str, ...], Olmo3YarnSettings | None]:
    """Parse and validate OLMo3's sequence length and per-layer SWA fields."""
    configured_sequence_length = getattr(
        config,
        "max_sequence_length",
        max_position_embeddings,
    )
    if (
        isinstance(configured_sequence_length, bool)
        or not isinstance(configured_sequence_length, int)
        or configured_sequence_length != max_position_embeddings
    ):
        raise ValueError(
            "OLMo3 max_sequence_length must equal max_position_embeddings; "
            f"got {configured_sequence_length!r} and {max_position_embeddings}."
        )

    if not hasattr(config, "sliding_window"):
        raise ValueError("OLMo3 config is missing required field 'sliding_window'.")
    sliding_window = getattr(config, "sliding_window")
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int):
        raise ValueError("OLMo3 sliding_window must be an integer token count.")
    if sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1, got {sliding_window}.")
    if sliding_window != 4096:
        raise ValueError(
            "OLMo3 requires the official 4096-token sliding window; "
            f"got {sliding_window}."
        )
    if sliding_window > max_position_embeddings:
        raise ValueError(
            "sliding_window cannot exceed max_position_embeddings for this pretraining port; "
            f"got {sliding_window} > {max_position_embeddings}."
        )

    if not hasattr(config, "layer_types"):
        raise ValueError("OLMo3 config is missing required field 'layer_types'.")
    raw_layer_types = getattr(config, "layer_types")
    if not isinstance(raw_layer_types, (list, tuple)):
        raise ValueError("OLMo3 layer_types must be a list or tuple.")
    layer_types = tuple(raw_layer_types)
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "OLMo3 layer_types must contain exactly one entry per transformer layer; "
            f"got {len(layer_types)} entries for {num_hidden_layers} layers."
        )

    expected = default_olmo3_layer_types(num_hidden_layers)
    if layer_types != expected:
        mismatches = [
            f"{index}:{actual!r}!={wanted!r}"
            for index, (actual, wanted) in enumerate(zip(layer_types, expected))
            if actual != wanted
        ]
        raise ValueError(
            "OLMo3 requires the official 3:1 SWA pattern "
            "(sliding, sliding, sliding, full); mismatches: "
            + ", ".join(mismatches[:8])
        )

    rope_scaling = getattr(config, "rope_scaling", None)
    yarn = None
    if rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            raise ValueError("OLMo3 rope_scaling must be null or a YaRN dictionary.")
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type != "yarn":
            raise ValueError(
                "OLMo3 long context supports only Full-attention YaRN scaling; "
                f"got rope_type={rope_type!r}."
            )
        expected_values = {
            "factor": 8.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 8192,
        }
        aliases = {
            "original_max_position_embeddings": "old_context_len",
        }
        parsed: dict[str, Any] = {}
        mismatches = []
        for key, expected_value in expected_values.items():
            value = rope_scaling.get(key, rope_scaling.get(aliases.get(key, "")))
            parsed[key] = value
            if value != expected_value:
                mismatches.append(f"{key}: expected {expected_value!r}, got {value!r}")
        if mismatches:
            raise ValueError(
                "OLMo3 Stage-3 YaRN must match the official recipe: "
                + "; ".join(mismatches)
            )
        if max_position_embeddings != 65536:
            raise ValueError(
                "OLMo3 Stage-3 YaRN requires max_position_embeddings=65536; "
                f"got {max_position_embeddings}."
            )
        theta = getattr(config, "rope_theta", None)
        if (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float))
            or float(theta) != OLMO3_ROPE_THETA
        ):
            raise ValueError(
                "OLMo3 YaRN requires rope_theta=500000; "
                f"got {theta!r}."
            )
        yarn = Olmo3YarnSettings(
            factor=float(parsed["factor"]),
            beta_fast=int(parsed["beta_fast"]),
            beta_slow=int(parsed["beta_slow"]),
            old_context_len=int(parsed["original_max_position_embeddings"]),
            theta=float(theta),
        )
    elif max_position_embeddings != 8192:
        raise ValueError(
            "OLMo3 without YaRN must keep the Stage-1/Stage-2 context at 8192; "
            f"got {max_position_embeddings}."
        )
    return sliding_window, layer_types, yarn


@dataclass(frozen=True)
class Olmo3Settings(Olmo3BaseSettings):
    """Validated OLMo3 backbone settings plus per-layer SWA."""

    sliding_window: int
    layer_types: tuple[str, ...]
    yarn: Olmo3YarnSettings | None

    @classmethod
    def from_model_config(cls, config: Any) -> "Olmo3Settings":
        """Parse the OLMo3 backbone contract and exact SWA pattern."""
        base = Olmo3BaseSettings.from_model_config(config)
        sliding_window, layer_types, yarn = parse_olmo3_swa_fields(
            config,
            num_hidden_layers=base.num_hidden_layers,
            max_position_embeddings=base.max_position_embeddings,
        )
        return cls(
            **asdict(base),
            sliding_window=sliding_window,
            layer_types=layer_types,
            yarn=yarn,
        )
