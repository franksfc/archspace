"""Strict base configuration contract for the OLMo3 architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


OLMO3_ROPE_THETA = 500_000.0
OLMO3_INIT_STD = 0.02
OLMO3_TRUNCATED_NORMAL_FACTOR = 3.0
OLMO3_RMS_NORM_EPS = 1.0e-6
OLMO3_TRUE_VOCAB_SIZE = 100_278
OLMO3_PADDED_VOCAB_SIZE = 100_352
OLMO3_PAD_TOKEN_ID = 100_277

# These are resolved directly from the locked OLMo-core factories:
# ``olmo3_1B``, ``olmo3_3B``, and ``olmo3_7B``.  In particular the 7B
# feed-forward width is 11008 (the Llama-2 7B width), not 16384.
OLMO3_SUPPORTED_GEOMETRIES = frozenset(
    {
        (16, 2_048, 8_192, 16, 16),
        (16, 3_328, 13_312, 16, 16),
        (32, 4_096, 11_008, 32, 32),
    }
)


def validate_official_olmo3_backbone(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    vocab_size: int,
    padded_vocab_size: int,
    pad_token_id: int,
    attention_dropout: float,
    hidden_dropout: float,
    initializer_range: float,
    truncated_normal_factor: float,
    rms_norm_eps: float,
    vocab_z_loss_coeff: float,
) -> None:
    """Reject silent drift from the locked 1B/3B/7B OLMo3 contracts."""

    geometry = (
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
    )
    if geometry not in OLMO3_SUPPORTED_GEOMETRIES:
        raise ValueError(
            "OLMo3 geometry must match the locked official 1B, 3B, or 7B "
            f"factory; got {geometry!r}."
        )
    exact_values = {
        "vocab_size": (vocab_size, OLMO3_TRUE_VOCAB_SIZE),
        "padded_vocab_size": (padded_vocab_size, OLMO3_PADDED_VOCAB_SIZE),
        "pad_token_id": (pad_token_id, OLMO3_PAD_TOKEN_ID),
        "attention_dropout": (attention_dropout, 0.0),
        "hidden_dropout": (hidden_dropout, 0.0),
        "initializer_range": (initializer_range, OLMO3_INIT_STD),
        "truncated_normal_factor": (
            truncated_normal_factor,
            OLMO3_TRUNCATED_NORMAL_FACTOR,
        ),
        "rms_norm_eps": (rms_norm_eps, OLMO3_RMS_NORM_EPS),
    }
    mismatches = [
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in exact_values.items()
        if actual != expected
    ]
    if vocab_z_loss_coeff not in (0.0, 1.0e-5):
        mismatches.append(
            "vocab_z_loss_coeff: expected 1e-5 for Stage 1/2/3 or 0.0 "
            f"for SFT, got {vocab_z_loss_coeff!r}"
        )
    if mismatches:
        raise ValueError(
            "OLMo3 common architecture/recipe fields differ from the locked "
            "official contract:\n  " + "\n  ".join(mismatches)
        )


def _required(config: Any, name: str) -> Any:
    if not hasattr(config, name):
        raise ValueError(f"OLMo3 config is missing required field {name!r}.")
    return getattr(config, name)


def _required_bool(config: Any, name: str) -> bool:
    value = _required(config, name)
    if not isinstance(value, bool):
        raise ValueError(f"OLMo3 config field {name!r} must be boolean.")
    return value


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return value


def _finite_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return value


def _dropout_probability(name: str, value: float) -> float:
    value = _finite_non_negative(name, value)
    if value >= 1.0:
        raise ValueError(f"{name} must be smaller than 1.0, got {value!r}.")
    return value


@dataclass(frozen=True)
class Olmo3BaseSettings:
    """Validated settings consumed by the standalone OLMo3 MCore model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    padded_vocab_size: int
    pad_token_id: int
    max_position_embeddings: int
    attention_dropout: float
    hidden_dropout: float
    initializer_range: float
    truncated_normal_factor: float
    rms_norm_eps: float
    rope_theta: float
    rope_full_precision: bool
    vocab_z_loss_coeff: float
    olmo3_weight_decay: bool
    use_cache: bool

    @classmethod
    def from_model_config(cls, config: Any) -> "Olmo3BaseSettings":
        """Parse OLMo3 fields and reject accidentally enabled experiment modules."""
        required_true = ("qk_norm", "olmo3_weight_decay", "rope_full_precision")
        for name in required_true:
            if not _required_bool(config, name):
                raise ValueError(f"{name} must be true for model_impl=olmo3.")

        required_false = (
            "scale_embeds",
            "scale_output_layer_init",
            "tie_word_embeddings",
            "attention_bias",
            "mlp_bias",
            "use_depth_attention",
            "use_siamese_norm",
        )
        for name in required_false:
            if _required_bool(config, name):
                raise ValueError(f"{name} must be false for model_impl=olmo3.")

        if _required(config, "qk_norm_mode") != "full_projection":
            raise ValueError("qk_norm_mode must be 'full_projection'.")
        if _required(config, "siamese_norm_variant") is not None:
            raise ValueError("siamese_norm_variant must be null when SiameseNorm is disabled.")
        if _required(config, "depth_attention_stride") is not None:
            raise ValueError("depth_attention_stride must be null when Depth-Attention is disabled.")
        if int(_required(config, "depth_attention_recent_window")) != 0:
            raise ValueError(
                "depth_attention_recent_window must be 0 when Depth-Attention is disabled."
            )
        if _required(config, "hidden_act") != "silu":
            raise ValueError("OLMo3 dense MLP requires hidden_act='silu'.")

        settings = cls(
            hidden_size=int(_required(config, "hidden_size")),
            intermediate_size=int(_required(config, "intermediate_size")),
            num_hidden_layers=int(_required(config, "num_hidden_layers")),
            num_attention_heads=int(_required(config, "num_attention_heads")),
            num_key_value_heads=int(_required(config, "num_key_value_heads")),
            vocab_size=int(_required(config, "vocab_size")),
            padded_vocab_size=int(_required(config, "padded_vocab_size")),
            pad_token_id=int(_required(config, "pad_token_id")),
            max_position_embeddings=int(_required(config, "max_position_embeddings")),
            attention_dropout=_dropout_probability(
                "attention_dropout", _required(config, "attention_dropout")
            ),
            hidden_dropout=_dropout_probability(
                "hidden_dropout", _required(config, "hidden_dropout")
            ),
            initializer_range=_finite_positive(
                "initializer_range", _required(config, "initializer_range")
            ),
            truncated_normal_factor=_finite_positive(
                "truncated_normal_factor", _required(config, "truncated_normal_factor")
            ),
            rms_norm_eps=_finite_positive("rms_norm_eps", _required(config, "rms_norm_eps")),
            rope_theta=_finite_positive("rope_theta", _required(config, "rope_theta")),
            rope_full_precision=_required_bool(config, "rope_full_precision"),
            vocab_z_loss_coeff=_finite_non_negative(
                "vocab_z_loss_coeff", _required(config, "vocab_z_loss_coeff")
            ),
            olmo3_weight_decay=_required_bool(config, "olmo3_weight_decay"),
            use_cache=_required_bool(config, "use_cache"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        positive_integers = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")
        if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
            raise ValueError(
                f"pad_token_id must be in [0, {self.vocab_size}), got {self.pad_token_id}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA/MHA."
            )
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size cannot be smaller than vocab_size.")
        if self.rope_theta != OLMO3_ROPE_THETA:
            raise ValueError(
                "OLMo3 requires rope_theta=500000; "
                f"got {self.rope_theta!r}."
            )
        validate_official_olmo3_backbone(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            initializer_range=self.initializer_range,
            truncated_normal_factor=self.truncated_normal_factor,
            rms_norm_eps=self.rms_norm_eps,
            vocab_z_loss_coeff=self.vocab_z_loss_coeff,
        )

    def validate_runtime(
        self, runtime_config: Any, *, padded_vocab_size: int, max_sequence_length: int
    ) -> None:
        """Ensure the JSON contract and parsed Megatron command describe one model."""
        expected_runtime_values = {
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.intermediate_size,
            "num_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_key_value_heads,
            "rotary_base": self.rope_theta,
            "rope_full_precision": self.rope_full_precision,
        }
        errors = [
            f"{name}: config.json={expected}, Megatron={getattr(runtime_config, name, None)}"
            for name, expected in expected_runtime_values.items()
            if getattr(runtime_config, name, None) != expected
        ]
        if padded_vocab_size != self.padded_vocab_size:
            errors.append(
                "padded_vocab_size: "
                f"config.json={self.padded_vocab_size}, Megatron={padded_vocab_size}"
            )
        if max_sequence_length != self.max_position_embeddings:
            errors.append(
                "max_position_embeddings: "
                f"config.json={self.max_position_embeddings}, "
                f"Megatron={max_sequence_length}"
            )
        if errors:
            raise ValueError(
                "OLMo3 model config and Megatron runtime arguments disagree:\n  "
                + "\n  ".join(errors)
            )

    @property
    def truncated_normal_cutoff(self) -> float:
        return self.initializer_range * self.truncated_normal_factor
