"""Hugging Face configuration for the OLMo 3 Siamese-Norm/Depth-Attention model."""

from __future__ import annotations

from transformers.models.olmo3.configuration_olmo3 import Olmo3Config


class Olmo3SiameseDepthConfig(Olmo3Config):
    """OLMo 3 plus the checkpoint-compatible Siamese/Depth extensions."""

    model_type = "olmo3_siamese_depth"

    def __init__(
        self,
        *args,
        vocab_size: int | None = None,
        true_vocab_size: int = 100278,
        padded_vocab_size: int = 100352,
        qk_norm_mode: str = "full_projection",
        use_siamese_norm: bool = True,
        siamese_norm_variant: str = "hybrid_pre",
        use_depth_attention: bool = True,
        depth_attention_stride: int = 8,
        depth_attention_recent_window: int = 0,
        rope_full_precision: bool = True,
        **kwargs,
    ):
        # ``PretrainedConfig.to_diff_dict()`` constructs a no-argument instance
        # of this class.  Make that default instance internally consistent
        # instead of inheriting OLMo3's unrelated 50304-token default.
        if vocab_size is None:
            vocab_size = true_vocab_size
        super().__init__(*args, vocab_size=vocab_size, **kwargs)
        self.true_vocab_size = int(true_vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)
        self.qk_norm = True
        self.qk_norm_mode = qk_norm_mode
        self.use_siamese_norm = bool(use_siamese_norm)
        self.siamese_norm_variant = siamese_norm_variant
        self.use_depth_attention = bool(use_depth_attention)
        self.depth_attention_stride = int(depth_attention_stride)
        self.depth_attention_recent_window = int(depth_attention_recent_window)
        self.rope_full_precision = bool(rope_full_precision)
        self._validate_siamese_depth()

    def _validate_siamese_depth(self) -> None:
        if self.vocab_size != self.true_vocab_size:
            raise ValueError(
                "HF vocab_size must equal true_vocab_size after padded-row removal; "
                f"got {self.vocab_size} and {self.true_vocab_size}."
            )
        if self.padded_vocab_size < self.true_vocab_size:
            raise ValueError("padded_vocab_size cannot be smaller than true_vocab_size.")
        if self.qk_norm_mode != "full_projection":
            raise ValueError("qk_norm_mode must be 'full_projection'.")
        if not self.use_siamese_norm or self.siamese_norm_variant != "hybrid_pre":
            raise ValueError("This remote model requires Hybrid-Pre Siamese Norm.")
        if not self.use_depth_attention:
            raise ValueError("This remote model requires Depth Attention.")
        if self.depth_attention_stride < 1:
            raise ValueError("depth_attention_stride must be positive.")
        if self.depth_attention_recent_window < 0:
            raise ValueError("depth_attention_recent_window must be non-negative.")
        if not self.rope_full_precision:
            raise ValueError("OLMo 3 requires FP32 Q/K RoPE.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        expected_layer_types = [
            "sliding_attention" if (index + 1) % 4 else "full_attention"
            for index in range(self.num_hidden_layers)
        ]
        if list(self.layer_types) != expected_layer_types:
            raise ValueError("OLMo 3 requires the repeating SWA,SWA,SWA,Full pattern.")


__all__ = ["Olmo3SiameseDepthConfig"]
