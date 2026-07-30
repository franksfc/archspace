"""Layer-local RoPE selection for OLMo3 long-context training.

OLMo3 extends only full-attention layers with YaRN. Sliding-window layers keep
the Stage-1 RoPE frequencies, so a single GPT-level rotary tensor is not
sufficient for the long-context stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
    get_pos_emb_on_this_cp_rank,
)
from modeling.olmo3_config import FULL_ATTENTION, SLIDING_ATTENTION, Olmo3YarnSettings


@dataclass(frozen=True)
class Olmo3RotaryEmbeddingOutput:
    """One layer's rotary frequencies and per-Q/K YaRN m-scale."""

    freqs: Tensor
    mscale: float = 1.0

    # Attention.forward intentionally uses this marker instead of importing a
    # modeling-specific type into Megatron Core.
    _olmo3_rotary_output: bool = True


@dataclass(frozen=True)
class Olmo3LayerwiseRotaryEmbeddingOutput:
    """Separate rotary outputs for SWA and full-attention layers."""

    sliding: Olmo3RotaryEmbeddingOutput
    full: Olmo3RotaryEmbeddingOutput


def select_olmo3_layer_rotary(
    rotary_pos_emb: Any,
    *,
    config: Any,
    layer_number: int,
) -> Any:
    """Select the layer's RoPE while leaving the Stage-1 tensor path unchanged."""

    if not isinstance(rotary_pos_emb, Olmo3LayerwiseRotaryEmbeddingOutput):
        return rotary_pos_emb
    layer_types = tuple(getattr(config, "olmo3_layer_types", ()))
    if len(layer_types) != int(config.num_layers):
        raise RuntimeError("OLMo3 layer-local RoPE requires one layer type per layer.")
    layer_index = int(layer_number) - 1
    if not 0 <= layer_index < len(layer_types):
        raise ValueError(
            f"OLMo3 layer_number must be in [1, {len(layer_types)}], got {layer_number}."
        )
    layer_type = layer_types[layer_index]
    if layer_type == SLIDING_ATTENTION:
        return rotary_pos_emb.sliding
    if layer_type == FULL_ATTENTION:
        return rotary_pos_emb.full
    raise ValueError(f"Unsupported OLMo3 attention type {layer_type!r}.")


class Olmo3LayerwiseRotaryEmbedding(torch.nn.Module):
    """Checkpoint-neutral Stage-3 Full/SWA rotary dispatcher.

    ``RotaryEmbedding.inv_freq`` is deliberately not a persistent buffer in
    MCore 0.12. This wrapper likewise registers no new parameter or buffer, so
    enabling it does not add, remove, or rename any Stage-1 checkpoint key.
    """

    def __init__(
        self,
        base_rotary: RotaryEmbedding,
        yarn: Olmo3YarnSettings,
    ) -> None:
        super().__init__()
        self.base_rotary = base_rotary
        self.yarn = yarn
        if self.base_rotary.seq_len_interpolation_factor is not None:
            raise ValueError("OLMo3 YaRN cannot be combined with linear RoPE interpolation.")

    def get_rotary_seq_len(self, *args: Any, **kwargs: Any) -> int:
        return int(self.base_rotary.get_rotary_seq_len(*args, **kwargs))

    def get_cos_sin(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        del args, kwargs
        raise NotImplementedError(
            "Mixed Full/SWA RoPE cannot use GPTModel's one-table flash-decode cache."
        )

    def _scaled_inv_freq(self) -> Tensor:
        base_inv_freq = self.base_rotary.inv_freq
        if base_inv_freq.device.type == "cpu" and not self.base_rotary.training:
            # Keep CPU construction usable in checkpoint-key and formula tests.
            device = base_inv_freq.device
        elif base_inv_freq.device.type == "cpu":
            if hasattr(torch, "npu") and torch.npu.is_available():
                device = torch.device("npu", torch.npu.current_device())
            elif torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = base_inv_freq.device
            if device.type != "cpu":
                self.base_rotary.inv_freq = base_inv_freq = base_inv_freq.to(
                    device=device
                )
        else:
            device = base_inv_freq.device

        interpolation = base_inv_freq / self.yarn.factor
        half_dim = int(base_inv_freq.numel())
        rotary_dim = half_dim * 2
        indices = torch.arange(half_dim, device=device, dtype=torch.float32)

        def dimension_from_rotations(rotations: int) -> float:
            return (
                rotary_dim
                * math.log(
                    self.yarn.old_context_len / (rotations * 2.0 * math.pi)
                )
                / (2.0 * math.log(self.yarn.theta))
            )

        low = max(int(math.floor(dimension_from_rotations(self.yarn.beta_fast))), 0)
        high = min(
            int(math.ceil(dimension_from_rotations(self.yarn.beta_slow))),
            half_dim - 1,
        )
        ramp = ((indices - low) / max(high - low, 1.0e-3)).clamp_(0.0, 1.0)
        return interpolation * ramp + base_inv_freq * (1.0 - ramp)

    def _yarn_forward(
        self,
        max_seq_len: int,
        offset: int,
        *,
        packed_seq: bool,
    ) -> Tensor:
        inv_freq = self._scaled_inv_freq()
        positions = (
            torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
            + offset
        )
        freqs = torch.outer(positions, inv_freq)
        if not self.base_rotary.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    @lru_cache(maxsize=32)
    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
    ) -> Olmo3LayerwiseRotaryEmbeddingOutput:
        sliding = self.base_rotary(
            max_seq_len,
            offset=offset,
            packed_seq=packed_seq,
        )
        full = self._yarn_forward(
            max_seq_len,
            offset,
            packed_seq=packed_seq,
        )
        return Olmo3LayerwiseRotaryEmbeddingOutput(
            sliding=Olmo3RotaryEmbeddingOutput(sliding),
            full=Olmo3RotaryEmbeddingOutput(full, self.yarn.attention_rescale_factor),
        )

    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None:
        # Old implementations may have serialized the non-persistent frequency
        # tensor. Ignore it just as MCore's RotaryEmbedding does.
        type(self).forward.cache_clear()
        state_dict.pop(f"{prefix}inv_freq", None)
        state_dict.pop(f"{prefix}base_rotary.inv_freq", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
