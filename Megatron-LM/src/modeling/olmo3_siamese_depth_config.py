"""Strict configuration contract for OLMo3 + SiameseNorm + Depth-Attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from modeling.olmo3_siamese_depth_base_config import Olmo3SiameseDepthBaseSettings
from modeling.olmo3_config import Olmo3YarnSettings, parse_olmo3_swa_fields


@dataclass(frozen=True)
class Olmo3SiameseDepthSettings(Olmo3SiameseDepthBaseSettings):
    """Validated Siamese/Depth settings plus OLMo3 per-layer SWA."""

    sliding_window: int
    layer_types: tuple[str, ...]
    yarn: Olmo3YarnSettings | None

    @classmethod
    def from_model_config(cls, config: Any) -> "Olmo3SiameseDepthSettings":
        """Parse both the existing Siamese/Depth contract and OLMo3 SWA."""
        base = Olmo3SiameseDepthBaseSettings.from_model_config(config)
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
