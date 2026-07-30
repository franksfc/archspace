"""Checkpoint helpers for last-layer-only OLMo3 SiameseNorm modules."""

import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.utils import sharded_state_dict_default


FINAL_SIAMESE_NORM_NAMES = (
    "siamese_post_final_layernorm",
    "siamese_pre_final_layernorm",
)


def remap_final_siamese_norms_for_checkpoint(
    owner: torch.nn.Module,
    sharded_state_dict: ShardedStateDict,
    prefix: str,
    metadata: dict | None = None,
) -> None:
    """Replace erroneous layer-sharded final-norm checkpoint entries.

    Args:
        owner: Last transformer layer that owns the two final norms.
        sharded_state_dict: Layer state produced with the homogeneous layer offset.
        prefix: Local state-dict prefix for the last transformer layer.
        metadata: Optional distributed-checkpoint metadata.

    The replacement keeps each local state-dict key unchanged while rebuilding
    its ShardedTensor without the homogeneous layer offset. TransformerBlock can
    therefore continue applying its normal checkpoint-key prefix mapping.
    """
    for module_name in FINAL_SIAMESE_NORM_NAMES:
        module = getattr(owner, module_name)
        if module is None:
            raise RuntimeError(f"Final SiameseNorm module {module_name} was not initialized.")
        module_prefix = f"{prefix}{module_name}."
        replacement_state = sharded_state_dict_default(
            module,
            prefix=module_prefix,
            sharded_offsets=(),
            metadata=metadata,
        )
        existing_keys = {key for key in sharded_state_dict if key.startswith(module_prefix)}
        if existing_keys != replacement_state.keys():
            raise RuntimeError(
                f"Unexpected checkpoint entries for {module_name}: "
                f"existing={sorted(existing_keys)}, replacement={sorted(replacement_state)}"
            )
        sharded_state_dict.update(replacement_state)
