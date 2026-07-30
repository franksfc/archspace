"""Standalone OLMo3 architecture for MindSpeed/Megatron Core 0.12."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from modeling.olmo3_base import (
    Olmo3BaseModel,
    Olmo3BaseSelfAttention,
    Olmo3BaseTransformerLayer,
    _validate_architecture_config,
    make_olmo3_truncated_normal,
)
from modeling.olmo3_siamese_depth_components import (
    apply_fused_tp_full_projection_qk_norm,
)
from modeling.olmo3_config import Olmo3Settings
from modeling.olmo3_runtime import (
    attach_olmo3_swa_config,
    validate_olmo3_mindspeed_runtime_args,
)
from modeling.olmo3_swa import Olmo3DotProductAttention


class Olmo3SelfAttention(Olmo3BaseSelfAttention):
    """OLMo3 full-projection Q/K RMSNorm plus per-layer OLMo3 attention."""

    def _apply_qk_norm(
        self, query: Tensor, key: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fuse independent Q/K statistics into one TP collective."""
        return apply_fused_tp_full_projection_qk_norm(
            query,
            key,
            self.q_layernorm,
            self.k_layernorm,
        )


class Olmo3TransformerLayer(Olmo3BaseTransformerLayer):
    """OLMo3 reordered-norm block whose token attention is selected per layer."""


def get_olmo3_layer_spec() -> ModuleSpec:
    """Build the OLMo3 layer spec with native Full/SWA dispatch."""
    return ModuleSpec(
        module=Olmo3TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=Olmo3DotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


class Olmo3Model(Olmo3BaseModel):
    """Single-pass causal LM implementing the pretraining OLMo3 architecture."""

    # MCore's dynamic/flash-decode branches bypass the per-layer SWA core.
    cache_static_only = True
    cache_supports_flash_decode = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        embedding_weight = self.backbone.embedding.word_embeddings.weight
        setattr(embedding_weight, "_olmo3_input_embedding", True)


def build_model(
    *,
    config: TransformerConfig,
    model_config: Any,
    vocab_size: int,
    max_sequence_length: int,
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    use_transformer_engine_spec: bool = False,
) -> Olmo3Model:
    """Validate and construct OLMo3 with official 3:1 sliding attention."""
    settings = Olmo3Settings.from_model_config(model_config)
    config.rope_full_precision = settings.rope_full_precision
    settings.validate_runtime(
        config,
        padded_vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
    )
    init_method = make_olmo3_truncated_normal(
        settings.initializer_range,
        settings.truncated_normal_factor,
    )
    config.normalization = "RMSNorm"
    config.layernorm_epsilon = settings.rms_norm_eps
    config.qk_layernorm = True
    config.activation_func = F.silu
    config.gated_linear_unit = True
    config.add_bias_linear = False
    config.attention_dropout = settings.attention_dropout
    config.hidden_dropout = settings.hidden_dropout
    config.init_method_std = settings.initializer_range
    config.init_method = init_method
    config.output_layer_init_method = init_method
    config.vocab_z_loss_coeff = settings.vocab_z_loss_coeff
    config.olmo3_weight_decay = settings.olmo3_weight_decay
    _validate_architecture_config(
        config,
        pre_process=pre_process,
        post_process=post_process,
        use_transformer_engine_spec=use_transformer_engine_spec,
        allow_context_parallel=True,
        allow_sequence_parallel=True,
    )
    attach_olmo3_swa_config(config, settings)
    validate_olmo3_mindspeed_runtime_args(settings, model_impl="olmo3")
    return Olmo3Model(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
        transformer_layer_spec=get_olmo3_layer_spec(),
        rotary_base=settings.rope_theta,
    )
