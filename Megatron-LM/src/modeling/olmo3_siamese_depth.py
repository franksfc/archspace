"""OLMo3 + Hybrid-Pre SiameseNorm + Depth-Attention for MindSpeed."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from modeling.olmo3_siamese_depth_base import (
    Olmo3DepthSelfAttentionBase,
    Olmo3SiameseDepthModelBase,
    Olmo3SiameseDepthTransformerLayerBase,
    _validate_architecture_config,
    make_olmo3_truncated_normal,
)
from modeling.olmo3_siamese_depth_components import (
    apply_fused_tp_full_projection_qk_norm,
)
from modeling.olmo3_runtime import (
    attach_olmo3_swa_config,
    validate_olmo3_mindspeed_runtime_args,
)
from modeling.olmo3_siamese_depth_config import Olmo3SiameseDepthSettings
from modeling.olmo3_swa import Olmo3DotProductAttention


class Olmo3DepthSelfAttention(Olmo3DepthSelfAttentionBase):
    """Depth-Attention V mixing followed by the layer's OLMo3 token attention."""

    optimize_depth_attention_mha = True
    validate_depth_partitioning = True

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


def _olmo3_siamese_residual_update(
    post_stream: Tensor,
    pre_stream: Tensor,
    residual_update: Tensor,
    *,
    post_residual_scale: float,
) -> tuple[Tensor, Tensor]:
    """Apply SiameseNorm's two updates without a residual-sized temporary."""
    if (
        post_stream.shape != pre_stream.shape
        or post_stream.shape != residual_update.shape
    ):
        raise ValueError(
            "SiameseNorm streams and residual update must have identical shapes; "
            f"got {tuple(post_stream.shape)}, {tuple(pre_stream.shape)}, and "
            f"{tuple(residual_update.shape)}."
        )
    return (
        torch.add(post_stream, residual_update, alpha=post_residual_scale),
        torch.add(pre_stream, residual_update),
    )


class Olmo3SiameseDepthTransformerLayer(Olmo3SiameseDepthTransformerLayerBase):
    """Hybrid-Pre SiameseNorm block with OLMo3 Full/SWA attention."""

    clone_initial_siamese_stream = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._siamese_post_residual_scale = 1.0 / math.sqrt(
            2.0 * self.layer_number
        )

    def _residual_branch(
        self,
        output_with_bias: tuple[Tensor, Tensor | None],
        residual_reference: Tensor,
    ) -> Tensor:
        """Fast-path the frozen OLMo3 no-bias, zero-dropout residual branch."""
        if (
            isinstance(output_with_bias, (tuple, list))
            and len(output_with_bias) == 2
            and isinstance(output_with_bias[0], Tensor)
            and output_with_bias[1] is None
            and output_with_bias[0].dtype == residual_reference.dtype
            and self.hidden_dropout == 0.0
        ):
            return output_with_bias[0]
        return super()._residual_branch(output_with_bias, residual_reference)

    def _combine_attention_streams(
        self,
        post_stream: Tensor,
        pre_normalized: Tensor,
    ) -> Tensor:
        """Fuse ``pre + post * scale`` into one elementwise operation."""
        attention_scale = self.siamese_hybrid_attn_scale.to(
            dtype=post_stream.dtype
        )
        return torch.addcmul(pre_normalized, post_stream, attention_scale)

    def _update_siamese_streams(
        self,
        post_stream: Tensor,
        pre_stream: Tensor,
        residual_update: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Use two axpy-style kernels for the two Siamese residual streams."""
        return _olmo3_siamese_residual_update(
            post_stream,
            pre_stream,
            residual_update,
            post_residual_scale=self._siamese_post_residual_scale,
        )


def get_olmo3_siamese_depth_layer_spec() -> ModuleSpec:
    """Build the composed Siamese/Depth/OLMo3 layer spec."""
    return ModuleSpec(
        module=Olmo3SiameseDepthTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3DepthSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=Olmo3DotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


class Olmo3SiameseDepthModel(Olmo3SiameseDepthModelBase):
    """Single-pass OLMo3 LM with SiameseNorm and Depth-Attention."""

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
) -> Olmo3SiameseDepthModel:
    """Validate and construct the composed OLMo3/Siamese/Depth model."""
    settings = Olmo3SiameseDepthSettings.from_model_config(model_config)
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
    config.depth_attention_stride = settings.depth_attention_stride
    config.depth_attention_recent_window = settings.depth_attention_recent_window
    _validate_architecture_config(
        config,
        pre_process=pre_process,
        post_process=post_process,
        use_transformer_engine_spec=use_transformer_engine_spec,
        allow_context_parallel=True,
        allow_sequence_parallel=True,
    )
    attach_olmo3_swa_config(config, settings)
    validate_olmo3_mindspeed_runtime_args(
        settings,
        model_impl="olmo3_siamese_depth",
    )
    return Olmo3SiameseDepthModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
        transformer_layer_spec=get_olmo3_siamese_depth_layer_spec(),
        rotary_base=settings.rope_theta,
    )
