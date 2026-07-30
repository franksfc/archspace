"""Shared OLMo3 transformer topology for MindSpeed/Megatron Core 0.12."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor
from modeling.olmo3_mcore_common import Olmo3MCoreLMBase
from modeling.olmo3_rotary import select_olmo3_layer_rotary
from modeling.olmo3_siamese_depth_components import (
    TensorParallelFullProjectionRMSNorm,
    apply_full_projection_qk_norm,
)
from modeling.olmo3_swa import unwrap_external_ulysses_for_olmo3


def make_olmo3_truncated_normal(standard_deviation: float, cutoff_factor: float) -> Any:
    """Build the official truncated-normal initializer used by this OLMo3 run."""
    cutoff = standard_deviation * cutoff_factor

    def init_(tensor: Tensor) -> Tensor:
        return torch.nn.init.trunc_normal_(
            tensor, mean=0.0, std=standard_deviation, a=-cutoff, b=cutoff
        )

    return init_


def reordered_norm_residual(
    residual: Tensor, branch_output: Tensor, norm: torch.nn.Module
) -> Tensor:
    """Apply OLMo3 reordered norm: residual plus normalized branch output."""
    if residual.shape != branch_output.shape:
        raise ValueError(
            "OLMo3 residual and branch output must have identical shapes; "
            f"got {tuple(residual.shape)} and {tuple(branch_output.shape)}."
        )
    normalized = norm(branch_output)
    if not isinstance(normalized, Tensor):
        raise TypeError(
            "OLMo3 reordered norm must return a Tensor, "
            f"got {type(normalized).__name__}."
        )
    return residual + normalized


class Olmo3BaseSelfAttention(SelfAttention):
    """Self-attention with independent full-projection Q and K RMSNorm."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )
        # MindSpeed-LLM patches Attention.__init__ and otherwise performs a
        # full-sequence Ulysses A2A before the OLMo3 core.  Restore the single
        # communication owner so SWA sees CP-local Q/K/V and Full Attention
        # continues to use the core's fused QKV Ulysses path.
        self.core_attention = unwrap_external_ulysses_for_olmo3(
            self.core_attention
        )
        if int(self.config.tensor_model_parallel_size) > 1:
            self.q_layernorm = TensorParallelFullProjectionRMSNorm(
                self.config,
                self.query_projection_size,
                self.config.layernorm_epsilon,
            )
            self.k_layernorm = TensorParallelFullProjectionRMSNorm(
                self.config,
                self.kv_projection_size,
                self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = build_module(
                TENorm,
                config=self.config,
                hidden_size=self.query_projection_size,
                eps=self.config.layernorm_epsilon,
            )
            self.k_layernorm = build_module(
                TENorm,
                config=self.config,
                hidden_size=self.kv_projection_size,
                eps=self.config.layernorm_epsilon,
            )

    def _apply_qk_norm(
        self, query: Tensor, key: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Keep the OLMo3 path on its original independent TP collectives."""
        return apply_full_projection_qk_norm(
            query,
            key,
            self.q_layernorm,
            self.k_layernorm,
        )

    def get_query_key_value_tensors(
        self, hidden_states: Tensor, key_value_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V, then normalize complete Q and K projections independently."""
        if key_value_states is not None:
            raise ValueError("Olmo3BaseSelfAttention supports decoder self-attention only.")
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        mixed_qkv = mixed_qkv.view(
            *mixed_qkv.size()[:-1],
            self.num_query_groups_per_partition,
            (heads_per_group + 2) * self.hidden_size_per_attention_head,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )
        query = query.reshape(
            query.size(0),
            query.size(1),
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query, key = self._apply_qk_norm(query, key)
        return query, key, value


class Olmo3BaseTransformerLayer(TransformerLayer):
    """Official OLMo3 reordered-norm decoder layer."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )
        self.post_attention_layernorm = self._new_norm()
        self.post_feedforward_layernorm = self._new_norm()

    def _new_norm(self) -> torch.nn.Module:
        return build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def _branch_output(
        self, output_with_bias: tuple[Tensor, Tensor | None], residual_reference: Tensor
    ) -> Tensor:
        if not isinstance(output_with_bias, (tuple, list)) or len(output_with_bias) != 2:
            raise TypeError("OLMo3 residual blocks must return an (output, bias) pair.")
        output, bias = output_with_bias
        if not isinstance(output, Tensor) or (bias is not None and not isinstance(bias, Tensor)):
            raise TypeError("OLMo3 residual output and optional bias must be tensors.")
        if output.dtype != residual_reference.dtype:
            output = output.to(residual_reference.dtype)
            if bias is not None:
                bias = bias.to(residual_reference.dtype)
        if bias is not None:
            output = output + bias
        return F.dropout(output, p=self.hidden_dropout, training=self.training, inplace=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Any = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: Tensor | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Any]:
        """Run Attention and MLP with branch-output RMSNorm before each residual add."""
        del kwargs
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if context_mask is not None:
            raise ValueError("OLMo3 is decoder-only and does not support cross-attention.")

        residual = hidden_states
        rotary_pos_emb = select_olmo3_layer_rotary(
            rotary_pos_emb,
            config=self.config,
            layer_number=self.layer_number,
        )
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        attention_output = self._branch_output(attention_output_with_bias, residual)
        hidden_states = reordered_norm_residual(
            residual, attention_output, self.post_attention_layernorm
        )

        residual = hidden_states
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, hidden_states
            )
        else:
            mlp_output_with_bias = self.mlp(hidden_states)
        mlp_output = self._branch_output(mlp_output_with_bias, residual)
        hidden_states = reordered_norm_residual(
            residual, mlp_output, self.post_feedforward_layernorm
        )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output, context


def _validate_architecture_config(
    config: TransformerConfig,
    *,
    pre_process: bool,
    post_process: bool,
    use_transformer_engine_spec: bool,
    allow_context_parallel: bool = False,
    allow_sequence_parallel: bool = False,
) -> None:
    """Reject runtime modes outside the supported production OLMo3 path."""
    if not use_transformer_engine_spec:
        raise ValueError("OLMo3 requires the MindSpeed Transformer Engine spec path.")
    tensor_parallel_size = int(config.tensor_model_parallel_size)
    if tensor_parallel_size < 1:
        raise ValueError("Tensor parallel size must be positive.")
    if (
        config.num_attention_heads % tensor_parallel_size != 0
        or config.num_query_groups % tensor_parallel_size != 0
    ):
        raise ValueError(
            "Full-projection Q/K RMSNorm requires attention heads and query groups "
            "to be divisible by tensor parallel size."
        )
    if int(config.pipeline_model_parallel_size) != 1 or not pre_process or not post_process:
        raise ValueError("The supported OLMo3 production path requires PP=1.")
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("The supported OLMo3 path does not use virtual pipeline parallelism.")
    if int(config.context_parallel_size) != 1 and not allow_context_parallel:
        raise ValueError("This OLMo3 path requires CP=1 unless CP support is enabled.")
    if config.sequence_parallel and not allow_sequence_parallel:
        raise ValueError(
            "This OLMo3 path requires sequence_parallel=False unless SP support is enabled."
        )
    if config.sequence_parallel:
        if tensor_parallel_size < 2:
            raise ValueError("Sequence parallelism requires tensor parallel size >= 2.")
        if int(config.seq_length) % (
            tensor_parallel_size * int(config.context_parallel_size)
        ):
            raise ValueError(
                "Sequence length must be divisible by TP x CP when sequence "
                "parallelism is enabled."
            )
    if config.fp8 is not None:
        raise ValueError("The supported OLMo3 path does not use FP8.")
    if config.cross_entropy_loss_fusion:
        raise ValueError("Dense vocabulary z-loss requires cross_entropy_loss_fusion=False.")
    if getattr(config, "mtp_num_layers", None):
        raise ValueError("The supported OLMo3 path does not use MTP layers.")
    if config.num_moe_experts is not None:
        raise ValueError("This OLMo3 architecture is dense and does not use MoE layers.")
    if config.normalization != "RMSNorm":
        raise ValueError("OLMo3 requires normalization='RMSNorm'.")
    if config.num_query_groups is None or config.num_query_groups < 1:
        raise ValueError("OLMo3 requires at least one Q/K/V query group.")
    if config.num_attention_heads % config.num_query_groups != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_query_groups for OLMo3 GQA/MHA."
        )
    if config.multi_latent_attention:
        raise ValueError("OLMo3 full-projection Q/K RMSNorm is incompatible with MLA.")


class Olmo3BaseModel(Olmo3MCoreLMBase):
    """Single-pass causal LM using official OLMo3 reordered normalization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        embedding_weight = self.backbone.embedding.word_embeddings.weight
        setattr(embedding_weight, "_olmo3_input_embedding", True)

    def _embed_tokens_sbh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        """Embed tokens without scaling, matching OLMo3."""
        if position_ids is None:
            position_ids = torch.arange(
                tokens.shape[1], device=tokens.device, dtype=torch.long
            ).unsqueeze(0)
        return self.backbone.embedding(input_ids=tokens, position_ids=position_ids)

    def forward(
        self,
        tokens: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        global_step: int | None = None,
        labels_are_shifted: bool = False,
        return_per_sample_loss_sums: bool = False,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        runtime_gather_output: bool | None = None,
        use_cache: bool | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> Tensor:
        """Return training losses or inference logits with optional MCore KV cache."""
        del loss_mask, global_step
        if tokens.dim() != 2 or tokens.shape[1] < 1:
            raise ValueError(
                "tokens must have non-empty [batch, sequence] shape; "
                f"got {tuple(tokens.shape)}."
            )
        if labels is None:
            return self._forward_inference_logits(
                tokens,
                position_ids,
                attention_mask,
                inference_context=inference_context,
                inference_params=inference_params,
                use_cache=use_cache,
                packed_seq_params=packed_seq_params,
                runtime_gather_output=runtime_gather_output,
            )
        active_context = self._resolve_inference_context(
            inference_context,
            inference_params,
            use_cache,
            batch_size=tokens.shape[0],
            query_length=tokens.shape[1],
            packed_seq_params=packed_seq_params,
        )
        if active_context is not None:
            raise ValueError("labels and KV-cache inference cannot be used in the same forward.")
        if labels.shape != tokens.shape:
            raise ValueError(
                "tokens and labels must share [batch, sequence] shape; "
                f"got {tuple(tokens.shape)} and {tuple(labels.shape)}."
            )
        if position_ids is None:
            position_ids = (
                torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long)
                .unsqueeze(0)
                .expand_as(tokens)
            )
        elif position_ids.shape != tokens.shape:
            raise ValueError("position_ids must match the tokens shape.")
        if attention_mask is not None and (
            attention_mask.dim() != 2 or attention_mask.shape != tokens.shape
        ):
            raise ValueError("attention_mask must be None or a 2D tensor matching tokens.")

        hidden_states = self._embed_tokens_sbh(tokens, position_ids)
        hidden_states = self._decoder_forward_sbh(
            hidden_states,
            attention_mask,
            position_ids,
            packed_seq_params=packed_seq_params,
        )
        return self._compute_shifted_loss_components(
            hidden_states.transpose(0, 1).contiguous(),
            labels,
            labels_are_shifted=labels_are_shifted,
            return_per_sample_sums=return_per_sample_loss_sums,
        )
