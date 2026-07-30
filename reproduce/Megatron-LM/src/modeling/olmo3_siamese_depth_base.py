"""Shared OLMo3 Hybrid-Pre SiameseNorm and Depth-Attention topology."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor
from modeling.olmo3_mcore_common import Olmo3MCoreLMBase
from modeling.olmo3_rotary import select_olmo3_layer_rotary
from modeling.olmo3_siamese_depth_checkpointing import (
    remap_final_siamese_norms_for_checkpoint,
)
from modeling.olmo3_siamese_depth_components import (
    DepthAttentionContext,
    Olmo3ArchitectureContext,
    TensorParallelFullProjectionRMSNorm,
    apply_full_projection_qk_norm,
    depth_attention_mix,
    siamese_norm_residual_update,
)
from modeling.olmo3_swa import unwrap_external_ulysses_for_olmo3


def make_olmo3_truncated_normal(standard_deviation: float, cutoff_factor: float) -> Any:
    """Build the OLMo3 initializer selected by the dedicated model config."""
    cutoff = standard_deviation * cutoff_factor

    def init_(tensor: Tensor) -> Tensor:
        return torch.nn.init.trunc_normal_(
            tensor, mean=0.0, std=standard_deviation, a=-cutoff, b=cutoff
        )

    return init_


class DepthAwareCoreAttention(torch.nn.Module):
    """Inject Depth-Attention into the V path immediately after RoPE."""

    def __init__(
        self,
        core_attention: torch.nn.Module,
        layer_index: int,
        *,
        optimize_mha: bool = False,
        expected_sequence_length: int | None = None,
        expected_query_heads: int | None = None,
        expected_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.core_attention = core_attention
        self.layer_index = layer_index
        self.optimize_mha = optimize_mha
        self.expected_sequence_length = expected_sequence_length
        self.expected_query_heads = expected_query_heads
        self.expected_kv_heads = expected_kv_heads
        self.olmo3_keeps_packed_sbhd = bool(
            getattr(core_attention, "olmo3_keeps_packed_sbhd", False)
        )
        self.olmo3_owns_ulysses = bool(
            getattr(core_attention, "olmo3_owns_ulysses", False)
        )
        self._active_context: DepthAttentionContext | None = None
        self._active_inference_context: BaseInferenceContext | None = None

    @contextmanager
    def activate(
        self,
        context: DepthAttentionContext,
        inference_context: BaseInferenceContext | None = None,
    ) -> Iterator[None]:
        """Attach forward-local state for exactly one core-attention call."""
        if self._active_context is not None:
            raise RuntimeError("Depth-Attention core wrapper does not support reentrant forwards.")
        self._active_context = context
        self._active_inference_context = inference_context
        try:
            yield
        finally:
            self._active_context = None
            self._active_inference_context = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Mix V across depth, record post-RoPE K/V, then call MindSpeed attention."""
        if self._active_context is None:
            raise RuntimeError("Depth-Attention requires fresh forward-local context.")
        expected_shape = (
            self.expected_sequence_length,
            self.expected_query_heads,
            self.expected_kv_heads,
        )
        query_head_axis = 1 if query.ndim == 3 else 2
        key_head_axis = 1 if key.ndim == 3 else 2
        if (
            self.training
            and self._active_inference_context is None
            and all(expected is not None for expected in expected_shape)
            and (
            query.shape[0] != self.expected_sequence_length
            or key.shape[0] != self.expected_sequence_length
            or query.shape[query_head_axis] != self.expected_query_heads
            or key.shape[key_head_axis] != self.expected_kv_heads
            )
        ):
            raise RuntimeError(
                "OLMo3 Depth-Attention must run before Ulysses on CP-local, "
                "TP-head-sharded Q/K/V: "
                f"got query={tuple(query.shape)}, key={tuple(key.shape)}, "
                f"expected sequence={self.expected_sequence_length}, "
                f"query_heads={self.expected_query_heads}, "
                f"kv_heads={self.expected_kv_heads}."
            )
        if self._active_inference_context is None:
            mixed_value = depth_attention_mix(
                query,
                key,
                value,
                self._active_context.sources_for(self.layer_index),
                optimize_mha=self.optimize_mha,
            )
            self._active_context.record(self.layer_index, key, mixed_value)
            attention_value = mixed_value.contiguous()
        else:
            if not self._active_inference_context.is_static_batching():
                raise NotImplementedError(
                    "Depth-Attention KV cache supports static batching only."
                )
            current_sequence_length = query.shape[0]
            if key.shape[0] < current_sequence_length:
                raise RuntimeError(
                    "Cached Depth-Attention received fewer keys than current queries."
                )
            current_key = key[-current_sequence_length:]
            current_value = value[-current_sequence_length:]
            mixed_current_value = depth_attention_mix(
                query,
                current_key,
                current_value,
                self._active_context.sources_for(self.layer_index),
                optimize_mha=self.optimize_mha,
            )
            # MCore has already appended the raw current V into its static cache.
            # Replace only that new slice with the depth-mixed V. The returned
            # ``value`` is a view of the cache, so future decode steps observe
            # the exact V used by this attention call.
            current_value.copy_(mixed_current_value)
            self._active_context.record(
                self.layer_index,
                current_key,
                mixed_current_value,
            )
            attention_value = value
        return self.core_attention(
            query, key, attention_value, *args, **kwargs
        )


class Olmo3DepthSelfAttentionBase(SelfAttention):
    """Self-attention with independent full-Q/full-K RMSNorm and depth V mixing."""

    optimize_depth_attention_mha = False
    validate_depth_partitioning = False

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
        # Depth-Attention must mix CP-local Q/K/V before the OLMo3 attention
        # core chooses SWA halo or Full fused-Ulysses communication.  Remove
        # only MindSpeed's redundant parameter-free outer Ulysses wrapper.
        self.core_attention = unwrap_external_ulysses_for_olmo3(
            self.core_attention
        )
        expected_core_attention_recompute = (
            self.config.recompute_granularity == "selective"
            and "core_attn" in (self.config.recompute_modules or ())
        )
        if self.checkpoint_core_attention != expected_core_attention_recompute:
            raise RuntimeError(
                "MindSpeed core-attention recompute does not match the explicit "
                "recompute_modules contract."
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
        expected_sequence_length = None
        expected_query_heads = None
        expected_kv_heads = None
        if self.validate_depth_partitioning:
            context_parallel_size = int(self.config.context_parallel_size)
            if int(self.config.seq_length) % context_parallel_size:
                raise ValueError(
                    "OLMo3 sequence length must be divisible by context parallel size."
                )
            expected_sequence_length = (
                int(self.config.seq_length) // context_parallel_size
            )
            expected_query_heads = self.num_attention_heads_per_partition
            expected_kv_heads = self.num_query_groups_per_partition
        self.core_attention = DepthAwareCoreAttention(
            self.core_attention,
            layer_index=self.layer_number - 1,
            optimize_mha=self.optimize_depth_attention_mha,
            expected_sequence_length=expected_sequence_length,
            expected_query_heads=expected_query_heads,
            expected_kv_heads=expected_kv_heads,
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
        """Project Q/K/V and normalize complete Q and K projections independently."""
        if key_value_states is not None:
            raise ValueError("Olmo3DepthSelfAttentionBase supports decoder self-attention only.")
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: Union[Tensor, Tuple[Tensor, Tensor], None] = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        architecture_context: Olmo3ArchitectureContext | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run ordinary MindSpeed attention with Depth-Attention's modified V."""
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if architecture_context is None:
            raise ValueError("Olmo3DepthSelfAttentionBase requires Olmo3ArchitectureContext.")
        with self.core_attention.activate(
            architecture_context.depth_attention,
            inference_context,
        ):
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )


class Olmo3SiameseDepthTransformerLayerBase(TransformerLayer):
    """One Hybrid-Pre SiameseNorm layer with Depth-Attention self-attention."""

    clone_initial_siamese_stream = True

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
        self.siamese_attn_pre_norm = self._new_norm()
        self.siamese_mlp_pre_norm = self._new_norm()
        self.siamese_mlp_input_norm = self._new_norm()
        self.siamese_hybrid_attn_scale = torch.nn.Parameter(
            torch.ones(self.config.hidden_size, dtype=self.config.params_dtype)
        )
        setattr(self.siamese_hybrid_attn_scale, "sequence_parallel", self.config.sequence_parallel)

        self.siamese_post_final_layernorm = None
        self.siamese_pre_final_layernorm = None
        if self.layer_number == self.config.num_layers:
            self.siamese_post_final_layernorm = self._new_norm()
            self.siamese_pre_final_layernorm = self._new_norm()

    def _new_norm(self) -> torch.nn.Module:
        return build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Describe final Siamese norms as model-level checkpoint tensors.

        TransformerBlock normally prepends a homogeneous layer axis to every
        parameter owned by a transformer layer. The two final Siamese norms
        exist only on the last layer, so treating them as per-layer parameters
        leaves the first ``num_layers - 1`` shards missing. Rebuild only those
        entries without the layer offset while preserving ordinary layer
        sharding for every other parameter.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.layer_number == self.config.num_layers:
            remap_final_siamese_norms_for_checkpoint(
                self,
                sharded_state_dict,
                prefix,
                metadata,
            )
        return sharded_state_dict

    @staticmethod
    def _apply_norm(norm: torch.nn.Module | None, hidden_states: Tensor, name: str) -> Tensor:
        if norm is None:
            raise RuntimeError(f"SiameseNorm module {name} was not initialized.")
        output = norm(hidden_states)
        if not isinstance(output, Tensor):
            raise TypeError(
                f"SiameseNorm module {name} must return a Tensor, got {type(output).__name__}."
            )
        return output

    def _residual_branch(
        self, output_with_bias: tuple[Tensor, Tensor | None], residual_reference: Tensor
    ) -> Tensor:
        if not isinstance(output_with_bias, (tuple, list)) or len(output_with_bias) != 2:
            raise TypeError("SiameseNorm residual blocks must return an (output, bias) pair.")
        output, bias = output_with_bias
        if not isinstance(output, Tensor) or (bias is not None and not isinstance(bias, Tensor)):
            raise TypeError("SiameseNorm residual output and optional bias must be tensors.")
        if output.dtype != residual_reference.dtype:
            output = output.to(residual_reference.dtype)
            if bias is not None:
                bias = bias.to(residual_reference.dtype)
        if bias is not None:
            output = output + bias
        return F.dropout(output, p=self.hidden_dropout, training=self.training, inplace=False)

    def _combine_attention_streams(
        self,
        post_stream: Tensor,
        pre_normalized: Tensor,
    ) -> Tensor:
        """Combine Hybrid-Pre streams using the original OLMo3 expression."""
        attention_scale = self.siamese_hybrid_attn_scale.to(
            dtype=post_stream.dtype
        )
        return post_stream * attention_scale + pre_normalized

    def _update_siamese_streams(
        self,
        post_stream: Tensor,
        pre_stream: Tensor,
        residual_update: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the original SiameseNorm residual update."""
        return siamese_norm_residual_update(
            post_stream,
            pre_stream,
            residual_update,
            self.layer_number,
        )

    def _resolve_context(self, hidden_states: Tensor, context: Any) -> Olmo3ArchitectureContext:
        if context is None:
            if self.layer_number != 1:
                raise RuntimeError(
                    "OLMo3 architecture context was lost before transformer "
                    f"layer {self.layer_number}."
                )
            return Olmo3ArchitectureContext.create(
                hidden_states,
                num_layers=self.config.num_layers,
                depth_stride=self.config.depth_attention_stride,
                depth_recent_window=self.config.depth_attention_recent_window,
                clone_initial_stream=self.clone_initial_siamese_stream,
            )
        if not isinstance(context, Olmo3ArchitectureContext):
            raise TypeError(
                "OLMo3 decoder reserves TransformerBlock.context for forward-local architecture "
                f"state, got {type(context).__name__}."
            )
        return context

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
    ) -> tuple[Tensor, Olmo3ArchitectureContext]:
        """Run Attention and MLP while advancing both forward-local streams."""
        del kwargs
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if context_mask is not None:
            raise ValueError(
                "OLMo3 SiameseNorm + Depth-Attention is decoder-only without cross-attention."
            )

        architecture_context = self._resolve_context(hidden_states, context)
        pre_stream = architecture_context.siamese_norm.stream_for(self.layer_number)
        pre_normalized = self._apply_norm(
            self.siamese_attn_pre_norm, pre_stream, "siamese_attn_pre_norm"
        )
        post_stream = hidden_states
        attention_input = self._combine_attention_streams(
            post_stream,
            pre_normalized,
        )
        rotary_pos_emb = select_olmo3_layer_rotary(
            rotary_pos_emb,
            config=self.config,
            layer_number=self.layer_number,
        )
        attention_output_with_bias = self.self_attention(
            attention_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            architecture_context=architecture_context,
        )
        residual_update = self._residual_branch(attention_output_with_bias, post_stream)
        post_stream, pre_stream = self._update_siamese_streams(
            post_stream,
            pre_stream,
            residual_update,
        )
        architecture_context.siamese_norm.update(self.layer_number, pre_stream)

        post_stream = self._apply_norm(self.pre_mlp_layernorm, post_stream, "pre_mlp_layernorm")
        pre_normalized = self._apply_norm(
            self.siamese_mlp_pre_norm, pre_stream, "siamese_mlp_pre_norm"
        )
        mlp_input = self._apply_norm(
            self.siamese_mlp_input_norm, post_stream + pre_normalized, "siamese_mlp_input_norm"
        )
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, mlp_input
            )
        else:
            mlp_output_with_bias = self.mlp(mlp_input)
        residual_update = self._residual_branch(mlp_output_with_bias, post_stream)
        post_stream, pre_stream = self._update_siamese_streams(
            post_stream,
            pre_stream,
            residual_update,
        )
        architecture_context.siamese_norm.advance(self.layer_number, pre_stream)
        output = make_viewless_tensor(
            inp=post_stream, requires_grad=post_stream.requires_grad, keep_graph=True
        )

        if self.layer_number == self.config.num_layers:
            post_final = self._apply_norm(
                self.siamese_post_final_layernorm, output, "siamese_post_final_layernorm"
            )
            pre_final = self._apply_norm(
                self.siamese_pre_final_layernorm,
                architecture_context.siamese_norm.final_stream(),
                "siamese_pre_final_layernorm",
            )
            output = post_final + pre_final
        return output, architecture_context


def _validate_architecture_config(
    config: TransformerConfig,
    *,
    pre_process: bool,
    post_process: bool,
    use_transformer_engine_spec: bool,
    allow_context_parallel: bool = False,
    allow_sequence_parallel: bool = False,
) -> None:
    """Reject execution modes that cannot preserve forward-local cross-layer state."""
    if not use_transformer_engine_spec:
        raise ValueError("OLMo3 SiameseNorm + Depth-Attention requires the MindSpeed TE spec path.")
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
        raise ValueError("Forward-local SiameseNorm state currently requires PP=1.")
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError(
            "Forward-local SiameseNorm state does not support virtual pipeline parallelism."
        )
    if int(config.context_parallel_size) != 1 and not allow_context_parallel:
        raise ValueError("This Depth-Attention port requires CP=1.")
    if config.sequence_parallel and not allow_sequence_parallel:
        raise ValueError("This architecture currently requires sequence_parallel=False.")
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
    if config.recompute_granularity is not None:
        recompute_modules = set(config.recompute_modules or ())
        if (
            config.recompute_granularity != "selective"
            or recompute_modules != {"mlp"}
        ):
            raise ValueError(
                "Mutable Siamese/Depth state supports only official Stage-2 "
                "selective MLP recomputation. Attention/layer/block recomputation "
                "would replay cross-layer state."
            )
    if config.fp8 is not None:
        raise ValueError("Independent SiameseNorm boundaries are not implemented for FP8.")
    if config.cross_entropy_loss_fusion:
        raise ValueError("Dense vocabulary z-loss requires cross_entropy_loss_fusion=False.")
    if config.enable_cuda_graph or config.external_cuda_graph:
        raise ValueError("Forward-local SiameseNorm state does not support CUDA graphs.")
    if config.cpu_offloading:
        raise ValueError("Forward-local SiameseNorm state does not support CPU offloading.")
    if getattr(config, "mtp_num_layers", None):
        raise ValueError("OLMo3 SiameseNorm + Depth-Attention does not support MTP layers.")
    if config.num_moe_experts is not None:
        raise ValueError("This OLMo3 architecture is dense and does not support MoE layers.")
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


class Olmo3SiameseDepthModelBase(Olmo3MCoreLMBase):
    """Single-pass causal LM using the custom OLMo3/Siamese/Depth layer spec."""

    cache_static_only = True
    cache_supports_flash_decode = False

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
