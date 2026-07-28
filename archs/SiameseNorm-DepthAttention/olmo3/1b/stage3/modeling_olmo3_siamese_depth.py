"""Transformers remote code for OLMo 3 + Siamese Norm + Depth Attention.

The Full/SWA masks, Full-only YaRN rotary embeddings, and bounded sliding KV
cache are delegated to the official Transformers OLMo 3 implementation.  This
file adds only the two checkpoint-persistent mechanisms used during training:
Hybrid-Pre Siamese Norm and recursive same-position Depth Attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3MLP,
    Olmo3PreTrainedModel,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .configuration_olmo3_siamese_depth import Olmo3SiameseDepthConfig


@dataclass
class _DepthSource:
    layer_index: int
    key: torch.Tensor
    value: torch.Tensor


class _DepthContext:
    def __init__(self, config: Olmo3SiameseDepthConfig):
        self.num_layers = int(config.num_hidden_layers)
        self.stride = int(config.depth_attention_stride)
        self.recent_window = int(config.depth_attention_recent_window)
        self.records: dict[int, _DepthSource] = {}

    def sources_for(self, layer_index: int) -> tuple[_DepthSource, ...]:
        recent_start = max(0, layer_index - self.recent_window)
        return tuple(
            self.records[index]
            for index in sorted(self.records)
            if index < layer_index
            and (index % self.stride == 0 or index >= recent_start)
        )

    def record(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if layer_index in self.records:
            raise RuntimeError(f"Depth Attention layer {layer_index} was recorded twice.")
        self.records[layer_index] = _DepthSource(layer_index, key, value)
        next_recent_start = layer_index + 1 - self.recent_window
        stale = [
            index
            for index in self.records
            if index % self.stride != 0 and index < next_recent_start
        ]
        for index in stale:
            del self.records[index]


def _depth_attention_mix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sources: tuple[_DepthSource, ...],
) -> torch.Tensor:
    """Mix current V with recursively mixed V from selected earlier layers.

    Tensors use Transformers' ``[batch, heads, sequence, head_dim]`` layout.
    """

    if not sources:
        return value
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Depth Attention requires Q heads divisible by KV heads.")
    group_size = query_heads // kv_heads
    grouped_query = (
        query
        if group_size == 1
        else query.reshape(
            query.shape[0],
            kv_heads,
            group_size,
            query.shape[2],
            query.shape[3],
        ).mean(dim=2)
    )
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = [
        (grouped_query * source.key).sum(dim=-1, keepdim=True) * scale
        for source in sources
    ]
    scores.append((grouped_query * key).sum(dim=-1, keepdim=True) * scale)
    weights = torch.softmax(torch.cat(scores, dim=-1), dim=-1, dtype=torch.float32)
    weights = weights.to(dtype=value.dtype)
    mixed = weights[..., -1:].mul(value)
    for source_index, source in enumerate(sources):
        mixed = mixed + weights[..., source_index : source_index + 1] * source.value
    return mixed


class Olmo3SiameseDepthAttention(nn.Module):
    def __init__(self, config: Olmo3SiameseDepthConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # These norms cover the complete Q and K projections, not one head.
        self.q_norm = Olmo3RMSNorm(
            config.num_attention_heads * self.head_dim, config.rms_norm_eps
        )
        self.k_norm = Olmo3RMSNorm(
            config.num_key_value_heads * self.head_dim, config.rms_norm_eps
        )
        self.attention_type = config.layer_types[self.layer_idx]
        self.sliding_window = (
            int(config.sliding_window)
            if self.attention_type == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        depth_context: _DepthContext,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_norm(self.q_proj(hidden_states)).view(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states)).view(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query.float(), key.float(), cos.float(), sin.float())
        query = query.to(dtype=hidden_states.dtype)
        key = key.to(dtype=hidden_states.dtype)

        mixed_value = _depth_attention_mix(
            query,
            key,
            value,
            depth_context.sources_for(self.layer_idx),
        )
        depth_context.record(self.layer_idx, key, mixed_value)

        attention_key = key
        attention_value = mixed_value
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            attention_key, attention_value = past_key_values.update(
                key, mixed_value, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        attention_output, attention_weights = attention_interface(
            self,
            query,
            attention_key,
            attention_value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            batch_size, sequence_length, -1
        ).contiguous()
        attention_output = self.o_proj(attention_output)
        return attention_output, attention_weights if output_attentions else None


class Olmo3SiameseDepthDecoderLayer(nn.Module):
    def __init__(self, config: Olmo3SiameseDepthConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = Olmo3SiameseDepthAttention(config, layer_idx)
        self.mlp = Olmo3MLP(config)
        self.pre_mlp_layernorm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.siamese_attn_pre_norm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.siamese_mlp_pre_norm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.siamese_mlp_input_norm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.siamese_hybrid_attn_scale = nn.Parameter(
            torch.ones(config.hidden_size)
        )
        self.post_residual_scale = 1.0 / math.sqrt(2.0 * (self.layer_idx + 1))

    def forward(
        self,
        post_stream: torch.Tensor,
        pre_stream: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        depth_context: _DepthContext,
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        output_attentions: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pre_normalized = self.siamese_attn_pre_norm(pre_stream)
        attention_input = torch.addcmul(
            pre_normalized,
            post_stream,
            self.siamese_hybrid_attn_scale.to(dtype=post_stream.dtype),
        )
        attention_output, attention_weights = self.self_attn(
            hidden_states=attention_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            depth_context=depth_context,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
            **kwargs,
        )
        post_stream = torch.add(
            post_stream, attention_output, alpha=self.post_residual_scale
        )
        pre_stream = pre_stream + attention_output

        # The native Hybrid-Pre block advances its post stream to the
        # pre-MLP-normalized value before both the MLP input construction and
        # the MLP residual update.  This is intentionally not a conventional
        # pre-norm residual that adds the branch back to the unnormalized
        # stream.
        post_stream = self.pre_mlp_layernorm(post_stream)
        pre_normalized = self.siamese_mlp_pre_norm(pre_stream)
        mlp_input = self.siamese_mlp_input_norm(
            post_stream + pre_normalized
        )
        mlp_output = self.mlp(mlp_input)
        post_stream = torch.add(
            post_stream, mlp_output, alpha=self.post_residual_scale
        )
        pre_stream = pre_stream + mlp_output
        return post_stream, pre_stream, attention_weights


class Olmo3SiameseDepthPreTrainedModel(Olmo3PreTrainedModel):
    config_class = Olmo3SiameseDepthConfig
    base_model_prefix = "model"
    _no_split_modules = ["Olmo3SiameseDepthDecoderLayer"]
    # Whole-layer replay is invalid for the mutable cross-layer Depth context.
    # The native trainer supports only its dedicated selective-MLP checkpoint
    # path; do not advertise Transformers' generic layer checkpointing.
    supports_gradient_checkpointing = False


class Olmo3SiameseDepthModel(Olmo3SiameseDepthPreTrainedModel):
    def __init__(self, config: Olmo3SiameseDepthConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Olmo3SiameseDepthDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.siamese_post_final_layernorm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.siamese_pre_final_layernorm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.norm = Olmo3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_embs = nn.ModuleDict(
            {
                "sliding_attention": Olmo3RotaryEmbedding(
                    config=config, rope_type="default"
                ),
                "full_attention": Olmo3RotaryEmbedding(config=config),
            }
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(
                    **mask_kwargs
                ),
            }
        else:
            causal_masks = attention_mask

        post_stream = inputs_embeds
        # The optimized training path intentionally aliases both initial streams.
        pre_stream = inputs_embeds
        depth_context = _DepthContext(self.config)
        position_embeddings = {
            kind: rotary(post_stream, position_ids)
            for kind, rotary in self.rotary_embs.items()
        }
        hidden_history = () if output_hidden_states else None
        attention_history = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                hidden_history += (post_stream,)
            attention_type = layer.self_attn.attention_type
            post_stream, pre_stream, attention_weights = layer(
                post_stream=post_stream,
                pre_stream=pre_stream,
                attention_mask=causal_masks[attention_type],
                position_embeddings=position_embeddings[attention_type],
                depth_context=depth_context,
                past_key_values=past_key_values,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                attention_history += (attention_weights,)

        hidden_states = (
            self.siamese_post_final_layernorm(post_stream)
            + self.siamese_pre_final_layernorm(pre_stream)
        )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            hidden_history += (hidden_states,)
        if not return_dict:
            values = (hidden_states, past_key_values, hidden_history, attention_history)
            return tuple(value for value in values if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=hidden_history,
            attentions=attention_history,
        )


class Olmo3SiameseDepthForCausalLM(
    Olmo3SiameseDepthPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = []

    def __init__(self, config: Olmo3SiameseDepthConfig):
        super().__init__(config)
        self.model = Olmo3SiameseDepthModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        if not return_dict:
            result = (logits, outputs.past_key_values)
            return ((loss,) + result) if loss is not None else result
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Olmo3SiameseDepthForCausalLM",
    "Olmo3SiameseDepthModel",
    "Olmo3SiameseDepthPreTrainedModel",
]
