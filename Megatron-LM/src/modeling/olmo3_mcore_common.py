"""Shared MCore language-model helpers for the OLMo3 implementations."""

from __future__ import annotations

import os
import inspect
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import (
    BaseInferenceContext,
    StaticInferenceContext,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from modeling.lm_loss_utils import (
    align_causal_lm_loss_inputs,
    align_sequence_parallel_causal_lm_loss_inputs,
)
from modeling.olmo3_rotary import Olmo3LayerwiseRotaryEmbedding


class Olmo3MCoreLMBase(LanguageModule):
    """MCore GPT backbone, loss, and KV-cache helpers for OLMo3."""

    cache_static_only = False
    cache_supports_flash_decode = True

    def __init__(
        self,
        config: TransformerConfig,
        model_config: Any,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        use_transformer_engine_spec: bool = False,
        transformer_layer_spec: Any | None = None,
        rotary_base: float | None = None,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.model_config = model_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.use_transformer_engine_spec = use_transformer_engine_spec
        self.share_embeddings_and_output_weights = False
        self.model_type = ModelType.encoder_or_decoder
        self.use_null_attention_mask = (
            os.getenv(
                "MODEL_USE_NULL_ATTENTION_MASK",
                "1" if use_transformer_engine_spec else "0",
            )
            == "1"
        )

        if transformer_layer_spec is not None:
            layer_spec = transformer_layer_spec
        elif use_transformer_engine_spec:
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                qk_layernorm=config.qk_layernorm,
                multi_latent_attention=config.multi_latent_attention,
                moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            )
        else:
            layer_spec = get_gpt_layer_local_spec(normalization=config.normalization)

        self.backbone = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=(
                float(rotary_base)
                if rotary_base is not None
                else getattr(model_config, "rope_theta", 10000)
            ),
            scatter_embedding_sequence_parallel=True,
        )
        yarn = getattr(config, "olmo3_yarn", None)
        if yarn is not None:
            if bool(getattr(config, "flash_decode", False)):
                raise ValueError(
                    "Full/SWA-separated YaRN is incompatible with the one-table flash-decode path."
                )
            self.backbone.rotary_pos_emb = Olmo3LayerwiseRotaryEmbedding(
                self.backbone.rotary_pos_emb,
                yarn,
            )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        self.backbone.set_input_tensor(input_tensor)

    def _build_causal_mask(
        self,
        attention_mask: Tensor | None,
        query_length: int,
        device: torch.device,
        *,
        key_length: int | None = None,
    ) -> Tensor:
        """Build a right-aligned causal mask for full or cached attention."""
        if query_length < 1:
            raise ValueError(f"query_length must be positive, got {query_length}.")
        if key_length is None:
            key_length = query_length
        if key_length < query_length:
            raise ValueError(
                "Cached causal attention requires key_length >= query_length; "
                f"got {key_length} and {query_length}."
            )
        query_positions = (
            torch.arange(query_length, device=device) + key_length - query_length
        ).unsqueeze(1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        causal_mask = (key_positions > query_positions).view(
            1, 1, query_length, key_length
        )
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[1] != key_length:
                raise ValueError(
                    "A 2D attention_mask must have [batch, cached_key_length] shape; "
                    f"got {tuple(attention_mask.shape)} for key length {key_length}."
                )
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask | padding_mask
        return causal_mask

    def build_inference_context(
        self,
        max_batch_size: int,
        max_sequence_length: int | None = None,
        *,
        materialize_only_last_token_logits: bool = False,
    ) -> StaticInferenceContext:
        """Create the MCore-owned static KV-cache state used for prefill/decode."""
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer.")
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length
        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length < 1
        ):
            raise ValueError("max_sequence_length must be a positive integer.")
        if max_sequence_length > int(self.max_sequence_length):
            raise ValueError(
                "KV-cache max_sequence_length cannot exceed the model context length; "
                f"got {max_sequence_length} > {self.max_sequence_length}."
            )
        context = StaticInferenceContext(
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
        )
        context.materialize_only_last_token_logits = bool(
            materialize_only_last_token_logits
        )
        return context

    def _validate_inference_context(
        self,
        inference_context: BaseInferenceContext,
        *,
        batch_size: int,
        query_length: int,
        packed_seq_params: PackedSeqParams | None,
    ) -> None:
        """Reject cache modes that this architecture cannot execute correctly."""
        if self.training:
            raise RuntimeError("KV cache is inference-only; call model.eval() first.")
        if packed_seq_params is not None:
            raise NotImplementedError("KV cache does not support packed sequences.")
        if int(self.config.context_parallel_size) != 1:
            raise NotImplementedError(
                "KV-cache inference currently requires context_parallel_size=1. "
                "CP remains supported by the training path."
            )
        if bool(self.config.sequence_parallel):
            raise NotImplementedError(
                "KV-cache inference currently requires sequence_parallel=False. "
                "TP itself remains supported."
            )
        if self.cache_static_only and not inference_context.is_static_batching():
            raise NotImplementedError(
                f"{type(self).__name__} supports static KV-cache batching only."
            )
        if bool(getattr(self.config, "flash_decode", False)) and not (
            self.cache_supports_flash_decode
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support flash_decode because it "
                "would bypass the custom attention semantics."
            )
        if inference_context.is_static_batching():
            batch_end = int(inference_context.batch_size_offset) + batch_size
            sequence_end = int(inference_context.sequence_len_offset) + query_length
            if batch_end > int(inference_context.max_batch_size):
                raise ValueError(
                    "KV-cache batch exceeds max_batch_size; "
                    f"need {batch_end}, allocated {inference_context.max_batch_size}."
                )
            if sequence_end > int(inference_context.max_sequence_length):
                raise ValueError(
                    "KV-cache sequence exceeds max_sequence_length; "
                    f"need {sequence_end}, allocated {inference_context.max_sequence_length}."
                )

    def _resolve_inference_context(
        self,
        inference_context: BaseInferenceContext | None,
        inference_params: BaseInferenceContext | None,
        use_cache: bool | None,
        *,
        batch_size: int,
        query_length: int,
        packed_seq_params: PackedSeqParams | None,
    ) -> BaseInferenceContext | None:
        """Resolve the current and legacy cache APIs without ambiguous state."""
        if (
            inference_context is not None
            and inference_params is not None
            and inference_context is not inference_params
        ):
            raise ValueError(
                "inference_context and deprecated inference_params refer to different objects."
            )
        if inference_context is None:
            inference_context = inference_params
        if use_cache is not None and not isinstance(use_cache, bool):
            raise TypeError("use_cache must be bool or None.")
        if use_cache is False:
            return None
        if inference_context is None:
            if use_cache:
                raise ValueError(
                    "use_cache=True requires an inference_context. Create one with "
                    "model.build_inference_context(...) so cache ownership is explicit."
                )
            return None
        if not isinstance(inference_context, BaseInferenceContext):
            raise TypeError(
                "inference_context must inherit BaseInferenceContext, got "
                f"{type(inference_context).__name__}."
            )
        self._validate_inference_context(
            inference_context,
            batch_size=batch_size,
            query_length=query_length,
            packed_seq_params=packed_seq_params,
        )
        return inference_context

    def _prepare_inference_position_ids(
        self,
        tokens: Tensor,
        position_ids: Tensor | None,
        inference_context: BaseInferenceContext | None,
    ) -> Tensor:
        if position_ids is not None:
            if position_ids.shape != tokens.shape:
                raise ValueError(
                    "position_ids must match tokens [batch, sequence]; "
                    f"got {tuple(position_ids.shape)} and {tuple(tokens.shape)}."
                )
            return position_ids
        offset = 0
        if inference_context is not None and inference_context.is_static_batching():
            offset = int(inference_context.sequence_len_offset)
        return (
            torch.arange(
                offset,
                offset + tokens.shape[1],
                device=tokens.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(tokens)
        )

    def _prepare_inference_attention_mask(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None,
        inference_context: BaseInferenceContext | None,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        key_length = tokens.shape[1]
        if inference_context is not None and inference_context.is_static_batching():
            key_length += int(inference_context.sequence_len_offset)
        if attention_mask.dim() == 4:
            valid_batch_sizes = (1, tokens.shape[0])
            if (
                attention_mask.shape[0] not in valid_batch_sizes
                or attention_mask.shape[1] != 1
                or attention_mask.shape[2] != tokens.shape[1]
                or attention_mask.shape[3] != key_length
            ):
                raise ValueError(
                    "A 4D inference attention_mask must have "
                    "[1|batch, 1, query_length, cached_key_length] shape; "
                    f"got {tuple(attention_mask.shape)}, expected batch "
                    f"{valid_batch_sizes}, query {tokens.shape[1]}, key {key_length}."
                )
            return attention_mask
        if attention_mask.dim() != 2 or attention_mask.shape[0] != tokens.shape[0]:
            raise ValueError(
                "attention_mask must be None, [batch, key_length], or a 4D MCore mask."
            )
        if attention_mask.shape[1] != key_length:
            raise ValueError(
                "A cached 2D attention_mask must cover the prompt plus current tokens; "
                f"got width {attention_mask.shape[1]}, expected {key_length}."
            )
        if self.use_null_attention_mask and bool(torch.all(attention_mask != 0)):
            return None
        return self._build_causal_mask(
            attention_mask,
            tokens.shape[1],
            tokens.device,
            key_length=key_length,
        )

    @contextmanager
    def _mindspeed_kv_cache_scope(
        self,
        inference_context: BaseInferenceContext | None,
    ):
        """Select MindSpeed's prompt/incremental kernels for one cache forward."""
        if inference_context is None:
            yield
            return
        try:
            from megatron.training import get_args

            args = get_args()
        except (AssertionError, ImportError, RuntimeError):
            yield
            return

        missing = object()
        previous = getattr(args, "use_kv_cache", missing)
        setattr(args, "use_kv_cache", True)
        try:
            yield
        finally:
            if previous is missing:
                delattr(args, "use_kv_cache")
            else:
                setattr(args, "use_kv_cache", previous)

    @contextmanager
    def _runtime_gather_output_scope(
        self,
        runtime_gather_output: bool | None,
    ):
        """Bridge MCore's runtime gather option to MindSpeed's older GPT API."""
        if runtime_gather_output is not None and not isinstance(
            runtime_gather_output, bool
        ):
            raise TypeError("runtime_gather_output must be bool or None.")

        try:
            forward_parameters = inspect.signature(
                self.backbone.forward
            ).parameters
        except (TypeError, ValueError):
            forward_parameters = {}
        accepts_runtime_gather_output = (
            "runtime_gather_output" in forward_parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters.values()
            )
        )
        if accepts_runtime_gather_output:
            kwargs = (
                {}
                if runtime_gather_output is None
                else {"runtime_gather_output": runtime_gather_output}
            )
            yield kwargs
            return

        # MindSpeed-LLM's GPTModel predates this MCore forward keyword but its
        # ColumnParallelLinear still exposes the same gather_output switch.
        # Scope the mutation to this synchronous inference call and restore it
        # even when attention or output projection raises.
        if runtime_gather_output is None:
            yield {}
            return
        output_layer = getattr(self.backbone, "output_layer", None)
        if output_layer is None or not hasattr(output_layer, "gather_output"):
            raise NotImplementedError(
                "The active GPT backend cannot override output gathering at runtime."
            )
        previous = output_layer.gather_output
        output_layer.gather_output = runtime_gather_output
        try:
            yield {}
        finally:
            output_layer.gather_output = previous

    def _forward_inference_logits(
        self,
        tokens: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        inference_context: BaseInferenceContext | None = None,
        inference_params: BaseInferenceContext | None = None,
        use_cache: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        runtime_gather_output: bool | None = None,
    ) -> Tensor:
        """Return logits through the OLMo3 prefill/decode implementation.

        MindSpeed-LLM replaces ``GPTModel.forward`` with an older implementation
        that does not honor ``StaticInferenceContext.materialize_only_last_token_logits``.
        Projecting all 65,536 hidden states into the vocabulary during prefill
        is both unnecessary and large enough to exhaust an otherwise valid
        inference run. Cache-enabled forwards therefore use the same OLMo3
        embedding/decoder modules directly and slice hidden states before the
        output projection. The no-cache path remains the backend-native GPT
        forward used before Stage-3 cache support.
        """
        if tokens.dim() != 2 or tokens.shape[1] < 1:
            raise ValueError(
                "tokens must have non-empty [batch, sequence] shape; "
                f"got {tuple(tokens.shape)}."
            )
        active_context = self._resolve_inference_context(
            inference_context,
            inference_params,
            use_cache,
            batch_size=tokens.shape[0],
            query_length=tokens.shape[1],
            packed_seq_params=packed_seq_params,
        )
        position_ids = self._prepare_inference_position_ids(
            tokens, position_ids, active_context
        )
        attention_mask = self._prepare_inference_attention_mask(
            tokens, attention_mask, active_context
        )
        if active_context is None:
            with self._runtime_gather_output_scope(
                runtime_gather_output
            ) as output_kwargs:
                return self.backbone(
                    input_ids=tokens,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    **output_kwargs,
                )

        with (
            torch.no_grad(),
            self._mindspeed_kv_cache_scope(active_context),
            self._runtime_gather_output_scope(runtime_gather_output) as output_kwargs,
        ):
            hidden_states = self._embed_tokens_sbh(tokens, position_ids)
            hidden_states = self._decoder_forward_sbh(
                hidden_states,
                attention_mask,
                position_ids,
                inference_context=active_context,
                packed_seq_params=packed_seq_params,
            )
            if (
                active_context.is_static_batching()
                and bool(
                    getattr(
                        active_context,
                        "materialize_only_last_token_logits",
                        False,
                    )
                )
            ):
                hidden_states = hidden_states[-1:, :, :]
            logits, _ = self.backbone.output_layer(
                hidden_states,
                weight=None,
                **output_kwargs,
            )
            return logits.transpose(0, 1).contiguous()

    def _embed_tokens_sbh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long).unsqueeze(0)
        return self.backbone.embedding(input_ids=tokens, position_ids=position_ids)

    def _build_rotary_pos_emb(
        self,
        seq_len: int,
        position_ids: Tensor | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Any:
        del position_ids
        # RotaryEmbedding slices the full table onto the active CP rank. SP
        # additionally shards decoder input over TP before the QKV projection
        # gathers it, so reconstruct both dimensions here.
        sequence_parallel_multiplier = (
            parallel_state.get_tensor_model_parallel_world_size()
            if self.config.sequence_parallel
            else 1
        )
        global_seq_len = (
            seq_len
            * sequence_parallel_multiplier
            * parallel_state.get_context_parallel_world_size()
        )
        return self.backbone.rotary_pos_emb(
            global_seq_len,
            packed_seq=packed_seq_params is not None,
        )

    def _decoder_forward_sbh(
        self,
        hidden_states_sbh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        *,
        attention_bias: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        decoder_input = hidden_states_sbh.contiguous()
        rotary_sequence_length = hidden_states_sbh.shape[0]
        if inference_context is not None:
            rotary_sequence_length = int(inference_context.max_sequence_length)
        rotary_pos_emb = self._build_rotary_pos_emb(
            rotary_sequence_length,
            position_ids,
            packed_seq_params=packed_seq_params,
        )
        causal_mask = None
        if not self.use_null_attention_mask:
            attention_sequence_length = decoder_input.shape[0]
            if self.config.sequence_parallel:
                attention_sequence_length *= (
                    parallel_state.get_tensor_model_parallel_world_size()
                )
            causal_mask = self._build_causal_mask(
                attention_mask,
                attention_sequence_length,
                decoder_input.device,
            )
        decoder_kwargs = {
            "hidden_states": decoder_input,
            "attention_mask": causal_mask,
            "inference_context": inference_context,
            "rotary_pos_emb": rotary_pos_emb,
            "packed_seq_params": packed_seq_params,
        }
        if attention_bias is not None:
            decoder_kwargs["attention_bias"] = attention_bias
        return self.backbone.decoder(**decoder_kwargs)

    def _compute_lm_loss_component_sums(
        self,
        hidden_states_bsh: Tensor,
        labels: Tensor,
        loss_mask: Tensor,
        per_sample: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return masked sums for total, pure LM, and scaled z-loss."""
        hidden_states_sbh = hidden_states_bsh.transpose(0, 1).contiguous()
        logits, _ = self.backbone.output_layer(
            hidden_states_sbh, weight=None, runtime_gather_output=False
        )
        total_loss, lm_loss, z_loss = self.compute_language_model_loss_components(labels, logits)
        loss_mask = loss_mask.to(total_loss.dtype)
        reduction_dims = (1,) if per_sample else None
        return (
            (total_loss * loss_mask).sum(dim=reduction_dims),
            (lm_loss * loss_mask).sum(dim=reduction_dims),
            (z_loss * loss_mask).sum(dim=reduction_dims),
        )

    def _loss_denominator(self, loss_mask: Tensor) -> Tensor:
        return loss_mask.sum(dtype=torch.float32).clamp_min(1.0)

    def _compute_shifted_loss_components(
        self,
        hidden_states_bsh: Tensor,
        labels: Tensor,
        *,
        labels_are_shifted: bool = False,
        return_per_sample_sums: bool = False,
    ) -> Tensor:
        """Return mean losses or per-sample ``[total, lm, z, tokens]`` sums."""
        if self.config.sequence_parallel:
            shifted_hidden, shifted_labels = (
                align_sequence_parallel_causal_lm_loss_inputs(
                    hidden_states_bsh,
                    labels,
                    tensor_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
                    labels_are_shifted=labels_are_shifted,
                )
            )
        else:
            shifted_hidden, shifted_labels = align_causal_lm_loss_inputs(
                hidden_states_bsh,
                labels,
                labels_are_shifted=labels_are_shifted,
            )

        loss_mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
        token_count = (
            loss_mask.sum(dim=1, dtype=torch.float32)
            if return_per_sample_sums
            else self._loss_denominator(loss_mask)
        )
        component_sums = self._compute_lm_loss_component_sums(
            shifted_hidden,
            safe_labels,
            loss_mask,
            return_per_sample_sums,
        )
        if return_per_sample_sums:
            return torch.stack((*component_sums, token_count), dim=1)
        if self.config.calculate_per_token_loss:
            # Megatron sums these values over CP, DP, and all gradient-
            # accumulation microbatches, then divides once by the global
            # valid-token count in finalize_model_grads(). This is required
            # for packed CP where local padding counts can differ.
            return torch.stack((*component_sums, token_count))
        return torch.stack(
            tuple(component_sum / token_count for component_sum in component_sums)
        )
