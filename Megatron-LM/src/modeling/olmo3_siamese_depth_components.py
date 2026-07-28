"""Forward-local tensor state for OLMo3 + SiameseNorm + Depth-Attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


class _TensorParallelFullProjectionRMSNorm(torch.autograd.Function):
    """RMSNorm a last dimension sharded across one tensor-parallel group."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        weight: Tensor,
        epsilon: float,
        full_hidden_size: int,
        tensor_parallel_group: torch.distributed.ProcessGroup,
    ) -> Tensor:
        local_sum_squares = input_.float().pow(2).sum(dim=-1, keepdim=True)
        torch.distributed.all_reduce(local_sum_squares, group=tensor_parallel_group)
        inverse_rms = torch.rsqrt(local_sum_squares / full_hidden_size + epsilon)
        normalized = input_.float() * inverse_rms
        ctx.full_hidden_size = full_hidden_size
        ctx.tensor_parallel_group = tensor_parallel_group
        ctx.input_dtype = input_.dtype
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(input_, inverse_rms, weight)
        return (normalized * weight.float()).to(dtype=input_.dtype)

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> tuple[Tensor | None, ...]:
        input_, inverse_rms, weight = ctx.saved_tensors
        normalized = input_.float() * inverse_rms
        grad_normalized = output_grad.float() * weight.float()
        local_dot = (grad_normalized * normalized).sum(dim=-1, keepdim=True)
        dot_reduce = torch.distributed.all_reduce(
            local_dot,
            group=ctx.tensor_parallel_group,
            async_op=True,
        )
        reduction_dims = tuple(range(output_grad.ndim - 1))
        grad_weight = (output_grad.float() * normalized).sum(dim=reduction_dims)
        if dot_reduce is not None:
            dot_reduce.wait()
        grad_input = (
            grad_normalized
            - normalized * (local_dot / ctx.full_hidden_size)
        ) * inverse_rms
        return (
            grad_input.to(dtype=ctx.input_dtype),
            grad_weight.to(dtype=ctx.weight_dtype),
            None,
            None,
            None,
        )


class _TensorParallelFusedQKFullProjectionRMSNorm(torch.autograd.Function):
    """Normalize TP-sharded Q and K with one collective in each pass.

    Q and K have different projection widths and affine weights, but their RMS
    statistics share the same leading ``[sequence, batch]`` dimensions. Packing
    the two scalar statistics into one tensor preserves the independent
    normalization equations while halving latency-bound TP collectives.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        query_weight: Tensor,
        key_weight: Tensor,
        query_epsilon: float,
        key_epsilon: float,
        query_full_hidden_size: int,
        key_full_hidden_size: int,
        tensor_parallel_group: torch.distributed.ProcessGroup,
    ) -> tuple[Tensor, Tensor]:
        local_sum_squares = torch.cat(
            (
                query.float().pow(2).sum(dim=-1, keepdim=True),
                key.float().pow(2).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        torch.distributed.all_reduce(local_sum_squares, group=tensor_parallel_group)
        query_inverse_rms = torch.rsqrt(
            local_sum_squares[..., :1] / query_full_hidden_size + query_epsilon
        )
        key_inverse_rms = torch.rsqrt(
            local_sum_squares[..., 1:] / key_full_hidden_size + key_epsilon
        )
        normalized_query = query.float() * query_inverse_rms
        normalized_key = key.float() * key_inverse_rms

        ctx.query_full_hidden_size = query_full_hidden_size
        ctx.key_full_hidden_size = key_full_hidden_size
        ctx.tensor_parallel_group = tensor_parallel_group
        ctx.query_dtype = query.dtype
        ctx.key_dtype = key.dtype
        ctx.query_weight_dtype = query_weight.dtype
        ctx.key_weight_dtype = key_weight.dtype
        ctx.save_for_backward(
            query,
            key,
            query_inverse_rms,
            key_inverse_rms,
            query_weight,
            key_weight,
        )
        return (
            (normalized_query * query_weight.float()).to(dtype=query.dtype),
            (normalized_key * key_weight.float()).to(dtype=key.dtype),
        )

    @staticmethod
    def backward(
        ctx: Any,
        query_output_grad: Tensor,
        key_output_grad: Tensor,
    ) -> tuple[Tensor | None, ...]:
        (
            query,
            key,
            query_inverse_rms,
            key_inverse_rms,
            query_weight,
            key_weight,
        ) = ctx.saved_tensors
        normalized_query = query.float() * query_inverse_rms
        normalized_key = key.float() * key_inverse_rms
        query_grad_normalized = query_output_grad.float() * query_weight.float()
        key_grad_normalized = key_output_grad.float() * key_weight.float()
        projection_dots = torch.cat(
            (
                (query_grad_normalized * normalized_query).sum(
                    dim=-1, keepdim=True
                ),
                (key_grad_normalized * normalized_key).sum(
                    dim=-1, keepdim=True
                ),
            ),
            dim=-1,
        )
        dot_reduce = torch.distributed.all_reduce(
            projection_dots,
            group=ctx.tensor_parallel_group,
            async_op=True,
        )

        query_reduction_dims = tuple(range(query_output_grad.ndim - 1))
        key_reduction_dims = tuple(range(key_output_grad.ndim - 1))
        query_weight_grad = (
            query_output_grad.float() * normalized_query
        ).sum(dim=query_reduction_dims)
        key_weight_grad = (
            key_output_grad.float() * normalized_key
        ).sum(dim=key_reduction_dims)
        if dot_reduce is not None:
            dot_reduce.wait()

        query_input_grad = (
            query_grad_normalized
            - normalized_query
            * (projection_dots[..., :1] / ctx.query_full_hidden_size)
        ) * query_inverse_rms
        key_input_grad = (
            key_grad_normalized
            - normalized_key
            * (projection_dots[..., 1:] / ctx.key_full_hidden_size)
        ) * key_inverse_rms
        return (
            query_input_grad.to(dtype=ctx.query_dtype),
            key_input_grad.to(dtype=ctx.key_dtype),
            query_weight_grad.to(dtype=ctx.query_weight_dtype),
            key_weight_grad.to(dtype=ctx.key_weight_dtype),
            None,
            None,
            None,
            None,
            None,
        )


class TensorParallelFullProjectionRMSNorm(torch.nn.Module):
    """Exact full-projection RMSNorm with a TP-sharded learned scale.

    Q and K projections are column-sharded by Megatron. Their RMS statistic must
    still cover the complete projection, so each forward and backward performs
    one small all-reduce inside the tensor-parallel group. The scale parameter
    is sharded on the same last dimension and checkpointed as a TP tensor.
    """

    def __init__(self, config: Any, full_hidden_size: int, eps: float) -> None:
        super().__init__()
        tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        if tensor_parallel_size <= 1:
            raise ValueError(
                "TensorParallelFullProjectionRMSNorm requires tensor parallel size > 1."
            )
        if full_hidden_size % tensor_parallel_size != 0:
            raise ValueError(
                f"Full projection size {full_hidden_size} must be divisible by "
                f"tensor parallel size {tensor_parallel_size}."
            )
        self.full_hidden_size = int(full_hidden_size)
        self.local_hidden_size = self.full_hidden_size // tensor_parallel_size
        self.eps = float(eps)
        self.tensor_parallel_size = int(tensor_parallel_size)
        if bool(config.use_cpu_initialization):
            device: torch.device | int = torch.device("cpu")
        elif hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu", torch.npu.current_device())
        else:
            device = torch.cuda.current_device()
        self.weight = Parameter(
            torch.ones(
                self.local_hidden_size,
                dtype=config.params_dtype,
                device=device,
            )
        )
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, input_: Tensor) -> Tensor:
        if input_.shape[-1] != self.local_hidden_size:
            raise ValueError(
                "TP full-projection RMSNorm received the wrong local width: "
                f"expected {self.local_hidden_size}, got {input_.shape[-1]}."
            )
        return _TensorParallelFullProjectionRMSNorm.apply(
            input_,
            self.weight,
            self.eps,
            self.full_hidden_size,
            parallel_state.get_tensor_model_parallel_group(),
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        del metadata
        key = f"{prefix}weight"
        return {
            key: make_tp_sharded_tensor_for_checkpoint(
                self.weight,
                key,
                tp_axis=0,
                prepend_offsets=sharded_offsets,
            )
        }


def apply_full_projection_qk_norm(
    query: Tensor, key: Tensor, q_norm: torch.nn.Module, k_norm: torch.nn.Module
) -> tuple[Tensor, Tensor]:
    """Apply independent RMSNorms over all Q channels and all K channels."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Full-projection Q/K RMSNorm expects four-dimensional tensors.")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Full-projection Q/K RMSNorm requires matching sequence, batch, and head_dim."
        )
    query_shape = query.shape
    key_shape = key.shape
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = q_norm(query.reshape(query.size(0), query.size(1), -1)).to(
        dtype=query_dtype
    )
    key = k_norm(key.reshape(key.size(0), key.size(1), -1)).to(
        dtype=key_dtype
    )
    return query.reshape(query_shape), key.reshape(key_shape)


def apply_fused_tp_full_projection_qk_norm(
    query: Tensor,
    key: Tensor,
    q_norm: torch.nn.Module,
    k_norm: torch.nn.Module,
) -> tuple[Tensor, Tensor]:
    """Apply exact OLMo3 Q/K RMSNorm with fused TP statistic collectives.

    TP=1 continues to use Transformer Engine's fused RMSNorm modules. The
    optimized path is selected only when both affine norms are the repository's
    TP-sharded full-projection implementation, so checkpoint parameter names
    and sharding remain unchanged.
    """
    if not isinstance(
        q_norm, TensorParallelFullProjectionRMSNorm
    ) or not isinstance(k_norm, TensorParallelFullProjectionRMSNorm):
        return apply_full_projection_qk_norm(query, key, q_norm, k_norm)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Fused TP Q/K RMSNorm expects four-dimensional tensors.")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Fused TP Q/K RMSNorm requires matching sequence, batch, and head_dim."
        )

    query_shape = query.shape
    key_shape = key.shape
    normalized_query, normalized_key = (
        _TensorParallelFusedQKFullProjectionRMSNorm.apply(
            query.reshape(query.size(0), query.size(1), -1),
            key.reshape(key.size(0), key.size(1), -1),
            q_norm.weight,
            k_norm.weight,
            q_norm.eps,
            k_norm.eps,
            q_norm.full_hidden_size,
            k_norm.full_hidden_size,
            parallel_state.get_tensor_model_parallel_group(),
        )
    )
    return (
        normalized_query.reshape(query_shape),
        normalized_key.reshape(key_shape),
    )


@dataclass(frozen=True)
class DepthAttentionSource:
    """One earlier layer's post-RoPE key and recursively mixed value."""

    layer_index: int
    key: Tensor
    value: Tensor


def depth_attention_mix(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sources: Sequence[DepthAttentionSource],
    *,
    optimize_mha: bool = False,
) -> Tensor:
    """Mix same-position values across layers before causal self-attention.

    Args:
        query: Current query in ``[sequence, batch, query_heads, head_dim]`` format.
        key: Current key in ``[sequence, batch, kv_heads, head_dim]`` format.
        value: Current value in ``[sequence, batch, kv_heads, head_dim]`` format.
        sources: Earlier keys and recursively mixed values in ascending layer order.

    Returns:
        The recursively mixed current value with the same shape and dtype as ``value``.
    """
    if query.ndim == key.ndim == value.ndim == 3:
        packed_sources = tuple(
            DepthAttentionSource(
                source.layer_index,
                source.key.unsqueeze(1),
                source.value.unsqueeze(1),
            )
            for source in sources
        )
        return depth_attention_mix(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            packed_sources,
            optimize_mha=optimize_mha,
        ).squeeze(1)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "Depth-Attention expects SBHD tensors or packed THD tensors."
        )
    if key.shape != value.shape:
        raise ValueError(
            "Depth-Attention key and value shapes must match; "
            f"got {tuple(key.shape)} and {tuple(value.shape)}."
        )
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Depth-Attention query and key must share sequence, batch, and head dimensions; "
            f"got {tuple(query.shape)} and {tuple(key.shape)}."
        )

    query_heads = query.shape[2]
    kv_heads = key.shape[2]
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"Depth-Attention query heads ({query_heads}) must be divisible by "
            f"KV heads ({kv_heads})."
        )
    if not sources:
        return value

    previous_layer_index = -1
    for source in sources:
        if source.layer_index <= previous_layer_index:
            raise ValueError("Depth-Attention sources must be in strictly increasing layer order.")
        if source.key.shape != key.shape or source.value.shape != value.shape:
            raise ValueError(
                "Every Depth-Attention source must match the current key/value shape; "
                f"layer {source.layer_index} has "
                f"{tuple(source.key.shape)}/{tuple(source.value.shape)}, "
                f"current tensors have {tuple(key.shape)}/{tuple(value.shape)}."
            )
        previous_layer_index = source.layer_index

    query_group_size = query_heads // kv_heads
    if optimize_mha and query_group_size == 1:
        # OLMo3 1B uses MHA (Q heads == KV heads). A mean over a dimension of
        # length one is an avoidable full-activation reduction kernel.
        grouped_query = query
    else:
        grouped_query = query.reshape(
            query.shape[0],
            query.shape[1],
            kv_heads,
            query_group_size,
            query.shape[-1],
        ).mean(dim=3)
    scale = 1.0 / math.sqrt(query.shape[-1])

    # Do not stack retained K/V tensors. At the target 52-layer, micro-batch-4,
    # sequence-2048 shape, those stack results become large autograd-saved copies
    # at every later layer and exhaust a 64 GiB NPU. Computing one source at a
    # time is mathematically identical while retaining only the original source
    # tensors plus the very small per-depth score/weight tensors.
    cross_scores = [
        (grouped_query * source.key).sum(dim=-1, keepdim=True) * scale for source in sources
    ]
    self_score = (grouped_query * key).sum(dim=-1, keepdim=True) * scale
    depth_scores = torch.cat((*cross_scores, self_score), dim=-1)
    depth_weights = torch.softmax(depth_scores, dim=-1, dtype=torch.float32).to(value.dtype)
    mixed_value = depth_weights[..., -1:] * value
    for source_index, source in enumerate(sources):
        mixed_value = (
            mixed_value + depth_weights[..., source_index : source_index + 1] * source.value
        )
    return mixed_value


class DepthAttentionContext:
    """Forward-local sparse storage for Depth-Attention sources."""

    def __init__(self, num_layers: int, stride: int | None, recent_window: int) -> None:
        if num_layers < 1:
            raise ValueError(f"Depth-Attention requires num_layers >= 1, got {num_layers}.")
        if stride is None:
            stride = max(1, num_layers // 2)
        if stride < 1:
            raise ValueError(f"Depth-Attention stride must be >= 1, got {stride}.")
        if recent_window < 0:
            raise ValueError(f"Depth-Attention recent_window must be >= 0, got {recent_window}.")
        self.num_layers = num_layers
        self.stride = stride
        self.recent_window = recent_window
        self._records: dict[int, DepthAttentionSource] = {}

    def selected_layer_indices(self, layer_index: int) -> tuple[int, ...]:
        """Return retained source indices selected for ``layer_index``."""
        self._validate_layer_index(layer_index)
        recent_start = max(0, layer_index - self.recent_window)
        return tuple(
            index
            for index in sorted(self._records)
            if index < layer_index and (index % self.stride == 0 or index >= recent_start)
        )

    def sources_for(self, layer_index: int) -> tuple[DepthAttentionSource, ...]:
        """Return selected source tensors in ascending layer order."""
        return tuple(self._records[index] for index in self.selected_layer_indices(layer_index))

    def record(self, layer_index: int, key: Tensor, mixed_value: Tensor) -> None:
        """Record a layer and discard non-stride values outside the recent window."""
        self._validate_layer_index(layer_index)
        if layer_index in self._records:
            raise ValueError(
                f"Depth-Attention layer {layer_index} was recorded twice in one forward pass."
            )
        self._records[layer_index] = DepthAttentionSource(layer_index, key, mixed_value)

        next_recent_start = layer_index + 1 - self.recent_window
        stale_indices = [
            index
            for index in self._records
            if index % self.stride != 0 and index < next_recent_start
        ]
        for index in stale_indices:
            del self._records[index]

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(
                f"Depth-Attention layer index must be in [0, {self.num_layers}), got {layer_index}."
            )


def siamese_norm_depth_scale(layer_number: int) -> float:
    """Return the official one-based SiameseNorm residual scale."""
    if layer_number < 1:
        raise ValueError(f"SiameseNorm layer_number must be >= 1, got {layer_number}.")
    return math.sqrt(2.0 * layer_number)


def siamese_norm_residual_update(
    post_stream: Tensor, pre_stream: Tensor, residual_update: Tensor, layer_number: int
) -> tuple[Tensor, Tensor]:
    """Inject one shared branch output into both SiameseNorm streams."""
    if post_stream.shape != pre_stream.shape or post_stream.shape != residual_update.shape:
        raise ValueError(
            "SiameseNorm streams and residual update must have identical shapes; "
            f"got {tuple(post_stream.shape)}, {tuple(pre_stream.shape)}, and "
            f"{tuple(residual_update.shape)}."
        )
    scale = siamese_norm_depth_scale(layer_number)
    return post_stream + residual_update / scale, pre_stream + residual_update


class SiameseNormContext:
    """Forward-local storage for SiameseNorm's Pre-Norm-like stream."""

    def __init__(
        self,
        hidden_states: Tensor,
        num_layers: int,
        *,
        clone_initial_stream: bool = True,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"SiameseNorm requires num_layers >= 1, got {num_layers}.")
        self.num_layers = num_layers
        self._pre_stream = (
            hidden_states.clone() if clone_initial_stream else hidden_states
        )
        self._next_layer_number = 1

    def stream_for(self, layer_number: int) -> Tensor:
        """Return the Pre-Norm-like stream for the next transformer layer."""
        self._validate_next_layer(layer_number)
        return self._pre_stream

    def update(self, layer_number: int, pre_stream: Tensor) -> None:
        """Update the current layer stream between its Attention and MLP blocks."""
        self._validate_next_layer(layer_number)
        if pre_stream.shape != self._pre_stream.shape:
            raise ValueError(
                "SiameseNorm cannot change the residual-stream shape between layers; "
                f"got {tuple(self._pre_stream.shape)} then {tuple(pre_stream.shape)}."
            )
        self._pre_stream = pre_stream

    def advance(self, layer_number: int, pre_stream: Tensor) -> None:
        """Record a completed layer's Pre-Norm-like stream."""
        self.update(layer_number, pre_stream)
        self._next_layer_number += 1

    def final_stream(self) -> Tensor:
        """Return the completed stream after all transformer layers ran."""
        if self._next_layer_number != self.num_layers + 1:
            raise ValueError(
                "SiameseNorm final stream was requested before every layer completed; "
                f"completed {self._next_layer_number - 1} of {self.num_layers} layers."
            )
        return self._pre_stream

    def _validate_next_layer(self, layer_number: int) -> None:
        if layer_number != self._next_layer_number:
            raise ValueError(
                "SiameseNorm layers must share one forward-local context in order; "
                f"expected layer {self._next_layer_number}, got {layer_number}."
            )
        if layer_number > self.num_layers:
            raise ValueError(
                f"SiameseNorm context for {self.num_layers} layers cannot run layer {layer_number}."
            )


@dataclass
class Olmo3ArchitectureContext:
    """Combined state carried through MCore 0.12's existing ``context`` slot."""

    siamese_norm: SiameseNormContext
    depth_attention: DepthAttentionContext

    @classmethod
    def create(
        cls,
        hidden_states: Tensor,
        *,
        num_layers: int,
        depth_stride: int,
        depth_recent_window: int,
        clone_initial_stream: bool = True,
    ) -> "Olmo3ArchitectureContext":
        """Create fresh state for one decoder forward."""
        return cls(
            siamese_norm=SiameseNormContext(
                hidden_states,
                num_layers,
                clone_initial_stream=clone_initial_stream,
            ),
            depth_attention=DepthAttentionContext(
                num_layers, stride=depth_stride, recent_window=depth_recent_window
            ),
        )
