"""Per-layer OLMo3 sliding-window attention for TE and Ascend MindSpeed."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.olmo3_runtime_markers import emit_rank0_runtime_marker_once
from modeling.olmo3_config import FULL_ATTENTION, SLIDING_ATTENTION
from modeling.olmo3_swa_utils import (
    build_olmo3_sliding_causal_mask,
    olmo3_te_window,
)


_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_COMPRESSED_CAUSAL_MASKS: dict[tuple[str, int | None], Tensor] = {}
_MINDSPEED_ATTENTION_MODULE_PREFIXES = ("mindspeed.", "mindspeed_llm.")
_MIN_SWA_HALO_OVERLAP_QUERY_TOKENS = 1024


def unwrap_external_ulysses_for_olmo3(
    core_attention: torch.nn.Module,
) -> torch.nn.Module:
    """Remove MindSpeed's outer Ulysses owner from an OLMo3-owned core.

    MindSpeed-LLM patches ``Attention.__init__`` after MCore has constructed
    ``submodules.core_attention``.  For CP>1 that patch wraps the requested
    module in ``UlyssesContextAttention`` and therefore gathers the complete
    sequence *before* calling it.  OLMo3 cannot use that ordering: SWA must
    retain CP-local Q/output and exchange only the adjacent K/V halo, while
    Full Attention performs its own fused QKV Ulysses all-to-all.

    Keep this adapter narrow and fail closed.  It unwraps exactly the
    parameter-free MindSpeed owner around a core which explicitly advertises
    ``olmo3_owns_ulysses``; ordinary attention modules and direct OLMo3 cores
    are returned unchanged.
    """

    local_attention = getattr(core_attention, "local_attn", None)
    if local_attention is None:
        return core_attention
    if not bool(getattr(local_attention, "olmo3_owns_ulysses", False)):
        return core_attention
    wrapper_type = type(core_attention)
    if (
        wrapper_type.__name__ != "UlyssesContextAttention"
        or not wrapper_type.__module__.startswith(
            _MINDSPEED_ATTENTION_MODULE_PREFIXES
        )
    ):
        raise RuntimeError(
            "OLMo3 found an unknown outer attention wrapper around its "
            "Ulysses-owning core; refusing to bypass it."
        )
    children = dict(core_attention.named_children())
    if children != {"local_attn": local_attention}:
        raise RuntimeError(
            "MindSpeed Ulysses wrapper structure changed; OLMo3 requires a "
            "parameter-free wrapper containing only local_attn."
        )
    if tuple(core_attention.named_parameters(recurse=False)) or tuple(
        core_attention.named_buffers(recurse=False)
    ):
        raise RuntimeError(
            "MindSpeed Ulysses wrapper unexpectedly owns persistent state; "
            "OLMo3 will not discard it."
        )
    local_state_keys = tuple(
        local_attention.state_dict(keep_vars=True).keys()
    )
    if local_state_keys:
        raise RuntimeError(
            "OLMo3 Ulysses-owning attention unexpectedly has checkpoint "
            "state, so removing MindSpeed's outer wrapper would change its "
            f"keyspace: {local_state_keys}."
        )
    communication = getattr(local_attention, "ulysses_comm_para", None)
    if not isinstance(communication, dict) or "spg" not in communication:
        raise RuntimeError(
            "MindSpeed Ulysses wrapper did not attach its process group to "
            "the OLMo3 core."
        )
    setattr(local_attention, "olmo3_external_ulysses_unwrapped", True)
    return local_attention


def _swa_overlap_min_query_tokens() -> int:
    """Return an attested positive split-kernel amortization threshold."""

    name = "OLMO3_SWA_HALO_OVERLAP_MIN_QUERY_TOKENS"
    default = _MIN_SWA_HALO_OVERLAP_QUERY_TOKENS
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_swa_cp_mode(context_parallel_size: int) -> None:
    """Reject stale launchers that request a non-production SWA CP algorithm."""

    if context_parallel_size < 1:
        raise ValueError("OLMo3 context parallel size must be positive.")
    mode = os.environ.get("OLMO3_SWA_CP_MODE", "halo").strip().lower()
    allowed = {"halo", "single_halo"}
    if context_parallel_size == 1:
        # The generic command renderer uses "none" when no CP transport runs.
        allowed.add("none")
    if mode not in allowed:
        raise ValueError(
            "OLMO3_SWA_CP_MODE must select the production single halo "
            f"(allowed={sorted(allowed)}); got {mode!r}."
        )


def _swa_halo_overlap_enabled() -> bool:
    """Return the launcher-attested exact overlap path (enabled by default)."""

    value = os.environ.get("OLMO3_SWA_HALO_OVERLAP", "1").strip()
    if value not in ("0", "1"):
        raise ValueError("OLMO3_SWA_HALO_OVERLAP must be exactly 0 or 1.")
    return value == "1"


def _swa_halo_backward_overlap_enabled() -> bool:
    """Return whether the exact CANN backward may hide reverse halo P2P.

    This switch is intentionally independent from the forward switch so a
    launcher can compare the new manually scheduled CANN backward against the
    original torch-npu autograd path without changing checkpoint state or the
    mathematical attention contract.
    """

    value = os.environ.get("OLMO3_SWA_HALO_BACKWARD_OVERLAP", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(
            "OLMO3_SWA_HALO_BACKWARD_OVERLAP must be exactly 0 or 1."
        )
    return value == "1"


def _fused_qkv_a2a_packing_enabled() -> bool:
    """Return whether Full Attention packs Q/K/V directly into A2A layout."""

    value = os.environ.get("OLMO3_FUSED_QKV_A2A_PACKING", "1").strip()
    if value not in ("0", "1"):
        raise ValueError(
            "OLMO3_FUSED_QKV_A2A_PACKING must be exactly 0 or 1."
        )
    return value == "1"


@dataclass(frozen=True)
class Olmo3CannAttentionContract:
    """Arguments whose meaning must stay identical at every CANN boundary."""

    pre_tokens: int
    next_tokens: int
    sparse_mode: int


@dataclass(frozen=True)
class Olmo3PackedSwaHaloPlan:
    """Host-side packing plan for one CP-local packed SWA shard.

    Each segment is ``(batch, local_start, local_end, left_prefix_length)``.
    The optional prefix is always the trailing same-document portion of the
    single-hop left halo. Query segments stay in batch-major local token order,
    while K/V segments prepend only history belonging to that query document.
    """

    segments: tuple[tuple[int, int, int, int], ...]
    query_endpoints: tuple[int, ...]
    kv_endpoints: tuple[int, ...]
    max_kv_length: int


@dataclass(frozen=True)
class Olmo3PackedSwaOverlapPlan:
    """Batch-one work split used to hide a one-hop halo behind local SWA."""

    boundary_query_length: int
    boundary_prefix_length: int
    independent_segments: tuple[tuple[int, int, int, int], ...]
    independent_query_endpoints: tuple[int, ...]
    independent_kv_endpoints: tuple[int, ...]
    independent_max_kv_length: int


@dataclass(frozen=True)
class Olmo3SwaHalo:
    """The ordered causal history collected from the adjacent left CP rank.

    ``history`` contains only real tokens preceding the local CP shard, in
    oldest-to-newest order. ``autograd_anchor`` is numerically zero but keeps
    the exchange in every rank's graph so backward executes the exact
    transposed communication, including on causal boundary ranks.
    """

    history: Tensor
    autograd_anchor: Tensor


@dataclass
class _Olmo3PendingP2P:
    """Own asynchronous P2P requests and their live send buffer."""

    requests: tuple[Any, ...] = ()
    send_tensor: Tensor | None = None
    waited: bool = False

    def wait(self) -> None:
        if self.waited:
            self.send_tensor = None
            return
        for request in self.requests:
            request.wait()
        self.waited = True
        # HCCL no longer needs the source storage after every request finishes.
        self.send_tensor = None


@dataclass(frozen=True)
class Olmo3PendingSwaHalo:
    """A one-hop halo whose HCCL transfer can overlap local attention."""

    history: Tensor
    work: _Olmo3PendingP2P

    def wait(self) -> Tensor:
        self.work.wait()
        return self.history


@dataclass(frozen=True)
class _Olmo3BatchedP2P:
    """Separately track send and receive completion from one backend batch."""

    send_requests: tuple[Any, ...] = ()
    recv_requests: tuple[Any, ...] = ()

    def wait_receives(self) -> None:
        for request in self.recv_requests:
            request.wait()

    def wait_sends(self) -> None:
        for request in self.send_requests:
            request.wait()

    def wait_all(self) -> None:
        self.wait_receives()
        self.wait_sends()


def flatten_packed_sbhd_to_tnd(tensor: Tensor) -> Tensor:
    """Flatten ``[sequence, batch, heads, dim]`` in batch-major TND order."""

    if tensor.ndim != 4:
        raise ValueError(
            "Packed OLMo3 SBHD-to-TND conversion requires a four-dimensional "
            f"tensor, got {tuple(tensor.shape)}."
        )
    sequence_length, batch_size, num_heads, head_dim = tensor.shape
    if min(sequence_length, batch_size, num_heads, head_dim) < 1:
        raise ValueError(
            f"Packed OLMo3 Q/K/V dimensions must be positive, got {tuple(tensor.shape)}."
        )
    if batch_size == 1:
        # Removing a size-one batch axis preserves the exact batch-major TND
        # order and is a view for the contiguous SBHD tensors produced by
        # MCore. Avoid materializing a second 65K activation solely for layout.
        return tensor[:, 0]
    return (
        tensor.permute(1, 0, 2, 3)
        .contiguous()
        .reshape(batch_size * sequence_length, num_heads, head_dim)
    )


def restore_packed_tnd_to_sbh(
    tensor: Tensor,
    *,
    sequence_length: int,
    batch_size: int,
) -> Tensor:
    """Restore batch-major TND output to MCore's ``[sequence, batch, hidden]``."""

    if tensor.ndim != 3:
        raise ValueError(
            "Packed OLMo3 TND output must have [tokens, heads, dim] shape; "
            f"got {tuple(tensor.shape)}."
        )
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("sequence_length and batch_size must both be positive")
    expected_tokens = sequence_length * batch_size
    if tensor.shape[0] != expected_tokens:
        raise ValueError(
            "Packed OLMo3 TND output token count does not match sequence*batch: "
            f"tokens={tensor.shape[0]}, sequence={sequence_length}, batch={batch_size}."
        )
    if batch_size == 1:
        # This is the inverse view of the batch-one fast path above.
        return tensor.reshape(sequence_length, 1, -1)
    return (
        tensor.reshape(batch_size, sequence_length, -1)
        .permute(1, 0, 2)
        .contiguous()
    )


def _fused_swa_halo_tail(
    key: Tensor,
    value: Tensor,
    *,
    halo_length: int,
) -> Tensor:
    """Fuse only the K/V suffix that can cross a causal SWA boundary.

    Stage 3 CP4 owns 16K local tokens but its 4096-token SWA window can consume
    at most 4095 tokens from the previous rank. Fusing full-length local K/V
    before slicing therefore copies 75% data that can never be communicated.
    Slice first, then fuse; the resulting values and autograd mapping are
    identical to slicing the old full fused buffer.
    """

    if key.shape != value.shape:
        raise ValueError(
            "OLMo3 SWA halo requires matching K/V shapes; "
            f"got key={tuple(key.shape)}, value={tuple(value.shape)}."
        )
    if isinstance(halo_length, bool) or halo_length < 0:
        raise ValueError("OLMo3 SWA halo length must be non-negative.")
    if halo_length == 0:
        return torch.cat((key[:0], value[:0]), dim=-1)
    tail_length = min(int(key.shape[0]), halo_length)
    return torch.cat(
        (key[-tail_length:], value[-tail_length:]),
        dim=-1,
    )


def _cann_cumulative_endpoints(
    packed_seq_params: PackedSeqParams,
    field_name: str,
    *,
    total_tokens: int,
) -> list[int]:
    """Return validated endpoint-only lengths for CANN's TND API.

    MCore stores canonical FlashAttention offsets as ``[0, end_0, ...]``.
    CANN consumes ``[end_0, ...]``.  Cache the host list on the per-batch
    metadata object so the conversion and device synchronization happen once
    per microbatch, not once per transformer layer.
    """

    cache_name = f"_olmo3_cann_{field_name}_endpoints"
    cached = getattr(packed_seq_params, cache_name, None)
    if cached is not None:
        if not cached or cached[-1] != total_tokens:
            raise ValueError(
                f"Cached {field_name} endpoints do not cover {total_tokens} tokens."
            )
        return cached

    value = getattr(packed_seq_params, field_name, None)
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError(f"Packed OLMo3 requires one-dimensional {field_name}.")
    if value.dtype != torch.int32:
        raise ValueError(f"Packed OLMo3 {field_name} must use torch.int32.")
    canonical = [int(item) for item in value.tolist()]
    if len(canonical) < 2 or canonical[0] != 0:
        raise ValueError(
            f"Packed OLMo3 {field_name} must start at zero and contain an endpoint."
        )
    if any(right <= left for left, right in zip(canonical, canonical[1:])):
        raise ValueError(f"Packed OLMo3 {field_name} must be strictly increasing.")
    if canonical[-1] != total_tokens:
        raise ValueError(
            f"Packed OLMo3 {field_name} ends at {canonical[-1]}, "
            f"but reconstructed TND has {total_tokens} tokens."
        )
    endpoints = canonical[1:]
    setattr(packed_seq_params, cache_name, endpoints)
    return endpoints


def _ulysses_all_to_all(
    tensor: Tensor,
    process_group: Any,
    scatter_dim: int,
    gather_dim: int,
    gather_size: int | None = None,
) -> Tensor:
    """Call MindSpeed's autograd-aware all-to-all primitive lazily."""

    from mindspeed.te.pytorch.attention.dot_product_attention.ulysses_context_parallel import (
        all_to_all,
    )

    return all_to_all(
        tensor,
        process_group,
        scatter_dim,
        gather_dim,
        gather_size,
    )


class _FusedQkvUlyssesAllToAll(torch.autograd.Function):
    """Pack MHA Q/K/V directly into Ulysses' head-scattered send layout.

    The generic path first materializes ``cat(Q,K,V)`` in SBHD order and then
    copies that complete buffer again while transposing the CP head dimension.
    This exact specialization fuses those two memory transforms into one
    packed send buffer, while retaining one QKV collective and its transposed
    backward collective.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        process_group: Any,
    ) -> Tensor:
        if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "Fused Ulysses QKV packing requires equal four-dimensional Q/K/V."
            )
        world_size = torch.distributed.get_world_size(process_group)
        sequence_length, batch_size, num_heads, head_dim = (
            int(size) for size in query.shape
        )
        if world_size < 2 or num_heads % world_size:
            raise ValueError(
                "Fused Ulysses QKV packing requires heads divisible by CP>1: "
                f"heads={num_heads}, CP={world_size}."
            )
        heads_per_rank = num_heads // world_size

        # Each non-contiguous view exposes [destination CP rank, local
        # sequence, batch, destination heads, dim]. torch.cat writes the three
        # projections once into the final contiguous HCCL send layout.
        packed_send = torch.cat(
            tuple(
                tensor.reshape(
                    sequence_length,
                    batch_size,
                    world_size,
                    heads_per_rank,
                    head_dim,
                ).permute(2, 0, 1, 3, 4)
                for tensor in (query, key, value)
            ),
            dim=-1,
        )
        packed_recv = torch.empty_like(packed_send)
        torch.distributed.all_to_all_single(
            packed_recv,
            packed_send,
            group=process_group,
        )
        ctx.process_group = process_group
        ctx.world_size = world_size
        ctx.sequence_length = sequence_length
        ctx.batch_size = batch_size
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.heads_per_rank = heads_per_rank
        return packed_recv.reshape(
            sequence_length * world_size,
            batch_size,
            heads_per_rank,
            3 * head_dim,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        expected_shape = (
            ctx.sequence_length * ctx.world_size,
            ctx.batch_size,
            ctx.heads_per_rank,
            3 * ctx.head_dim,
        )
        if tuple(grad_output.shape) != expected_shape:
            raise RuntimeError(
                "Fused Ulysses QKV backward received an invalid shape: "
                f"got={tuple(grad_output.shape)}, expected={expected_shape}."
            )
        packed_send = grad_output.reshape(
            ctx.world_size,
            ctx.sequence_length,
            ctx.batch_size,
            ctx.heads_per_rank,
            3 * ctx.head_dim,
        ).contiguous()
        packed_recv = torch.empty_like(packed_send)
        torch.distributed.all_to_all_single(
            packed_recv,
            packed_send,
            group=ctx.process_group,
        )
        local_qkv = (
            packed_recv.permute(1, 2, 0, 3, 4)
            .contiguous()
            .reshape(
                ctx.sequence_length,
                ctx.batch_size,
                ctx.num_heads,
                3 * ctx.head_dim,
            )
        )
        grad_query, grad_key, grad_value = local_qkv.chunk(3, dim=-1)
        return grad_query, grad_key, grad_value, None


def _fused_qkv_ulysses_all_to_all(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    process_group: Any,
) -> Tensor:
    return _FusedQkvUlyssesAllToAll.apply(
        query,
        key,
        value,
        process_group,
    )


def _begin_ordered_directional_p2p(
    *,
    send_tensor: Tensor,
    recv_tensor: Tensor,
    send_peer: int | None,
    recv_peer: int | None,
    process_group: Any,
    group_rank: int,
) -> tuple[Any, ...]:
    """Start one adjacent, non-wrapping exchange without waiting.

    Use parity ordering to avoid a blocking send/receive cycle, while leaving
    both group boundaries open because a causal left halo must never wrap from
    the last CP rank to rank zero.
    """

    operations = []

    def _send() -> None:
        if send_peer is not None:
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_tensor,
                    send_peer,
                    group=process_group,
                )
            )

    def _recv() -> None:
        if recv_peer is not None:
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_tensor,
                    recv_peer,
                    group=process_group,
                )
            )

    if group_rank % 2 == 0:
        _send()
        _recv()
    else:
        _recv()
        _send()
    if not operations:
        return ()
    # Submit the matched adjacent send/receive pair as one backend batch. This
    # preserves the proven parity order while removing one Python dispatch and
    # one HCCL launch boundary from both forward and backward.
    return tuple(torch.distributed.batch_isend_irecv(operations))


def _ordered_directional_p2p(
    *,
    send_tensor: Tensor,
    recv_tensor: Tensor,
    send_peer: int | None,
    recv_peer: int | None,
    process_group: Any,
    group_rank: int,
) -> None:
    """Exchange one adjacent tensor and wait for both directions."""

    requests = _begin_ordered_directional_p2p(
        send_tensor=send_tensor,
        recv_tensor=recv_tensor,
        send_peer=send_peer,
        recv_peer=recv_peer,
        process_group=process_group,
        group_rank=group_rank,
    )
    for request in requests:
        request.wait()


class _LeftHaloExchange(torch.autograd.Function):
    """Autograd-aware one-hop left halo exchange.

    Forward maps rank ``r``'s tensor to rank ``r + 1``. Backward performs the
    exact transpose communication, returning the halo gradient from
    ``r + 1`` to its source rank ``r``. Boundary ranks exchange no wrapped
    data, preserving causal sequence order.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        process_group: Any,
        global_ranks: tuple[int, ...],
    ) -> Tensor:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("OLMo3 SWA halo exchange requires initialized distributed.")
        actual_size = torch.distributed.get_world_size(process_group)
        if actual_size != len(global_ranks):
            raise RuntimeError(
                "OLMo3 SWA halo rank list disagrees with its process group: "
                f"group={actual_size}, ranks={len(global_ranks)}."
            )
        group_rank = torch.distributed.get_rank(process_group)
        if not 0 <= group_rank < actual_size:
            raise RuntimeError(
                f"Invalid true-overlap CP rank {group_rank} for CP={actual_size}."
            )

        send_tensor = input_.contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        previous_peer = global_ranks[group_rank - 1] if group_rank > 0 else None
        next_peer = (
            global_ranks[group_rank + 1]
            if group_rank + 1 < actual_size
            else None
        )
        if previous_peer is None:
            recv_tensor.zero_()
        _ordered_directional_p2p(
            send_tensor=send_tensor,
            recv_tensor=recv_tensor,
            send_peer=next_peer,
            recv_peer=previous_peer,
            process_group=process_group,
            group_rank=group_rank,
        )
        ctx.process_group = process_group
        ctx.global_ranks = global_ranks
        ctx.group_rank = group_rank
        return recv_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        global_ranks = ctx.global_ranks
        group_rank = ctx.group_rank
        actual_size = len(global_ranks)
        grad_input = torch.zeros_like(grad_output)
        previous_peer = global_ranks[group_rank - 1] if group_rank > 0 else None
        next_peer = (
            global_ranks[group_rank + 1]
            if group_rank + 1 < actual_size
            else None
        )
        _ordered_directional_p2p(
            send_tensor=grad_output.contiguous(),
            recv_tensor=grad_input,
            send_peer=previous_peer,
            recv_peer=next_peer,
            process_group=ctx.process_group,
            group_rank=group_rank,
        )
        return grad_input, None, None


def _left_halo_exchange(
    tensor: Tensor,
    process_group: Any,
    global_ranks: tuple[int, ...],
) -> Tensor:
    return _LeftHaloExchange.apply(tensor, process_group, global_ranks)


class _VariableLeftHaloExchange(torch.autograd.Function):
    """Autograd-aware one-hop halo with exact document-dependent lengths.

    A packed document boundary can require fewer than ``W - 1`` remote
    tokens, including zero. Every rank knows the global document IDs, so rank
    ``r`` can send exactly the suffix that rank ``r + 1`` will consume. The
    reverse pass exchanges the transposed, correspondingly ragged gradients.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        recv_length: int,
        process_group: Any,
        global_ranks: tuple[int, ...],
    ) -> Tensor:
        if input_.ndim < 1:
            raise ValueError("Variable OLMo3 SWA halo requires a token dimension.")
        if isinstance(recv_length, bool) or recv_length < 0:
            raise ValueError("Variable OLMo3 SWA receive length must be non-negative.")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Variable OLMo3 SWA halo requires initialized distributed.")
        actual_size = torch.distributed.get_world_size(process_group)
        if actual_size != len(global_ranks):
            raise RuntimeError(
                "Variable OLMo3 SWA halo rank list disagrees with its process group: "
                f"group={actual_size}, ranks={len(global_ranks)}."
            )
        # The process group already owns the global-to-local rank mapping.
        # Avoid scanning the CP rank tuple in every one of the 12 SWA layers.
        group_rank = torch.distributed.get_rank(process_group)
        if not 0 <= group_rank < actual_size:
            raise RuntimeError(
                f"Invalid true-overlap CP rank {group_rank} for CP={actual_size}."
            )

        send_length = int(input_.shape[0])
        recv_length = int(recv_length)
        recv_tensor = input_.new_empty((recv_length, *input_.shape[1:]))
        send_peer = (
            global_ranks[group_rank + 1]
            if send_length > 0 and group_rank + 1 < actual_size
            else None
        )
        recv_peer = (
            global_ranks[group_rank - 1]
            if recv_length > 0 and group_rank > 0
            else None
        )
        _ordered_directional_p2p(
            send_tensor=input_.contiguous(),
            recv_tensor=recv_tensor,
            send_peer=send_peer,
            recv_peer=recv_peer,
            process_group=process_group,
            group_rank=group_rank,
        )
        ctx.process_group = process_group
        ctx.global_ranks = global_ranks
        ctx.group_rank = group_rank
        ctx.send_length = send_length
        ctx.recv_length = recv_length
        return recv_tensor

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None]:
        global_ranks = ctx.global_ranks
        group_rank = ctx.group_rank
        actual_size = len(global_ranks)
        grad_input = grad_output.new_zeros(
            (ctx.send_length, *grad_output.shape[1:])
        )
        send_peer = (
            global_ranks[group_rank - 1]
            if ctx.recv_length > 0 and group_rank > 0
            else None
        )
        recv_peer = (
            global_ranks[group_rank + 1]
            if ctx.send_length > 0 and group_rank + 1 < actual_size
            else None
        )
        _ordered_directional_p2p(
            send_tensor=grad_output.contiguous(),
            recv_tensor=grad_input,
            send_peer=send_peer,
            recv_peer=recv_peer,
            process_group=ctx.process_group,
            group_rank=group_rank,
        )
        return grad_input, None, None, None


class _AsyncVariableLeftHaloExchange(torch.autograd.Function):
    """Start a ragged halo in forward and wait only when its data is consumed."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: Tensor,
        recv_length: int,
        process_group: Any,
        global_ranks: tuple[int, ...],
        work: _Olmo3PendingP2P,
    ) -> Tensor:
        if input_.ndim < 1:
            raise ValueError("Asynchronous OLMo3 SWA halo requires a token dimension.")
        if isinstance(recv_length, bool) or recv_length < 0:
            raise ValueError(
                "Asynchronous OLMo3 SWA receive length must be non-negative."
            )
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "Asynchronous OLMo3 SWA halo requires initialized distributed."
            )
        actual_size = torch.distributed.get_world_size(process_group)
        if actual_size != len(global_ranks):
            raise RuntimeError(
                "Asynchronous OLMo3 SWA halo rank list disagrees with its "
                f"process group: group={actual_size}, ranks={len(global_ranks)}."
            )
        group_rank = torch.distributed.get_rank(process_group)
        if not 0 <= group_rank < actual_size:
            raise RuntimeError(
                f"Invalid asynchronous halo CP rank {group_rank} for CP={actual_size}."
            )

        send_length = int(input_.shape[0])
        recv_length = int(recv_length)
        recv_tensor = input_.new_empty((recv_length, *input_.shape[1:]))
        send_peer = (
            global_ranks[group_rank + 1]
            if send_length > 0 and group_rank + 1 < actual_size
            else None
        )
        recv_peer = (
            global_ranks[group_rank - 1]
            if recv_length > 0 and group_rank > 0
            else None
        )
        send_tensor = input_.contiguous()
        work.send_tensor = send_tensor
        work.requests = _begin_ordered_directional_p2p(
            send_tensor=send_tensor,
            recv_tensor=recv_tensor,
            send_peer=send_peer,
            recv_peer=recv_peer,
            process_group=process_group,
            group_rank=group_rank,
        )
        work.waited = not work.requests

        ctx.process_group = process_group
        ctx.global_ranks = global_ranks
        ctx.group_rank = group_rank
        ctx.send_length = send_length
        ctx.recv_length = recv_length
        ctx.work = work
        return recv_tensor

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None]:
        if not ctx.work.waited:
            raise RuntimeError(
                "Asynchronous OLMo3 SWA halo reached backward before forward "
                "communication completed."
            )
        global_ranks = ctx.global_ranks
        group_rank = ctx.group_rank
        actual_size = len(global_ranks)
        grad_input = grad_output.new_zeros(
            (ctx.send_length, *grad_output.shape[1:])
        )
        send_peer = (
            global_ranks[group_rank - 1]
            if ctx.recv_length > 0 and group_rank > 0
            else None
        )
        recv_peer = (
            global_ranks[group_rank + 1]
            if ctx.send_length > 0 and group_rank + 1 < actual_size
            else None
        )
        _ordered_directional_p2p(
            send_tensor=grad_output.contiguous(),
            recv_tensor=grad_input,
            send_peer=send_peer,
            recv_peer=recv_peer,
            process_group=ctx.process_group,
            group_rank=group_rank,
        )
        return grad_input, None, None, None, None


def _variable_left_halo_exchange(
    tensor: Tensor,
    *,
    recv_length: int,
    process_group: Any,
    global_ranks: tuple[int, ...],
) -> Tensor:
    return _VariableLeftHaloExchange.apply(
        tensor,
        recv_length,
        process_group,
        global_ranks,
    )


def _begin_variable_left_halo_exchange(
    tensor: Tensor,
    *,
    recv_length: int,
    process_group: Any,
    global_ranks: tuple[int, ...],
) -> Olmo3PendingSwaHalo:
    work = _Olmo3PendingP2P()
    history = _AsyncVariableLeftHaloExchange.apply(
        tensor,
        recv_length,
        process_group,
        global_ranks,
        work,
    )
    return Olmo3PendingSwaHalo(history=history, work=work)


def _begin_adjacent_halo_p2p(
    *,
    send_items: tuple[tuple[int, Tensor], ...],
    recv_items: tuple[tuple[int, Tensor], ...],
    process_group: Any,
    group_rank: int,
) -> _Olmo3BatchedP2P:
    """Submit the single adjacent send/receive pair as one backend batch."""

    if len(send_items) > 1 or len(recv_items) > 1:
        raise ValueError(
            "OLMo3 production SWA supports one adjacent halo peer per direction."
        )

    send_operations = [
        torch.distributed.P2POp(
            torch.distributed.isend,
            tensor,
            peer,
            group=process_group,
        )
        for peer, tensor in send_items
    ]
    recv_operations = [
        torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor,
            peer,
            group=process_group,
        )
        for peer, tensor in recv_items
    ]
    send_first = group_rank % 2 == 0
    operations = (
        send_operations + recv_operations
        if send_first
        else recv_operations + send_operations
    )
    if not operations:
        return _Olmo3BatchedP2P()
    requests = tuple(torch.distributed.batch_isend_irecv(operations))
    if not requests:
        raise RuntimeError(
            "Adjacent halo P2P submitted operations but returned no work handle."
        )
    if len(requests) == len(operations):
        # Stock ProcessGroup implementations expose one Work per P2POp, so the
        # original operation order gives exact directional completion handles.
        if send_first:
            send_requests = requests[: len(send_operations)]
            recv_requests = requests[len(send_operations) :]
        else:
            recv_requests = requests[: len(recv_operations)]
            send_requests = requests[len(recv_operations) :]
    elif send_operations and recv_operations:
        # PTA/HCCL coalesces a complete batch_isend_irecv call into one Work.
        # Such a handle cannot prove receive completion independently from send
        # completion. Treat it conservatively as the receive dependency; the
        # later send wait is then a no-op because the same Work already covered
        # both directions. Calls containing only one direction (used by the
        # backward pre-post path) remain independently waitable below.
        recv_requests = requests
        send_requests = ()
    elif send_operations:
        send_requests = requests
        recv_requests = ()
    else:
        send_requests = ()
        recv_requests = requests
    return _Olmo3BatchedP2P(
        send_requests=tuple(send_requests),
        recv_requests=tuple(recv_requests),
    )


def _require_single_hop_halo_capacity(
    *,
    local_sequence_length: int,
    halo_length: int,
) -> None:
    """Fail closed when a SWA window would require more than one CP hop."""

    if local_sequence_length < 1:
        raise ValueError("OLMo3 SWA local sequence length must be positive.")
    if isinstance(halo_length, bool) or halo_length < 0:
        raise ValueError("OLMo3 SWA halo length must be non-negative.")
    if local_sequence_length < halo_length:
        raise ValueError(
            "OLMo3 production SWA CP supports a single adjacent K/V halo only: "
            f"local_sequence_length={local_sequence_length} must be at least "
            f"window_minus_one={halo_length}."
        )


def _single_hop_left_halo_exchange(
    tensor: Tensor,
    process_group: Any,
    global_ranks: tuple[int, ...],
    *,
    halo_length: int,
) -> Olmo3SwaHalo:
    """Collect one fixed-length causal K/V halo from the adjacent left rank."""

    if tensor.ndim < 1:
        raise ValueError("OLMo3 SWA halo requires a token dimension.")
    if int(tensor.shape[0]) != halo_length:
        raise ValueError(
            "Fixed OLMo3 SWA halo tensor must contain exactly window_minus_one "
            f"tokens: got={tensor.shape[0]}, expected={halo_length}."
        )
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("OLMo3 SWA halo requires initialized distributed.")
    actual_size = torch.distributed.get_world_size(process_group)
    if actual_size != len(global_ranks):
        raise RuntimeError(
            "OLMo3 SWA halo rank list disagrees with its process group: "
            f"group={actual_size}, ranks={len(global_ranks)}."
        )
    group_rank = torch.distributed.get_rank(process_group)
    if not 0 <= group_rank < actual_size:
        raise RuntimeError(
            f"Invalid OLMo3 SWA halo group rank {group_rank} for CP={actual_size}."
        )
    if halo_length == 0 or actual_size == 1:
        return Olmo3SwaHalo(
            history=tensor[:0],
            autograd_anchor=tensor.sum() * 0.0,
        )

    received = _left_halo_exchange(tensor, process_group, global_ranks)
    history = received[:0] if group_rank == 0 else received
    return Olmo3SwaHalo(
        history=history,
        # Rank zero consumes no history but still feeds rank one. The zero
        # anchor keeps its exact transposed gradient exchange in the graph.
        autograd_anchor=received.sum() * 0.0,
    )


def _packed_swa_halo_plan(
    packed_seq_params: PackedSeqParams,
    *,
    cp_rank: int,
    cp_size: int,
    local_sequence_length: int,
    sliding_window: int,
) -> Olmo3PackedSwaHaloPlan:
    """Build and cache document-safe TND segments for a local SWA shard."""

    if min(cp_size, local_sequence_length, sliding_window) < 1:
        raise ValueError("CP size, local sequence length, and SWA window must be positive.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid CP rank {cp_rank} for CP size {cp_size}.")
    halo_length = sliding_window - 1
    _require_single_hop_halo_capacity(
        local_sequence_length=local_sequence_length,
        halo_length=halo_length,
    )

    cache = getattr(packed_seq_params, "_olmo3_swa_halo_plans", None)
    if cache is None:
        cache = {}
        setattr(packed_seq_params, "_olmo3_swa_halo_plans", cache)
    cache_key = (cp_rank, cp_size, local_sequence_length, sliding_window)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    global_document_ids = getattr(
        packed_seq_params,
        "olmo3_global_document_ids",
        None,
    )
    if isinstance(global_document_ids, Tensor):
        if global_document_ids.ndim != 2:
            raise ValueError(
                "Packed SWA global document IDs must be two-dimensional."
            )
        batch_size, global_sequence_length = (
            int(global_document_ids.shape[0]),
            int(global_document_ids.shape[1]),
        )
    else:
        batch_size = int(
            getattr(packed_seq_params, "olmo3_micro_batch_size", 0)
        )
        global_sequence_length = int(
            getattr(
                packed_seq_params,
                "olmo3_global_tokens_per_sample",
                0,
            )
        )
    if batch_size < 1 or global_sequence_length < 1:
        raise ValueError(
            "Packed SWA halo CP requires positive compact global batch metadata."
        )
    expected_global_length = local_sequence_length * cp_size
    if global_sequence_length != expected_global_length:
        raise ValueError(
            "Packed SWA global/local sequence lengths disagree with CP: "
            f"global={global_sequence_length}, local={local_sequence_length}, "
            f"CP={cp_size}."
        )

    # Reuse the canonical document endpoints that CANN consumes instead of
    # copying every global document ID to the host. Real Longmino batches
    # contain only tens of endpoints but 65,536 token IDs, so this also removes
    # a large device synchronization from the first SWA layer of every step.
    total_tokens = batch_size * global_sequence_length
    canonical_endpoints = [
        0,
        *_cann_cumulative_endpoints(
            packed_seq_params,
            "cu_seqlens_q",
            total_tokens=total_tokens,
        ),
    ]
    document_intervals = tuple(
        zip(canonical_endpoints[:-1], canonical_endpoints[1:])
    )

    segments: list[tuple[int, int, int, int]] = []
    query_endpoints: list[int] = []
    kv_endpoints: list[int] = []
    query_total = 0
    kv_total = 0
    max_kv_length = 0
    for batch_index in range(batch_size):
        row_start = batch_index * global_sequence_length
        local_global_start = row_start + cp_rank * local_sequence_length
        local_global_end = local_global_start + local_sequence_length
        covered_local_tokens = 0
        for document_start, document_end in document_intervals:
            if document_end <= local_global_start:
                continue
            if document_start >= local_global_end:
                break
            intersection_start = max(document_start, local_global_start)
            intersection_end = min(document_end, local_global_end)
            if intersection_end <= intersection_start:
                continue
            local_start = intersection_start - local_global_start
            local_end = intersection_end - local_global_start
            left_prefix_length = (
                min(halo_length, local_global_start - document_start)
                if local_start == 0 and document_start < local_global_start
                else 0
            )
            query_length = local_end - local_start
            kv_length = left_prefix_length + query_length
            query_total += query_length
            kv_total += kv_length
            max_kv_length = max(max_kv_length, kv_length)
            query_endpoints.append(query_total)
            kv_endpoints.append(kv_total)
            segments.append(
                (
                    batch_index,
                    local_start,
                    local_end,
                    left_prefix_length,
                )
            )
            covered_local_tokens += query_length
        if covered_local_tokens != local_sequence_length:
            raise RuntimeError(
                "Packed SWA document endpoints do not cover the complete local "
                f"row: batch={batch_index}, covered={covered_local_tokens}, "
                f"expected={local_sequence_length}."
            )

    if query_total != batch_size * local_sequence_length:
        raise RuntimeError(
            "Packed SWA halo plan does not cover every local query token: "
            f"covered={query_total}, expected={batch_size * local_sequence_length}."
        )
    plan = Olmo3PackedSwaHaloPlan(
        segments=tuple(segments),
        query_endpoints=tuple(query_endpoints),
        kv_endpoints=tuple(kv_endpoints),
        max_kv_length=max_kv_length,
    )
    cache[cache_key] = plan
    return plan


def _pack_swa_halo_tnd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    halo_key: Tensor,
    halo_value: Tensor,
    plan: Olmo3PackedSwaHaloPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a halo plan while preserving local batch-major query order."""

    query_tnd = flatten_packed_sbhd_to_tnd(query)
    sequence_length, batch_size = int(key.shape[0]), int(key.shape[1])
    if batch_size == 1:
        if (
            not plan.segments
            or plan.segments[0][0] != 0
            or plan.segments[0][1] != 0
            or plan.segments[-1][0] != 0
            or plan.segments[-1][2] != sequence_length
        ):
            raise RuntimeError(
                "Packed SWA halo plan does not cover the batch-one local row."
            )
        prefix_length = plan.segments[0][3]
        key_tnd = key[:, 0]
        value_tnd = value[:, 0]
        if prefix_length:
            key_tnd = torch.cat(
                (halo_key[-prefix_length:, 0], key_tnd),
                dim=0,
            )
            value_tnd = torch.cat(
                (halo_value[-prefix_length:, 0], value_tnd),
                dim=0,
            )
        return query_tnd, key_tnd, value_tnd

    key_rows = []
    value_rows = []
    for batch_index in range(batch_size):
        row_segments = tuple(
            segment
            for segment in plan.segments
            if segment[0] == batch_index
        )
        if not row_segments:
            raise RuntimeError(
                f"Packed SWA halo plan is missing batch row {batch_index}."
            )
        expected_start = 0
        for _, local_start, local_end, prefix_length in row_segments:
            if local_start != expected_start or local_end <= local_start:
                raise RuntimeError(
                    "Packed SWA halo segments must cover each batch row "
                    "contiguously and in order."
                )
            if local_start > 0 and prefix_length:
                raise RuntimeError(
                    "Only the first local document may consume a remote halo."
                )
            expected_start = local_end
        if expected_start != sequence_length:
            raise RuntimeError(
                "Packed SWA halo segments do not cover the complete local row."
            )

        prefix_length = row_segments[0][3]
        local_key = key[:, batch_index]
        local_value = value[:, batch_index]
        if prefix_length:
            local_key = torch.cat(
                (halo_key[-prefix_length:, batch_index], local_key),
                dim=0,
            )
            local_value = torch.cat(
                (halo_value[-prefix_length:, batch_index], local_value),
                dim=0,
            )
        key_rows.append(local_key)
        value_rows.append(local_value)
    if not key_rows:
        raise RuntimeError("Packed SWA halo plan contains no attention segments.")
    key_tnd = torch.cat(key_rows, dim=0)
    value_tnd = torch.cat(value_rows, dim=0)
    return (
        query_tnd,
        key_tnd,
        value_tnd,
    )


def _maximum_halo_prefix(plan: Olmo3PackedSwaHaloPlan) -> int:
    """Return the largest same-document prefix consumed by any batch row."""

    return max(
        (prefix_length for _, _, _, prefix_length in plan.segments),
        default=0,
    )


def _packed_swa_group_recv_lengths(
    packed_seq_params: PackedSeqParams,
    *,
    cp_size: int,
    local_sequence_length: int,
    sliding_window: int,
) -> tuple[int, ...]:
    """Cache the exact document-ragged receive schedule for all CP ranks.

    The schedule is shared by every SWA layer in a microbatch.  Building it in
    each of 12 local-attention layers repeats the same CP-sized Python scan even
    though the compact document endpoints are immutable for that batch.
    """

    cache = getattr(
        packed_seq_params,
        "_olmo3_swa_group_recv_length_schedules",
        None,
    )
    if cache is None:
        cache = {}
        setattr(
            packed_seq_params,
            "_olmo3_swa_group_recv_length_schedules",
            cache,
        )
    cache_key = (cp_size, local_sequence_length, sliding_window)
    cached = cache.get(cache_key)
    if cached is not None:
        return tuple(cached)
    schedule = tuple(
        _maximum_halo_prefix(
            _packed_swa_halo_plan(
                packed_seq_params,
                cp_rank=rank,
                cp_size=cp_size,
                local_sequence_length=local_sequence_length,
                sliding_window=sliding_window,
            )
        )
        for rank in range(cp_size)
    )
    cache[cache_key] = schedule
    return schedule


def _packed_swa_overlap_plan(
    packed_seq_params: PackedSeqParams,
    plan: Olmo3PackedSwaHaloPlan,
    *,
    local_sequence_length: int,
    halo_length: int,
) -> Olmo3PackedSwaOverlapPlan:
    """Split batch-one packed queries into halo-dependent and local work."""

    cache = getattr(packed_seq_params, "_olmo3_swa_overlap_plans", None)
    if cache is None:
        cache = {}
        setattr(packed_seq_params, "_olmo3_swa_overlap_plans", cache)
    cache_key = (id(plan), local_sequence_length, halo_length)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if local_sequence_length < 1 or halo_length < 0:
        raise ValueError("Packed SWA overlap lengths must be non-negative.")
    if not plan.segments:
        raise RuntimeError("Packed SWA overlap requires at least one segment.")
    if any(batch_index != 0 for batch_index, *_ in plan.segments):
        raise ValueError("Packed SWA overlap fast path currently requires micro-batch one.")
    first_batch, first_start, first_end, first_prefix = plan.segments[0]
    if first_batch != 0 or first_start != 0 or first_end <= first_start:
        raise RuntimeError("Packed SWA overlap plan has an invalid first segment.")

    boundary_query_length = (
        min(halo_length, first_end - first_start)
        if first_prefix > 0
        else 0
    )
    independent_segments: list[tuple[int, int, int, int]] = []
    query_endpoints: list[int] = []
    kv_endpoints: list[int] = []
    query_total = 0
    kv_total = 0
    max_kv_length = 0
    for segment_index, (_, local_start, local_end, _) in enumerate(plan.segments):
        query_start = local_start
        kv_start = local_start
        if segment_index == 0:
            query_start += boundary_query_length
            # The independent suffix of the first document still attends to
            # its earlier CP-local tokens. Right alignment plus sparse_mode=4
            # applies the exact W-token window without any remote K/V.
            kv_start = local_start
        if query_start >= local_end:
            continue
        query_length = local_end - query_start
        kv_length = local_end - kv_start
        query_total += query_length
        kv_total += kv_length
        max_kv_length = max(max_kv_length, kv_length)
        query_endpoints.append(query_total)
        kv_endpoints.append(kv_total)
        independent_segments.append(
            (query_start, local_end, kv_start, local_end)
        )
    if boundary_query_length + query_total != local_sequence_length:
        raise RuntimeError(
            "Packed SWA overlap split does not cover every local query: "
            f"boundary={boundary_query_length}, independent={query_total}, "
            f"local={local_sequence_length}."
        )
    overlap_plan = Olmo3PackedSwaOverlapPlan(
        boundary_query_length=boundary_query_length,
        boundary_prefix_length=first_prefix,
        independent_segments=tuple(independent_segments),
        independent_query_endpoints=tuple(query_endpoints),
        independent_kv_endpoints=tuple(kv_endpoints),
        independent_max_kv_length=max_kv_length,
    )
    cache[cache_key] = overlap_plan
    return overlap_plan


def _packed_swa_group_overlap_is_worthwhile(
    packed_seq_params: PackedSeqParams,
    *,
    cp_size: int,
    local_sequence_length: int,
    sliding_window: int,
) -> bool:
    """Choose the split SWA path once for the complete CP group.

    All ranks enter the same custom autograd path and post exactly matched P2P
    operations. Individual ranks may have different packed-document splits;
    ranks without independent work simply wait, while other ranks hide their
    transfer behind local CANN attention. Enable the split only when the
    group-average independent work is large enough to amortize the extra
    kernel launch.
    """

    if cp_size < 2:
        return False
    if local_sequence_length < 1 or sliding_window < 2:
        raise ValueError(
            "Packed SWA group overlap requires positive local length and window >= 2."
        )
    cache = getattr(packed_seq_params, "_olmo3_swa_group_overlap_decisions", None)
    if cache is None:
        cache = {}
        setattr(packed_seq_params, "_olmo3_swa_group_overlap_decisions", cache)
    cache_key = (cp_size, local_sequence_length, sliding_window)
    cached = cache.get(cache_key)
    if cached is not None:
        return bool(cached)

    halo_length = sliding_window - 1
    plans = tuple(
        _packed_swa_halo_plan(
            packed_seq_params,
            cp_rank=rank,
            cp_size=cp_size,
            local_sequence_length=local_sequence_length,
            sliding_window=sliding_window,
        )
        for rank in range(cp_size)
    )
    recv_lengths = _packed_swa_group_recv_lengths(
        packed_seq_params,
        cp_size=cp_size,
        local_sequence_length=local_sequence_length,
        sliding_window=sliding_window,
    )
    has_exchange = False
    independent_tokens_by_rank: list[int] = []
    for rank, plan in enumerate(plans):
        send_length = recv_lengths[rank + 1] if rank + 1 < cp_size else 0
        recv_length = recv_lengths[rank]
        has_exchange = has_exchange or send_length > 0 or recv_length > 0
        overlap_plan = _packed_swa_overlap_plan(
            packed_seq_params,
            plan,
            local_sequence_length=local_sequence_length,
            halo_length=halo_length,
        )
        independent_tokens = (
            overlap_plan.independent_query_endpoints[-1]
            if overlap_plan.independent_query_endpoints
            else 0
        )
        independent_tokens_by_rank.append(independent_tokens)
    threshold = _swa_overlap_min_query_tokens()
    decision = (
        has_exchange
        and sum(independent_tokens_by_rank) >= threshold * cp_size
    )
    cache[cache_key] = decision
    return decision


def _pack_swa_independent_tnd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    plan: Olmo3PackedSwaOverlapPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack batch-one query segments that do not consume a remote halo."""

    if not plan.independent_segments:
        return query[:0, 0], key[:0, 0], value[:0, 0]

    first_query_start = plan.independent_segments[0][0]
    first_kv_start = plan.independent_segments[0][2]
    previous_query_end = plan.independent_segments[-1][1]
    previous_kv_end = plan.independent_segments[-1][3]
    if (
        previous_query_end <= first_query_start
        or previous_kv_end <= first_kv_start
    ):
        raise RuntimeError("Packed SWA independent slice is empty or reversed.")
    # The CANN endpoint arrays, rather than physical gaps in Q/K/V, separate
    # documents. All independent pieces are adjacent views of the original
    # batch-one tensors, so concatenating them would only copy the same large
    # activation back into its original order.
    return (
        query[first_query_start:previous_query_end, 0],
        key[first_kv_start:previous_kv_end, 0],
        value[first_kv_start:previous_kv_end, 0],
    )


def _pack_swa_boundary_tnd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    halo_key: Tensor,
    halo_value: Tensor,
    plan: Olmo3PackedSwaOverlapPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack the only batch-one query prefix that can consume remote K/V."""

    query_length = plan.boundary_query_length
    prefix_length = plan.boundary_prefix_length
    if query_length < 1 or prefix_length < 1:
        return query[:0, 0], key[:0, 0], value[:0, 0]
    return (
        query[:query_length, 0],
        torch.cat(
            (halo_key[-prefix_length:, 0], key[:query_length, 0]),
            dim=0,
        ),
        torch.cat(
            (halo_value[-prefix_length:, 0], value[:query_length, 0]),
            dim=0,
        ),
    )


def _is_mindspeed_attention(
    attention_cls: type[Any], config: TransformerConfig
) -> bool:
    """Return the explicit runtime backend, with an import-only fallback.

    MindSpeed can patch methods on Megatron's TE wrapper without replacing the
    class object.  In that case ``__module__`` still starts with
    ``megatron.`` and is not a reliable capability probe.  Production attaches
    ``olmo3_mindspeed_runtime`` while constructing the MCore config; the module
    check remains only for isolated tests and third-party callers that do not
    use the MindSpeed entrypoint.
    """

    explicit_backend = getattr(config, "olmo3_mindspeed_runtime", None)
    if explicit_backend is not None:
        return bool(explicit_backend)
    return attention_cls.__module__.startswith(_MINDSPEED_ATTENTION_MODULE_PREFIXES)


def _layer_attention_type(config: TransformerConfig, layer_number: int) -> str:
    layer_types = tuple(getattr(config, "olmo3_layer_types", ()))
    if len(layer_types) != int(config.num_layers):
        raise ValueError(
            "OLMo3 runtime config must carry one olmo3_layer_types entry per layer."
        )
    layer_index = layer_number - 1
    if not 0 <= layer_index < len(layer_types):
        raise ValueError(
            f"OLMo3 layer_number must be in [1, {len(layer_types)}], got {layer_number}."
        )
    attention_type = layer_types[layer_index]
    if attention_type not in (SLIDING_ATTENTION, FULL_ATTENTION):
        raise ValueError(f"Unsupported OLMo3 attention type {attention_type!r}.")
    return attention_type


def _compressed_causal_mask(device: torch.device) -> Tensor:
    key = (device.type, device.index)
    mask = _COMPRESSED_CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        _COMPRESSED_CAUSAL_MASKS[key] = mask
    return mask


def _run_raw_cann_tnd_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    contract: Olmo3CannAttentionContract,
    actual_seq_qlen: tuple[int, ...],
    actual_seq_kvlen: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    """Run CANN TND attention while retaining its exact backward statistics."""

    import torch_npu

    result = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        num_heads,
        "TND",
        pse=None,
        padding_mask=None,
        atten_mask=_compressed_causal_mask(query.device),
        scale=softmax_scale,
        pre_tockens=contract.pre_tokens,
        next_tockens=contract.next_tokens,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=contract.sparse_mode,
        actual_seq_qlen=list(actual_seq_qlen),
        actual_seq_kvlen=list(actual_seq_kvlen),
    )
    return result[0], result[1], result[2]


def _run_raw_cann_tnd_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    attention_output: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    *,
    num_heads: int,
    softmax_scale: float,
    contract: Olmo3CannAttentionContract,
    actual_seq_qlen: tuple[int, ...],
    actual_seq_kvlen: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the exact CANN gradient used by torch-npu's attention autograd."""

    import torch_npu

    result = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        grad_output,
        num_heads,
        "TND",
        pse=None,
        padding_mask=None,
        atten_mask=_compressed_causal_mask(query.device),
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_output,
        scale_value=softmax_scale,
        pre_tockens=contract.pre_tokens,
        next_tockens=contract.next_tokens,
        sparse_mode=contract.sparse_mode,
        keep_prob=1.0,
        actual_seq_qlen=list(actual_seq_qlen),
        actual_seq_kvlen=list(actual_seq_kvlen),
    )
    return result[0], result[1], result[2]


class _PackedSwaOneHopOverlap(torch.autograd.Function):
    """Exact packed SWA with forward and reverse-halo compute overlap.

    The first local document prefix is the only query block that can consume
    remote K/V.  Forward computes the remaining independent queries while the
    adjacent left halo is in flight. Backward computes the boundary gradient,
    starts the exact adjacent transposed exchange, then computes the independent
    attention gradient while reverse communication remains in flight.

    The function owns the complete split attention operation rather than only
    the halo tensor.  This is necessary because a tensor-only autograd Function
    cannot return an unready gradient to its caller and therefore must wait
    before any useful downstream backward computation can overlap.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        process_group: Any,
        global_ranks: tuple[int, ...],
        recv_lengths: tuple[int, ...],
        overlap_plan: Olmo3PackedSwaOverlapPlan,
        softmax_scale: float,
        contract: Olmo3CannAttentionContract,
    ) -> Tensor:
        if query.ndim != 4 or query.shape[1] != 1:
            raise ValueError(
                "True-overlap packed SWA requires batch-one SBHD query tensors."
            )
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "True-overlap packed SWA requires equal Q/K/V shapes; got "
                f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        if contract.sparse_mode != 4:
            raise ValueError("True-overlap packed SWA requires CANN sparse_mode=4.")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("True-overlap packed SWA requires initialized distributed.")
        actual_size = torch.distributed.get_world_size(process_group)
        if actual_size != len(global_ranks):
            raise RuntimeError(
                "True-overlap SWA rank list disagrees with its process group: "
                f"group={actual_size}, ranks={len(global_ranks)}."
            )
        group_rank = torch.distributed.get_rank(process_group)
        if not 0 <= group_rank < actual_size:
            raise RuntimeError(
                f"Invalid true-overlap CP rank {group_rank} for CP={actual_size}."
            )

        local_sequence_length = int(query.shape[0])
        num_heads = int(query.shape[2])
        head_dim = int(query.shape[3])
        recv_lengths = tuple(int(length) for length in recv_lengths)
        if len(recv_lengths) != actual_size:
            raise ValueError(
                "Single-hop SWA receive schedule must have one entry per CP rank."
            )
        if recv_lengths[0] != 0 or any(
            isinstance(length, bool)
            or length < 0
            or length > local_sequence_length
            for length in recv_lengths
        ):
            raise ValueError(
                "Single-hop SWA receive lengths must be non-negative, fit in one "
                "local shard, and be zero on CP rank zero."
            )
        recv_length = recv_lengths[group_rank]
        send_length = (
            recv_lengths[group_rank + 1]
            if group_rank + 1 < actual_size
            else 0
        )
        boundary_query_length = int(overlap_plan.boundary_query_length)
        boundary_prefix_length = int(overlap_plan.boundary_prefix_length)
        if boundary_prefix_length != recv_length:
            raise RuntimeError(
                "True-overlap SWA boundary prefix and receive length disagree: "
                f"prefix={boundary_prefix_length}, recv={recv_length}."
            )
        local_kv = _fused_swa_halo_tail(
            key,
            value,
            halo_length=send_length,
        )
        history = query.new_empty(
            (recv_length, 1, num_heads, 2 * head_dim)
        )
        send_items = (
            ((global_ranks[group_rank + 1], local_kv),)
            if send_length > 0 and group_rank + 1 < actual_size
            else ()
        )
        recv_items = (
            ((global_ranks[group_rank - 1], history),)
            if recv_length > 0 and group_rank > 0
            else ()
        )
        requests = _begin_adjacent_halo_p2p(
            send_items=send_items,
            recv_items=recv_items,
            process_group=process_group,
            group_rank=group_rank,
        )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_FORWARD_ACTIVE",
            cp=actual_size,
            transport="adjacent_p2p",
        )

        independent_query, independent_key, independent_value = (
            _pack_swa_independent_tnd(
                query,
                key,
                value,
                overlap_plan,
            )
        )
        if int(independent_query.shape[0]) > 0:
            (
                independent_output,
                independent_softmax_max,
                independent_softmax_sum,
            ) = _run_raw_cann_tnd_attention(
                independent_query,
                independent_key,
                independent_value,
                num_heads=num_heads,
                softmax_scale=softmax_scale,
                contract=contract,
                actual_seq_qlen=overlap_plan.independent_query_endpoints,
                actual_seq_kvlen=overlap_plan.independent_kv_endpoints,
            )
        else:
            independent_output = independent_query
            independent_softmax_max = query.new_empty((0,))
            independent_softmax_sum = query.new_empty((0,))

        # Only the left-side receives are a dependency of this rank's boundary
        # attention. Keep outgoing K/V alive and in flight while CANN computes
        # that boundary block.
        requests.wait_receives()
        if int(history.shape[0]) != recv_length:
            raise RuntimeError(
                "True-overlap SWA contiguous receive buffer has the wrong length: "
                f"got={history.shape[0]}, expected={recv_length}."
            )
        halo_key, halo_value = history.split((head_dim, head_dim), dim=-1)
        boundary_query, boundary_key, boundary_value = _pack_swa_boundary_tnd(
            query,
            key,
            value,
            halo_key,
            halo_value,
            overlap_plan,
        )
        if boundary_query_length:
            boundary_kv_length = boundary_prefix_length + boundary_query_length
            boundary_output, boundary_softmax_max, boundary_softmax_sum = (
                _run_raw_cann_tnd_attention(
                    boundary_query,
                    boundary_key,
                    boundary_value,
                    num_heads=num_heads,
                    softmax_scale=softmax_scale,
                    contract=contract,
                    actual_seq_qlen=(boundary_query_length,),
                    actual_seq_kvlen=(boundary_kv_length,),
                )
            )
            output = torch.cat((boundary_output, independent_output), dim=0)
        else:
            empty = query.new_empty((0,))
            boundary_output = empty
            boundary_softmax_max = empty
            boundary_softmax_sum = empty
            output = independent_output
        if int(output.shape[0]) != local_sequence_length:
            raise RuntimeError(
                "True-overlap packed SWA returned the wrong query count: "
                f"output={output.shape[0]}, local={local_sequence_length}."
            )
        requests.wait_sends()

        ctx.save_for_backward(
            independent_query,
            independent_key,
            independent_value,
            independent_output,
            independent_softmax_max,
            independent_softmax_sum,
            boundary_query,
            boundary_key,
            boundary_value,
            boundary_output,
            boundary_softmax_max,
            boundary_softmax_sum,
        )
        ctx.process_group = process_group
        ctx.global_ranks = global_ranks
        ctx.group_rank = group_rank
        ctx.recv_length = int(recv_length)
        ctx.send_length = int(send_length)
        ctx.local_shape = tuple(int(size) for size in query.shape)
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.boundary_query_length = boundary_query_length
        ctx.boundary_prefix_length = boundary_prefix_length
        if overlap_plan.independent_segments:
            ctx.independent_query_start = int(
                overlap_plan.independent_segments[0][0]
            )
            ctx.independent_query_end = int(
                overlap_plan.independent_segments[-1][1]
            )
            ctx.independent_kv_start = int(
                overlap_plan.independent_segments[0][2]
            )
            ctx.independent_kv_end = int(
                overlap_plan.independent_segments[-1][3]
            )
        else:
            ctx.independent_query_start = local_sequence_length
            ctx.independent_query_end = local_sequence_length
            ctx.independent_kv_start = local_sequence_length
            ctx.independent_kv_end = local_sequence_length
        ctx.independent_query_endpoints = overlap_plan.independent_query_endpoints
        ctx.independent_kv_endpoints = overlap_plan.independent_kv_endpoints
        ctx.softmax_scale = float(softmax_scale)
        ctx.contract = contract
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            independent_query,
            independent_key,
            independent_value,
            independent_output,
            independent_softmax_max,
            independent_softmax_sum,
            boundary_query,
            boundary_key,
            boundary_value,
            boundary_output,
            boundary_softmax_max,
            boundary_softmax_sum,
        ) = ctx.saved_tensors
        if grad_output.ndim != 3 or int(grad_output.shape[0]) != ctx.local_shape[0]:
            raise RuntimeError(
                "True-overlap SWA backward received an invalid output gradient: "
                f"got={tuple(grad_output.shape)}, local={ctx.local_shape[0]}."
            )

        boundary_query_length = ctx.boundary_query_length
        boundary_prefix_length = ctx.boundary_prefix_length
        if (
            ctx.independent_query_start != boundary_query_length
            or ctx.independent_query_end != ctx.local_shape[0]
        ):
            raise RuntimeError(
                "True-overlap SWA independent queries are not a contiguous "
                "suffix of the local shard."
            )

        future_gradient = independent_query.new_empty(
            (
                ctx.send_length,
                1,
                ctx.num_heads,
                2 * ctx.head_dim,
            )
        )

        if boundary_query_length:
            boundary_dq, boundary_dk, boundary_dv = (
                _run_raw_cann_tnd_attention_backward(
                    boundary_query,
                    boundary_key,
                    boundary_value,
                    grad_output[:boundary_query_length].contiguous(),
                    boundary_output,
                    boundary_softmax_max,
                    boundary_softmax_sum,
                    num_heads=ctx.num_heads,
                    softmax_scale=ctx.softmax_scale,
                    contract=ctx.contract,
                    actual_seq_qlen=(boundary_query_length,),
                    actual_seq_kvlen=(
                        boundary_prefix_length + boundary_query_length,
                    ),
                )
            )
            halo_grad_send = torch.cat(
                (
                    boundary_dk[:boundary_prefix_length],
                    boundary_dv[:boundary_prefix_length],
                ),
                dim=-1,
            ).unsqueeze(1)
        else:
            boundary_dq = boundary_query
            boundary_dk = boundary_key
            boundary_dv = boundary_value
            halo_grad_send = independent_query.new_empty(
                (0, 1, ctx.num_heads, 2 * ctx.head_dim)
            )

        send_items = (
            ((ctx.global_ranks[ctx.group_rank - 1], halo_grad_send),)
            if ctx.recv_length > 0 and ctx.group_rank > 0
            else ()
        )
        recv_items = (
            ((ctx.global_ranks[ctx.group_rank + 1], future_gradient),)
            if ctx.send_length > 0
            and ctx.group_rank + 1 < len(ctx.global_ranks)
            else ()
        )
        requests = _begin_adjacent_halo_p2p(
            send_items=send_items,
            recv_items=recv_items,
            process_group=ctx.process_group,
            group_rank=ctx.group_rank,
        )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_BACKWARD_ACTIVE",
            cp=len(ctx.global_ranks),
            transport="reverse_adjacent_p2p",
        )

        # This CANN gradient is intentionally submitted after reverse P2P.
        # HCCL can transfer the boundary halo gradient while the independent
        # query block performs the much larger local attention backward.
        if int(independent_query.shape[0]) > 0:
            independent_dq, independent_dk, independent_dv = (
                _run_raw_cann_tnd_attention_backward(
                    independent_query,
                    independent_key,
                    independent_value,
                    grad_output[boundary_query_length:].contiguous(),
                    independent_output,
                    independent_softmax_max,
                    independent_softmax_sum,
                    num_heads=ctx.num_heads,
                    softmax_scale=ctx.softmax_scale,
                    contract=ctx.contract,
                    actual_seq_qlen=ctx.independent_query_endpoints,
                    actual_seq_kvlen=ctx.independent_kv_endpoints,
                )
            )
        else:
            independent_dq = independent_query
            independent_dk = independent_key
            independent_dv = independent_value

        if boundary_query_length:
            # Query blocks are disjoint and already in local token order.
            # Concatenating once avoids a zero-fill plus two scatter-add
            # kernels on every SWA layer.
            grad_query = torch.cat(
                (boundary_dq, independent_dq),
                dim=0,
            ).unsqueeze(1)
        else:
            grad_query = independent_dq.unsqueeze(1)

        if (
            ctx.independent_kv_start == 0
            and ctx.independent_kv_end == ctx.local_shape[0]
        ):
            # The common case already has one gradient for every local K/V;
            # reuse those fresh CANN outputs as the accumulation buffers.
            grad_key = independent_dk.unsqueeze(1)
            grad_value = independent_dv.unsqueeze(1)
        else:
            grad_key = independent_key.new_zeros(ctx.local_shape)
            grad_value = independent_value.new_zeros(ctx.local_shape)
            grad_key[
                ctx.independent_kv_start : ctx.independent_kv_end, 0
            ].copy_(independent_dk)
            grad_value[
                ctx.independent_kv_start : ctx.independent_kv_end, 0
            ].copy_(independent_dv)
        if boundary_query_length:
            grad_key[:boundary_query_length, 0].add_(
                boundary_dk[boundary_prefix_length:]
            )
            grad_value[:boundary_query_length, 0].add_(
                boundary_dv[boundary_prefix_length:]
            )

        # Future-consumer gradients are the only data dependency here. The
        # transposed gradients sent to source ranks may remain in flight while
        # we add those received gradients into the local source slices.
        requests.wait_receives()
        if ctx.send_length:
            remote_key_grad, remote_value_grad = future_gradient.split(
                (ctx.head_dim, ctx.head_dim),
                dim=-1,
            )
            source_start = ctx.local_shape[0] - ctx.send_length
            grad_key[source_start:].add_(remote_key_grad)
            grad_value[source_start:].add_(remote_value_grad)
        requests.wait_sends()
        return (
            grad_query,
            grad_key,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Olmo3DotProductAttention(torch.nn.Module):
    """Dispatch Full/SWA layers without globally changing every attention layer.

    Stock Transformer Engine receives its native inclusive ``window_size``.
    MindSpeed replaces ``TEDotProductAttention`` with a class that explicitly
    rejects that option, so local layers use Ascend's compressed band-sparse
    FlashAttention mode instead.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
    ) -> None:
        super().__init__()
        if attention_type != "self":
            raise ValueError("OLMo3 supports decoder self-attention only.")
        if attn_mask_type != AttnMaskType.causal:
            raise ValueError("OLMo3 supports causal attention only.")
        self.config = config
        self.layer_number = layer_number
        self.attention_type = attention_type
        self.layer_attention_type = _layer_attention_type(config, layer_number)
        self.sliding_window = int(getattr(config, "olmo3_sliding_window", 0))
        if self.sliding_window < 1:
            raise ValueError("OLMo3 runtime config is missing a positive sliding window.")
        self.is_sliding = self.layer_attention_type == SLIDING_ATTENTION
        self.mindspeed_band_sparse = _is_mindspeed_attention(
            TEDotProductAttention, config
        )
        _validate_swa_cp_mode(int(config.context_parallel_size))
        self.olmo3_swa_halo_overlap = _swa_halo_overlap_enabled()
        self.olmo3_swa_halo_backward_overlap = (
            _swa_halo_backward_overlap_enabled()
        )
        self.olmo3_fused_qkv_a2a_packing = (
            _fused_qkv_a2a_packing_enabled()
        )
        if (
            self.mindspeed_band_sparse
            and int(config.context_parallel_size) != 1
            and getattr(config, "context_parallel_algo", None) != "ulysses_cp_algo"
        ):
            raise ValueError(
                "OLMo3 MindSpeed band-sparse attention with CP>1 requires Ulysses CP."
            )

        inner_config = copy.copy(config)
        inner_config.window_size = None
        if self.is_sliding and not self.mindspeed_band_sparse:
            inner_config.window_size = olmo3_te_window(self.sliding_window)

        core_attention_kwargs: dict[str, Any] = dict(
            config=inner_config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            cp_comm_type=cp_comm_type,
        )
        if not self.mindspeed_band_sparse:
            # Transformer Engine supports asymmetric K/V channel overrides.
            # MindSpeed-LLM's replacement does not expose those parameters.
            core_attention_kwargs["k_channels"] = k_channels
            core_attention_kwargs["v_channels"] = v_channels
        self.core_attention = TEDotProductAttention(**core_attention_kwargs)
        self.attention_dropout = (
            float(config.attention_dropout)
            if attention_dropout is None
            else float(attention_dropout)
        )
        if softmax_scale is None:
            qk_channels: int | tuple[int, int] = (
                k_channels if k_channels is not None else config.kv_channels
            )
            if isinstance(qk_channels, tuple):
                qk_channels = qk_channels[0]
            softmax_scale = 1.0 / math.sqrt(qk_channels)
        self.softmax_scale = float(softmax_scale)
        # MCore keeps OLMo3 packed Q/K/V in SBHD until this module so that the
        # same adapter owns both the TND conversion and optional Ulysses A2A.
        # This is a runtime capability marker, not persistent model state.
        self.olmo3_keeps_packed_sbhd = self.mindspeed_band_sparse
        # MindSpeed-LLM wraps every CP attention module in its own
        # UlyssesContextAttention after construction.  The explicit OLMo3 CANN
        # adapter already owns that all-to-all so it can keep SWA CP-local and
        # convert Full Attention to document-aware TND exactly once.  The
        # OLMo3 SelfAttention constructors use this marker to remove only that
        # redundant, parameter-free outer owner.  This adds no checkpoint
        # state.
        self.olmo3_owns_ulysses = self.mindspeed_band_sparse

    @property
    def window_size(self) -> tuple[int, int]:
        """Expose the effective inclusive window for runtime introspection."""
        return olmo3_te_window(self.sliding_window) if self.is_sliding else (-1, 0)

    def _mindspeed_contract(self, *, max_kv_length: int) -> Olmo3CannAttentionContract:
        if self.is_sliding:
            return Olmo3CannAttentionContract(
                # CANN band mode counts tokens preceding the current token;
                # next_tokens=0 additionally includes the current token.
                # Match OLMo/TE's inclusive W-token window with W - 1.
                pre_tokens=self.sliding_window - 1,
                next_tokens=0,
                sparse_mode=4,
            )
        return Olmo3CannAttentionContract(
            pre_tokens=max(1, int(max_kv_length)),
            next_tokens=0,
            sparse_mode=2,
        )

    def _prepare_ulysses_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Any, int] | None]:
        """Reconstruct the global sequence while sharding heads across CP."""

        configured_cp_size = int(self.config.context_parallel_size)
        if configured_cp_size == 1:
            return query, key, value, None
        if getattr(self.config, "context_parallel_algo", None) != "ulysses_cp_algo":
            raise ValueError("OLMo3 CP>1 requires Ulysses context parallelism.")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("OLMo3 Ulysses requires initialized torch.distributed.")
        process_group = parallel_state.get_context_parallel_group()
        actual_cp_size = torch.distributed.get_world_size(process_group)
        if actual_cp_size != configured_cp_size:
            raise RuntimeError(
                "OLMo3 context-parallel group size disagrees with the model config: "
                f"group={actual_cp_size}, config={configured_cp_size}."
            )
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.ndim != 4:
                raise ValueError(f"Ulysses {name} must be SBHD, got {tuple(tensor.shape)}.")
            if tensor.shape[2] % actual_cp_size:
                raise ValueError(
                    f"Ulysses {name} heads ({tensor.shape[2]}) must be divisible "
                    f"by CP ({actual_cp_size})."
                )

        original_query_heads = int(query.shape[2])
        if query.shape == key.shape == value.shape:
            # OLMo3 uses MHA. The optimized path writes Q/K/V directly into
            # the head-scattered send layout, eliminating the otherwise
            # redundant full-size cat buffer plus transpose copy.
            if self.olmo3_fused_qkv_a2a_packing:
                fused_qkv = _fused_qkv_ulysses_all_to_all(
                    query,
                    key,
                    value,
                    process_group,
                )
                emit_rank0_runtime_marker_once(
                    "OLMO3_RUNTIME_FULL_FUSED_QKV_A2A_ACTIVE",
                    cp=actual_cp_size,
                    layer=self.layer_number,
                )
            else:
                fused_qkv = torch.cat((query, key, value), dim=-1)
                fused_qkv = _ulysses_all_to_all(
                    fused_qkv,
                    process_group,
                    2,
                    0,
                )
            query, key, value = fused_qkv.chunk(3, dim=-1)
        else:
            # Preserve correctness for a future GQA configuration whose Q and
            # K/V head counts cannot share the MHA packing layout.
            query = _ulysses_all_to_all(query, process_group, 2, 0)
            key = _ulysses_all_to_all(key, process_group, 2, 0)
            value = _ulysses_all_to_all(value, process_group, 2, 0)
        return query, key, value, (process_group, original_query_heads)

    @staticmethod
    def _restore_ulysses_output(
        output: Tensor,
        restore_context: tuple[Any, int] | None,
    ) -> Tensor:
        if restore_context is None:
            return output
        process_group, original_query_heads = restore_context
        return _ulysses_all_to_all(
            output,
            process_group,
            0,
            2,
            original_query_heads,
        )

    def _run_mindspeed_cann_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        num_heads: int,
        layout: str,
        contract: Olmo3CannAttentionContract,
        actual_seq_qlen: list[int] | None = None,
        actual_seq_kvlen: list[int] | None = None,
    ) -> Tensor:
        """Invoke the final NPU operator with no wrapper-inferred semantics."""

        import torch_npu

        kwargs: dict[str, Any] = {}
        if layout == "TND":
            if actual_seq_qlen is None or actual_seq_kvlen is None:
                raise ValueError("TND attention requires explicit cumulative endpoints.")
            kwargs.update(
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
        elif layout != "SBH":
            raise ValueError(f"Unsupported OLMo3 CANN layout {layout!r}.")

        return torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            num_heads,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=_compressed_causal_mask(query.device),
            scale=self.softmax_scale,
            pre_tockens=contract.pre_tokens,
            next_tockens=contract.next_tokens,
            keep_prob=(1.0 - self.attention_dropout if self.training else 1.0),
            inner_precise=0,
            sparse_mode=contract.sparse_mode,
            **kwargs,
        )[0]

    def _mindspeed_unpacked_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Fixed-length SBHD adapter used without any stage/length special case."""

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Unpacked OLMo3 attention expects SBHD Q/K/V; got "
                f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        if (
            self.is_sliding
            and int(self.config.context_parallel_size) > 1
        ):
            return self._mindspeed_unpacked_swa_halo_forward(
                query,
                key,
                value,
            )
        query, key, value, restore_context = self._prepare_ulysses_qkv(
            query, key, value
        )
        sequence_length, batch_size, num_heads, _ = query.shape
        if key.shape[0] != sequence_length or value.shape[0] != sequence_length:
            raise ValueError("Unpacked OLMo3 training requires equal Q/K/V lengths.")
        query_sbh = query.contiguous().reshape(sequence_length, batch_size, -1)
        key_sbh = key.contiguous().reshape(sequence_length, batch_size, -1)
        value_sbh = value.contiguous().reshape(sequence_length, batch_size, -1)
        output = self._run_mindspeed_cann_attention(
            query_sbh,
            key_sbh,
            value_sbh,
            num_heads=num_heads,
            layout="SBH",
            contract=self._mindspeed_contract(max_kv_length=sequence_length),
        )
        return self._restore_ulysses_output(output, restore_context)

    def _mindspeed_unpacked_swa_halo_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Run fixed-length SWA with local Q and one adjacent K/V halo.

        A causal W-token sliding window needs at most W-1 tokens preceding the
        local shard. The production topology guarantees each CP-local shard is
        at least W-1 tokens, so exactly one adjacent exchange covers that
        history; Q and the output stay local. Full-attention layers continue
        to use fused-QKV Ulysses.

        The TND contract gives every batch row independent Q/KV endpoints and
        correctly right-aligns the local queries after their optional prefix.
        It also keeps this path valid for micro-batches larger than one.
        """

        if query.shape != key.shape or key.shape != value.shape:
            raise ValueError(
                "Unpacked OLMo3 SWA halo requires matching MHA Q/K/V shapes; "
                f"got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        configured_cp_size = int(self.config.context_parallel_size)
        if configured_cp_size <= 1:
            raise ValueError("Unpacked SWA halo is only valid when CP is larger than one.")
        if getattr(self.config, "context_parallel_algo", None) != "ulysses_cp_algo":
            raise ValueError("Unpacked OLMo3 SWA halo requires Ulysses CP batch sharding.")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Unpacked OLMo3 SWA halo requires initialized distributed.")

        process_group = parallel_state.get_context_parallel_group()
        actual_cp_size = torch.distributed.get_world_size(process_group)
        if actual_cp_size != configured_cp_size:
            raise RuntimeError(
                "OLMo3 SWA halo group size disagrees with the model config: "
                f"group={actual_cp_size}, config={configured_cp_size}."
            )

        local_sequence_length = int(query.shape[0])
        batch_size = int(query.shape[1])
        halo_length = self.sliding_window - 1
        _require_single_hop_halo_capacity(
            local_sequence_length=local_sequence_length,
            halo_length=halo_length,
        )
        global_ranks_value = parallel_state.get_context_parallel_global_ranks()
        if global_ranks_value is None:
            raise RuntimeError("OLMo3 SWA halo is missing CP global ranks.")
        global_ranks = tuple(int(rank) for rank in global_ranks_value)
        if len(global_ranks) != actual_cp_size:
            raise RuntimeError(
                "OLMo3 SWA halo global-rank count disagrees with CP: "
                f"ranks={len(global_ranks)}, CP={actual_cp_size}."
            )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_CP_SINGLE_HALO_ACTIVE",
            cp=actual_cp_size,
            packed=0,
            window=self.sliding_window,
        )

        local_kv = _fused_swa_halo_tail(
            key,
            value,
            halo_length=halo_length,
        )
        halo = _single_hop_left_halo_exchange(
            local_kv,
            process_group,
            global_ranks,
            halo_length=halo_length,
        )
        halo_key, halo_value = halo.history.split(
            (int(key.shape[-1]), int(value.shape[-1])),
            dim=-1,
        )
        history_length = int(halo_key.shape[0])

        query_tnd = flatten_packed_sbhd_to_tnd(query)
        if history_length == 0:
            key_tnd = flatten_packed_sbhd_to_tnd(key)
            value_tnd = flatten_packed_sbhd_to_tnd(value)
            kv_length_per_sample = local_sequence_length
        else:
            key_tnd = torch.cat(
                [
                    torch.cat((halo_key[:, batch_index], key[:, batch_index]), dim=0)
                    for batch_index in range(batch_size)
                ],
                dim=0,
            )
            value_tnd = torch.cat(
                [
                    torch.cat(
                        (halo_value[:, batch_index], value[:, batch_index]),
                        dim=0,
                    )
                    for batch_index in range(batch_size)
                ],
                dim=0,
            )
            kv_length_per_sample = history_length + local_sequence_length

        query_endpoints = [
            local_sequence_length * (batch_index + 1)
            for batch_index in range(batch_size)
        ]
        kv_endpoints = [
            kv_length_per_sample * (batch_index + 1)
            for batch_index in range(batch_size)
        ]
        output = self._run_mindspeed_cann_attention(
            query_tnd,
            key_tnd,
            value_tnd,
            num_heads=int(query_tnd.shape[1]),
            layout="TND",
            contract=self._mindspeed_contract(
                max_kv_length=kv_length_per_sample,
            ),
            actual_seq_qlen=query_endpoints,
            actual_seq_kvlen=kv_endpoints,
        )
        output = restore_packed_tnd_to_sbh(
            output,
            sequence_length=local_sequence_length,
            batch_size=batch_size,
        )
        # Rank zero consumes no history but still retains the exchange in its
        # graph so its K/V receive gradients from rank one.
        return output + halo.autograd_anchor.to(output.dtype)

    def _mindspeed_packed_swa_overlapped_one_hop_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: PackedSeqParams,
        plan: Olmo3PackedSwaHaloPlan,
        *,
        process_group: Any,
        global_ranks: tuple[int, ...],
        local_sequence_length: int,
        halo_length: int,
        send_length: int,
        recv_length: int,
        group_recv_lengths: tuple[int, ...],
        overlap_plan: Olmo3PackedSwaOverlapPlan | None = None,
    ) -> Tensor:
        """Hide a ragged one-hop halo behind exact CP-local packed SWA."""

        if overlap_plan is None:
            overlap_plan = _packed_swa_overlap_plan(
                packed_seq_params,
                plan,
                local_sequence_length=local_sequence_length,
                halo_length=halo_length,
            )
        if (
            self.olmo3_swa_halo_backward_overlap
            and self.training
            and self.attention_dropout == 0.0
        ):
            output_tnd = _PackedSwaOneHopOverlap.apply(
                query,
                key,
                value,
                process_group,
                global_ranks,
                group_recv_lengths,
                overlap_plan,
                self.softmax_scale,
                self._mindspeed_contract(
                    max_kv_length=max(
                        overlap_plan.independent_max_kv_length,
                        overlap_plan.boundary_prefix_length
                        + overlap_plan.boundary_query_length,
                    )
                ),
            )
            return restore_packed_tnd_to_sbh(
                output_tnd,
                sequence_length=local_sequence_length,
                batch_size=1,
            )
        local_kv = _fused_swa_halo_tail(
            key,
            value,
            halo_length=send_length,
        )
        pending_halo = _begin_variable_left_halo_exchange(
            local_kv,
            recv_length=recv_length,
            process_group=process_group,
            global_ranks=global_ranks,
        )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_FORWARD_ACTIVE",
            cp=len(global_ranks),
            transport="adjacent_p2p",
        )

        independent_query, independent_key, independent_value = (
            _pack_swa_independent_tnd(
                query,
                key,
                value,
                overlap_plan,
            )
        )
        if independent_query.shape[0] > 0:
            independent_output = self._run_mindspeed_cann_attention(
                independent_query,
                independent_key,
                independent_value,
                num_heads=int(independent_query.shape[1]),
                layout="TND",
                contract=self._mindspeed_contract(
                    max_kv_length=overlap_plan.independent_max_kv_length,
                ),
                actual_seq_qlen=list(
                    overlap_plan.independent_query_endpoints
                ),
                actual_seq_kvlen=list(
                    overlap_plan.independent_kv_endpoints
                ),
            )
        else:
            independent_output = independent_query

        history = pending_halo.wait()
        halo_key, halo_value = history.split(
            (int(key.shape[-1]), int(value.shape[-1])),
            dim=-1,
        )
        boundary_query, boundary_key, boundary_value = _pack_swa_boundary_tnd(
            query,
            key,
            value,
            halo_key,
            halo_value,
            overlap_plan,
        )
        if boundary_query.shape[0] > 0:
            boundary_kv_length = (
                overlap_plan.boundary_prefix_length
                + overlap_plan.boundary_query_length
            )
            boundary_output = self._run_mindspeed_cann_attention(
                boundary_query,
                boundary_key,
                boundary_value,
                num_heads=int(boundary_query.shape[1]),
                layout="TND",
                contract=self._mindspeed_contract(
                    max_kv_length=boundary_kv_length,
                ),
                actual_seq_qlen=[
                    overlap_plan.boundary_query_length
                ],
                actual_seq_kvlen=[boundary_kv_length],
            )
            output_tnd = torch.cat(
                (boundary_output, independent_output),
                dim=0,
            )
        else:
            output_tnd = independent_output
        if int(output_tnd.shape[0]) != local_sequence_length:
            raise RuntimeError(
                "Overlapped packed SWA returned the wrong query count: "
                f"output={output_tnd.shape[0]}, local={local_sequence_length}."
            )
        output = restore_packed_tnd_to_sbh(
            output_tnd,
            sequence_length=local_sequence_length,
            batch_size=1,
        )
        # The zero anchor makes even a rank with no received prefix execute
        # the exact reverse P2P and collect gradients consumed by its right
        # neighbour.
        return output + (history.sum() * 0.0).to(output.dtype)

    def _mindspeed_packed_swa_halo_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tensor:
        """Run packed SWA with an exact document-safe adjacent K/V halo."""

        configured_cp_size = int(self.config.context_parallel_size)
        if configured_cp_size <= 1:
            raise ValueError("Packed SWA halo is only valid when CP is larger than one.")
        if getattr(self.config, "context_parallel_algo", None) != "ulysses_cp_algo":
            raise ValueError("Packed OLMo3 SWA halo requires Ulysses CP batch sharding.")
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Packed OLMo3 SWA halo requires initialized distributed.")
        process_group = parallel_state.get_context_parallel_group()
        actual_cp_size = torch.distributed.get_world_size(process_group)
        if actual_cp_size != configured_cp_size:
            raise RuntimeError(
                "OLMo3 SWA halo group size disagrees with the model config: "
                f"group={actual_cp_size}, config={configured_cp_size}."
            )

        local_sequence_length = int(query.shape[0])
        halo_length = self.sliding_window - 1
        _require_single_hop_halo_capacity(
            local_sequence_length=local_sequence_length,
            halo_length=halo_length,
        )
        cp_rank = torch.distributed.get_rank(process_group)
        global_ranks_value = parallel_state.get_context_parallel_global_ranks()
        if global_ranks_value is None:
            raise RuntimeError("OLMo3 SWA halo is missing CP global ranks.")
        global_ranks = tuple(int(rank) for rank in global_ranks_value)
        if len(global_ranks) != actual_cp_size:
            raise RuntimeError(
                "OLMo3 SWA halo global-rank count disagrees with CP: "
                f"ranks={len(global_ranks)}, CP={actual_cp_size}."
            )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_CP_SINGLE_HALO_ACTIVE",
            cp=actual_cp_size,
            packed=1,
            window=self.sliding_window,
        )

        plan = _packed_swa_halo_plan(
            packed_seq_params,
            cp_rank=cp_rank,
            cp_size=actual_cp_size,
            local_sequence_length=local_sequence_length,
            sliding_window=self.sliding_window,
        )
        group_recv_lengths = _packed_swa_group_recv_lengths(
            packed_seq_params,
            cp_size=actual_cp_size,
            local_sequence_length=local_sequence_length,
            sliding_window=self.sliding_window,
        )
        recv_length = group_recv_lengths[cp_rank]
        send_length = (
            group_recv_lengths[cp_rank + 1]
            if cp_rank + 1 < actual_cp_size
            else 0
        )
        # Use the canonical global document endpoints to make the adjacent edge
        # ragged: receive only this rank's same-document prefix and send only
        # the suffix that the next rank will consume.
        if self.olmo3_swa_halo_overlap and int(query.shape[1]) == 1:
            if _packed_swa_group_overlap_is_worthwhile(
                packed_seq_params,
                cp_size=actual_cp_size,
                local_sequence_length=local_sequence_length,
                sliding_window=self.sliding_window,
            ):
                overlap_plan = _packed_swa_overlap_plan(
                    packed_seq_params,
                    plan,
                    local_sequence_length=local_sequence_length,
                    halo_length=halo_length,
                )
                return self._mindspeed_packed_swa_overlapped_one_hop_forward(
                    query,
                    key,
                    value,
                    packed_seq_params,
                    plan,
                    process_group=process_group,
                    global_ranks=global_ranks,
                    local_sequence_length=local_sequence_length,
                    halo_length=halo_length,
                    send_length=send_length,
                    recv_length=recv_length,
                    group_recv_lengths=group_recv_lengths,
                    overlap_plan=overlap_plan,
                )
        local_kv = _fused_swa_halo_tail(
            key,
            value,
            halo_length=send_length,
        )
        history = _variable_left_halo_exchange(
            local_kv,
            recv_length=recv_length,
            process_group=process_group,
            global_ranks=global_ranks,
        )
        halo = Olmo3SwaHalo(
            history=history,
            autograd_anchor=history.sum() * 0.0,
        )
        halo_key, halo_value = halo.history.split(
            (int(key.shape[-1]), int(value.shape[-1])),
            dim=-1,
        )
        query_tnd, key_tnd, value_tnd = _pack_swa_halo_tnd(
            query,
            key,
            value,
            halo_key,
            halo_value,
            plan,
        )
        output = self._run_mindspeed_cann_attention(
            query_tnd,
            key_tnd,
            value_tnd,
            num_heads=int(query_tnd.shape[1]),
            layout="TND",
            contract=self._mindspeed_contract(
                max_kv_length=plan.max_kv_length,
            ),
            actual_seq_qlen=list(plan.query_endpoints),
            actual_seq_kvlen=list(plan.kv_endpoints),
        )
        output = restore_packed_tnd_to_sbh(
            output,
            sequence_length=local_sequence_length,
            batch_size=int(query.shape[1]),
        )
        # Keep the adjacent exchange in every rank's graph so transposed
        # backward returns remote K/V gradients without a boundary deadlock.
        return output + halo.autograd_anchor.to(output.dtype)

    def _mindspeed_packed_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tensor:
        """Packed TND adapter with document boundaries and optional Ulysses."""

        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Packed OLMo3 requires qkv_format='thd'.")
        if query.ndim == 3 and key.ndim == 3 and value.ndim == 3:
            metadata_batch_size = int(
                getattr(packed_seq_params, "olmo3_micro_batch_size", 1)
            )
            if metadata_batch_size != 1:
                raise ValueError(
                    "Batched packed OLMo3 attention must retain SBHD Q/K/V "
                    f"until TND conversion; metadata batch={metadata_batch_size}."
                )
            query, key, value = (
                query.unsqueeze(1),
                key.unsqueeze(1),
                value.unsqueeze(1),
            )
        elif query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Packed OLMo3 attention expects THD or SBHD Q/K/V; got "
                f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
            raise ValueError(
                "Packed OLMo3 Q/K/V must share sequence and batch dimensions; "
                f"got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
            raise ValueError(
                "Packed OLMo3 Q/K/V must share head_dim; "
                f"got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        metadata_batch_size = int(
            getattr(packed_seq_params, "olmo3_micro_batch_size", query.shape[1])
        )
        if metadata_batch_size != query.shape[1]:
            raise ValueError(
                "Packed OLMo3 metadata and Q/K/V batch sizes disagree: "
                f"metadata={metadata_batch_size}, tensors={query.shape[1]}."
            )

        if self.is_sliding and int(self.config.context_parallel_size) > 1:
            global_tokens_per_sample = int(
                getattr(
                    packed_seq_params,
                    "olmo3_global_tokens_per_sample",
                    0,
                )
            )
            if global_tokens_per_sample < 1:
                raise ValueError(
                    "Packed SWA with CP>1 requires global document-boundary "
                    "metadata; falling back to full-sequence Ulysses is disabled."
                )
            return self._mindspeed_packed_swa_halo_forward(
                query,
                key,
                value,
                packed_seq_params,
            )

        query, key, value, restore_context = self._prepare_ulysses_qkv(
            query, key, value
        )
        sequence_length = int(query.shape[0])
        batch_size = int(query.shape[1])
        total_query_tokens = sequence_length * batch_size
        total_kv_tokens = int(key.shape[0]) * int(key.shape[1])
        total_value_tokens = int(value.shape[0]) * int(value.shape[1])
        if (
            total_query_tokens != total_kv_tokens
            or total_value_tokens != total_kv_tokens
        ):
            raise ValueError(
                "Packed OLMo3 self-attention requires equal Q/K/V token counts."
            )
        query_tnd = flatten_packed_sbhd_to_tnd(query)
        key_tnd = flatten_packed_sbhd_to_tnd(key)
        value_tnd = flatten_packed_sbhd_to_tnd(value)
        actual_seq_qlen = _cann_cumulative_endpoints(
            packed_seq_params,
            "cu_seqlens_q",
            total_tokens=total_query_tokens,
        )
        actual_seq_kvlen = _cann_cumulative_endpoints(
            packed_seq_params,
            "cu_seqlens_kv",
            total_tokens=total_kv_tokens,
        )
        max_kv_length = int(getattr(packed_seq_params, "max_seqlen_kv", 0) or 0)
        if max_kv_length < 1:
            starts = [0, *actual_seq_kvlen[:-1]]
            max_kv_length = max(
                end - start for start, end in zip(starts, actual_seq_kvlen)
            )
        output = self._run_mindspeed_cann_attention(
            query_tnd,
            key_tnd,
            value_tnd,
            num_heads=int(query_tnd.shape[1]),
            layout="TND",
            contract=self._mindspeed_contract(max_kv_length=max_kv_length),
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )
        output = restore_packed_tnd_to_sbh(
            output,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        return self._restore_ulysses_output(output, restore_context)

    def _mindspeed_explicit_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attention_bias: Tensor | None,
        packed_seq_params: PackedSeqParams | None,
    ) -> Tensor:
        """Dispatch only on whether document packing is active."""

        if attention_bias is not None:
            raise ValueError("OLMo3 MindSpeed attention does not support an attention bias.")
        if attention_mask is not None:
            raise ValueError(
                "OLMo3 MindSpeed attention requires a null dataloader mask; "
                "the explicit CANN contract supplies causal/document masking."
            )
        if key.shape != value.shape:
            raise ValueError("OLMo3 MindSpeed attention requires matching K/V shapes.")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("OLMo3 MindSpeed attention requires equal Q/K head_dim.")
        if packed_seq_params is None:
            return self._mindspeed_unpacked_forward(query, key, value)
        return self._mindspeed_packed_forward(query, key, value, packed_seq_params)

    def _mindspeed_cached_swa_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attention_bias: Tensor | None,
        packed_seq_params: PackedSeqParams | None,
    ) -> Tensor:
        """Run right-aligned SWA when cached K/V is longer than the query."""
        if attention_bias is not None:
            raise ValueError("OLMo3 MindSpeed SWA does not support an attention bias.")
        if packed_seq_params is not None:
            raise NotImplementedError(
                "OLMo3 MindSpeed cached SWA does not support packed sequences."
            )
        if attention_mask is not None:
            raise NotImplementedError(
                "OLMo3 MindSpeed cached SWA currently requires a null attention mask. "
                "Use equal-length, unpadded static batches."
            )
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "OLMo3 MindSpeed cached SWA expects [sequence, batch, heads, head_dim]."
            )
        if key.shape != value.shape:
            raise ValueError("OLMo3 MindSpeed cached SWA requires matching key/value shapes.")
        if query.shape[0] >= key.shape[0]:
            raise ValueError("Cached SWA requires key_length > query_length.")
        if query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "OLMo3 MindSpeed cached SWA requires matching batch and head_dim."
            )
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError(
                "OLMo3 MindSpeed cached SWA requires query heads divisible by KV heads."
            )

        # The earliest current query can see W tokens. Discard every older
        # cached K/V row before invoking dense FA, so work remains bounded by
        # W + query_length - 1 instead of the total generated length.
        query_length = query.shape[0]
        key_length = key.shape[0]
        cache_start = max(
            0,
            key_length - query_length - self.sliding_window + 1,
        )
        key = key[cache_start:]
        value = value[cache_start:]
        effective_key_length = key.shape[0]
        sliding_mask = build_olmo3_sliding_causal_mask(
            query_length,
            effective_key_length,
            self.sliding_window,
            device=query.device,
        )
        if not bool(sliding_mask.any()):
            sliding_mask = None

        import torch_npu

        batch_size = query.shape[1]
        query_sbh = query.contiguous().reshape(query_length, batch_size, -1)
        key_sbh = key.contiguous().reshape(effective_key_length, batch_size, -1)
        value_sbh = value.contiguous().reshape(effective_key_length, batch_size, -1)
        return torch_npu.npu_fusion_attention(
            query_sbh,
            key_sbh,
            value_sbh,
            query.shape[2],
            "SBH",
            pse=None,
            padding_mask=None,
            atten_mask=sliding_mask,
            scale=self.softmax_scale,
            pre_tockens=effective_key_length,
            next_tockens=0,
            keep_prob=(
                1.0 - self.attention_dropout if self.training else 1.0
            ),
            inner_precise=0,
            sparse_mode=0,
        )[0]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run one Full or SWA attention layer."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected OLMo3 core-attention arguments: {unexpected}.")
        if attn_mask_type != AttnMaskType.causal:
            if not (
                attn_mask_type == AttnMaskType.no_mask
                and key.shape[0] > query.shape[0]
            ):
                raise ValueError("OLMo3 supports causal attention only.")
        if self.mindspeed_band_sparse and key.shape[0] > query.shape[0]:
            if not self.is_sliding:
                return self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            return self._mindspeed_cached_swa_forward(
                query,
                key,
                value,
                attention_mask,
                attention_bias,
                packed_seq_params,
            )
        if self.mindspeed_band_sparse:
            return self._mindspeed_explicit_forward(
                query,
                key,
                value,
                attention_mask,
                attention_bias,
                packed_seq_params,
            )
        return self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
