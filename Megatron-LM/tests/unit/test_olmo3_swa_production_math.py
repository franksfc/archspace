"""CPU mathematical tests for the production single-hop SWA/CP path."""

from __future__ import annotations

import socket
import sys
import types

import pytest


torch = pytest.importorskip("torch")


# CPU test images do not install Transformer Engine. The OLMo3 module only
# needs this symbol at import time; none of these tests instantiate the backend.
_TE_MODULE = "megatron.core.extensions.transformer_engine"
_inserted_te_stub = False
try:
    __import__(_TE_MODULE)
except ModuleNotFoundError:
    te_module = types.ModuleType(_TE_MODULE)
    te_module.TEDotProductAttention = type("TEDotProductAttention", (), {})
    sys.modules[_TE_MODULE] = te_module
    _inserted_te_stub = True

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from modeling import olmo3_swa  # noqa: E402

if _inserted_te_stub:
    del sys.modules[_TE_MODULE]


def _packed(endpoints: list[int], global_tokens: int) -> PackedSeqParams:
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(endpoints, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(endpoints, dtype=torch.int32),
    )
    packed.olmo3_micro_batch_size = 1
    packed.olmo3_global_tokens_per_sample = global_tokens
    return packed


def test_single_hop_capacity_is_exact_and_fails_closed() -> None:
    olmo3_swa._require_single_hop_halo_capacity(
        local_sequence_length=4095,
        halo_length=4095,
    )
    with pytest.raises(ValueError, match="single adjacent"):
        olmo3_swa._require_single_hop_halo_capacity(
            local_sequence_length=4094,
            halo_length=4095,
        )
    with pytest.raises(ValueError, match="single adjacent"):
        olmo3_swa._packed_swa_halo_plan(
            _packed([0, 8], global_tokens=8),
            cp_rank=0,
            cp_size=4,
            local_sequence_length=2,
            sliding_window=4,
        )


def test_non_production_swa_cp_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLMO3_SWA_CP_MODE", raising=False)
    olmo3_swa._validate_swa_cp_mode(16)
    monkeypatch.setenv("OLMO3_SWA_CP_MODE", "single_halo")
    olmo3_swa._validate_swa_cp_mode(16)
    monkeypatch.setenv("OLMO3_SWA_CP_MODE", "none")
    olmo3_swa._validate_swa_cp_mode(1)
    with pytest.raises(ValueError, match="production single halo"):
        olmo3_swa._validate_swa_cp_mode(16)
    monkeypatch.setenv("OLMO3_SWA_CP_MODE", "ulysses")
    with pytest.raises(ValueError, match="production single halo"):
        olmo3_swa._validate_swa_cp_mode(16)


def test_external_ulysses_owner_is_removed_only_for_an_olmo3_owned_core() -> None:
    """SWA must observe CP-local sequence, not an outer Ulysses gather."""

    class OwnedOlmo3Core(torch.nn.Module):
        olmo3_owns_ulysses = True

        def __init__(self) -> None:
            super().__init__()
            # This is attached by MindSpeed's UlyssesContextAttention.
            self.ulysses_comm_para = {"spg": object()}

        def forward(self, query, key, value):
            del key, value
            return query

    class UlyssesContextAttention(torch.nn.Module):
        def __init__(self, local_attn: torch.nn.Module) -> None:
            super().__init__()
            self.local_attn = local_attn

        def forward(self, query, key, value):
            # Model the faulty upstream ordering: CP=16 gathers S=4096 into
            # global S=65536 before the OLMo3 SWA core runs.
            query, key, value = (
                tensor.repeat(16, 1, 1, 1)
                for tensor in (query, key, value)
            )
            return self.local_attn(query, key, value)

    UlyssesContextAttention.__module__ = (
        "mindspeed.core.context_parallel.ulysses_context_parallel"
    )
    local = OwnedOlmo3Core()
    external = UlyssesContextAttention(local)
    local_qkv = torch.zeros(4, 1, 1, 1)
    assert external(local_qkv, local_qkv, local_qkv).shape[0] == 64

    owner = olmo3_swa.unwrap_external_ulysses_for_olmo3(external)
    assert owner is local
    assert owner(local_qkv, local_qkv, local_qkv).shape[0] == 4
    assert owner.olmo3_external_ulysses_unwrapped is True
    # A direct OLMo3 core and an unrelated wrapper remain untouched.
    assert olmo3_swa.unwrap_external_ulysses_for_olmo3(local) is local
    unrelated = UlyssesContextAttention(torch.nn.Identity())
    assert olmo3_swa.unwrap_external_ulysses_for_olmo3(unrelated) is unrelated


def test_external_ulysses_unwrap_fails_closed_on_unknown_wrapper() -> None:
    class OwnedOlmo3Core(torch.nn.Module):
        olmo3_owns_ulysses = True

        def __init__(self) -> None:
            super().__init__()
            self.ulysses_comm_para = {"spg": object()}

    class UlyssesContextAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local_attn = OwnedOlmo3Core()

    # Matching a class name is insufficient: only the pinned MindSpeed
    # implementation may be removed.
    assert not UlyssesContextAttention.__module__.startswith(
        ("mindspeed.", "mindspeed_llm.")
    )
    with pytest.raises(RuntimeError, match="unknown outer attention wrapper"):
        olmo3_swa.unwrap_external_ulysses_for_olmo3(
            UlyssesContextAttention()
        )


def test_external_ulysses_unwrap_preserves_checkpoint_keyspace() -> None:
    class StatefulOwnedOlmo3Core(torch.nn.Module):
        olmo3_owns_ulysses = True

        def __init__(self) -> None:
            super().__init__()
            self.ulysses_comm_para = {"spg": object()}
            self.register_buffer(
                "persistent_contract",
                torch.ones(1),
                persistent=True,
            )

    class UlyssesContextAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local_attn = StatefulOwnedOlmo3Core()

    UlyssesContextAttention.__module__ = "mindspeed_llm.core.context_parallel"
    wrapped = UlyssesContextAttention()
    assert tuple(wrapped.state_dict()) == (
        "local_attn.persistent_contract",
    )
    with pytest.raises(RuntimeError, match="change its keyspace"):
        olmo3_swa.unwrap_external_ulysses_for_olmo3(wrapped)
    # Fail-closed means the caller still owns the unchanged wrapper/keyspace.
    assert tuple(wrapped.state_dict()) == (
        "local_attn.persistent_contract",
    )


def test_packed_plan_never_prepends_halo_across_document_boundary() -> None:
    # Global documents are [0, 6) and [6, 8). CP rank one owns [4, 8):
    # only its first two local queries may consume the last three remote tokens.
    packed = _packed([0, 6, 8], global_tokens=8)
    plan = olmo3_swa._packed_swa_halo_plan(
        packed,
        cp_rank=1,
        cp_size=2,
        local_sequence_length=4,
        sliding_window=4,
    )
    assert plan.segments == ((0, 0, 2, 3), (0, 2, 4, 0))
    assert plan.query_endpoints == (2, 4)
    assert plan.kv_endpoints == (5, 7)

    query = torch.arange(100, 104, dtype=torch.float32).reshape(4, 1, 1, 1)
    key = torch.arange(10, 14, dtype=torch.float32).reshape(4, 1, 1, 1)
    value = torch.arange(20, 24, dtype=torch.float32).reshape(4, 1, 1, 1)
    halo_key = torch.tensor([1, 2, 3], dtype=torch.float32).reshape(3, 1, 1, 1)
    halo_value = torch.tensor([4, 5, 6], dtype=torch.float32).reshape(3, 1, 1, 1)
    query_tnd, key_tnd, value_tnd = olmo3_swa._pack_swa_halo_tnd(
        query,
        key,
        value,
        halo_key,
        halo_value,
        plan,
    )

    assert query_tnd[:, 0, 0].tolist() == [100, 101, 102, 103]
    # The second document starts with local token 12 and receives no old halo.
    assert key_tnd[:, 0, 0].tolist() == [1, 2, 3, 10, 11, 12, 13]
    assert value_tnd[:, 0, 0].tolist() == [4, 5, 6, 20, 21, 22, 23]


def test_overlap_split_covers_each_query_once_and_preserves_local_history() -> None:
    packed = _packed([0, 8], global_tokens=8)
    plan = olmo3_swa._packed_swa_halo_plan(
        packed,
        cp_rank=1,
        cp_size=2,
        local_sequence_length=4,
        sliding_window=4,
    )
    overlap = olmo3_swa._packed_swa_overlap_plan(
        packed,
        plan,
        local_sequence_length=4,
        halo_length=3,
    )
    assert overlap.boundary_query_length == 3
    assert overlap.boundary_prefix_length == 3
    assert overlap.independent_segments == ((3, 4, 0, 4),)
    assert overlap.independent_query_endpoints == (1,)
    assert overlap.independent_kv_endpoints == (4,)
    assert overlap.boundary_query_length + overlap.independent_query_endpoints[-1] == 4


def test_fused_qkv_ulysses_matches_reference_layout_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def identity_all_to_all(output, input_, *, group):
        del group
        output.copy_(input_)

    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        identity_all_to_all,
    )
    torch.manual_seed(7)
    tensors = [
        torch.randn(3, 1, 4, 2, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    references = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]

    actual = olmo3_swa._fused_qkv_ulysses_all_to_all(
        *tensors,
        process_group=object(),
    )
    expected = torch.cat(
        tuple(
            tensor.reshape(3, 1, 2, 2, 2).permute(2, 0, 1, 3, 4)
            for tensor in references
        ),
        dim=-1,
    ).reshape(6, 1, 2, 6)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for tensor, reference in zip(tensors, references):
        torch.testing.assert_close(tensor.grad, reference.grad, rtol=0.0, atol=0.0)


def test_adjacent_batch_rejects_more_than_one_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.ones(2)
    with pytest.raises(ValueError, match="one adjacent halo peer"):
        olmo3_swa._begin_adjacent_halo_p2p(
            send_items=((1, tensor), (2, tensor)),
            recv_items=(),
            process_group=object(),
            group_rank=0,
        )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ragged_halo_worker(rank: int, port: int) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 0:
            local = torch.tensor([10.0, 11.0], requires_grad=True)
            recv_length = 0
        else:
            local = torch.empty(0, requires_grad=True)
            recv_length = 2
        history = olmo3_swa._variable_left_halo_exchange(
            local,
            recv_length=recv_length,
            process_group=dist.group.WORLD,
            global_ranks=(0, 1),
        )
        if rank == 0:
            loss = history.sum() * 0.0
        else:
            torch.testing.assert_close(
                history,
                torch.tensor([10.0, 11.0]),
                rtol=0.0,
                atol=0.0,
            )
            loss = (history * torch.tensor([3.0, 5.0])).sum()
        loss.backward()
        if rank == 0:
            torch.testing.assert_close(
                local.grad,
                torch.tensor([3.0, 5.0]),
                rtol=0.0,
                atol=0.0,
            )
        else:
            assert local.grad is not None and local.grad.numel() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed is unavailable",
)
def test_ragged_adjacent_halo_backward_is_exact_on_cpu() -> None:
    # A real two-rank Gloo exchange verifies that the custom autograd operation
    # implements the transpose of the document-dependent forward message.
    torch.multiprocessing.spawn(
        _ragged_halo_worker,
        args=(_free_tcp_port(),),
        nprocs=2,
        join=True,
    )


def _reference_tnd_attention(
    query,
    key,
    value,
    *,
    softmax_scale: float,
    pre_tokens: int,
    query_endpoints: tuple[int, ...],
    key_endpoints: tuple[int, ...],
):
    outputs = []
    query_start = 0
    key_start = 0
    for query_end, key_end in zip(query_endpoints, key_endpoints):
        query_segment = query[query_start:query_end]
        key_segment = key[key_start:key_end]
        value_segment = value[key_start:key_end]
        query_length = int(query_segment.shape[0])
        key_length = int(key_segment.shape[0])
        alignment = key_length - query_length
        segment_outputs = []
        for query_index in range(query_length):
            key_right = alignment + query_index
            key_left = max(0, key_right - pre_tokens)
            scores = torch.einsum(
                "hd,khd->hk",
                query_segment[query_index],
                key_segment[key_left : key_right + 1],
            )
            probabilities = torch.softmax(scores * softmax_scale, dim=-1)
            segment_outputs.append(
                torch.einsum(
                    "hk,khd->hd",
                    probabilities,
                    value_segment[key_left : key_right + 1],
                )
            )
        outputs.append(torch.stack(segment_outputs))
        query_start = query_end
        key_start = key_end
    return torch.cat(outputs, dim=0)


def _fake_raw_attention(
    query,
    key,
    value,
    *,
    num_heads: int,
    softmax_scale: float,
    contract,
    actual_seq_qlen: tuple[int, ...],
    actual_seq_kvlen: tuple[int, ...],
):
    assert num_heads == int(query.shape[1])
    output = _reference_tnd_attention(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        pre_tokens=contract.pre_tokens,
        query_endpoints=actual_seq_qlen,
        key_endpoints=actual_seq_kvlen,
    )
    empty = query.new_empty((0,))
    return output, empty, empty


def _fake_raw_attention_backward(
    query,
    key,
    value,
    grad_output,
    attention_output,
    softmax_max,
    softmax_sum,
    *,
    num_heads: int,
    softmax_scale: float,
    contract,
    actual_seq_qlen: tuple[int, ...],
    actual_seq_kvlen: tuple[int, ...],
):
    del attention_output, softmax_max, softmax_sum
    with torch.enable_grad():
        local_query = query.detach().requires_grad_(True)
        local_key = key.detach().requires_grad_(True)
        local_value = value.detach().requires_grad_(True)
        output = _reference_tnd_attention(
            local_query,
            local_key,
            local_value,
            softmax_scale=softmax_scale,
            pre_tokens=contract.pre_tokens,
            query_endpoints=actual_seq_qlen,
            key_endpoints=actual_seq_kvlen,
        )
        return torch.autograd.grad(
            output,
            (local_query, local_key, local_value),
            grad_output,
        )


def _overlapped_halo_worker(rank: int, port: int) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        olmo3_swa._run_raw_cann_tnd_attention = _fake_raw_attention
        olmo3_swa._run_raw_cann_tnd_attention_backward = (
            _fake_raw_attention_backward
        )
        torch.manual_seed(29)
        global_query = torch.randn(8, 1, 1, 2, dtype=torch.float64)
        global_key = torch.randn(8, 1, 1, 2, dtype=torch.float64)
        global_value = torch.randn(8, 1, 1, 2, dtype=torch.float64)
        output_weight = torch.randn(8, 1, 1, 2, dtype=torch.float64)
        local_slice = slice(rank * 4, (rank + 1) * 4)
        query = global_query[local_slice].clone().requires_grad_(True)
        key = global_key[local_slice].clone().requires_grad_(True)
        value = global_value[local_slice].clone().requires_grad_(True)
        packed = _packed([0, 8], global_tokens=8)
        plan = olmo3_swa._packed_swa_halo_plan(
            packed,
            cp_rank=rank,
            cp_size=2,
            local_sequence_length=4,
            sliding_window=4,
        )
        overlap = olmo3_swa._packed_swa_overlap_plan(
            packed,
            plan,
            local_sequence_length=4,
            halo_length=3,
        )
        contract = olmo3_swa.Olmo3CannAttentionContract(
            pre_tokens=3,
            next_tokens=0,
            sparse_mode=4,
        )
        scale = 2**-0.5
        output = olmo3_swa._PackedSwaOneHopOverlap.apply(
            query,
            key,
            value,
            dist.group.WORLD,
            (0, 1),
            (0, 3),
            overlap,
            scale,
            contract,
        )
        (output * output_weight[local_slice, 0]).sum().backward()

        reference_query = global_query.clone().requires_grad_(True)
        reference_key = global_key.clone().requires_grad_(True)
        reference_value = global_value.clone().requires_grad_(True)
        reference_output = _reference_tnd_attention(
            reference_query[:, 0],
            reference_key[:, 0],
            reference_value[:, 0],
            softmax_scale=scale,
            pre_tokens=3,
            query_endpoints=(8,),
            key_endpoints=(8,),
        )
        (reference_output * output_weight[:, 0]).sum().backward()

        torch.testing.assert_close(
            output,
            reference_output[local_slice],
            rtol=1e-12,
            atol=1e-12,
        )
        for actual, expected in (
            (query.grad, reference_query.grad[local_slice]),
            (key.grad, reference_key.grad[local_slice]),
            (value.grad, reference_value.grad[local_slice]),
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed is unavailable",
)
def test_true_forward_and_backward_overlap_match_global_swa_on_cpu() -> None:
    torch.multiprocessing.spawn(
        _overlapped_halo_worker,
        args=(_free_tcp_port(),),
        nprocs=2,
        join=True,
    )
