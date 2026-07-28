"""CPU/mock contracts for packed SWA and Full-Attention CP ownership.

These tests pin the ordering required by the production Stage-3 path:

* packed SWA receives the pre-Ulysses CP-local SBHD shard;
* Full Attention retains OLMo3's fused QKV Ulysses all-to-all;
* compact global document endpoints are projected onto each local shard
  without allowing a halo to cross a document or microbatch-row boundary.
"""

from __future__ import annotations

import sys
import types
from types import MethodType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


# CPU test images do not install Transformer Engine.  The OLMo3 module only
# needs this symbol while importing; these tests mock the attention backend.
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


GLOBAL_SEQUENCE_LENGTH = 65_536
SLIDING_WINDOW = 4_096


def _packed(
    endpoints: list[int],
    *,
    global_tokens_per_sample: int = GLOBAL_SEQUENCE_LENGTH,
    micro_batch_size: int = 1,
) -> PackedSeqParams:
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(endpoints, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(endpoints, dtype=torch.int32),
    )
    packed.olmo3_micro_batch_size = micro_batch_size
    packed.olmo3_global_tokens_per_sample = global_tokens_per_sample
    return packed


@pytest.mark.parametrize("cp_size", (2, 8, 16))
def test_packed_swa_accepts_only_the_pre_ulysses_local_layout(
    cp_size: int,
) -> None:
    """Reject a global-S tensor at the local SWA halo boundary."""

    local_sequence_length = GLOBAL_SEQUENCE_LENGTH // cp_size
    packed = _packed([0, GLOBAL_SEQUENCE_LENGTH])

    local_plan = olmo3_swa._packed_swa_halo_plan(
        packed,
        cp_rank=1,
        cp_size=cp_size,
        local_sequence_length=local_sequence_length,
        sliding_window=SLIDING_WINDOW,
    )
    assert local_plan.segments == (
        (0, 0, local_sequence_length, SLIDING_WINDOW - 1),
    )
    assert local_plan.query_endpoints == (local_sequence_length,)
    assert local_plan.kv_endpoints == (
        local_sequence_length + SLIDING_WINDOW - 1,
    )

    # An outer Ulysses owner must not gather S_global before invoking the
    # local single-halo SWA core.
    with pytest.raises(
        ValueError,
        match=(
            rf"global={GLOBAL_SEQUENCE_LENGTH}, "
            rf"local={GLOBAL_SEQUENCE_LENGTH}, CP={cp_size}"
        ),
    ):
        olmo3_swa._packed_swa_halo_plan(
            _packed([0, GLOBAL_SEQUENCE_LENGTH]),
            cp_rank=1,
            cp_size=cp_size,
            local_sequence_length=GLOBAL_SEQUENCE_LENGTH,
            sliding_window=SLIDING_WINDOW,
        )


@pytest.mark.parametrize("cp_size", (2, 8, 16))
def test_packed_swa_dispatch_never_calls_full_ulysses(
    cp_size: int,
) -> None:
    """Exercise the runtime branch, not only its source-code ordering."""

    local_sequence_length = GLOBAL_SEQUENCE_LENGTH // cp_size
    packed = _packed([0, GLOBAL_SEQUENCE_LENGTH])
    calls: list[tuple[int, int, int, int]] = []

    attention = SimpleNamespace(
        is_sliding=True,
        config=SimpleNamespace(context_parallel_size=cp_size),
    )

    def local_halo_forward(
        self,
        query,
        key,
        value,
        packed_seq_params,
    ):
        del self
        assert query.shape == key.shape == value.shape
        calls.append(tuple(int(size) for size in query.shape))
        # The planner is the final fail-closed check that this is still the
        # local sequence shard when the halo implementation consumes it.
        olmo3_swa._packed_swa_halo_plan(
            packed_seq_params,
            cp_rank=1,
            cp_size=cp_size,
            local_sequence_length=int(query.shape[0]),
            sliding_window=SLIDING_WINDOW,
        )
        return query.reshape(
            local_sequence_length,
            1,
            -1,
        )

    def forbidden_full_ulysses(*args, **kwargs):
        del args, kwargs
        raise AssertionError("packed SWA must not enter Full-Attention Ulysses")

    attention._mindspeed_packed_swa_halo_forward = MethodType(
        local_halo_forward,
        attention,
    )
    attention._prepare_ulysses_qkv = forbidden_full_ulysses

    # Keeping CP heads on the local tensor also distinguishes the pre-Ulysses
    # [S/CP, B, H, D] layout from post-Ulysses [S, B, H/CP, D].
    local_qkv = torch.zeros(local_sequence_length, 1, cp_size, 1)
    output = olmo3_swa.Olmo3DotProductAttention._mindspeed_packed_forward(
        attention,
        local_qkv,
        local_qkv,
        local_qkv,
        packed,
    )
    assert calls == [(local_sequence_length, 1, cp_size, 1)]
    assert output.shape == (local_sequence_length, 1, cp_size)


@pytest.mark.parametrize("cp_size", (2, 8, 16))
def test_document_plan_math_is_exact_for_two_packed_batch_rows(
    cp_size: int,
) -> None:
    """Project global document intervals onto rank one without row leakage."""

    local_sequence_length = GLOBAL_SEQUENCE_LENGTH // cp_size
    row = GLOBAL_SEQUENCE_LENGTH
    # In row zero the document spanning rank one's left boundary contributes
    # seven remote tokens.  In row one it contributes three.  Both are below
    # the window cap and must remain independent even though cu_seqlens is a
    # single flattened list for the complete microbatch.
    endpoints = [
        0,
        local_sequence_length - 7,
        local_sequence_length + 19,
        row,
        row + local_sequence_length - 3,
        row + local_sequence_length + 23,
        2 * row,
    ]
    plan = olmo3_swa._packed_swa_halo_plan(
        _packed(endpoints, micro_batch_size=2),
        cp_rank=1,
        cp_size=cp_size,
        local_sequence_length=local_sequence_length,
        sliding_window=SLIDING_WINDOW,
    )

    assert plan.segments == (
        (0, 0, 19, 7),
        (0, 19, local_sequence_length, 0),
        (1, 0, 23, 3),
        (1, 23, local_sequence_length, 0),
    )
    assert plan.query_endpoints == (
        19,
        local_sequence_length,
        local_sequence_length + 23,
        2 * local_sequence_length,
    )
    assert plan.kv_endpoints == (
        26,
        local_sequence_length + 7,
        local_sequence_length + 33,
        2 * local_sequence_length + 10,
    )
    assert plan.max_kv_length == local_sequence_length - 19


@pytest.mark.parametrize("cp_size", (2, 8, 16))
def test_full_attention_keeps_fused_qkv_ulysses(
    cp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the Full path from local SBHD through fused gather and restore."""

    local_sequence_length = GLOBAL_SEQUENCE_LENGTH // cp_size
    process_group = object()
    calls: list[tuple[str, tuple[int, ...]]] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: cp_size if group is process_group else 0,
    )
    monkeypatch.setattr(
        olmo3_swa.parallel_state,
        "get_context_parallel_group",
        lambda: process_group,
    )
    monkeypatch.setattr(
        olmo3_swa,
        "emit_rank0_runtime_marker_once",
        lambda *args, **kwargs: None,
    )

    def fused_qkv(query, key, value, group):
        assert group is process_group
        assert query.shape == key.shape == value.shape
        calls.append(("fused", tuple(int(size) for size in query.shape)))
        return query.new_zeros(GLOBAL_SEQUENCE_LENGTH, 1, 1, 3)

    def restore_all_to_all(
        tensor,
        group,
        scatter_dim,
        gather_dim,
        gather_size=None,
    ):
        assert group is process_group
        assert (scatter_dim, gather_dim, gather_size) == (0, 2, cp_size)
        calls.append(("restore", tuple(int(size) for size in tensor.shape)))
        return tensor.new_zeros(local_sequence_length, 1, cp_size)

    monkeypatch.setattr(
        olmo3_swa,
        "_fused_qkv_ulysses_all_to_all",
        fused_qkv,
    )
    monkeypatch.setattr(olmo3_swa, "_ulysses_all_to_all", restore_all_to_all)

    attention = SimpleNamespace(
        is_sliding=False,
        config=SimpleNamespace(
            context_parallel_size=cp_size,
            context_parallel_algo="ulysses_cp_algo",
        ),
        olmo3_fused_qkv_a2a_packing=True,
        layer_number=4,
    )
    attention._prepare_ulysses_qkv = MethodType(
        olmo3_swa.Olmo3DotProductAttention._prepare_ulysses_qkv,
        attention,
    )
    # This production helper is a staticmethod; keep its two-argument calling
    # convention when attaching it to the lightweight mock owner.
    attention._restore_ulysses_output = (
        olmo3_swa.Olmo3DotProductAttention._restore_ulysses_output
    )
    attention._mindspeed_contract = lambda *, max_kv_length: (
        olmo3_swa.Olmo3CannAttentionContract(
            pre_tokens=max_kv_length,
            next_tokens=0,
            sparse_mode=2,
        )
    )

    def fake_cann_attention(
        query,
        key,
        value,
        *,
        num_heads,
        layout,
        contract,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
    ):
        del contract
        assert query.shape == key.shape == value.shape
        assert query.shape == (GLOBAL_SEQUENCE_LENGTH, 1, 1)
        assert num_heads == 1
        assert layout == "TND"
        assert actual_seq_qlen == [GLOBAL_SEQUENCE_LENGTH]
        assert actual_seq_kvlen == [GLOBAL_SEQUENCE_LENGTH]
        calls.append(("cann", tuple(int(size) for size in query.shape)))
        return query

    attention._run_mindspeed_cann_attention = fake_cann_attention
    local_qkv = torch.zeros(local_sequence_length, 1, cp_size, 1)
    output = olmo3_swa.Olmo3DotProductAttention._mindspeed_packed_forward(
        attention,
        local_qkv,
        local_qkv,
        local_qkv,
        _packed([0, GLOBAL_SEQUENCE_LENGTH]),
    )

    assert calls == [
        ("fused", (local_sequence_length, 1, cp_size, 1)),
        ("cann", (GLOBAL_SEQUENCE_LENGTH, 1, 1)),
        ("restore", (GLOBAL_SEQUENCE_LENGTH, 1, 1)),
    ]
    assert output.shape == (local_sequence_length, 1, cp_size)
