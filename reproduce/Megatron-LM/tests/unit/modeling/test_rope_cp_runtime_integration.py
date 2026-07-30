from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.models.common.embeddings import rope_utils
from megatron.core.models.common.embeddings import (
    rotary_pos_embedding as mcore_rotary_pos_embedding,
)
from megatron.core.transformer import attention as attention_module


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MINDSPEED_ROPE_PATCH = (
    _PROJECT_ROOT
    / "third_party"
    / "MindSpeed"
    / "mindspeed"
    / "core"
    / "context_parallel"
    / "rotary_pos_embedding_utils.py"
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_real_mindspeed_rope_patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: SimpleNamespace,
    cp_state: SimpleNamespace,
) -> ModuleType:
    """Load the pristine MindSpeed implementation without importing torch_npu.

    MindSpeed's module only needs these collaborators at function-call time,
    but importing its package normally initializes the entire Ascend stack.
    Small import stubs keep this a CPU audit while the implementation under
    test still comes directly from the frozen third-party source file.
    """

    def unused(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the Ulysses/TP-y=1 branch must not call this helper")

    class UnusedTensorParallelYUnionCP:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            unused()

    modules = {
        "mindspeed": _module("mindspeed"),
        "mindspeed.core": _module("mindspeed.core"),
        "mindspeed.core.context_parallel": _module(
            "mindspeed.core.context_parallel",
            mpu=cp_state,
            get_args=lambda: args,
        ),
        "mindspeed.core.tensor_parallel_y_union_cp": _module(
            "mindspeed.core.tensor_parallel_y_union_cp",
            TensorParallelYUnionCP=UnusedTensorParallelYUnionCP,
        ),
        "mindspeed.utils": _module(
            "mindspeed.utils",
            get_position_ids=unused,
            generate_rearrange_idx_tensor=unused,
        ),
        "mindspeed.core.context_parallel.model_parallel_utils": _module(
            "mindspeed.core.context_parallel.model_parallel_utils",
            get_context_parallel_for_hybrid_ulysses_world_size=unused,
            get_context_parallel_for_hybrid_ulysses_rank=unused,
            get_context_parallel_for_hybrid_ring_world_size=unused,
            get_context_parallel_for_hybrid_ring_rank=unused,
        ),
        "mindspeed.core.context_parallel.utils": _module(
            "mindspeed.core.context_parallel.utils",
            get_remapped_seq_order=unused,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert _MINDSPEED_ROPE_PATCH.is_file()
    module_name = "_olmo3_test_real_mindspeed_rotary_patch"
    spec = importlib.util.spec_from_file_location(module_name, _MINDSPEED_ROPE_PATCH)
    assert spec is not None and spec.loader is not None
    patch_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, patch_module)
    spec.loader.exec_module(patch_module)
    return patch_module


def test_stage2_mindspeed_ulysses_patch_uses_contiguous_cp_positions_and_freqs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-patch MCore symbol must match Ulysses' contiguous data split."""

    cp_size = 8
    cp_state = SimpleNamespace(
        rank=0,
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_state.rank,
    )
    args = SimpleNamespace(
        context_parallel_algo="ulysses_cp_algo",
        tp_y=1,
        attention_mask_type="causal",
        reset_position_ids=False,
    )
    mindspeed_patch = _load_real_mindspeed_rope_patch(
        monkeypatch,
        args=args,
        cp_state=cp_state,
    )

    # This assignment is the exact target registered by
    # ContextParallelFeature.register_patches at runtime.
    monkeypatch.setattr(
        mcore_rotary_pos_embedding,
        "get_pos_emb_on_this_cp_rank",
        mindspeed_patch.get_pos_emb_on_this_cp_rank,
    )
    assert (
        mcore_rotary_pos_embedding.get_pos_emb_on_this_cp_rank
        is mindspeed_patch.get_pos_emb_on_this_cp_rank
    )
    assert (
        Path(
            mcore_rotary_pos_embedding.get_pos_emb_on_this_cp_rank.__code__.co_filename
        ).resolve()
        == _MINDSPEED_ROPE_PATCH.resolve()
    )

    sequence_length = 64
    positions = torch.arange(sequence_length, dtype=torch.float32)
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float32)
    unrepeated_freqs = torch.outer(positions, inv_freq)
    full_freqs = torch.cat((unrepeated_freqs, unrepeated_freqs), dim=-1)[
        :, None, None, :
    ]

    local_frequency_tables: list[torch.Tensor] = []
    local_length = sequence_length // cp_size
    for cp_rank in range(cp_size):
        cp_state.rank = cp_rank
        actual = mcore_rotary_pos_embedding.get_pos_emb_on_this_cp_rank(
            full_freqs, 0
        )
        start = cp_rank * local_length
        stop = start + local_length
        expected = full_freqs[start:stop]

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[:, 0, 0, 0],
            positions[start:stop],
            rtol=0,
            atol=0,
        )

        # MCore's native ring-CP slicing would pair one leading and one
        # mirrored trailing segment.  That layout is wrong for Ulysses, whose
        # dataloader and Q/K/V all-to-all start from contiguous local chunks.
        symmetric = full_freqs.view(
            2 * cp_size, -1, 1, 1, full_freqs.shape[-1]
        )[[cp_rank, 2 * cp_size - cp_rank - 1]].reshape_as(expected)
        assert not torch.equal(actual, symmetric)
        local_frequency_tables.append(actual)

    # Across all CP ranks every position/frequency occurs exactly once and in
    # the same order as the Stage-2 global sequence.
    torch.testing.assert_close(
        torch.cat(local_frequency_tables, dim=0),
        full_freqs,
        rtol=0,
        atol=0,
    )


class _FakeNpuRotary:
    """CPU implementation of MindSpeed's four-argument fused Ascend ABI."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []

    def __call__(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        self.calls.append((tensor, cos, sin, mode))
        if mode == 0:
            first, second = torch.chunk(tensor, 2, dim=-1)
            rotated_half = torch.cat((-second, first), dim=-1)
        else:
            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            rotated_half = torch.stack((-odd, even), dim=-1).flatten(-2)
        return tensor * cos + rotated_half * sin


def _packed_rope_fp32_reference(
    tensor: torch.Tensor,
    frequency_table: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    mscale: float,
) -> torch.Tensor:
    sequence_length, batch_size = tensor.shape[:2]
    positions = position_ids.transpose(0, 1).contiguous().reshape(-1)
    local_freqs = frequency_table[:, 0, 0, :].index_select(0, positions)
    local_freqs = local_freqs.reshape(
        sequence_length, batch_size, 1, frequency_table.shape[-1]
    )

    rotary_dim = frequency_table.shape[-1]
    rotary = tensor[..., :rotary_dim].float()
    first, second = torch.chunk(rotary, 2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    cos = torch.cos(local_freqs.float()) * mscale
    sin = torch.sin(local_freqs.float()) * mscale
    rotated = (rotary * cos + rotated_half * sin).to(tensor.dtype)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


@pytest.mark.parametrize(
    ("layer_kind", "mscale"),
    [
        ("swa", 1.0),
        ("full_yarn", 1.0 + 0.1 * math.log(8.0)),
    ],
)
def test_stage3_cp16_packed_explicit_positions_match_fp32_fused_rope(
    monkeypatch: pytest.MonkeyPatch,
    layer_kind: str,
    mscale: float,
) -> None:
    """Arbitrary CP-local document resets drive both SWA and Full YaRN RoPE."""

    del layer_kind  # The parameter name makes failures identify the layer path.
    fake_npu = _FakeNpuRotary()
    monkeypatch.setattr(
        rope_utils, "_runtime_requests_npu_rope_fusion", lambda: True
    )
    monkeypatch.setattr(
        rope_utils, "_resolve_npu_rotary_backend", lambda _tensor: fake_npu
    )
    # Exercise the real attention entry while keeping backend resolution under
    # the project-owned rope_utils module, as it is after compatibility setup.
    monkeypatch.setattr(
        attention_module,
        "_apply_rotary_pos_emb_bshd",
        rope_utils._apply_rotary_pos_emb_bshd,
    )

    torch.manual_seed(1729)
    sequence_length = 9
    batch_size = 2
    heads = 3
    rotary_dim = 8
    head_dim = 10
    max_position = 16
    frequency_table = torch.randn(
        max_position, 1, 1, rotary_dim, dtype=torch.float32
    )
    base_positions = torch.tensor(
        [
            [0, 1, 2, 0, 1, 5, 2, 0, 3],
            [7, 0, 1, 2, 0, 6, 1, 4, 0],
        ],
        dtype=torch.long,
    )

    for cp_rank in range(16):
        # Rolling retains resets and non-monotonic document-local positions,
        # while ensuring the test covers distinct local layouts on all CP16
        # ranks instead of validating one rank and extrapolating.
        position_ids = torch.roll(
            base_positions, shifts=cp_rank % sequence_length, dims=1
        )
        assert bool((position_ids[:, 1:] < position_ids[:, :-1]).any())
        assert int((position_ids == 0).sum()) >= 4

        source = torch.randn(
            sequence_length,
            batch_size,
            heads,
            head_dim,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        weight = torch.randn_like(source)

        reference_input = source.detach().clone().requires_grad_(True)
        expected = _packed_rope_fp32_reference(
            reference_input,
            frequency_table,
            position_ids,
            mscale=mscale,
        )
        (expected.float() * weight.float()).sum().backward()

        actual_input = source.detach().clone().requires_grad_(True)
        actual = attention_module._apply_packed_document_rotary_pos_emb(
            actual_input,
            frequency_table,
            position_ids,
            SimpleNamespace(
                rotary_interleaved=False,
                multi_latent_attention=False,
                rope_full_precision=True,
            ),
            mscale=mscale,
        )
        (actual.float() * weight.float()).sum().backward()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=0, atol=0
        )
        assert actual.dtype == torch.bfloat16
        # RoPE must leave a partial-rotary tail bit-exact.
        torch.testing.assert_close(
            actual[..., rotary_dim:],
            source[..., rotary_dim:],
            rtol=0,
            atol=0,
        )

        fused_tensor, fused_cos, fused_sin, mode = fake_npu.calls[-1]
        assert fused_tensor.dtype == torch.float32
        assert fused_cos.dtype == torch.float32
        assert fused_sin.dtype == torch.float32
        assert fused_tensor.is_contiguous()
        assert fused_cos.is_contiguous()
        assert fused_sin.is_contiguous()
        assert mode == 0

        explicit_freqs = frequency_table[:, 0, 0, :].index_select(
            0, position_ids.transpose(0, 1).contiguous().reshape(-1)
        )
        explicit_freqs = explicit_freqs.reshape(
            sequence_length, batch_size, 1, rotary_dim
        )
        torch.testing.assert_close(
            fused_cos, torch.cos(explicit_freqs) * mscale, rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused_sin, torch.sin(explicit_freqs) * mscale, rtol=0, atol=0
        )

    assert len(fake_npu.calls) == 16
