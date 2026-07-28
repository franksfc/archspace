"""CPU math and shape tests for SiameseNorm and Depth Attention."""

from __future__ import annotations
# ruff: noqa: E402

import math
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from modeling.olmo3_siamese_depth_components import (
    DepthAttentionContext,
    DepthAttentionSource,
    Olmo3ArchitectureContext,
    SiameseNormContext,
    _TensorParallelFusedQKFullProjectionRMSNorm,
    _TensorParallelFullProjectionRMSNorm,
    apply_full_projection_qk_norm,
    depth_attention_mix,
    siamese_norm_residual_update,
)


class _RMSNorm(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (value.float() * inverse_rms * self.weight.float()).to(value.dtype)


def _reference_depth_mix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sources: tuple[DepthAttentionSource, ...],
) -> torch.Tensor:
    packed = query.ndim == 3
    if packed:
        query, key, value = query[:, None], key[:, None], value[:, None]
        sources = tuple(
            DepthAttentionSource(s.layer_index, s.key[:, None], s.value[:, None])
            for s in sources
        )
    grouped_query = query.reshape(
        query.shape[0],
        query.shape[1],
        key.shape[2],
        query.shape[2] // key.shape[2],
        query.shape[-1],
    ).mean(dim=3)
    keys = torch.stack([source.key for source in sources] + [key], dim=-2)
    values = torch.stack([source.value for source in sources] + [value], dim=-2)
    logits = (
        torch.einsum("sbhd,sbhnd->sbhn", grouped_query, keys)
        / math.sqrt(query.shape[-1])
    )
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(value.dtype)
    result = torch.einsum("sbhn,sbhnd->sbhd", weights, values)
    return result[:, 0] if packed else result


def test_full_projection_qk_norm_matches_reference_and_backward() -> None:
    query = torch.randn(7, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    key = torch.randn(7, 3, 2, 5, dtype=torch.float64, requires_grad=True)
    q_norm = _RMSNorm(20)
    k_norm = _RMSNorm(10)
    with torch.no_grad():
        q_norm.weight.copy_(torch.linspace(0.7, 1.3, 20))
        k_norm.weight.copy_(torch.linspace(1.4, 0.8, 10))

    actual_query, actual_key = apply_full_projection_qk_norm(
        query, key, q_norm, k_norm
    )
    expected_query = q_norm(query.reshape(7, 3, 20)).reshape_as(query)
    expected_key = k_norm(key.reshape(7, 3, 10)).reshape_as(key)
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    (actual_query.square().sum() + actual_key.square().sum()).backward()
    assert query.grad is not None
    assert key.grad is not None
    assert q_norm.weight.grad is not None
    assert k_norm.weight.grad is not None


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("micro_batch", (1, 2, 4))
@pytest.mark.parametrize(("query_heads", "kv_heads"), ((4, 4), (4, 2)))
def test_depth_attention_is_batch_local_and_packed_shape_safe(
    packed: bool,
    micro_batch: int,
    query_heads: int,
    kv_heads: int,
) -> None:
    torch.manual_seed(123)
    sequence = 9
    head_dim = 8
    if packed:
        leading = (sequence * micro_batch,)
    else:
        leading = (sequence, micro_batch)
    tensors = [
        torch.randn(*leading, heads, head_dim, requires_grad=True)
        for heads in (query_heads, kv_heads, kv_heads, kv_heads, kv_heads)
    ]
    query, key, value, source_key, source_value = tensors
    sources = (DepthAttentionSource(0, source_key, source_value),)

    actual = depth_attention_mix(query, key, value, sources)
    expected = _reference_depth_mix(query, key, value, sources)
    assert actual.shape == value.shape
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_depth_attention_commutes_with_cp_and_tp_local_partitioning() -> None:
    """Depth mixing is pointwise in sequence/head, before CP token attention."""

    torch.manual_seed(321)
    query = torch.randn(16, 2, 8, 4)
    key = torch.randn(16, 2, 8, 4)
    value = torch.randn(16, 2, 8, 4)
    source_key = torch.randn_like(key)
    source_value = torch.randn_like(value)
    full = depth_attention_mix(
        query,
        key,
        value,
        (DepthAttentionSource(0, source_key, source_value),),
    )

    local_outputs = []
    for sequence_shard in range(2):
        row = []
        sequence_slice = slice(sequence_shard * 8, (sequence_shard + 1) * 8)
        for head_shard in range(2):
            head_slice = slice(head_shard * 4, (head_shard + 1) * 4)
            row.append(
                depth_attention_mix(
                    query[sequence_slice, :, head_slice],
                    key[sequence_slice, :, head_slice],
                    value[sequence_slice, :, head_slice],
                    (
                        DepthAttentionSource(
                            0,
                            source_key[sequence_slice, :, head_slice],
                            source_value[sequence_slice, :, head_slice],
                        ),
                    ),
                )
            )
        local_outputs.append(torch.cat(row, dim=2))
    reconstructed = torch.cat(local_outputs, dim=0)
    torch.testing.assert_close(reconstructed, full)


def test_stride8_context_retains_exact_sources() -> None:
    context = DepthAttentionContext(num_layers=16, stride=8, recent_window=0)
    tensor = torch.zeros(1, 2, 1, 1)
    observed = {}
    for layer_index in range(16):
        observed[layer_index] = context.selected_layer_indices(layer_index)
        context.record(layer_index, tensor + layer_index, tensor + layer_index)
    assert observed[0] == ()
    assert observed[1] == (0,)
    assert observed[8] == (0,)
    assert observed[9] == (0, 8)
    assert observed[15] == (0, 8)


def test_siamese_context_supports_sp_local_shape_and_microbatch_gt_one() -> None:
    # This is the hidden-state shape after CP and SP sharding. The context must
    # preserve it exactly and never couple microbatch elements.
    hidden = torch.randn(8, 4, 32, requires_grad=True)
    context = SiameseNormContext(
        hidden,
        num_layers=2,
        clone_initial_stream=False,
    )
    assert context.stream_for(1).data_ptr() == hidden.data_ptr()
    first = context.stream_for(1) + 1
    context.advance(1, first)
    second = context.stream_for(2) * 2
    context.advance(2, second)
    output = context.final_stream()
    assert output.shape == hidden.shape
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 2.0))


def test_siamese_residual_scaling_and_combined_context() -> None:
    post = torch.tensor([1.0, 2.0])
    pre = torch.tensor([3.0, 4.0])
    update = torch.tensor([2.0, -2.0])
    actual_post, actual_pre = siamese_norm_residual_update(
        post, pre, update, layer_number=2
    )
    torch.testing.assert_close(actual_post, post + update / 2.0)
    torch.testing.assert_close(actual_pre, pre + update)

    context = Olmo3ArchitectureContext.create(
        torch.zeros(8, 4, 32),
        num_layers=16,
        depth_stride=8,
        depth_recent_window=0,
        clone_initial_stream=False,
    )
    assert context.siamese_norm.num_layers == 16
    assert context.depth_attention.stride == 8


def test_tp_rmsnorm_autograd_formula_world_one() -> None:
    class _Work:
        def wait(self) -> None:
            return None

    def all_reduce(
        tensor: torch.Tensor, group=None, async_op: bool = False
    ) -> _Work | None:
        del tensor, group
        return _Work() if async_op else None

    actual_input = torch.randn(3, 2, 7, requires_grad=True)
    actual_weight = torch.linspace(0.7, 1.3, 7, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    reference_weight = actual_weight.detach().clone().requires_grad_(True)
    output_grad = torch.randn_like(actual_input)
    epsilon = 1.0e-6

    with patch("torch.distributed.all_reduce", side_effect=all_reduce):
        actual = _TensorParallelFullProjectionRMSNorm.apply(
            actual_input,
            actual_weight,
            epsilon,
            7,
            object(),
        )
        actual.backward(output_grad)
    inverse_rms = torch.rsqrt(
        reference_input.square().mean(dim=-1, keepdim=True) + epsilon
    )
    expected = reference_input * inverse_rms * reference_weight
    expected.backward(output_grad)
    for left, right in (
        (actual, expected),
        (actual_input.grad, reference_input.grad),
        (actual_weight.grad, reference_weight.grad),
    ):
        torch.testing.assert_close(left, right, rtol=5.0e-5, atol=5.0e-6)


def test_fused_tp_qk_rmsnorm_uses_one_collective_per_pass() -> None:
    class _Work:
        def wait(self) -> None:
            return None

    def all_reduce(
        tensor: torch.Tensor, group=None, async_op: bool = False
    ) -> _Work | None:
        del tensor, group
        return _Work() if async_op else None

    query = torch.randn(5, 2, 7, requires_grad=True)
    key = torch.randn(5, 2, 3, requires_grad=True)
    q_weight = torch.ones(7, requires_grad=True)
    k_weight = torch.ones(3, requires_grad=True)
    with patch("torch.distributed.all_reduce", side_effect=all_reduce) as reduce:
        q_out, k_out = _TensorParallelFusedQKFullProjectionRMSNorm.apply(
            query,
            key,
            q_weight,
            k_weight,
            1.0e-6,
            1.0e-6,
            7,
            3,
            object(),
        )
        (q_out.sum() + k_out.sum()).backward()
    assert reduce.call_count == 2
    assert reduce.call_args_list[0].kwargs.get("async_op", False) is False
    assert reduce.call_args_list[1].kwargs["async_op"] is True
