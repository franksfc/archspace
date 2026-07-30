from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.embeddings import rope_utils


class _FakeNpuRotary:
    """CPU implementation of the four-argument CANN fused-RoPE ABI."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def __call__(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        self.calls.append((x, cos, sin, mode))
        return x * cos + rope_utils._rotate_half(x, mode == 1) * sin


def _config(
    *,
    apply_rope_fusion: bool = True,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        apply_rope_fusion=apply_rope_fusion,
        rope_full_precision=True,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mrope_section=None,
    )


@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("multi_latent_attention", [False, True])
def test_npu_bshd_fusion_matches_fp32_equation_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
    rotary_interleaved: bool,
    multi_latent_attention: bool,
) -> None:
    torch.manual_seed(7)
    source = torch.randn(6, 2, 3, 10, dtype=torch.float32).to(torch.bfloat16)
    frequencies = torch.randn(6, 1, 1, 8, dtype=torch.float32).to(torch.bfloat16)
    weights = torch.randn_like(source)

    expected_input = source.detach().clone().requires_grad_(True)
    monkeypatch.setattr(rope_utils, "_runtime_requests_npu_rope_fusion", lambda: False)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: None)
    expected = rope_utils._apply_rotary_pos_emb_bshd(
        expected_input,
        frequencies,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mscale=1.37,
        full_precision=True,
    )
    (expected.float() * weights.float()).sum().backward()

    fake = _FakeNpuRotary()
    actual_input = source.detach().clone().requires_grad_(True)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: fake)
    actual = rope_utils._apply_rotary_pos_emb_bshd(
        actual_input,
        frequencies,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mscale=1.37,
        full_precision=True,
        _force_npu_fusion=True,
    )
    (actual.float() * weights.float()).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
    assert actual.dtype == source.dtype
    assert len(fake.calls) == 1
    fused_x, fused_cos, fused_sin, mode = fake.calls[0]
    assert fused_x.dtype == fused_cos.dtype == fused_sin.dtype == torch.float32
    assert fused_x.is_contiguous()
    assert fused_cos.is_contiguous()
    assert fused_sin.is_contiguous()
    assert mode == int(rotary_interleaved)
    # The non-rotary tail is preserved byte-for-byte.
    torch.testing.assert_close(actual[..., 8:], source[..., 8:], rtol=0, atol=0)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_npu_thd_packed_cp_uses_document_splits_and_fp32(
    monkeypatch: pytest.MonkeyPatch, cp_rank: int
) -> None:
    torch.manual_seed(11 + cp_rank)
    source = torch.randn(8, 2, 8, dtype=torch.float32).to(torch.bfloat16)
    frequencies = torch.randn(8, 1, 1, 8, dtype=torch.float32).to(torch.bfloat16)
    # Two global documents of length eight; CP2 owns four tokens from each.
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    monkeypatch.setattr(
        rope_utils.parallel_state, "get_context_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        rope_utils.parallel_state, "get_context_parallel_rank", lambda: cp_rank
    )
    monkeypatch.setattr(rope_utils, "_runtime_requests_npu_rope_fusion", lambda: False)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: None)

    expected_input = source.detach().clone().requires_grad_(True)
    expected = rope_utils._apply_rotary_pos_emb_thd(
        expected_input,
        cu_seqlens,
        frequencies,
        mscale=1.23,
        full_precision=True,
    )
    expected.square().float().sum().backward()

    fake = _FakeNpuRotary()
    actual_input = source.detach().clone().requires_grad_(True)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: fake)
    actual = rope_utils.apply_rotary_pos_emb(
        actual_input,
        frequencies,
        _config(),
        cu_seqlens=cu_seqlens,
        mscale=1.23,
    )
    actual.square().float().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
    assert actual.shape == source.shape
    assert actual.dtype == source.dtype
    assert len(fake.calls) == 2
    assert all(call[0].shape == (4, 1, 2, 8) for call in fake.calls)
    assert all(call[0].dtype == torch.float32 for call in fake.calls)

    first_half = slice(cp_rank * 2, (cp_rank + 1) * 2)
    second_half = slice(8 - (cp_rank + 1) * 2, 8 - cp_rank * 2)
    selected_freqs = torch.cat(
        [frequencies[first_half], frequencies[second_half]]
    ).float()
    expected_cos = torch.cos(selected_freqs) * 1.23
    expected_sin = torch.sin(selected_freqs) * 1.23
    torch.testing.assert_close(fake.calls[0][1], expected_cos)
    torch.testing.assert_close(fake.calls[0][2], expected_sin)


def test_apply_rope_fusion_without_any_backend_falls_back_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(19)
    source = torch.randn(5, 1, 2, 8, dtype=torch.bfloat16)
    frequencies = torch.randn(5, 1, 1, 8, dtype=torch.bfloat16)

    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: None)
    monkeypatch.setattr(rope_utils, "_runtime_requests_npu_rope_fusion", lambda: False)
    monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb", None)
    config = _config(apply_rope_fusion=True)
    actual = rope_utils.apply_rotary_pos_emb(source, frequencies, config)
    expected = rope_utils._apply_rotary_pos_emb_bshd(
        source, frequencies, full_precision=True
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert config.apply_rope_fusion is True


def test_npu_backend_failure_is_not_silently_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_backend(*_args: object) -> torch.Tensor:
        raise ValueError("synthetic CANN failure")

    monkeypatch.setattr(
        rope_utils, "_resolve_npu_rotary_backend", lambda _t: broken_backend
    )
    source = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    frequencies = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Ascend fused RoPE backend failed") as error:
        rope_utils.apply_rotary_pos_emb(source, frequencies, _config())
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("rotary_interleaved", [False, True])
def test_inference_cos_sin_path_uses_fp32_npu_fusion(
    monkeypatch: pytest.MonkeyPatch, rotary_interleaved: bool
) -> None:
    torch.manual_seed(23)
    source = torch.randn(4, 2, 3, 10, dtype=torch.float32).to(torch.bfloat16)
    cos = torch.randn(4, 4, dtype=torch.bfloat16)
    sin = torch.randn(4, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(rope_utils, "_runtime_requests_npu_rope_fusion", lambda: False)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: None)
    expected = rope_utils.apply_rotary_pos_emb_with_cos_sin(
        source,
        cos,
        sin,
        rotary_interleaved=rotary_interleaved,
        full_precision=True,
    )

    fake = _FakeNpuRotary()
    monkeypatch.setattr(rope_utils, "_runtime_requests_npu_rope_fusion", lambda: True)
    monkeypatch.setattr(rope_utils, "_resolve_npu_rotary_backend", lambda _t: fake)
    actual = rope_utils.apply_rotary_pos_emb_with_cos_sin(
        source,
        cos,
        sin,
        rotary_interleaved=rotary_interleaved,
        full_precision=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == source.dtype
    assert len(fake.calls) == 1
    assert all(tensor.dtype == torch.float32 for tensor in fake.calls[0][:3])
    assert fake.calls[0][3] == int(rotary_interleaved)


def test_post_mindspeed_compat_installs_project_rope_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import megatron.core.transformer.attention as attention_module
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup,
    )
    from megatron.olmo3_mindspeed_compat import (
        install_olmo3_mindspeed_compatibility,
    )

    def replaced(*_args: object, **_kwargs: object) -> None:
        return None

    # Make the installer take its non-idempotent branch without retaining any
    # mutation after pytest's monkeypatch teardown.
    monkeypatch.setattr(_ParamAndGradBucketGroup, "start_grad_sync", replaced)
    monkeypatch.setattr(_ParamAndGradBucketGroup, "start_inter_grad_sync", replaced)
    monkeypatch.setattr(_ParamAndGradBucketGroup, "finish_grad_sync", replaced)
    monkeypatch.setattr(rope_utils, "_apply_rotary_pos_emb_bshd", replaced)
    monkeypatch.setattr(rope_utils, "apply_rotary_pos_emb", replaced)
    monkeypatch.setattr(attention_module, "_apply_rotary_pos_emb_bshd", replaced)
    monkeypatch.setattr(attention_module, "apply_rotary_pos_emb", replaced)

    install_olmo3_mindspeed_compatibility()

    assert (
        rope_utils._apply_rotary_pos_emb_bshd
        is rope_utils.olmo3_apply_rotary_pos_emb_bshd
    )
    assert rope_utils.apply_rotary_pos_emb is rope_utils.olmo3_apply_rotary_pos_emb
    assert (
        attention_module._apply_rotary_pos_emb_bshd
        is rope_utils.olmo3_apply_rotary_pos_emb_bshd
    )
    assert (
        attention_module.apply_rotary_pos_emb
        is rope_utils.olmo3_apply_rotary_pos_emb
    )
