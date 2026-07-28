from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.olmo3_mindspeed_compat import (
    install_npu_fused_infer_attention_v2_fallback,
)


class _FakeTorchNpu(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def npu_prompt_flash_attention(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("prompt", args, kwargs))
        return "prompt-output"

    def npu_incre_flash_attention(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("incremental", args, kwargs))
        return "incremental-output"


def test_v2_fallback_maps_prompt_kernel_arguments() -> None:
    fake = _FakeTorchNpu()
    assert install_npu_fused_infer_attention_v2_fallback(fake)

    query = torch.empty(2, 7, 32)
    key = torch.empty(2, 7, 32)
    value = torch.empty(2, 7, 32)
    result = fake.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        num_query_heads=4,
        num_key_value_heads=4,
        input_layout="BSH",
        pse_shift="pse",
        softmax_scale=0.125,
        sparse_mode=4,
        atten_mask="mask",
        pre_tokens=4095,
        next_tokens=0,
    )

    assert result == ("prompt-output",)
    kind, args, kwargs = fake.calls.pop()
    assert kind == "prompt"
    assert args == (query, key, value)
    assert kwargs == {
        "num_heads": 4,
        "input_layout": "BSH",
        "pse_shift": "pse",
        "sparse_mode": 4,
        "padding_mask": None,
        "atten_mask": "mask",
        "scale_value": 0.125,
        "pre_tokens": 4095,
        "next_tokens": 0,
    }


def test_v2_fallback_maps_incremental_kernel_arguments() -> None:
    fake = _FakeTorchNpu()
    assert install_npu_fused_infer_attention_v2_fallback(fake)

    query = torch.empty(2, 1, 32)
    key = torch.empty(2, 17, 32)
    value = torch.empty(2, 17, 32)
    result = fake.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        num_query_heads=4,
        num_key_value_heads=4,
        input_layout="BSH",
        softmax_scale=0.25,
    )

    assert result == ("incremental-output",)
    kind, args, kwargs = fake.calls.pop()
    assert kind == "incremental"
    assert args == (query, key, value)
    assert kwargs == {
        "num_heads": 4,
        "input_layout": "BSH",
        "pse_shift": None,
        "padding_mask": None,
        "scale_value": 0.25,
    }


@pytest.mark.parametrize(
    ("input_layout", "num_query_heads", "num_key_value_heads"),
    [("BNSD", 4, 4), ("BSH", 8, 4)],
)
def test_v2_fallback_rejects_unsupported_semantics(
    input_layout: str,
    num_query_heads: int,
    num_key_value_heads: int,
) -> None:
    fake = _FakeTorchNpu()
    install_npu_fused_infer_attention_v2_fallback(fake)

    with pytest.raises(NotImplementedError):
        fake.npu_fused_infer_attention_score_v2(
            torch.empty(1, 2, 8),
            torch.empty(1, 2, 8),
            torch.empty(1, 2, 8),
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            input_layout=input_layout,
        )


def test_v2_fallback_is_noop_when_native_symbol_exists() -> None:
    native = object()
    fake = _FakeTorchNpu()
    fake.npu_fused_infer_attention_score_v2 = native

    assert not install_npu_fused_infer_attention_v2_fallback(fake)
    assert fake.npu_fused_infer_attention_score_v2 is native
