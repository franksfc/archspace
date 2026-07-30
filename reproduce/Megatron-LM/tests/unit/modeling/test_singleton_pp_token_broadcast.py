"""Contracts for token-count synchronization across pipeline groups."""

from __future__ import annotations

import importlib

import pytest


torch = pytest.importorskip("torch")

finalize = importlib.import_module(  # noqa: E402
    "megatron.core.distributed.finalize_model_grads"
)


def test_singleton_pipeline_group_does_not_create_a_communicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    broadcasts: list[tuple[object, int, object]] = []
    monkeypatch.setattr(
        finalize.parallel_state,
        "get_pipeline_model_parallel_last_rank",
        lambda: 31,
    )
    monkeypatch.setattr(
        finalize.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: group,
    )
    monkeypatch.setattr(
        finalize.torch.distributed,
        "get_world_size",
        lambda *, group: 1,
    )
    monkeypatch.setattr(
        finalize.torch.distributed,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((tensor, src, group)),
    )

    token_count = torch.tensor(8192)
    copies = finalize._broadcast_num_tokens_across_pipeline(token_count)

    assert broadcasts == []
    assert len(copies) == 1
    assert copies[0].item() == 8192
    assert copies[0] is not token_count


def test_real_pipeline_group_still_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = object()
    broadcasts: list[tuple[object, int, object]] = []
    monkeypatch.setattr(
        finalize.parallel_state,
        "get_pipeline_model_parallel_last_rank",
        lambda: 7,
    )
    monkeypatch.setattr(
        finalize.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: group,
    )
    monkeypatch.setattr(
        finalize.torch.distributed,
        "get_world_size",
        lambda *, group: 2,
    )
    monkeypatch.setattr(
        finalize.torch.distributed,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((tensor, src, group)),
    )

    token_count = torch.tensor(8192)
    copies = finalize._broadcast_num_tokens_across_pipeline(token_count)

    assert broadcasts == [(token_count, 7, group)]
    assert copies[0].item() == 8192
