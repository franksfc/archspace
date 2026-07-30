"""Dataset, microbatch, and sequence-parallel target-alignment tests."""

from __future__ import annotations
# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from modeling.lm_loss_utils import (
    align_causal_lm_loss_inputs,
    align_sequence_parallel_causal_lm_loss_inputs,
    dataset_labels_are_shifted,
)


@pytest.mark.parametrize(
    "backend",
    (
        "megatron_indexed",
        "olmo3_numpy_fsl",
        "olmo3_numpy_packed",
        "olmo3_sft_numpy",
    ),
)
def test_training_backends_emit_next_token_labels(backend: str) -> None:
    assert dataset_labels_are_shifted(backend)


@pytest.mark.parametrize("batch_size", (1, 2, 4))
def test_preshifted_labels_are_not_shifted_twice(batch_size: int) -> None:
    hidden = torch.randn(batch_size, 8, 16)
    labels = torch.arange(batch_size * 8).reshape(batch_size, 8)
    actual_hidden, actual_labels = align_causal_lm_loss_inputs(
        hidden, labels, labels_are_shifted=True
    )
    torch.testing.assert_close(actual_hidden, hidden)
    torch.testing.assert_close(actual_labels, labels)


def test_generic_unshifted_labels_receive_one_model_side_shift() -> None:
    hidden = torch.randn(2, 8, 16)
    labels = torch.arange(16).reshape(2, 8)
    actual_hidden, actual_labels = align_causal_lm_loss_inputs(
        hidden, labels, labels_are_shifted=False
    )
    torch.testing.assert_close(actual_hidden, hidden[:, :-1])
    torch.testing.assert_close(actual_labels, labels[:, 1:])


@pytest.mark.parametrize("tensor_parallel_size", (2, 4))
@pytest.mark.parametrize("batch_size", (1, 2, 4))
def test_sp_alignment_keeps_full_cp_local_labels(
    tensor_parallel_size: int, batch_size: int
) -> None:
    hidden = torch.randn(batch_size, 4, 16)
    labels = torch.arange(batch_size * 4 * tensor_parallel_size).reshape(
        batch_size, 4 * tensor_parallel_size
    )
    actual_hidden, actual_labels = align_sequence_parallel_causal_lm_loss_inputs(
        hidden,
        labels,
        tensor_parallel_size=tensor_parallel_size,
        labels_are_shifted=True,
    )
    torch.testing.assert_close(actual_hidden, hidden)
    torch.testing.assert_close(actual_labels, labels)


def test_sp_unshifted_labels_mask_only_the_last_target() -> None:
    hidden = torch.randn(2, 4, 16)
    labels = torch.arange(16).reshape(2, 8)
    _, shifted = align_sequence_parallel_causal_lm_loss_inputs(
        hidden,
        labels,
        tensor_parallel_size=2,
        labels_are_shifted=False,
    )
    torch.testing.assert_close(shifted[:, :-1], labels[:, 1:])
    assert torch.equal(shifted[:, -1], torch.full((2,), -100))
