"""Dataset-aware causal language-model loss alignment helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def dataset_labels_are_shifted(dataset_backend: str) -> bool:
    """Return whether a dataset backend already emits next-token labels."""
    if dataset_backend in (
        "megatron_indexed",
        "olmo3_numpy_fsl",
        "olmo3_numpy_packed",
        "olmo3_sft_numpy",
    ):
        return True
    raise ValueError(f"Unsupported dataset backend: {dataset_backend!r}.")


def align_causal_lm_loss_inputs(
    hidden_states_bsh: Tensor,
    labels: Tensor,
    *,
    labels_are_shifted: bool,
) -> tuple[Tensor, Tensor]:
    """Align decoder states and targets without shifting labels twice.

    Production OLMo3 datasets emit ``tokens=text[:-1]`` and
    ``labels=text[1:]``, so every hidden state already has a matching
    next-token target. The unshifted branch remains a generic tensor helper,
    but no production data backend selects it.
    """
    if hidden_states_bsh.dim() != 3 or labels.dim() != 2:
        raise ValueError(
            "hidden states and labels must have [batch, sequence, hidden] and "
            "[batch, sequence] shapes."
        )
    if hidden_states_bsh.shape[:2] != labels.shape:
        raise ValueError(
            "hidden states and labels must share batch and sequence dimensions; "
            f"got {tuple(hidden_states_bsh.shape)} and {tuple(labels.shape)}."
        )

    if labels_are_shifted:
        return hidden_states_bsh.contiguous(), labels.contiguous()
    if labels.shape[1] < 2:
        raise ValueError("Unshifted causal LM labels require sequence length >= 2.")
    return (
        hidden_states_bsh[:, :-1, :].contiguous(),
        labels[:, 1:].contiguous(),
    )


def align_sequence_parallel_causal_lm_loss_inputs(
    hidden_states_bsh: Tensor,
    labels: Tensor,
    *,
    tensor_parallel_size: int,
    labels_are_shifted: bool,
) -> tuple[Tensor, Tensor]:
    """Align SP-local hidden states with full CP-local labels.

    Megatron's sequence-parallel output projection consumes an equal-sized
    sequence shard on every TP rank and gathers those shards internally before
    producing vocabulary-parallel logits. Labels therefore remain full
    CP-local sequences here.

    For an unshifted dataset, keep every hidden-state shard the same length and
    shift the full labels left, masking the final target. This is equivalent to
    dropping the final hidden state after the output projection while
    preserving the equal first dimension required by the TP all-gather.
    """
    if hidden_states_bsh.dim() != 3 or labels.dim() != 2:
        raise ValueError(
            "SP hidden states and labels must have [batch, local_sequence, hidden] "
            "and [batch, full_sequence] shapes."
        )
    if isinstance(tensor_parallel_size, bool) or tensor_parallel_size < 2:
        raise ValueError(
            "Sequence-parallel loss alignment requires tensor_parallel_size >= 2."
        )
    if hidden_states_bsh.shape[0] != labels.shape[0]:
        raise ValueError(
            "SP hidden states and labels must share the batch dimension; "
            f"got {tuple(hidden_states_bsh.shape)} and {tuple(labels.shape)}."
        )
    expected_full_sequence = hidden_states_bsh.shape[1] * tensor_parallel_size
    if labels.shape[1] != expected_full_sequence:
        raise ValueError(
            "Full CP-local label length must equal SP-local hidden length times TP; "
            f"got labels={labels.shape[1]}, local_hidden={hidden_states_bsh.shape[1]}, "
            f"TP={tensor_parallel_size}."
        )

    if labels_are_shifted:
        return hidden_states_bsh.contiguous(), labels.contiguous()
    if labels.shape[1] < 2:
        raise ValueError("Unshifted causal LM labels require sequence length >= 2.")

    shifted_labels = torch.full_like(labels, -100)
    shifted_labels[:, :-1] = labels[:, 1:]
    return hidden_states_bsh.contiguous(), shifted_labels.contiguous()
