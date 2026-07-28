"""Backend-independent OLMo3 sliding-window helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def olmo3_te_window(sliding_window: int) -> tuple[int, int]:
    """Map an OLMo3 visible-token count to TE/CANN's inclusive boundaries."""
    if sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1, got {sliding_window}.")
    # OLMo3 window W allows k > q-W, i.e. current token plus W-1 past tokens.
    return sliding_window - 1, 0


def build_olmo3_sliding_causal_mask(
    query_length: int,
    key_length: int,
    sliding_window: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build the exact right-aligned OLMo3 mask (``True`` means masked)."""
    if query_length < 1 or key_length < 1:
        raise ValueError("query_length and key_length must both be >= 1.")
    left_window, _ = olmo3_te_window(sliding_window)
    query_positions = (
        torch.arange(query_length, device=device) + key_length - query_length
    ).unsqueeze(1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    future = key_positions > query_positions
    too_old = key_positions < (query_positions - left_window)
    return future | too_old
