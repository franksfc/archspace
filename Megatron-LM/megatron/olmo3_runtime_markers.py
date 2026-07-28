"""One-shot rank-zero evidence for OLMo3 production optimization paths."""

from __future__ import annotations

from typing import Any


_EMITTED_RUNTIME_MARKERS: set[str] = set()


def emit_rank0_runtime_marker_once(marker: str, **fields: Any) -> None:
    """Print one deterministic marker after a distributed runtime path is selected.

    Callers invoke this only after distributed initialization. Every process
    keeps its own one-shot set; only global rank zero writes to the job log.
    The helper performs no collective and does not synchronize any stream.
    """

    if marker in _EMITTED_RUNTIME_MARKERS:
        return
    _EMITTED_RUNTIME_MARKERS.add(marker)

    import torch

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    suffix = " ".join(
        f"{name}={fields[name]}" for name in sorted(fields)
    )
    print(f"{marker}{' ' + suffix if suffix else ''}", flush=True)
