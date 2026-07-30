from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from megatron import olmo3_runtime_markers


def test_runtime_marker_is_rank_zero_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rank = 0
    distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_rank=lambda: rank,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(distributed=distributed),
    )

    olmo3_runtime_markers._EMITTED_RUNTIME_MARKERS.clear()
    olmo3_runtime_markers.emit_rank0_runtime_marker_once(
        "OLMO3_RUNTIME_TEST_ACTIVE",
        z=2,
        a=1,
    )
    olmo3_runtime_markers.emit_rank0_runtime_marker_once(
        "OLMO3_RUNTIME_TEST_ACTIVE",
        z=3,
    )
    assert capsys.readouterr().out == "OLMO3_RUNTIME_TEST_ACTIVE a=1 z=2\n"

    rank = 1
    olmo3_runtime_markers._EMITTED_RUNTIME_MARKERS.clear()
    olmo3_runtime_markers.emit_rank0_runtime_marker_once(
        "OLMO3_RUNTIME_NONZERO_TEST",
    )
    assert capsys.readouterr().out == ""
