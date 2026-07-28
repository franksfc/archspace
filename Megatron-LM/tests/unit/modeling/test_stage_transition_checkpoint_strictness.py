from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.core.dist_checkpointing.mapping import CheckpointingException
from megatron.training import checkpointing
from megatron.training.checkpointing import (
    _OLMO3_STAGE_TRANSITION_RESET_SHARDED_KEYS,
    _validate_olmo3_stage_transition_sharded_mismatches,
)


def test_stage_transition_accepts_only_rng_and_rerun_missing_keys() -> None:
    state = {"model": object(), "optimizer": object()}
    result = _validate_olmo3_stage_transition_sharded_mismatches(
        (
            state,
            {"rng_state", "rerun_state_machine_state"},
            set(),
        )
    )

    assert result is state
    assert _OLMO3_STAGE_TRANSITION_RESET_SHARDED_KEYS == {
        "rng_state",
        "rerun_state_machine_state",
    }


@pytest.mark.parametrize(
    ("missing", "unexpected"),
    (
        ({"optimizer.state.exp_avg", "rng_state"}, set()),
        ({"model.decoder.weight"}, set()),
        ({"rng_state"}, {"optimizer.state.exp_avg_sq"}),
        (set(), {"model.unexpected_weight"}),
    ),
)
def test_stage_transition_rejects_every_other_sharded_mismatch(
    missing: set[str],
    unexpected: set[str],
) -> None:
    with pytest.raises(
        CheckpointingException,
        match="outside the intentional RNG/rerun reset contract",
    ):
        _validate_olmo3_stage_transition_sharded_mismatches(
            ({"model": object()}, missing, unexpected)
        )


def test_stage_transition_requires_return_all_validation_result() -> None:
    with pytest.raises(
        CheckpointingException,
        match="state/missing/unexpected strict-validation tuple",
    ):
        _validate_olmo3_stage_transition_sharded_mismatches({"model": object()})


@pytest.mark.parametrize(
    ("stage_transition", "expected_strictness"),
    (
        (False, "raise_all"),
        (True, "return_all"),
    ),
)
def test_distributed_loader_uses_scoped_transition_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_transition: bool,
    expected_strictness: str,
) -> None:
    args = SimpleNamespace(
        ckpt_fully_parallel_load=False,
        dist_ckpt_strictness="raise_all",
        olmo3_stage_transition=stage_transition,
    )
    seen: dict[str, object] = {}
    loaded = {"model": object(), "optimizer": object()}

    monkeypatch.setattr(
        checkpointing,
        "get_default_load_sharded_strategy",
        lambda _checkpoint_name: object(),
    )

    def fake_load(
        _state: object,
        _checkpoint_name: object,
        _strategy: object,
        *,
        strict: str,
    ) -> object:
        seen["strict"] = strict
        if stage_transition:
            return (
                loaded,
                {"rng_state", "rerun_state_machine_state"},
                set(),
            )
        return loaded

    monkeypatch.setattr(checkpointing.dist_checkpointing, "load", fake_load)

    state, _name, _release, kind = checkpointing._load_global_dist_base_checkpoint(
        tmp_path,
        args,
        rank0=False,
        sharded_state_dict={"model": object()},
        iteration=12,
        release=False,
    )

    assert state is loaded
    assert seen["strict"] == expected_strictness
    assert kind is checkpointing.CheckpointType.GLOBAL


def _tracker(root: Path, value: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "latest_checkpointed_iteration.txt"
    path.write_text(f"{value}\n", encoding="utf-8")
    return path


def test_zero_tracker_accepts_real_distributed_bootstrap_checkpoint(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path, "0")
    iteration = tmp_path / "iter_0000000"
    iteration.mkdir()
    (iteration / "metadata.json").write_text(
        json.dumps(
            {
                "sharded_backend": "torch_dist",
                "sharded_backend_version": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert checkpointing.read_metadata(tracker) == (0, False)


@pytest.mark.parametrize("value", ("0", "-1", "-999"))
def test_nonpositive_tracker_without_bootstrap_checkpoint_is_rejected(
    tmp_path: Path,
    value: str,
) -> None:
    tracker = _tracker(tmp_path / value.replace("-", "negative-"), value)

    with pytest.raises(AssertionError, match="error parsing metadata file"):
        checkpointing.read_metadata(tracker)


def test_release_tracker_remains_valid(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, "release")

    assert checkpointing.read_metadata(tracker) == (0, True)


def test_invalid_tracker_text_remains_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _tracker(tmp_path, "not-an-iteration")
    monkeypatch.setattr(checkpointing, "print_rank_0", lambda *_args: None)

    with pytest.raises(SystemExit):
        checkpointing.read_metadata(tracker)
