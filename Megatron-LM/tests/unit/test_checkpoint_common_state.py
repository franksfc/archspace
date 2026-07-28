"""Checkpoint common-state validation for CP loader-owner runtimes."""

from __future__ import annotations

from argparse import Namespace

from megatron.training.training import preprocess_common_state_dict


def _common_state(**runtime_args: object) -> dict[str, object]:
    return {
        "args": Namespace(
            dataset_backend="olmo3_sft_numpy",
            hidden_size=2048,
            seq_length=32768,
            **runtime_args,
        ),
        "checkpoint_version": 3.0,
        "iteration": 1,
    }


def test_cp_loader_owner_dataset_lengths_are_validation_only_differences() -> None:
    """Owner and non-owner ranks normalize to the same validation state."""

    owner = _common_state(
        rank=0,
        local_rank=0,
        dataset_train_len=32,
        dataset_valid_len=0,
    )
    non_owner = _common_state(rank=1, local_rank=1)

    assert preprocess_common_state_dict(owner) == preprocess_common_state_dict(
        non_owner
    )


def test_common_state_preprocessing_does_not_mutate_saved_state() -> None:
    """The rank-0 common.pt keeps its existing informational fields."""

    common_state = _common_state(
        rank=0,
        local_rank=0,
        dataset_train_len=32,
        dataset_valid_len=0,
    )

    normalized = preprocess_common_state_dict(common_state)

    assert vars(common_state["args"])["dataset_train_len"] == 32
    assert vars(common_state["args"])["dataset_valid_len"] == 0
    assert "dataset_train_len" not in normalized["args"]
    assert "dataset_valid_len" not in normalized["args"]
    assert normalized["args"]["hidden_size"] == 2048
    assert normalized["args"]["seq_length"] == 32768


def test_real_common_configuration_differences_remain_visible() -> None:
    """Only known rank-local metadata is ignored by validation."""

    left = preprocess_common_state_dict(
        _common_state(dataset_train_len=32, dataset_valid_len=0)
    )
    right = preprocess_common_state_dict(
        {
            "args": Namespace(hidden_size=4096, seq_length=32768),
            "checkpoint_version": 3.0,
            "iteration": 1,
        }
    )

    assert left != right
    assert left["args"]["hidden_size"] == 2048
    assert right["args"]["hidden_size"] == 4096


def test_all_rank_dataset_backend_still_validates_dataset_lengths() -> None:
    """Stage-2/other all-rank providers must not mask real length drift."""

    common_state = _common_state(
        dataset_train_len=32,
        dataset_valid_len=0,
    )
    common_state["args"].dataset_backend = "olmo3_numpy_fsl"

    normalized = preprocess_common_state_dict(common_state)

    assert normalized["args"]["dataset_train_len"] == 32
    assert normalized["args"]["dataset_valid_len"] == 0
