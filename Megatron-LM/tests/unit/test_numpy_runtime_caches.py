"""End-to-end CPU regressions for Stage-2 and Stage-3 prepared caches."""

from __future__ import annotations

from pathlib import Path
import hashlib

import pytest


def _source_contract(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime_modules():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from runtime.olmo3_packed_dataset import (
        Olmo3NumpyFSLDataset,
        Olmo3NumpyPackedFSLDataset,
        prepare_olmo3_long_context_cache,
        prepare_olmo3_midtraining_cache,
    )

    return (
        np,
        Olmo3NumpyFSLDataset,
        Olmo3NumpyPackedFSLDataset,
        prepare_olmo3_long_context_cache,
        prepare_olmo3_midtraining_cache,
    )


def test_stage2_prepared_cache_is_consumed_without_runtime_rebuild(
    tmp_path: Path,
) -> None:
    (
        np,
        Olmo3NumpyFSLDataset,
        _,
        _,
        prepare_olmo3_midtraining_cache,
    ) = _runtime_modules()
    source = tmp_path / "midtraining.npy"
    np.arange(8_192, dtype=np.uint32).tofile(source)
    work_dir = tmp_path / "midtraining-work"

    source_contract = _source_contract(source)
    cache_root = prepare_olmo3_midtraining_cache(
        [source],
        work_dir,
        source_contract_sha256=source_contract,
    )
    dataset = Olmo3NumpyFSLDataset(
        [source],
        work_dir,
        num_samples=1,
        global_batch_size=1,
        source_contract_sha256=source_contract,
    )
    sample = dataset[0]

    assert dataset.cache_root == cache_root
    assert dataset.base_instances == 1
    assert tuple(sample["tokens"].shape) == (8_192,)
    assert tuple(sample["labels"].shape) == (8_192,)
    assert int(sample["loss_mask"].sum().item()) == 8_191


def test_stage3_prepared_cache_preserves_documents_and_padding(
    tmp_path: Path,
) -> None:
    (
        np,
        _,
        Olmo3NumpyPackedFSLDataset,
        prepare_olmo3_long_context_cache,
        _,
    ) = _runtime_modules()
    sources = []
    for source_id in range(8):
        source = tmp_path / f"long-context-{source_id:02d}.npy"
        np.asarray([source_id + 1, 100_257], dtype=np.uint32).tofile(source)
        sources.append(source)
    work_dir = tmp_path / "long-context-work"

    cache_root = prepare_olmo3_long_context_cache(
        sources,
        work_dir,
        source_contract_sha256=_source_contract(*sources),
        workers=1,
    )
    dataset = Olmo3NumpyPackedFSLDataset(
        sources,
        work_dir,
        num_samples=1,
        global_batch_size=1,
        source_contract_sha256=_source_contract(*sources),
    )
    sample = dataset[0]

    assert dataset.cache_root == cache_root
    assert dataset.base_instances == 1
    assert tuple(sample["tokens"].shape) == (65_536,)
    assert tuple(sample["labels"].shape) == (65_536,)
    assert int(sample["loss_mask"].sum().item()) == 15
    assert int(sample["document_ids"].max().item()) == 8
    assert int(sample["position_ids"][16].item()) == 0


def test_stage2_cache_identity_is_bound_to_content_contract(
    tmp_path: Path,
) -> None:
    (
        np,
        _,
        _,
        _,
        prepare_olmo3_midtraining_cache,
    ) = _runtime_modules()
    source = tmp_path / "midtraining.npy"
    np.zeros(8_192, dtype=np.uint32).tofile(source)
    work_dir = tmp_path / "work"
    first_contract = _source_contract(source)
    first_root = prepare_olmo3_midtraining_cache(
        [source],
        work_dir,
        source_contract_sha256=first_contract,
    )

    np.ones(8_192, dtype=np.uint32).tofile(source)
    second_contract = _source_contract(source)
    second_root = prepare_olmo3_midtraining_cache(
        [source],
        work_dir,
        source_contract_sha256=second_contract,
    )

    assert first_contract != second_contract
    assert first_root != second_root


def test_stage2_content_contract_survives_mount_remapping(
    tmp_path: Path,
) -> None:
    (
        np,
        Olmo3NumpyFSLDataset,
        _,
        _,
        prepare_olmo3_midtraining_cache,
    ) = _runtime_modules()
    preparation_source = tmp_path / "prepare-mount" / "tokens.npy"
    runtime_source = tmp_path / "runtime-mount" / "tokens.npy"
    preparation_source.parent.mkdir()
    runtime_source.parent.mkdir()
    values = np.arange(8_192, dtype=np.uint32)
    values.tofile(preparation_source)
    values.tofile(runtime_source)
    contract = _source_contract(preparation_source)
    work_dir = tmp_path / "work"
    cache_root = prepare_olmo3_midtraining_cache(
        [preparation_source],
        work_dir,
        source_contract_sha256=contract,
    )

    dataset = Olmo3NumpyFSLDataset(
        [runtime_source],
        work_dir,
        num_samples=1,
        global_batch_size=1,
        source_contract_sha256=contract,
    )

    assert dataset.cache_root == cache_root
    assert int(dataset[0]["tokens"][17]) == 17
