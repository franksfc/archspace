"""End-to-end CPU regression for the prepared Stage-4 cache contract."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sft_preparation_and_training_reader_share_one_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from olmo3_pipeline.data_pipeline import sft_packed_work_dir
    from runtime.olmo3_sft_dataset import (
        Olmo3SFTNumpyPackedDataset,
        describe_olmo3_sft_cache,
        prepare_olmo3_sft_cache,
    )

    converted = tmp_path / "converted"
    converted.mkdir()
    np.asarray(
        [11, 12, 100_257, 21, 22, 23, 100_257],
        dtype=np.uint32,
    ).tofile(converted / "token_ids_part_0000.npy")
    np.asarray([0, 1, 1, 0, 1, 1, 1], dtype=np.uint8).tofile(
        converted / "labels_mask_part_0000.npy"
    )
    (converted / "token_ids_part_0000.csv.gz").write_bytes(b"metadata")
    (converted / "dataset_statistics.json").write_text("{}\n", encoding="utf-8")
    (converted / "tokenizer").mkdir()
    (converted / "tokenizer" / "tokenizer_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    work_root = tmp_path / "sft-work"
    packed_work = sft_packed_work_dir(work_root)
    cache_root = prepare_olmo3_sft_cache(converted, packed_work, workers=1)
    description = describe_olmo3_sft_cache(converted, packed_work)
    dataset = Olmo3SFTNumpyPackedDataset(
        converted,
        packed_work,
        num_samples=1,
        global_batch_size=1,
    )

    assert Path(description["cache_root"]) == cache_root
    assert cache_root.parent == packed_work
    assert description["base_instances"] == dataset.base_instances == 1
    assert description["fingerprint"] == cache_root.name.removeprefix(
        "olmo3-sft-packed-"
    )
    sample = dataset[0]
    assert tuple(sample["tokens"].shape) == (32_768,)
    assert tuple(sample["labels"].shape) == (32_768,)
    assert int(sample["loss_mask"].sum().item()) == 5

    first_fingerprint = str(description["fingerprint"])
    import runtime.olmo3_sft_dataset as sft_runtime

    monkeypatch.setattr(
        sft_runtime,
        "_file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("runtime cache lookup must not re-hash source content")
        ),
    )
    runtime_dataset = Olmo3SFTNumpyPackedDataset(
        converted,
        packed_work,
        num_samples=1,
        global_batch_size=1,
        expected_fingerprint=first_fingerprint,
    )
    assert runtime_dataset.base_instances == 1
    monkeypatch.undo()

    np.asarray([1, 0, 1, 0, 1, 1, 1], dtype=np.uint8).tofile(
        converted / "labels_mask_part_0000.npy"
    )
    replacement_root = prepare_olmo3_sft_cache(
        converted,
        packed_work,
        workers=1,
    )
    assert replacement_root != cache_root
    assert replacement_root.name != f"olmo3-sft-packed-{first_fingerprint}"
