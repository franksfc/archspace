"""Official-format packed SFT data for the OLMo3 MindSpeed runtime.

The companion Open-Instruct conversion script writes raw token arrays,
assistant-token label masks, and document metadata using this layout::

    token_ids_part_XXXX.npy
    labels_mask_part_XXXX.npy
    token_ids_part_XXXX.csv.gz
    tokenizer/
    dataset_statistics.json

OLMo-core applies OBFD independently to every source array by default.  This
module mirrors that contract, including EOS-delimited documents, truncation,
PCG64 epoch shuffling, shifted labels, and shifted assistant-only label masks.
The expensive EOS scan and OBFD packing are performed explicitly before a
large job; training fails closed if the immutable cache is absent or stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from runtime.olmo3_packed_dataset import (
    _atomic_save_numpy,
    _atomic_write_json,
    _prepare_source_group,
)


OLMO3_SFT_SEQUENCE_LENGTH = 32768
OLMO3_SFT_DATA_SEED = 34521
OLMO3_SFT_SHUFFLE_SEED = 42
OLMO3_SFT_EOS_TOKEN_ID = 100257
OLMO3_SFT_PAD_TOKEN_ID = 100277
OLMO3_SFT_TOKEN_DTYPE = np.dtype(np.uint32)
OLMO3_SFT_SOURCE_GROUP_SIZE = 1
OLMO3_SFT_CACHE_ALGORITHM = "olmo_core_obfd_truncate_sft_v1"


def discover_olmo3_sft_sources(
    data_dir: str | Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Discover and validate paired Open-Instruct token and label-mask files."""

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OLMo3 SFT converted-data directory is missing: {root}")
    token_paths = tuple(sorted(root.glob("token_ids_part_*.npy")))
    if not token_paths:
        raise FileNotFoundError(f"No token_ids_part_*.npy files found in {root}")

    label_paths: list[Path] = []
    expected_label_names: set[str] = set()
    for token_path in token_paths:
        suffix = token_path.name.removeprefix("token_ids")
        label_name = f"labels_mask{suffix}"
        label_path = root / label_name
        metadata_path = token_path.with_suffix(".csv.gz")
        for path in (token_path, label_path, metadata_path):
            if path.is_symlink():
                raise RuntimeError(f"OLMo3 SFT converted data rejects symlinks: {path}")
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing assistant label mask for {token_path.name}: {label_path}"
            )
        if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Missing Open-Instruct document metadata for {token_path.name}: "
                f"{metadata_path}"
            )
        label_paths.append(label_path)
        expected_label_names.add(label_name)

    actual_label_names = {
        path.name for path in root.glob("labels_mask_part_*.npy")
    }
    if actual_label_names != expected_label_names:
        raise RuntimeError(
            "Token/label-mask inventories differ: "
            f"missing={sorted(expected_label_names - actual_label_names)}, "
            f"unexpected={sorted(actual_label_names - expected_label_names)}"
        )
    if not (root / "dataset_statistics.json").is_file():
        raise FileNotFoundError(
            f"Missing Open-Instruct dataset_statistics.json in {root}"
        )
    if (root / "dataset_statistics.json").is_symlink():
        raise RuntimeError("OLMo3 SFT dataset_statistics.json must not be a symlink")
    tokenizer_root = root / "tokenizer"
    if not (tokenizer_root / "tokenizer_config.json").is_file():
        raise FileNotFoundError(
            f"Missing frozen SFT tokenizer snapshot in {tokenizer_root}"
        )
    if tokenizer_root.is_symlink():
        raise RuntimeError("OLMo3 SFT tokenizer snapshot must not be a symlink")

    for token_path, label_path in zip(token_paths, label_paths):
        token_bytes = token_path.stat().st_size
        label_bytes = label_path.stat().st_size
        if token_bytes <= 0 or token_bytes % OLMO3_SFT_TOKEN_DTYPE.itemsize:
            raise RuntimeError(
                f"Invalid raw uint32 token file size: {token_path}={token_bytes}"
            )
        token_count = token_bytes // OLMO3_SFT_TOKEN_DTYPE.itemsize
        if label_bytes != token_count:
            raise RuntimeError(
                "Assistant label mask must contain one byte per token: "
                f"{label_path}={label_bytes}, tokens={token_count}"
            )
    return token_paths, tuple(label_paths)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sft_contract_fingerprint(
    token_paths: Sequence[Path],
    label_paths: Sequence[Path],
    *,
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int,
    dtype: np.dtype[Any],
) -> str:
    """Hash every converted training input and all packing semantics.

    Preparation pays this I/O cost once. Distributed training supplies the
    frozen expected fingerprint and therefore never re-hashes the corpus.
    """

    if len(token_paths) != len(label_paths) or not token_paths:
        raise ValueError("SFT token and label-mask paths must be paired and non-empty.")
    digest = hashlib.sha256()
    digest.update(
        (
            f"algorithm={OLMO3_SFT_CACHE_ALGORITHM},"
            f"sequence_length={sequence_length},"
            f"eos_token_id={eos_token_id},"
            f"pad_token_id={pad_token_id},"
            f"dtype={dtype.str},"
            f"source_group_size={OLMO3_SFT_SOURCE_GROUP_SIZE},"
        ).encode()
    )
    data_root = token_paths[0].parent
    for token_path, label_path in zip(token_paths, label_paths):
        metadata_path = token_path.with_suffix(".csv.gz")
        for kind, path in (
            ("tokens", token_path),
            ("labels", label_path),
            ("metadata", metadata_path),
        ):
            digest.update(
                (
                    f"{kind}_path={path.relative_to(data_root).as_posix()},"
                    f"{kind}_size={path.stat().st_size},"
                    f"{kind}_sha256={_file_sha256(path)},"
                ).encode()
            )
    statistics_path = data_root / "dataset_statistics.json"
    digest.update(
        (
            "statistics_path=dataset_statistics.json,"
            f"statistics_size={statistics_path.stat().st_size},"
            f"statistics_sha256={_file_sha256(statistics_path)},"
        ).encode()
    )
    tokenizer_root = data_root / "tokenizer"
    tokenizer_entries = tuple(sorted(tokenizer_root.rglob("*")))
    symlinks = [path for path in tokenizer_entries if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"SFT tokenizer snapshot rejects symlinks: {symlinks[0]}")
    tokenizer_files = tuple(path for path in tokenizer_entries if path.is_file())
    if not tokenizer_files:
        raise FileNotFoundError(
            f"Frozen SFT tokenizer snapshot contains no files: {tokenizer_root}"
        )
    for path in tokenizer_files:
        digest.update(
            (
                f"tokenizer_path={path.relative_to(tokenizer_root).as_posix()},"
                f"tokenizer_size={path.stat().st_size},"
                f"tokenizer_sha256={_file_sha256(path)},"
            ).encode()
        )
    return digest.hexdigest()


def _sft_cache_root(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int,
    dtype: str | np.dtype[Any],
    expected_fingerprint: str | None = None,
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...], str]:
    token_paths, label_paths = discover_olmo3_sft_sources(data_dir)
    resolved_dtype = np.dtype(dtype)
    if expected_fingerprint is None:
        fingerprint = _sft_contract_fingerprint(
            token_paths,
            label_paths,
            sequence_length=sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dtype=resolved_dtype,
        )
    else:
        if (
            not isinstance(expected_fingerprint, str)
            or len(expected_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_fingerprint
            )
        ):
            raise ValueError(
                "expected_fingerprint must be a lowercase 64-character SHA256"
            )
        fingerprint = expected_fingerprint
    cache_root = (
        Path(work_dir).expanduser().resolve() / f"olmo3-sft-packed-{fingerprint}"
    )
    return cache_root, token_paths, label_paths, fingerprint


def _validate_sft_cache_source_sizes(
    cache_root: Path,
    token_paths: Sequence[Path],
    label_paths: Sequence[Path],
) -> None:
    token_sizes = np.load(
        cache_root / "source_file_sizes.npy", mmap_mode="r", allow_pickle=False
    )
    label_sizes = np.load(
        cache_root / "label_mask_file_sizes.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    actual_token_sizes = np.asarray(
        [path.stat().st_size for path in token_paths], dtype=np.uint64
    )
    actual_label_sizes = np.asarray(
        [path.stat().st_size for path in label_paths], dtype=np.uint64
    )
    if not np.array_equal(token_sizes, actual_token_sizes):
        raise RuntimeError("OLMo3 SFT token sources changed after cache preparation.")
    if not np.array_equal(label_sizes, actual_label_sizes):
        raise RuntimeError(
            "OLMo3 SFT assistant label masks changed after cache preparation."
        )


def prepare_olmo3_sft_cache(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    sequence_length: int = OLMO3_SFT_SEQUENCE_LENGTH,
    eos_token_id: int = OLMO3_SFT_EOS_TOKEN_ID,
    pad_token_id: int = OLMO3_SFT_PAD_TOKEN_ID,
    dtype: str | np.dtype[Any] = OLMO3_SFT_TOKEN_DTYPE,
    workers: int = 1,
) -> Path:
    """Prepare an immutable per-source OLMo-core-compatible SFT OBFD cache."""

    if sequence_length != OLMO3_SFT_SEQUENCE_LENGTH:
        raise ValueError("Strict OLMo3 SFT requires sequence_length=32768.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype != OLMO3_SFT_TOKEN_DTYPE:
        raise ValueError("The OLMo3 tokenizer requires raw uint32 SFT token IDs.")

    cache_root, token_paths, label_paths, fingerprint = _sft_cache_root(
        data_dir,
        work_dir,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        dtype=resolved_dtype,
    )
    complete_path = cache_root / "complete.json"
    if complete_path.is_file():
        _validate_sft_cache_source_sizes(cache_root, token_paths, label_paths)
        describe_olmo3_sft_cache(
            data_dir,
            work_dir,
            expected_fingerprint=fingerprint,
            sequence_length=sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dtype=resolved_dtype,
        )
        return cache_root

    token_sizes = np.asarray(
        [path.stat().st_size for path in token_paths], dtype=np.uint64
    )
    label_sizes = np.asarray(
        [path.stat().st_size for path in label_paths], dtype=np.uint64
    )

    # OLMo-core's SFT recipe leaves source_group_size at its default of one.
    # Running one source per worker also bounds preparation memory.
    if workers == 1:
        groups = [
            _prepare_source_group(
                source_id,
                (source_path,),
                (source_id,),
                (int(token_sizes[source_id]),),
                cache_root,
                sequence_length=sequence_length,
                eos_token_id=eos_token_id,
                dtype=resolved_dtype,
            )
            for source_id, source_path in enumerate(token_paths)
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        groups_by_id: list[dict[str, Any] | None] = [None] * len(token_paths)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _prepare_source_group,
                    source_id,
                    (source_path,),
                    (source_id,),
                    (int(token_sizes[source_id]),),
                    cache_root,
                    sequence_length=sequence_length,
                    eos_token_id=eos_token_id,
                    dtype=resolved_dtype,
                ): source_id
                for source_id, source_path in enumerate(token_paths)
            }
            for future in as_completed(futures):
                source_id = futures[future]
                groups_by_id[source_id] = future.result()
        if any(group is None for group in groups_by_id):
            raise RuntimeError("OLMo3 SFT preparation did not produce every source group.")
        groups = [group for group in groups_by_id if group is not None]

    instance_counts = np.asarray(
        [int(group["instances"]) for group in groups], dtype=np.uint64
    )
    group_offsets = np.zeros(len(groups) + 1, dtype=np.uint64)
    group_offsets[1:] = np.cumsum(instance_counts, dtype=np.uint64)
    if int(group_offsets[-1]) < 1:
        raise RuntimeError("OLMo3 SFT sources contain no EOS-delimited instances.")
    if int(group_offsets[-1]) >= int(np.iinfo(np.uint32).max):
        raise RuntimeError("OLMo3 SFT packed order exceeds uint32.")

    _atomic_save_numpy(cache_root / "group_offsets.npy", group_offsets)
    _atomic_save_numpy(cache_root / "source_file_sizes.npy", token_sizes)
    _atomic_save_numpy(cache_root / "label_mask_file_sizes.npy", label_sizes)
    statistics_path = token_paths[0].parent / "dataset_statistics.json"
    _atomic_write_json(
        complete_path,
        {
            "algorithm": OLMO3_SFT_CACHE_ALGORITHM,
            "assistant_only_loss": True,
            "data_seed": OLMO3_SFT_DATA_SEED,
            "dtype": resolved_dtype.str,
            "eos_token_id": eos_token_id,
            "fingerprint": fingerprint,
            "groups": groups,
            "instances": int(group_offsets[-1]),
            "label_mask_paths": [str(path) for path in label_paths],
            "label_mask_names": [path.name for path in label_paths],
            "label_mask_shift": "olmo_core_get_labels_v1",
            "pad_token_id": pad_token_id,
            "sequence_length": sequence_length,
            "source_count": len(token_paths),
            "source_group_size": OLMO3_SFT_SOURCE_GROUP_SIZE,
            "source_paths": [str(path) for path in token_paths],
            "source_names": [path.name for path in token_paths],
            "statistics_sha256": _file_sha256(statistics_path),
        },
    )
    return cache_root


class Olmo3SFTNumpyPackedDataset(Dataset[dict[str, Tensor]]):
    """Read a prepared packed SFT cache with assistant-only next-token loss."""

    official_olmo3_order = True

    def __init__(
        self,
        data_dir: str | Path,
        work_dir: str | Path,
        *,
        num_samples: int | None,
        global_batch_size: int,
        expected_fingerprint: str | None = None,
        data_seed: int = OLMO3_SFT_DATA_SEED,
        sequence_length: int = OLMO3_SFT_SEQUENCE_LENGTH,
        eos_token_id: int = OLMO3_SFT_EOS_TOKEN_ID,
        pad_token_id: int = OLMO3_SFT_PAD_TOKEN_ID,
        dtype: str | np.dtype[Any] = OLMO3_SFT_TOKEN_DTYPE,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.data_seed = int(data_seed)
        self.dtype = np.dtype(dtype)
        (
            self.cache_root,
            self.source_paths,
            self.label_mask_paths,
            fingerprint,
        ) = _sft_cache_root(
            data_dir,
            work_dir,
            sequence_length=self.sequence_length,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            dtype=self.dtype,
            expected_fingerprint=expected_fingerprint,
        )
        complete_path = self.cache_root / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared OLMo3 SFT cache {complete_path}. "
                "Run runtime.olmo3_sft_dataset prepare before submission."
            )
        metadata = json.loads(complete_path.read_text(encoding="utf-8"))
        expected = {
            "algorithm": OLMO3_SFT_CACHE_ALGORITHM,
            "assistant_only_loss": True,
            "dtype": self.dtype.str,
            "eos_token_id": self.eos_token_id,
            "fingerprint": fingerprint,
            "label_mask_shift": "olmo_core_get_labels_v1",
            "pad_token_id": self.pad_token_id,
            "sequence_length": OLMO3_SFT_SEQUENCE_LENGTH,
            "source_count": len(self.source_paths),
            "source_group_size": OLMO3_SFT_SOURCE_GROUP_SIZE,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Non-official OLMo3 SFT cache settings: {mismatches}")
        if tuple(metadata.get("source_names", ())) != tuple(
            path.name for path in self.source_paths
        ):
            raise RuntimeError("OLMo3 SFT cache token-source order changed.")
        if tuple(metadata.get("label_mask_names", ())) != tuple(
            path.name for path in self.label_mask_paths
        ):
            raise RuntimeError("OLMo3 SFT cache label-mask order changed.")
        _validate_sft_cache_source_sizes(
            self.cache_root, self.source_paths, self.label_mask_paths
        )

        self.source_file_sizes = np.load(
            self.cache_root / "source_file_sizes.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.label_mask_file_sizes = np.load(
            self.cache_root / "label_mask_file_sizes.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.group_offsets = np.load(
            self.cache_root / "group_offsets.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            self.group_offsets.ndim != 1
            or self.group_offsets.size != len(self.source_paths) + 1
            or int(self.group_offsets[0]) != 0
            or not bool(np.all(self.group_offsets[1:] >= self.group_offsets[:-1]))
            or int(self.group_offsets[-1]) != int(metadata.get("instances", -1))
        ):
            raise RuntimeError("OLMo3 SFT group-offset index is corrupt.")

        self.base_instances = int(self.group_offsets[-1])
        self.global_batch_size = int(global_batch_size)
        self.num_samples = (
            self.base_instances if num_samples is None else int(num_samples)
        )
        if self.global_batch_size < 1 or self.num_samples < 1:
            raise ValueError("SFT global_batch_size and num_samples must be positive.")
        self.instances_per_epoch = (
            self.base_instances // self.global_batch_size
        ) * self.global_batch_size
        if self.instances_per_epoch < 1:
            raise ValueError("SFT corpus is smaller than one global batch.")

        self._source_arrays: dict[int, np.memmap[Any, Any]] = {}
        self._label_mask_arrays: dict[int, np.memmap[Any, Any]] = {}
        self._group_arrays: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def __len__(self) -> int:
        return self.num_samples

    @lru_cache(maxsize=2)
    def _epoch_order(self, epoch: int) -> np.ndarray:
        order = np.arange(self.base_instances, dtype=np.uint32)
        np.random.Generator(
            np.random.PCG64(self.data_seed + epoch + 1)
        ).shuffle(order)
        return order[: self.instances_per_epoch]

    def _resolve_instance(self, index: int) -> tuple[int, int]:
        epoch, epoch_index = divmod(index, self.instances_per_epoch)
        shuffled_index = int(self._epoch_order(epoch)[epoch_index])
        group_id = int(
            np.searchsorted(self.group_offsets, shuffled_index, side="right") - 1
        )
        if not 0 <= group_id < len(self.source_paths):
            raise IndexError(index)
        return group_id, shuffled_index - int(self.group_offsets[group_id])

    def _load_group(
        self, group_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._group_arrays.get(group_id)
        if arrays is None:
            group_root = self.cache_root / f"group-{group_id:04d}"
            arrays = (
                np.load(
                    group_root / "documents.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    group_root / "instance_offsets.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    group_root / "document_ids.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
            )
            documents, offsets, document_ids = arrays
            expected_instances = int(
                self.group_offsets[group_id + 1] - self.group_offsets[group_id]
            )
            if (
                documents.ndim != 2
                or documents.shape[1:] != (3,)
                or offsets.ndim != 1
                or offsets.size != expected_instances + 1
                or int(offsets[0]) != 0
                or not bool(np.all(offsets[1:] >= offsets[:-1]))
                or document_ids.ndim != 1
                or int(offsets[-1]) != document_ids.size
                or (
                    document_ids.size > 0
                    and int(document_ids.max()) >= documents.shape[0]
                )
            ):
                raise RuntimeError(f"OLMo3 SFT source group {group_id} is corrupt.")
            self._group_arrays[group_id] = arrays
        return arrays

    def _load_source(self, source_id: int) -> np.memmap[Any, Any]:
        source = self._source_arrays.get(source_id)
        if source is None:
            path = self.source_paths[source_id]
            if path.stat().st_size != int(self.source_file_sizes[source_id]):
                raise RuntimeError(f"OLMo3 SFT token source changed: {path}")
            source = np.memmap(path, mode="r", dtype=self.dtype)
            self._source_arrays[source_id] = source
        return source

    def _load_label_mask(self, source_id: int) -> np.memmap[Any, Any]:
        mask = self._label_mask_arrays.get(source_id)
        if mask is None:
            path = self.label_mask_paths[source_id]
            if path.stat().st_size != int(self.label_mask_file_sizes[source_id]):
                raise RuntimeError(f"OLMo3 SFT label mask changed: {path}")
            mask = np.memmap(path, mode="r", dtype=np.bool_)
            self._label_mask_arrays[source_id] = mask
        return mask

    def __getitem__(self, index: int | None) -> dict[str, Tensor]:
        padding_sample = index is None
        index = 0 if padding_sample else int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        group_id, instance_id = self._resolve_instance(index)
        documents, offsets, document_ids = self._load_group(group_id)
        begin, end = int(offsets[instance_id]), int(offsets[instance_id + 1])
        instance_document_ids = document_ids[begin:end]

        tokens_numpy = np.full(
            self.sequence_length, self.pad_token_id, dtype=self.dtype
        )
        label_mask_numpy = np.zeros(self.sequence_length, dtype=np.bool_)
        document_lengths: list[int] = []
        cursor = 0
        for document_id in instance_document_ids:
            source_id, start, stop = (
                int(value) for value in documents[int(document_id)]
            )
            length = stop - start
            next_cursor = cursor + length
            if next_cursor > self.sequence_length:
                raise RuntimeError("Prepared OLMo3 SFT instance exceeds sequence length.")
            tokens_numpy[cursor:next_cursor] = self._load_source(source_id)[
                start:stop
            ]
            label_mask_numpy[cursor:next_cursor] = self._load_label_mask(
                source_id
            )[start:stop]
            document_lengths.append(length)
            cursor = next_cursor

        real_tokens = cursor
        padding_tokens = self.sequence_length - real_tokens
        tokens = torch.from_numpy(
            tokens_numpy.astype(np.int64, copy=False)
        ).long()
        labels = torch.empty_like(tokens)
        labels[:-1] = tokens[1:]
        labels[-1] = self.pad_token_id

        # This is OLMo-core data.utils.get_labels(): apply the assistant mask
        # to token-aligned labels first, then shift both labels and mask left.
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        if self.sequence_length > 1:
            loss_mask[:-1] = torch.from_numpy(
                label_mask_numpy[1:].astype(np.float32, copy=False)
            )
        if padding_sample:
            loss_mask.zero_()

        lengths_with_padding = list(document_lengths)
        if padding_tokens:
            lengths_with_padding.append(padding_tokens)
        lengths = torch.tensor(lengths_with_padding, dtype=torch.long)
        if lengths.numel() < 1 or int(lengths.sum()) != self.sequence_length:
            raise RuntimeError("OLMo3 SFT document lengths do not cover the instance.")
        packed_document_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), dtype=torch.long), lengths
        )
        starts = torch.cumsum(lengths, dim=0) - lengths
        position_ids = torch.arange(self.sequence_length, dtype=torch.long)
        position_ids -= torch.repeat_interleave(starts, lengths)

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "document_ids": packed_document_ids,
        }


def describe_olmo3_sft_cache(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    sequence_length: int = OLMO3_SFT_SEQUENCE_LENGTH,
    eos_token_id: int = OLMO3_SFT_EOS_TOKEN_ID,
    pad_token_id: int = OLMO3_SFT_PAD_TOKEN_ID,
    dtype: str | np.dtype[Any] = OLMO3_SFT_TOKEN_DTYPE,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    cache_root, token_paths, label_paths, fingerprint = _sft_cache_root(
        data_dir,
        work_dir,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        dtype=dtype,
        expected_fingerprint=expected_fingerprint,
    )
    complete_path = cache_root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"Prepared SFT cache is missing: {complete_path}")
    metadata = json.loads(complete_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        raise RuntimeError("Prepared SFT cache fingerprint mismatch.")
    _validate_sft_cache_source_sizes(cache_root, token_paths, label_paths)
    dataset = Olmo3SFTNumpyPackedDataset(
        data_dir,
        work_dir,
        num_samples=1,
        global_batch_size=1,
        expected_fingerprint=fingerprint,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        dtype=dtype,
    )
    for group_id in range(len(dataset.source_paths)):
        dataset._load_group(group_id)
    return {
        "base_instances": dataset.base_instances,
        "cache_root": str(cache_root),
        "fingerprint": fingerprint,
        "sequence_length": dataset.sequence_length,
        "source_count": len(dataset.source_paths),
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "describe"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, required=True)
        subparser.add_argument("--work-dir", type=Path, required=True)
        subparser.add_argument(
            "--sequence-length", type=int, default=OLMO3_SFT_SEQUENCE_LENGTH
        )
        subparser.add_argument(
            "--eos-token-id", type=int, default=OLMO3_SFT_EOS_TOKEN_ID
        )
        subparser.add_argument(
            "--pad-token-id", type=int, default=OLMO3_SFT_PAD_TOKEN_ID
        )
        subparser.add_argument("--dtype", default="uint32")
        if command == "prepare":
            subparser.add_argument("--workers", type=int, default=1)
        else:
            subparser.add_argument("--expected-fingerprint")
            subparser.add_argument(
                "--field",
                choices=(
                    "base_instances",
                    "cache_root",
                    "fingerprint",
                    "sequence_length",
                    "source_count",
                ),
                default=None,
            )
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    common = {
        "sequence_length": args.sequence_length,
        "eos_token_id": args.eos_token_id,
        "pad_token_id": args.pad_token_id,
        "dtype": args.dtype,
    }
    if args.command == "prepare":
        cache_root = prepare_olmo3_sft_cache(
            args.data_dir,
            args.work_dir,
            workers=args.workers,
            **common,
        )
        result = describe_olmo3_sft_cache(
            args.data_dir,
            args.work_dir,
            expected_fingerprint=cache_root.name.removeprefix(
                "olmo3-sft-packed-"
            ),
            **common,
        )
        if result["cache_root"] != str(cache_root):
            raise RuntimeError("Prepared and described SFT cache roots differ.")
        print(json.dumps(result, sort_keys=True))
    else:
        result = describe_olmo3_sft_cache(
            args.data_dir,
            args.work_dir,
            expected_fingerprint=args.expected_fingerprint,
            **common,
        )
        print(result[args.field] if args.field else json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
