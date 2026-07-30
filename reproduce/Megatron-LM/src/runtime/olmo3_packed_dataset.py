"""Document-packed inputs for OLMo3 Stage-2/3 training.

The official OLMo 3 long-context recipe does not concatenate the corpus and
cut arbitrary 65,536-token windows. It permutes the frozen source list, groups
eight consecutive sources, and applies Optimized Best-Fit Decreasing (OBFD)
packing to whole documents inside every group. The source list itself is
supplied by the self-contained runtime data manifest; this module never
infers a file count or a storage layout.

The cache format is intentionally immutable and content-addressed.  Training
only opens an already prepared cache; the expensive document scan and packing
must be completed by the companion preparation script before a large job is
submitted.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.packed_seq_params import PackedSeqParams


OLMO3_LONGMINO_SOURCE_GROUP_SIZE = 8
OLMO3_LONGMINO_SOURCE_PERMUTATION_SEED = 123
OLMO3_LONGMINO_DATA_SEED = 4123
OLMO3_LONG_CONTEXT_SEQUENCE_LENGTH = 65536
OLMO3_MIDTRAINING_DATA_SEED = 1337
OLMO3_MIDTRAINING_SEQUENCE_LENGTH = 8192
OLMO3_EOS_TOKEN_ID = 100257
OLMO3_PAD_TOKEN_ID = 100277


def _validate_source_contract_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "source_contract_sha256 must be a lowercase 64-character SHA256"
        )
    return value


def _manifest_contract_fingerprint(
    *,
    source_contract_sha256: str,
    algorithm: str,
    sequence_length: int,
    dtype: np.dtype[Any],
) -> str:
    """Fingerprint a raw-Numpy cache from its sealed content manifest.

    The runtime data manifest contains an ordered SHA256 for every token source.
    Binding to its contract digest makes cache lookup independent of mount-point
    remapping without re-hashing hundreds of GiB on every training rank.
    """

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "algorithm": algorithm,
                "dtype": np.dtype(dtype).str,
                "sequence_length": int(sequence_length),
                "source_contract_sha256": _validate_source_contract_sha256(
                    source_contract_sha256
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _source_fingerprint(
    *,
    source_contract_sha256: str,
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int,
    source_group_size: int,
    source_permutation_seed: int,
    dtype: np.dtype[Any],
) -> str:
    """Fingerprint packed data from its sealed content manifest and settings."""

    digest = hashlib.sha256()
    contract = {
        "algorithm": "olmo_core_obfd_truncate_v1",
        "dtype": np.dtype(dtype).str,
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "sequence_length": int(sequence_length),
        "source_group_size": int(source_group_size),
        "source_permutation_seed": int(source_permutation_seed),
        "source_contract_sha256": _validate_source_contract_sha256(
            source_contract_sha256
        ),
    }
    digest.update(json.dumps(contract, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _validate_prepared_source_sizes(
    cache_root: Path,
    current_sizes: np.ndarray,
    *,
    stage_name: str,
) -> None:
    """Fail closed when a path-addressed cache no longer matches its files."""

    size_index_path = cache_root / "source_file_sizes.npy"
    if not size_index_path.is_file():
        raise RuntimeError(
            f"Completed {stage_name} cache is missing {size_index_path}. "
            "Prepare it in a new work directory."
        )
    try:
        prepared_sizes = np.load(size_index_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Completed {stage_name} cache has an unreadable source-size index: "
            f"{size_index_path}. Prepare it in a new work directory."
        ) from exc
    if prepared_sizes.shape != current_sizes.shape or not np.array_equal(
        prepared_sizes, current_sizes
    ):
        raise RuntimeError(
            f"{stage_name} source files changed after cache preparation. The cache "
            f"at {cache_root} is immutable; prepare a new cache (or remove this "
            "generated cache after confirming no training process uses it)."
        )


def _source_group_contract_fingerprint(
    source_paths: Sequence[Path],
    source_ids: Sequence[int],
    source_file_sizes: Sequence[int],
    *,
    sequence_length: int,
    eos_token_id: int,
    dtype: np.dtype[Any],
) -> str:
    """Identify all inputs that make an OBFD source-group index reusable."""

    payload = {
        "algorithm": "olmo_core_obfd_truncate_group_v1",
        "dtype": np.dtype(dtype).str,
        "eos_token_id": int(eos_token_id),
        "sequence_length": int(sequence_length),
        "source_file_sizes": [int(size) for size in source_file_sizes],
        "source_ids": [int(source_id) for source_id in source_ids],
        "source_paths": [str(path.resolve()) for path in source_paths],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class _SegmentTreeNode:
    weight: int = 0
    parent: "_SegmentTreeNode | None" = None
    children: "tuple[_SegmentTreeNode, _SegmentTreeNode] | None" = None
    leaf_id: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def update(self, weight: int | None = None) -> None:
        if weight is not None:
            if not self.is_leaf:
                raise AssertionError("Only a segment-tree leaf has a direct weight.")
            self.weight = int(weight)
        else:
            if self.children is None:
                raise AssertionError("A leaf needs an explicit weight.")
            self.weight = max(self.children[0].weight, self.children[1].weight)
        if self.parent is not None:
            self.parent.update()


class _SegmentTree:
    """Exact OLMo-core OBFD segment tree (Appendix B of arXiv:2404.10830)."""

    def __init__(self, size: int) -> None:
        if size < 1 or not math.log2(size).is_integer():
            raise ValueError("OBFD sequence length must be a positive power of two.")
        self.root = _SegmentTreeNode()
        self.leaves: list[_SegmentTreeNode] = []
        max_depth = int(math.log2(size))
        leaf_id = 0
        queue: deque[tuple[_SegmentTreeNode, int]] = deque([(self.root, 0)])
        while queue:
            parent, depth = queue.popleft()
            if depth < max_depth:
                parent.children = (
                    _SegmentTreeNode(parent=parent),
                    _SegmentTreeNode(parent=parent),
                )
                queue.append((parent.children[0], depth + 1))
                queue.append((parent.children[1], depth + 1))
            else:
                parent.leaf_id = leaf_id
                self.leaves.append(parent)
                leaf_id += 1
        self.leaves[-1].update(size)

    def query(self, weight: int) -> _SegmentTreeNode:
        node = self.root
        while not node.is_leaf:
            if weight > node.weight or node.children is None:
                raise RuntimeError(f"No OBFD bin can fit a document of length {weight}.")
            left, right = node.children
            node = left if weight <= left.weight else right
        return node


class _InstancePacker:
    """Exact OLMo-core Optimized Best-Fit Decreasing packer."""

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = int(sequence_length)
        self.tree = _SegmentTree(self.sequence_length)
        self.bins: list[list[int]] = []
        self.space_to_bins: dict[int, deque[int]] = defaultdict(deque)

    def _pack_document(self, document_id: int, document_length: int) -> None:
        leaf_id = self.tree.query(document_length).leaf_id
        if leaf_id is None:
            raise AssertionError("OBFD query did not reach a leaf.")
        capacity = leaf_id + 1
        if capacity == self.sequence_length:
            self.bins.append([])
            bin_id = len(self.bins) - 1
        else:
            matching_bins = self.space_to_bins[capacity]
            bin_id = matching_bins.popleft()
            if not matching_bins:
                self.tree.leaves[capacity - 1].update(0)
        self.bins[bin_id].append(document_id)
        remaining = capacity - document_length
        if remaining > 0:
            matching_bins = self.space_to_bins[remaining]
            if not matching_bins:
                self.tree.leaves[remaining - 1].update(remaining)
            matching_bins.append(bin_id)

    def pack(self, documents: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        """Pack ``[source_id, start, end]`` rows and return sorted rows."""

        if documents.ndim != 2 or documents.shape[1] != 3:
            raise ValueError("OBFD document table must have [N, 3] shape.")
        lengths = documents[:, 2] - documents[:, 1]
        # Keep OLMo-core's default np.argsort semantics, including tie order.
        sorted_index = np.argsort(-1 * lengths.astype(np.int64))
        sorted_documents = np.take(documents, sorted_index, axis=0)
        for document_id, document in enumerate(sorted_documents):
            self._pack_document(
                document_id,
                int(document[2] - document[1]),
            )
        return self.bins, sorted_documents


def _iter_document_offsets(
    path: Path,
    *,
    eos_token_id: int,
    dtype: np.dtype[Any],
) -> Iterator[tuple[int, int]]:
    """Yield OLMo-core's local-file EOS-delimited document offsets.

    OLMo-core deliberately scans the token array whenever it is local and an
    EOS ID/dtype are available. A neighbouring ``.csv.gz`` is therefore not
    trusted here: this keeps AFS behavior byte-for-byte tied to the raw data.
    """

    tokens = np.memmap(path, mode="r", dtype=dtype)
    start = 0
    # Chunking avoids materialising a corpus-sized boolean array.
    scan_tokens = 16 * 1024 * 1024
    for chunk_start in range(0, int(tokens.shape[0]), scan_tokens):
        chunk = tokens[chunk_start : chunk_start + scan_tokens]
        for local_eos in np.flatnonzero(chunk == eos_token_id):
            end = chunk_start + int(local_eos) + 1
            yield start, end
            start = end
    # OLMo-core intentionally ignores a trailing fragment without EOS.


def _group_documents(
    source_paths: Sequence[Path],
    source_ids: Sequence[int],
    *,
    sequence_length: int,
    eos_token_id: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    for source_id, source_path in zip(source_ids, source_paths):
        for start, end in _iter_document_offsets(
            source_path,
            eos_token_id=eos_token_id,
            dtype=dtype,
        ):
            # Official Stage 3 uses LongDocStrategy.truncate.
            rows.append((source_id, start, min(end, start + sequence_length)))
    if not rows:
        raise RuntimeError(f"No EOS-delimited documents found in {source_paths!r}.")
    return np.asarray(rows, dtype=np.uint64)


def _atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.save(temporary, array, allow_pickle=False)
        temporary.flush()
        os.fsync(temporary.fileno())
    temporary_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    temporary_path.replace(path)


def prepare_olmo3_midtraining_cache(
    source_paths: Sequence[str | Path],
    work_dir: str | Path,
    *,
    source_contract_sha256: str,
    sequence_length: int = OLMO3_MIDTRAINING_SEQUENCE_LENGTH,
    dtype: str | np.dtype[Any] = np.uint32,
) -> Path:
    """Validate and index the official Stage-2 raw-Numpy FSL sources.

    This reproduces ``NumpyFSLDataset``: every source is independently split
    into full 8,192-token instances and the per-file remainder is ignored.
    The completion marker is written last, so training can fail closed rather
    than observing a partially prepared index.
    """

    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if sequence_length != OLMO3_MIDTRAINING_SEQUENCE_LENGTH:
        raise ValueError("Strict OLMo3 Stage 2 requires sequence_length=8192.")
    if not paths:
        raise ValueError("At least one OLMo3 midtraining source is required.")
    dtype = np.dtype(dtype)
    if dtype != np.dtype(np.uint32):
        raise ValueError("Official Dolma2 midtraining arrays use uint32 token IDs.")

    item_size = dtype.itemsize
    file_sizes = np.zeros(len(paths), dtype=np.uint64)
    instance_offsets = np.zeros(len(paths) + 1, dtype=np.uint64)
    missing: list[str] = []
    for source_id, path in enumerate(paths):
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            if len(missing) < 8:
                missing.append(str(path))
            continue
        if file_size % item_size:
            raise ValueError(
                f"Raw uint32 source {path} has a non-aligned size: {file_size} bytes."
            )
        file_sizes[source_id] = file_size
        token_count = file_size // item_size
        instance_offsets[source_id + 1] = (
            instance_offsets[source_id] + token_count // sequence_length
        )
    if missing:
        raise FileNotFoundError(
            "Missing official midtraining source files; first missing: "
            + ", ".join(missing)
        )
    if int(instance_offsets[-1]) < 1:
        raise RuntimeError("OLMo3 midtraining sources contain no full FSL instances.")
    if int(instance_offsets[-1]) >= int(np.iinfo(np.uint32).max):
        raise RuntimeError("OLMo3 Stage-2 FSL index exceeds the official uint32 order.")

    source_contract_sha256 = _validate_source_contract_sha256(
        source_contract_sha256
    )
    fingerprint = _manifest_contract_fingerprint(
        source_contract_sha256=source_contract_sha256,
        algorithm="olmo_core_numpy_fsl_v1",
        sequence_length=sequence_length,
        dtype=dtype,
    )
    cache_root = Path(work_dir).expanduser().resolve() / f"olmo3-fsl-{fingerprint}"
    complete_path = cache_root / "complete.json"
    if complete_path.is_file():
        _validate_prepared_source_sizes(
            cache_root,
            file_sizes,
            stage_name="OLMo3 Stage-2",
        )
        describe_olmo3_midtraining_cache(
            paths,
            work_dir,
            global_batch_size=1,
            source_contract_sha256=source_contract_sha256,
            sequence_length=sequence_length,
            dtype=dtype,
        )
        return cache_root

    _atomic_save_numpy(cache_root / "source_file_sizes.npy", file_sizes)
    _atomic_save_numpy(cache_root / "source_instance_offsets.npy", instance_offsets)
    _atomic_write_json(
        complete_path,
        {
            "algorithm": "olmo_core_numpy_fsl_v1",
            "dtype": dtype.str,
            "fingerprint": fingerprint,
            "instances": int(instance_offsets[-1]),
            "sequence_length": sequence_length,
            "source_contract_sha256": source_contract_sha256,
            "source_count": len(paths),
            "source_paths": [str(path) for path in paths],
        },
    )
    return cache_root


class Olmo3NumpyFSLDataset(Dataset[dict[str, Tensor]]):
    """Raw uint32 Stage-2 dataset matching OLMo-core ``NumpyFSLDataset``."""

    official_olmo3_order = True

    def __init__(
        self,
        source_paths: Sequence[str | Path],
        work_dir: str | Path,
        *,
        num_samples: int,
        global_batch_size: int,
        source_contract_sha256: str,
        data_seed: int = OLMO3_MIDTRAINING_DATA_SEED,
        sequence_length: int = OLMO3_MIDTRAINING_SEQUENCE_LENGTH,
        pad_token_id: int = OLMO3_PAD_TOKEN_ID,
        dtype: str | np.dtype[Any] = np.uint32,
    ) -> None:
        super().__init__()
        self.source_paths = tuple(
            Path(path).expanduser().resolve() for path in source_paths
        )
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)
        self.data_seed = int(data_seed)
        self.dtype = np.dtype(dtype)
        source_contract_sha256 = _validate_source_contract_sha256(
            source_contract_sha256
        )
        fingerprint = _manifest_contract_fingerprint(
            source_contract_sha256=source_contract_sha256,
            algorithm="olmo_core_numpy_fsl_v1",
            sequence_length=self.sequence_length,
            dtype=self.dtype,
        )
        self.cache_root = (
            Path(work_dir).expanduser().resolve() / f"olmo3-fsl-{fingerprint}"
        )
        complete_path = self.cache_root / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared OLMo3 Stage-2 cache {complete_path}. Run the "
                "midtraining preparation script before submitting Stage 2."
            )
        metadata = json.loads(complete_path.read_text(encoding="utf-8"))
        expected = {
            "algorithm": "olmo_core_numpy_fsl_v1",
            "dtype": self.dtype.str,
            "fingerprint": fingerprint,
            "sequence_length": OLMO3_MIDTRAINING_SEQUENCE_LENGTH,
            "source_contract_sha256": source_contract_sha256,
            "source_count": len(self.source_paths),
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Non-official OLMo3 Stage-2 cache settings: {mismatches}")
        self.source_file_sizes = np.load(
            self.cache_root / "source_file_sizes.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.source_instance_offsets = np.load(
            self.cache_root / "source_instance_offsets.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            self.source_file_sizes.ndim != 1
            or len(self.source_file_sizes) != len(self.source_paths)
            or bool(np.any(self.source_file_sizes % self.dtype.itemsize))
            or self.source_instance_offsets.ndim != 1
            or len(self.source_instance_offsets) != len(self.source_paths) + 1
            or int(self.source_instance_offsets[0]) != 0
            or not bool(
                np.all(
                    self.source_instance_offsets[1:]
                    >= self.source_instance_offsets[:-1]
                )
            )
        ):
            raise RuntimeError("OLMo3 Stage-2 source-offset index is corrupt.")
        self.base_instances = int(self.source_instance_offsets[-1])
        expected_instances_by_source = (
            self.source_file_sizes
            // self.dtype.itemsize
            // self.sequence_length
        )
        if not np.array_equal(
            np.diff(self.source_instance_offsets), expected_instances_by_source
        ) or self.base_instances != int(metadata.get("instances", -1)):
            raise RuntimeError(
                "OLMo3 Stage-2 source-offset counts do not match the prepared sources."
            )
        self.num_samples = int(num_samples)
        self.global_batch_size = int(global_batch_size)
        if self.base_instances < 1 or self.num_samples < 1:
            raise ValueError("OLMo3 Stage-2 dataset sizes must be positive.")
        if self.global_batch_size < 1:
            raise ValueError("OLMo3 Stage-2 global_batch_size must be positive.")
        # NumpyFSLDataLoader.total_size drops the shuffled tail that cannot
        # form a complete *global* batch. Do the same before advancing the
        # one-based epoch; otherwise a local Dataset would incorrectly join
        # the old epoch's tail to the next epoch's first batch.
        self.instances_per_epoch = (
            self.base_instances // self.global_batch_size
        ) * self.global_batch_size
        if self.instances_per_epoch < 1:
            raise ValueError(
                "OLMo3 Stage-2 corpus has fewer instances than one global batch."
            )
        self._source_arrays: dict[int, np.memmap[Any, Any]] = {}

    def __len__(self) -> int:
        return self.num_samples

    @lru_cache(maxsize=2)
    def _epoch_order(self, epoch: int) -> np.ndarray:
        # OLMo-core epochs are one-based and use PCG64(seed + epoch).
        order = np.arange(self.base_instances, dtype=np.uint32)
        np.random.Generator(np.random.PCG64(self.data_seed + epoch + 1)).shuffle(order)
        return order[: self.instances_per_epoch]

    def _resolve_instance(self, index: int) -> tuple[int, int]:
        epoch, epoch_index = divmod(index, self.instances_per_epoch)
        shuffled_index = int(self._epoch_order(epoch)[epoch_index])
        source_id = bisect.bisect_right(
            self.source_instance_offsets, shuffled_index
        ) - 1
        if not 0 <= source_id < len(self.source_paths):
            raise IndexError(index)
        return source_id, shuffled_index - int(self.source_instance_offsets[source_id])

    def _load_source(self, source_id: int) -> np.memmap[Any, Any]:
        source = self._source_arrays.get(source_id)
        if source is None:
            path = self.source_paths[source_id]
            expected_size = int(self.source_file_sizes[source_id])
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"OLMo3 Stage-2 source changed after preparation: {path} "
                    f"({expected_size} -> {actual_size} bytes)."
                )
            source = np.memmap(path, mode="r", dtype=self.dtype)
            self._source_arrays[source_id] = source
        return source

    def __getitem__(self, index: int | None) -> dict[str, Tensor]:
        padding_sample = index is None
        index = 0 if padding_sample else int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_id, local_instance = self._resolve_instance(index)
        start = local_instance * self.sequence_length
        stop = start + self.sequence_length
        token_view = self._load_source(source_id)[start:stop]
        if token_view.size != self.sequence_length:
            raise RuntimeError("Prepared OLMo3 Stage-2 instance is unexpectedly short.")
        tokens = torch.from_numpy(
            np.asarray(token_view).astype(np.int64, copy=True)
        ).long()
        labels = torch.empty_like(tokens)
        labels[:-1] = tokens[1:]
        labels[-1] = self.pad_token_id
        loss_mask = torch.ones(self.sequence_length, dtype=torch.float32)
        loss_mask[-1] = 0.0
        if padding_sample:
            loss_mask.zero_()
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(self.sequence_length, dtype=torch.long),
        }


def describe_olmo3_midtraining_cache(
    source_paths: Sequence[str | Path],
    work_dir: str | Path,
    *,
    source_contract_sha256: str,
    global_batch_size: int,
    data_seed: int = OLMO3_MIDTRAINING_DATA_SEED,
    sequence_length: int = OLMO3_MIDTRAINING_SEQUENCE_LENGTH,
    pad_token_id: int = OLMO3_PAD_TOKEN_ID,
    dtype: str | np.dtype[Any] = np.uint32,
) -> dict[str, Any]:
    """Validate and summarize one already prepared Stage-2 cache."""

    dataset = Olmo3NumpyFSLDataset(
        source_paths,
        work_dir,
        num_samples=1,
        global_batch_size=global_batch_size,
        source_contract_sha256=source_contract_sha256,
        data_seed=data_seed,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        dtype=dtype,
    )
    return {
        "base_instances": dataset.base_instances,
        "cache_root": str(dataset.cache_root),
        "sequence_length": dataset.sequence_length,
        "source_count": len(dataset.source_paths),
    }


def _prepare_source_group(
    group_id: int,
    source_paths: Sequence[Path],
    source_ids: Sequence[int],
    source_file_sizes: Sequence[int],
    cache_root: Path,
    *,
    sequence_length: int,
    eos_token_id: int,
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    group_root = cache_root / f"group-{group_id:04d}"
    metadata_path = group_root / "metadata.json"
    documents_path = group_root / "documents.npy"
    offsets_path = group_root / "instance_offsets.npy"
    document_ids_path = group_root / "document_ids.npy"
    contract_fingerprint = _source_group_contract_fingerprint(
        source_paths,
        source_ids,
        source_file_sizes,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        dtype=dtype,
    )
    if all(
        path.is_file()
        for path in (metadata_path, documents_path, offsets_path, document_ids_path)
    ):
        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                prepared_metadata = json.load(metadata_file)
            if prepared_metadata.get("contract_fingerprint") == contract_fingerprint:
                prepared_documents = np.load(
                    documents_path, mmap_mode="r", allow_pickle=False
                )
                prepared_offsets = np.load(
                    offsets_path, mmap_mode="r", allow_pickle=False
                )
                prepared_document_ids = np.load(
                    document_ids_path, mmap_mode="r", allow_pickle=False
                )
                structurally_valid = (
                    prepared_documents.ndim == 2
                    and prepared_documents.shape[1:] == (3,)
                    and prepared_offsets.ndim == 1
                    and prepared_document_ids.ndim == 1
                    and prepared_offsets.size > 0
                    and prepared_offsets.size
                    == int(prepared_metadata.get("instances", -1)) + 1
                    and int(prepared_offsets[0]) == 0
                    and int(prepared_offsets[-1]) == prepared_document_ids.size
                    and bool(
                        np.all(prepared_offsets[1:] >= prepared_offsets[:-1])
                    )
                    and (
                        prepared_document_ids.size == 0
                        or int(prepared_document_ids.max())
                        < prepared_documents.shape[0]
                    )
                    and int(prepared_metadata.get("documents", -1))
                    == prepared_documents.shape[0]
                )
                if structurally_valid:
                    return prepared_metadata
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            # A stale or interrupted group is generated data. Rebuild it below;
            # metadata.json is written last so completed peer groups remain safe.
            pass

    documents = _group_documents(
        source_paths,
        source_ids,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        dtype=dtype,
    )
    bins, documents = _InstancePacker(sequence_length).pack(documents)
    offsets = np.zeros(len(bins) + 1, dtype=np.uint64)
    if bins:
        offsets[1:] = np.cumsum(
            np.fromiter((len(instance) for instance in bins), dtype=np.uint64),
            dtype=np.uint64,
        )
    flat_document_ids = np.fromiter(
        (document_id for instance in bins for document_id in instance),
        dtype=np.uint64,
        count=int(offsets[-1]),
    )
    token_count = int((documents[:, 2] - documents[:, 1]).sum())
    padding_count = int(len(bins) * sequence_length - token_count)
    metadata = {
        "contract_fingerprint": contract_fingerprint,
        "documents": int(documents.shape[0]),
        "instances": int(len(bins)),
        "padding_tokens": padding_count,
        "tokens": token_count,
    }
    _atomic_save_numpy(documents_path, documents)
    _atomic_save_numpy(offsets_path, offsets)
    _atomic_save_numpy(document_ids_path, flat_document_ids)
    _atomic_write_json(metadata_path, metadata)
    return metadata


def prepare_olmo3_long_context_cache(
    source_paths: Sequence[str | Path],
    work_dir: str | Path,
    *,
    source_contract_sha256: str,
    sequence_length: int = OLMO3_LONG_CONTEXT_SEQUENCE_LENGTH,
    eos_token_id: int = OLMO3_EOS_TOKEN_ID,
    pad_token_id: int = OLMO3_PAD_TOKEN_ID,
    source_group_size: int = OLMO3_LONGMINO_SOURCE_GROUP_SIZE,
    source_permutation_seed: int = OLMO3_LONGMINO_SOURCE_PERMUTATION_SEED,
    dtype: str | np.dtype[Any] = np.uint32,
    workers: int = 1,
) -> Path:
    """Build the immutable OLMo3 packed-data cache and return its root."""

    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if sequence_length != 65536 or source_group_size != 8 or source_permutation_seed != 123:
        raise ValueError(
            "Strict OLMo3 Stage 3 requires sequence_length=65536, "
            "source_group_size=8, and source_permutation_seed=123."
        )
    if not paths:
        raise ValueError("At least one Longmino source is required.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} Longmino source files; first missing: {preview}"
        )
    dtype = np.dtype(dtype)
    if dtype != np.dtype(np.uint32):
        raise ValueError("Official Dolma2 Longmino arrays use uint32 token IDs.")

    source_file_sizes = np.asarray(
        [path.stat().st_size for path in paths], dtype=np.uint64
    )
    if np.any(source_file_sizes % dtype.itemsize):
        bad_index = int(np.flatnonzero(source_file_sizes % dtype.itemsize)[0])
        raise ValueError(
            f"Raw uint32 source {paths[bad_index]} has a non-aligned size: "
            f"{int(source_file_sizes[bad_index])} bytes."
        )

    source_contract_sha256 = _validate_source_contract_sha256(
        source_contract_sha256
    )
    fingerprint = _source_fingerprint(
        source_contract_sha256=source_contract_sha256,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        source_group_size=source_group_size,
        source_permutation_seed=source_permutation_seed,
        dtype=dtype,
    )
    cache_root = Path(work_dir).expanduser().resolve() / f"olmo3-packed-{fingerprint}"
    complete_path = cache_root / "complete.json"
    if complete_path.is_file():
        _validate_prepared_source_sizes(
            cache_root,
            source_file_sizes,
            stage_name="OLMo3 Stage-3",
        )
        describe_olmo3_long_context_cache(
            paths,
            work_dir,
            global_batch_size=1,
            source_contract_sha256=source_contract_sha256,
            sequence_length=sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            source_group_size=source_group_size,
            source_permutation_seed=source_permutation_seed,
            dtype=dtype,
        )
        return cache_root

    source_order = list(range(len(paths)))
    random.Random(source_permutation_seed).shuffle(source_order)
    permuted_paths = tuple(paths[index] for index in source_order)
    group_specs = [
        (
            group_id,
            permuted_paths[group_start : group_start + source_group_size],
            source_order[group_start : group_start + source_group_size],
            tuple(
                int(source_file_sizes[index])
                for index in source_order[
                    group_start : group_start + source_group_size
                ]
            ),
        )
        for group_id, group_start in enumerate(
            range(0, len(paths), source_group_size)
        )
    ]
    groups: list[dict[str, Any] | None] = [None] * len(group_specs)
    if workers == 1:
        for group_id, group_paths, group_source_ids, group_source_sizes in group_specs:
            groups[group_id] = _prepare_source_group(
                group_id,
                group_paths,
                group_source_ids,
                group_source_sizes,
                cache_root,
                sequence_length=sequence_length,
                eos_token_id=eos_token_id,
                dtype=dtype,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _prepare_source_group,
                    group_id,
                    group_paths,
                    group_source_ids,
                    group_source_sizes,
                    cache_root,
                    sequence_length=sequence_length,
                    eos_token_id=eos_token_id,
                    dtype=dtype,
                ): group_id
                for (
                    group_id,
                    group_paths,
                    group_source_ids,
                    group_source_sizes,
                ) in group_specs
            }
            for future in as_completed(futures):
                group_id = futures[future]
                groups[group_id] = future.result()
    if any(group is None for group in groups):
        raise RuntimeError("OLMo3 packing did not produce every source group.")
    complete_groups = [group for group in groups if group is not None]

    instance_counts = np.asarray(
        [group["instances"] for group in complete_groups], dtype=np.uint64
    )
    group_offsets = np.zeros(len(groups) + 1, dtype=np.uint64)
    group_offsets[1:] = np.cumsum(instance_counts, dtype=np.uint64)
    if int(group_offsets[-1]) >= int(np.iinfo(np.uint32).max):
        raise RuntimeError("OLMo3 Stage-3 packed index exceeds the official uint32 order.")
    _atomic_save_numpy(cache_root / "group_offsets.npy", group_offsets)
    _atomic_save_numpy(cache_root / "source_file_sizes.npy", source_file_sizes)
    _atomic_write_json(
        complete_path,
        {
            "algorithm": "olmo_core_obfd_truncate_v1",
            "dtype": dtype.str,
            "eos_token_id": eos_token_id,
            "fingerprint": fingerprint,
            "groups": complete_groups,
            "instances": int(group_offsets[-1]),
            "pad_token_id": pad_token_id,
            "sequence_length": sequence_length,
            "source_count": len(paths),
            "source_contract_sha256": source_contract_sha256,
            "source_group_size": source_group_size,
            "source_order": source_order,
            "source_paths": [str(path) for path in paths],
            "source_permutation_seed": source_permutation_seed,
        },
    )
    return cache_root


class Olmo3NumpyPackedFSLDataset(Dataset[dict[str, Tensor]]):
    """Read an already prepared official OLMo3 Longmino OBFD cache."""

    official_olmo3_order = True

    def __init__(
        self,
        source_paths: Sequence[str | Path],
        work_dir: str | Path,
        *,
        num_samples: int | None,
        global_batch_size: int,
        source_contract_sha256: str,
        data_seed: int = OLMO3_LONGMINO_DATA_SEED,
        sequence_length: int = OLMO3_LONG_CONTEXT_SEQUENCE_LENGTH,
        eos_token_id: int = OLMO3_EOS_TOKEN_ID,
        pad_token_id: int = OLMO3_PAD_TOKEN_ID,
        source_group_size: int = OLMO3_LONGMINO_SOURCE_GROUP_SIZE,
        source_permutation_seed: int = OLMO3_LONGMINO_SOURCE_PERMUTATION_SEED,
        dtype: str | np.dtype[Any] = np.uint32,
    ) -> None:
        super().__init__()
        self.source_paths = tuple(
            Path(path).expanduser().resolve() for path in source_paths
        )
        self.sequence_length = int(sequence_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.data_seed = int(data_seed)
        self.dtype = np.dtype(dtype)
        source_contract_sha256 = _validate_source_contract_sha256(
            source_contract_sha256
        )
        fingerprint = _source_fingerprint(
            source_contract_sha256=source_contract_sha256,
            sequence_length=self.sequence_length,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            source_group_size=source_group_size,
            source_permutation_seed=source_permutation_seed,
            dtype=self.dtype,
        )
        self.cache_root = (
            Path(work_dir).expanduser().resolve() / f"olmo3-packed-{fingerprint}"
        )
        complete_path = self.cache_root / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared OLMo3 cache {complete_path}. Run the data "
                "preparation script before submitting Stage 3."
            )
        with complete_path.open("r", encoding="utf-8") as complete_file:
            metadata = json.load(complete_file)
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("OLMo3 packed cache fingerprint does not match its sources.")
        expected = {
            "algorithm": "olmo_core_obfd_truncate_v1",
            "dtype": self.dtype.str,
            "sequence_length": 65536,
            "source_count": len(self.source_paths),
            "source_contract_sha256": source_contract_sha256,
            "source_group_size": 8,
            "source_permutation_seed": 123,
            "eos_token_id": 100257,
            "pad_token_id": 100277,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Non-official OLMo3 packed cache settings: {mismatches}")
        self.source_file_sizes = np.load(
            self.cache_root / "source_file_sizes.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            self.source_file_sizes.ndim != 1
            or len(self.source_file_sizes) != len(self.source_paths)
            or bool(np.any(self.source_file_sizes % self.dtype.itemsize))
        ):
            raise RuntimeError("OLMo3 packed source-size index is corrupt.")
        self.group_offsets = np.load(
            self.cache_root / "group_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        expected_groups = math.ceil(len(self.source_paths) / source_group_size)
        metadata_groups = metadata.get("groups")
        if (
            self.group_offsets.ndim != 1
            or len(self.group_offsets) != expected_groups + 1
            or int(self.group_offsets[0]) != 0
            or not bool(np.all(self.group_offsets[1:] >= self.group_offsets[:-1]))
            or not isinstance(metadata_groups, list)
            or len(metadata_groups) != expected_groups
            or not all(isinstance(group, dict) for group in metadata_groups)
            or int(self.group_offsets[-1]) != int(metadata.get("instances", -1))
        ):
            raise RuntimeError("OLMo3 packed group-offset index is corrupt.")
        metadata_group_counts = np.asarray(
            [group.get("instances", -1) for group in metadata_groups],
            dtype=np.int64,
        )
        if np.any(metadata_group_counts < 0) or not np.array_equal(
            np.diff(self.group_offsets).astype(np.int64, copy=False),
            metadata_group_counts,
        ):
            raise RuntimeError(
                "OLMo3 packed group counts do not match the prepared group offsets."
            )
        self.base_instances = int(self.group_offsets[-1])
        if self.base_instances < 1:
            raise RuntimeError("OLMo3 packed cache contains no instances.")
        self.num_samples = (
            self.base_instances if num_samples is None else int(num_samples)
        )
        self.global_batch_size = int(global_batch_size)
        if self.num_samples < 1:
            raise ValueError("OLMo3 packed dataset num_samples must be positive.")
        if self.global_batch_size < 1:
            raise ValueError("OLMo3 Stage-3 global_batch_size must be positive.")
        self.instances_per_epoch = (
            self.base_instances // self.global_batch_size
        ) * self.global_batch_size
        if self.instances_per_epoch < 1:
            raise ValueError(
                "OLMo3 Stage-3 corpus has fewer packed instances than one global batch."
            )
        self._source_arrays: dict[int, np.memmap[Any, Any]] = {}
        self._group_arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.num_samples

    @lru_cache(maxsize=2)
    def _epoch_order(self, epoch: int) -> np.ndarray:
        # OLMo-core epochs are one-based and use PCG64(seed + epoch).
        order = np.arange(self.base_instances, dtype=np.uint32)
        np.random.Generator(np.random.PCG64(self.data_seed + epoch + 1)).shuffle(order)
        return order[: self.instances_per_epoch]

    def _resolve_instance(self, index: int) -> tuple[int, int]:
        epoch, epoch_index = divmod(index, self.instances_per_epoch)
        shuffled_index = int(self._epoch_order(epoch)[epoch_index])
        group_id = bisect.bisect_right(self.group_offsets, shuffled_index) - 1
        if not 0 <= group_id < len(self.group_offsets) - 1:
            raise IndexError(index)
        return group_id, shuffled_index - int(self.group_offsets[group_id])

    def _load_group(self, group_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._group_arrays.get(group_id)
        if arrays is None:
            group_root = self.cache_root / f"group-{group_id:04d}"
            arrays = (
                np.load(group_root / "documents.npy", mmap_mode="r", allow_pickle=False),
                np.load(group_root / "instance_offsets.npy", mmap_mode="r", allow_pickle=False),
                np.load(group_root / "document_ids.npy", mmap_mode="r", allow_pickle=False),
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
                raise RuntimeError(
                    f"OLMo3 packed source group {group_id} has a corrupt index."
                )
            self._group_arrays[group_id] = arrays
        return arrays

    def _load_source(self, source_id: int) -> np.memmap[Any, Any]:
        source = self._source_arrays.get(source_id)
        if source is None:
            path = self.source_paths[source_id]
            expected_size = int(self.source_file_sizes[source_id])
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"OLMo3 Longmino source changed after preparation: {path} "
                    f"({expected_size} -> {actual_size} bytes)."
                )
            source = np.memmap(path, mode="r", dtype=self.dtype)
            self._source_arrays[source_id] = source
        return source

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

        token_parts: list[np.ndarray] = []
        document_lengths: list[int] = []
        for document_id in instance_document_ids:
            source_id, start, stop = (
                int(value) for value in documents[int(document_id)]
            )
            token_part = np.asarray(self._load_source(source_id)[start:stop])
            token_parts.append(token_part)
            document_lengths.append(int(stop - start))
        real_tokens = sum(document_lengths)
        padding_tokens = self.sequence_length - real_tokens
        if padding_tokens < 0:
            raise RuntimeError("Prepared OLMo3 instance exceeds sequence length.")
        tokens_numpy = np.full(
            self.sequence_length,
            self.pad_token_id,
            dtype=self.dtype,
        )
        cursor = 0
        for token_part in token_parts:
            next_cursor = cursor + token_part.size
            tokens_numpy[cursor:next_cursor] = token_part
            cursor = next_cursor
        tokens = torch.from_numpy(tokens_numpy.astype(np.int64, copy=False)).long()

        labels = torch.empty_like(tokens)
        labels[:-1] = tokens[1:]
        labels[-1] = self.pad_token_id
        # OLMo-core constructs an explicit label_mask from document occupancy,
        # then shifts it left with the labels. Do not infer occupancy from the
        # token value: a real token equal to pad_token_id would still be data.
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        if real_tokens > 1:
            loss_mask[: real_tokens - 1] = 1.0
        if padding_sample:
            loss_mask.zero_()

        lengths_with_padding = list(document_lengths)
        if padding_tokens:
            lengths_with_padding.append(padding_tokens)
        lengths = torch.tensor(lengths_with_padding, dtype=torch.long)
        packed_document_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), dtype=torch.long), lengths
        )
        starts = torch.cumsum(lengths, dim=0) - lengths
        packed_position_ids = torch.arange(self.sequence_length, dtype=torch.long)
        packed_position_ids -= torch.repeat_interleave(starts, lengths)
        if packed_document_ids.numel() != self.sequence_length:
            raise RuntimeError("OLMo3 packed document lengths do not cover the instance.")

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": packed_position_ids,
            "document_ids": packed_document_ids,
        }


def describe_olmo3_long_context_cache(
    source_paths: Sequence[str | Path],
    work_dir: str | Path,
    *,
    source_contract_sha256: str,
    global_batch_size: int,
    data_seed: int = OLMO3_LONGMINO_DATA_SEED,
    sequence_length: int = OLMO3_LONG_CONTEXT_SEQUENCE_LENGTH,
    eos_token_id: int = OLMO3_EOS_TOKEN_ID,
    pad_token_id: int = OLMO3_PAD_TOKEN_ID,
    source_group_size: int = OLMO3_LONGMINO_SOURCE_GROUP_SIZE,
    source_permutation_seed: int = OLMO3_LONGMINO_SOURCE_PERMUTATION_SEED,
    dtype: str | np.dtype[Any] = np.uint32,
) -> dict[str, Any]:
    """Validate and summarize one already prepared Stage-3 cache."""

    dataset = Olmo3NumpyPackedFSLDataset(
        source_paths,
        work_dir,
        num_samples=1,
        global_batch_size=global_batch_size,
        source_contract_sha256=source_contract_sha256,
        data_seed=data_seed,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        source_group_size=source_group_size,
        source_permutation_seed=source_permutation_seed,
        dtype=dtype,
    )
    for group_id in range(len(dataset.group_offsets) - 1):
        dataset._load_group(group_id)
    return {
        "base_instances": dataset.base_instances,
        "cache_root": str(dataset.cache_root),
        "sequence_length": dataset.sequence_length,
        "source_count": len(dataset.source_paths),
    }


def document_ids_and_position_ids(tokens: Tensor, eod_token_id: int) -> tuple[Tensor, Tensor]:
    """Return fixed-width document IDs and document-relative positions.

    The first token in each rank sample is treated as a fresh document because
    no context from the preceding sample is available. Every token immediately
    following EOD starts another document. EOD itself remains part of the
    preceding document, matching OLMo-core's packed-FSL convention.
    """

    if tokens.ndim != 1 or tokens.numel() < 1:
        raise ValueError(
            "Packed OLMo3 tokens must be a non-empty one-dimensional tensor."
        )
    starts = torch.zeros_like(tokens, dtype=torch.bool)
    starts[0] = True
    starts[1:] = tokens[:-1].eq(int(eod_token_id))
    document_ids = starts.to(torch.long).cumsum(dim=0) - 1

    token_indices = torch.arange(tokens.numel(), device=tokens.device, dtype=torch.long)
    start_indices = torch.where(starts, token_indices, torch.zeros_like(token_indices))
    latest_start = torch.cummax(start_indices, dim=0).values
    position_ids = token_indices - latest_start
    return document_ids, position_ids


def _one_row_cumulative_document_lengths(document_ids: Tensor) -> Tensor:
    """Return canonical ``[0, end_0, ...]`` offsets for one packed row."""

    if document_ids.ndim != 1 or document_ids.numel() < 1:
        raise ValueError("each document_ids row must be a non-empty sequence")
    changes = (
        torch.nonzero(
            document_ids[1:] != document_ids[:-1],
            as_tuple=False,
        ).flatten()
        + 1
    )
    endpoints = torch.tensor(
        [0, document_ids.numel()],
        device=document_ids.device,
        dtype=torch.int32,
    )
    if changes.numel() == 0:
        return endpoints
    return torch.cat(
        (endpoints[:1], changes.to(dtype=torch.int32), endpoints[1:]),
        dim=0,
    ).contiguous()


def cumulative_document_lengths(document_ids: Tensor) -> Tensor:
    """Convert one or more packed rows to flattened CANN/TE cu_seqlens.

    TND concatenates samples in batch-major order. Each row therefore owns an
    independent document namespace, and every row after the first receives a
    fixed token offset. The boundary between adjacent microbatch samples is
    always retained even when their terminal/initial document IDs match.
    """

    if document_ids.ndim == 1:
        return _one_row_cumulative_document_lengths(document_ids)
    if document_ids.ndim != 2 or document_ids.shape[0] < 1 or document_ids.shape[1] < 1:
        raise ValueError(
            "document_ids must have [micro_batch, sequence] shape with both "
            "dimensions positive"
        )
    total_tokens = int(document_ids.numel())
    if total_tokens > 2**31 - 1:
        raise ValueError(
            f"packed document offsets exceed signed int32: tokens={total_tokens}"
        )

    row_length = int(document_ids.shape[1])
    pieces = [
        torch.zeros(1, device=document_ids.device, dtype=torch.int32)
    ]
    for batch_index, row in enumerate(document_ids):
        row_endpoints = _one_row_cumulative_document_lengths(row)
        pieces.append(row_endpoints[1:] + batch_index * row_length)
    return torch.cat(pieces, dim=0).contiguous()


def build_packed_seq_params(
    global_document_ids: Tensor,
    local_position_ids: Tensor,
) -> PackedSeqParams:
    """Build full-document metadata plus CP-local rotary positions.

    Ulysses first reconstructs the global token sequence with all-to-all, so
    its TND kernel must receive global cu_seqlens. RoPE is applied before that
    all-to-all and therefore consumes the CP-local position row separately.
    """

    if local_position_ids.ndim != 2 or min(local_position_ids.shape) < 1:
        raise ValueError(
            "Packed OLMo3 local position_ids must have "
            "[micro_batch, local_tokens] shape with both dimensions positive."
        )
    if global_document_ids.ndim == 1:
        global_document_ids = global_document_ids.unsqueeze(0)
    if global_document_ids.ndim != 2 or min(global_document_ids.shape) < 1:
        raise ValueError(
            "Packed OLMo3 global document_ids must have "
            "[micro_batch, global_tokens] shape with both dimensions positive."
        )
    if global_document_ids.shape[0] != local_position_ids.shape[0]:
        raise ValueError(
            "Packed OLMo3 document IDs and local positions must share the "
            "microbatch dimension; "
            f"got {tuple(global_document_ids.shape)} and "
            f"{tuple(local_position_ids.shape)}."
        )
    cu_seqlens = cumulative_document_lengths(global_document_ids)
    packed = build_packed_seq_params_from_cu_seqlens(
        cu_seqlens,
        local_position_ids,
        micro_batch_size=int(global_document_ids.shape[0]),
        global_tokens_per_sample=int(global_document_ids.shape[1]),
    )
    # Preserve the original public helper's identity/reference contract for
    # CP1 and callers that still provide token-wide document IDs.
    packed.olmo3_global_document_ids = global_document_ids
    return packed


def build_packed_seq_params_from_cu_seqlens(
    cu_seqlens: Tensor,
    local_position_ids: Tensor,
    *,
    micro_batch_size: int,
    global_tokens_per_sample: int,
) -> PackedSeqParams:
    """Build packed metadata from compact canonical document endpoints.

    Stage-3 CP ranks do not need a 65,536-element document-ID row after the
    DataLoader owner has converted it to endpoints.  This entry point lets the
    owner scatter only local tokens/positions and broadcast the tens of int32
    endpoints consumed by CANN, RoPE, and the exact SWA halo planner.
    """

    if local_position_ids.ndim != 2 or min(local_position_ids.shape) < 1:
        raise ValueError(
            "Packed OLMo3 local position_ids must have "
            "[micro_batch, local_tokens] shape with both dimensions positive."
        )
    if isinstance(micro_batch_size, bool) or micro_batch_size < 1:
        raise ValueError("Packed OLMo3 micro_batch_size must be positive.")
    if (
        isinstance(global_tokens_per_sample, bool)
        or global_tokens_per_sample < 1
    ):
        raise ValueError("Packed OLMo3 global_tokens_per_sample must be positive.")
    if int(local_position_ids.shape[0]) != micro_batch_size:
        raise ValueError(
            "Packed OLMo3 local positions and compact metadata disagree on "
            f"microbatch: positions={local_position_ids.shape[0]}, "
            f"metadata={micro_batch_size}."
        )
    if not isinstance(cu_seqlens, Tensor) or cu_seqlens.ndim != 1:
        raise ValueError("Packed OLMo3 cu_seqlens must be a one-dimensional tensor.")
    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    if cu_seqlens.numel() < micro_batch_size + 1:
        raise ValueError(
            "Packed OLMo3 cu_seqlens must contain at least one document per sample."
        )
    endpoints = [int(value) for value in cu_seqlens.tolist()]
    expected_tokens = micro_batch_size * global_tokens_per_sample
    if endpoints[0] != 0 or endpoints[-1] != expected_tokens:
        raise ValueError(
            "Packed OLMo3 compact endpoints do not span the global batch: "
            f"start={endpoints[0]}, end={endpoints[-1]}, "
            f"expected={expected_tokens}."
        )
    if any(right <= left for left, right in zip(endpoints, endpoints[1:])):
        raise ValueError("Packed OLMo3 compact endpoints must be strictly increasing.")
    row_boundaries = {
        row_index * global_tokens_per_sample
        for row_index in range(1, micro_batch_size)
    }
    if not row_boundaries.issubset(endpoints):
        raise ValueError(
            "Packed OLMo3 compact endpoints must retain every microbatch row boundary."
        )

    max_seqlen = max(
        right - left for left, right in zip(endpoints, endpoints[1:])
    )
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    # This is intentionally a dynamic attribute. MindSpeed forwards only the
    # dataclass fields above to FA, while MCore Attention consumes this row for
    # exact document-relative RoPE before CP communication.
    packed.position_ids = local_position_ids
    packed.olmo3_micro_batch_size = micro_batch_size
    packed.olmo3_global_tokens_per_sample = global_tokens_per_sample
    packed.olmo3_local_tokens_per_sample = int(local_position_ids.shape[1])
    # The builder already paid the one compact device-to-host validation.
    # Seed both attention caches so the first Full and SWA layer do not repeat
    # a synchronization merely to strip the canonical leading zero.
    cann_endpoints = endpoints[1:]
    packed._olmo3_cann_cu_seqlens_q_endpoints = cann_endpoints
    packed._olmo3_cann_cu_seqlens_kv_endpoints = cann_endpoints
    return packed


class Olmo3PackedGPTDataset(GPTDataset):
    """GPTDataset that exports fixed-width document boundaries for batching."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.reset_position_ids or self.config.reset_attention_mask:
            raise ValueError(
                "OLMo3 packed data owns document boundaries; Megatron reset flags must be off."
            )
        if self.config.create_attention_mask:
            raise ValueError(
                "OLMo3 packed data requires the fused TND kernel, not a dense dataloader mask."
            )
        if self.config.eod_mask_loss:
            raise ValueError(
                "Official OLMo3 packed labels retain EOD targets; --eod-mask-loss must be off."
            )

    def __getitem__(self, idx: int | None) -> dict[str, Tensor]:
        sample = super().__getitem__(idx)
        document_ids, position_ids = document_ids_and_position_ids(
            sample["tokens"], self.config.tokenizer.eod
        )
        sample["document_ids"] = document_ids
        sample["position_ids"] = position_ids
        return sample
