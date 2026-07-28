"""OLMo3 document-aware PPL validation for Megatron/MindSpeed."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset


_RAW_TOKEN_DTYPE = np.dtype("<u4")
_INDEX_CACHE_VERSION = 1


@dataclass(frozen=True)
class PPLManifestEntry:
    """One named raw-token stream from a PPL validation manifest."""

    name: str
    path: Path
    token_count: int


@dataclass(frozen=True)
class PPLDocumentIndex:
    """Cached document offsets and exact official-eval accounting."""

    source_ids: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    source_document_counts: np.ndarray
    source_valid_target_counts: np.ndarray
    source_truncated_token_counts: np.ndarray
    cache_path: Path | None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_ppl_manifest(manifest_path: Path) -> tuple[PPLManifestEntry, ...]:
    """Validate and load ``name,relative/path`` raw-token manifest entries."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PPL validation manifest does not exist: {manifest_path}")

    root = manifest_path.parent
    entries: list[PPLManifestEntry] = []
    names: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, relative_path = line.partition(",")
            name = name.strip()
            relative_path = relative_path.strip()
            if not separator or not name or not relative_path:
                raise ValueError(
                    f"Invalid PPL manifest row {manifest_path}:{line_number}: "
                    "expected name,relative/path"
                )
            if name in names:
                raise ValueError(f"Duplicate PPL dataset name {name!r} in {manifest_path}")
            names.add(name)

            unresolved_path = Path(relative_path)
            if unresolved_path.is_absolute():
                raise ValueError(
                    f"PPL manifest paths must be relative: {manifest_path}:{line_number}"
                )
            token_path = (root / unresolved_path).resolve()
            if not _is_relative_to(token_path, root):
                raise ValueError(
                    f"PPL manifest path escapes its dataset root: "
                    f"{manifest_path}:{line_number}"
                )
            if not token_path.is_file():
                raise FileNotFoundError(f"PPL token stream does not exist: {token_path}")

            byte_count = token_path.stat().st_size
            if byte_count < _RAW_TOKEN_DTYPE.itemsize:
                raise ValueError(f"PPL token stream must contain at least one token: {token_path}")
            if byte_count % _RAW_TOKEN_DTYPE.itemsize:
                raise ValueError(
                    f"PPL token stream byte size is not uint32-aligned: {token_path}"
                )
            entries.append(
                PPLManifestEntry(
                    name=name,
                    path=token_path,
                    token_count=byte_count // _RAW_TOKEN_DTYPE.itemsize,
                )
            )

    if not entries:
        raise ValueError(f"PPL validation manifest is empty: {manifest_path}")
    return tuple(entries)


def _document_index_fingerprint(
    entries: Sequence[PPLManifestEntry], sequence_length: int, eos_token_id: int
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        f"v={_INDEX_CACHE_VERSION},seq={sequence_length},eos={eos_token_id}\n".encode()
    )
    for entry in entries:
        stat = entry.path.stat()
        hasher.update(
            (
                f"{entry.name}\0{entry.path}\0{entry.token_count}\0"
                f"{stat.st_size}\0{stat.st_mtime_ns}\n"
            ).encode()
        )
    return hasher.hexdigest()


def _scan_document_index(
    entries: Sequence[PPLManifestEntry], sequence_length: int, eos_token_id: int
) -> PPLDocumentIndex:
    source_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    source_document_counts = np.zeros(len(entries), dtype=np.int64)
    source_valid_target_counts = np.zeros(len(entries), dtype=np.int64)
    source_truncated_token_counts = np.zeros(len(entries), dtype=np.int64)

    for source_id, entry in enumerate(entries):
        tokens = np.memmap(entry.path, dtype=_RAW_TOKEN_DTYPE, mode="r")
        eos_ends = np.flatnonzero(tokens == eos_token_id).astype(np.int64) + 1
        if eos_ends.size == 0:
            raise ValueError(
                f"PPL source contains no EOS token {eos_token_id}: {entry.path}"
            )
        if int(eos_ends[-1]) != entry.token_count:
            trailing = entry.token_count - int(eos_ends[-1])
            raise ValueError(
                f"PPL source has {trailing} trailing tokens after its final EOS: {entry.path}"
            )

        source_starts = np.concatenate(
            (np.zeros(1, dtype=np.int64), eos_ends[:-1])
        )
        source_lengths = eos_ends - source_starts
        clipped_lengths = np.minimum(source_lengths, sequence_length)

        source_ids.append(
            np.full(source_lengths.shape, source_id, dtype=np.int16)
        )
        starts.append(source_starts.astype(np.int64, copy=False))
        lengths.append(source_lengths.astype(np.int32, copy=False))
        source_document_counts[source_id] = source_lengths.size
        source_valid_target_counts[source_id] = np.maximum(
            clipped_lengths - 1, 0
        ).sum(dtype=np.int64)
        source_truncated_token_counts[source_id] = np.maximum(
            source_lengths - sequence_length, 0
        ).sum(dtype=np.int64)

    return PPLDocumentIndex(
        source_ids=np.concatenate(source_ids),
        starts=np.concatenate(starts),
        lengths=np.concatenate(lengths),
        source_document_counts=source_document_counts,
        source_valid_target_counts=source_valid_target_counts,
        source_truncated_token_counts=source_truncated_token_counts,
        cache_path=None,
    )


def _write_document_index_cache(
    cache_path: Path,
    index: PPLDocumentIndex,
    *,
    fingerprint: str,
    sequence_length: int,
    eos_token_id: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.tmp-rank{dist.get_rank() if dist.is_initialized() else 0}-pid{os.getpid()}"
    )
    try:
        with temporary_path.open("wb") as handle:
            np.savez(
                handle,
                version=np.asarray(_INDEX_CACHE_VERSION, dtype=np.int64),
                fingerprint=np.asarray(fingerprint),
                sequence_length=np.asarray(sequence_length, dtype=np.int64),
                eos_token_id=np.asarray(eos_token_id, dtype=np.int64),
                source_ids=index.source_ids,
                starts=index.starts,
                lengths=index.lengths,
                source_document_counts=index.source_document_counts,
                source_valid_target_counts=index.source_valid_target_counts,
                source_truncated_token_counts=index.source_truncated_token_counts,
            )
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_document_index_cache(
    cache_path: Path,
    *,
    fingerprint: str,
    sequence_length: int,
    eos_token_id: int,
    entries: Sequence[PPLManifestEntry],
) -> PPLDocumentIndex:
    with np.load(cache_path, allow_pickle=False) as cache:
        if int(cache["version"]) != _INDEX_CACHE_VERSION:
            raise ValueError("document-index cache version mismatch")
        if str(cache["fingerprint"]) != fingerprint:
            raise ValueError("document-index cache fingerprint mismatch")
        if int(cache["sequence_length"]) != sequence_length:
            raise ValueError("document-index cache sequence length mismatch")
        if int(cache["eos_token_id"]) != eos_token_id:
            raise ValueError("document-index cache EOS token mismatch")
        source_ids = np.array(cache["source_ids"], dtype=np.int16, copy=True)
        starts = np.array(cache["starts"], dtype=np.int64, copy=True)
        lengths = np.array(cache["lengths"], dtype=np.int32, copy=True)
        source_document_counts = np.array(
            cache["source_document_counts"], dtype=np.int64, copy=True
        )
        source_valid_target_counts = np.array(
            cache["source_valid_target_counts"], dtype=np.int64, copy=True
        )
        source_truncated_token_counts = np.array(
            cache["source_truncated_token_counts"], dtype=np.int64, copy=True
        )

    if source_ids.ndim != 1 or starts.shape != source_ids.shape or lengths.shape != source_ids.shape:
        raise ValueError("document-index cache arrays have inconsistent shapes")
    source_count = len(entries)
    expected_source_shape = (source_count,)
    for name, values in (
        ("source_document_counts", source_document_counts),
        ("source_valid_target_counts", source_valid_target_counts),
        ("source_truncated_token_counts", source_truncated_token_counts),
    ):
        if values.shape != expected_source_shape:
            raise ValueError(
                f"document-index cache {name} shape mismatch: "
                f"expected {expected_source_shape}, got {values.shape}"
            )
    if source_ids.size == 0:
        raise ValueError("document-index cache contains no documents")
    if int(source_ids.min()) < 0 or int(source_ids.max()) >= source_count:
        raise ValueError("document-index cache contains an invalid source ID")
    if np.any(starts < 0) or np.any(lengths < 1):
        raise ValueError("document-index cache contains an invalid document span")
    for source_id, entry in enumerate(entries):
        positions = np.flatnonzero(source_ids == source_id)
        if positions.size != int(source_document_counts[source_id]):
            raise ValueError(
                "document-index cache source document count mismatch"
            )
        if positions.size == 0 or np.any(np.diff(positions) != 1):
            raise ValueError(
                "document-index cache source documents are missing or interleaved"
            )
        source_starts = starts[positions]
        source_lengths = lengths[positions].astype(np.int64, copy=False)
        if (
            int(source_starts[0]) != 0
            or np.any(source_starts[1:] != source_starts[:-1] + source_lengths[:-1])
            or int(source_starts[-1] + source_lengths[-1]) != entry.token_count
        ):
            raise ValueError(
                "document-index cache does not cover its source token stream exactly"
            )
        clipped_lengths = np.minimum(source_lengths, sequence_length)
        valid_targets = np.maximum(clipped_lengths - 1, 0).sum(dtype=np.int64)
        truncated_tokens = np.maximum(
            source_lengths - sequence_length, 0
        ).sum(dtype=np.int64)
        if valid_targets != source_valid_target_counts[source_id]:
            raise ValueError(
                "document-index cache valid-target accounting mismatch"
            )
        if truncated_tokens != source_truncated_token_counts[source_id]:
            raise ValueError(
                "document-index cache truncation accounting mismatch"
            )

    return PPLDocumentIndex(
        source_ids=source_ids,
        starts=starts,
        lengths=lengths,
        source_document_counts=source_document_counts,
        source_valid_target_counts=source_valid_target_counts,
        source_truncated_token_counts=source_truncated_token_counts,
        cache_path=cache_path,
    )


def load_or_build_document_index(
    entries: Sequence[PPLManifestEntry],
    sequence_length: int,
    eos_token_id: int,
    *,
    cache_dir: Path | None = None,
    sync_process_group: Any = None,
) -> PPLDocumentIndex:
    """Load a shared document index, building it once per distributed job."""

    if cache_dir is None:
        return _scan_document_index(entries, sequence_length, eos_token_id)

    fingerprint = _document_index_fingerprint(entries, sequence_length, eos_token_id)
    cache_path = Path(cache_dir).expanduser().resolve() / (
        f"olmo3-ppl-doc-index-v{_INDEX_CACHE_VERSION}-"
        f"s{sequence_length}-eos{eos_token_id}-{fingerprint[:20]}.npz"
    )
    distributed = dist.is_available() and dist.is_initialized()
    group_rank = dist.get_rank(group=sync_process_group) if distributed else 0
    group_world_size = (
        dist.get_world_size(group=sync_process_group) if distributed else 1
    )

    if group_rank == 0:
        rebuild = not cache_path.is_file()
        if not rebuild:
            try:
                _load_document_index_cache(
                    cache_path,
                    fingerprint=fingerprint,
                    sequence_length=sequence_length,
                    eos_token_id=eos_token_id,
                    entries=entries,
                )
            except (OSError, KeyError, ValueError):
                rebuild = True
        if rebuild:
            index = _scan_document_index(entries, sequence_length, eos_token_id)
            _write_document_index_cache(
                cache_path,
                index,
                fingerprint=fingerprint,
                sequence_length=sequence_length,
                eos_token_id=eos_token_id,
            )

    if group_world_size > 1:
        dist.barrier(group=sync_process_group)
    return _load_document_index_cache(
        cache_path,
        fingerprint=fingerprint,
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        entries=entries,
    )


class PPLManifestDataset(Dataset):
    """Expose official OLMo3 document-padded validation samples.

    EOS-delimited documents stay independent. Each document is truncated to
    ``sequence_length`` and right-padded, and only transitions within that
    document contribute to the loss. The dataset length is padded to a whole
    global batch with all-masked dummy samples so distributed evaluation never
    repeats or drops a real document.
    """

    deterministic_eval = True

    def __init__(
        self,
        manifest_path: Path,
        sequence_length: int,
        global_batch_size: int,
        *,
        eos_token_id: int,
        pad_token_id: int,
        split: Any = None,
        cache_dir: Path | None = None,
        sync_process_group: Any = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if eos_token_id < 0 or pad_token_id < 0:
            raise ValueError("EOS and padding token IDs must be non-negative")

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.entries = load_ppl_manifest(self.manifest_path)
        self.sequence_length = int(sequence_length)
        self.global_batch_size = int(global_batch_size)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.split = split
        self.dataset_names = tuple(entry.name for entry in self.entries)
        self.dummy_source_id = len(self.entries)

        self.document_index = load_or_build_document_index(
            self.entries,
            self.sequence_length,
            self.eos_token_id,
            cache_dir=cache_dir,
            sync_process_group=sync_process_group,
        )
        empty_target_sources = [
            name
            for name, count in zip(
                self.dataset_names,
                self.document_index.source_valid_target_counts,
            )
            if int(count) <= 0
        ]
        if empty_target_sources:
            raise ValueError(
                "Every PPL source must contain at least one valid next-token "
                f"target; empty sources: {empty_target_sources}"
            )
        self.real_sample_count = int(self.document_index.source_ids.size)
        self.eval_iters = (
            self.real_sample_count + self.global_batch_size - 1
        ) // self.global_batch_size
        self.padded_sample_count = self.eval_iters * self.global_batch_size
        self.dummy_sample_count = self.padded_sample_count - self.real_sample_count
        self.total_token_count = sum(entry.token_count for entry in self.entries)
        self.valid_target_count = int(
            self.document_index.source_valid_target_counts.sum()
        )
        self.truncated_token_count = int(
            self.document_index.source_truncated_token_counts.sum()
        )
        clipped_token_count = self.valid_target_count + self.real_sample_count
        self.padding_token_count = (
            self.real_sample_count * self.sequence_length - clipped_token_count
        )
        self._arrays: tuple[np.memmap, ...] | None = None
        self._position_ids = torch.arange(self.sequence_length, dtype=torch.long)

    def __len__(self) -> int:
        return self.padded_sample_count

    def _get_arrays(self) -> tuple[np.memmap, ...]:
        if self._arrays is None:
            self._arrays = tuple(
                np.memmap(entry.path, dtype=_RAW_TOKEN_DTYPE, mode="r")
                for entry in self.entries
            )
        return self._arrays

    def _dummy_sample(self) -> dict[str, Tensor]:
        return {
            "tokens": torch.full(
                (self.sequence_length,), self.pad_token_id, dtype=torch.long
            ),
            "labels": torch.full((self.sequence_length,), -100, dtype=torch.long),
            "loss_mask": torch.zeros(self.sequence_length, dtype=torch.float32),
            "position_ids": self._position_ids,
            "ppl_source_id": torch.tensor(self.dummy_source_id, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        index = int(idx)
        if index < 0:
            index += self.padded_sample_count
        if index < 0 or index >= self.padded_sample_count:
            raise IndexError(index)
        if index >= self.real_sample_count:
            return self._dummy_sample()

        source_id = int(self.document_index.source_ids[index])
        start = int(self.document_index.starts[index])
        document_length = min(
            int(self.document_index.lengths[index]), self.sequence_length
        )
        document = np.asarray(
            self._get_arrays()[source_id][start : start + document_length],
            dtype=np.int64,
        )

        tokens = torch.full(
            (self.sequence_length,), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((self.sequence_length,), -100, dtype=torch.long)
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        tokens[:document_length] = torch.from_numpy(document.copy())
        if document_length > 1:
            labels[: document_length - 1] = torch.from_numpy(document[1:].copy())
            loss_mask[: document_length - 1] = 1.0

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self._position_ids,
            "ppl_source_id": torch.tensor(source_id, dtype=torch.long),
        }

    def validate_expected_counts(
        self,
        *,
        documents: int | None = None,
        valid_targets: int | None = None,
        truncated_tokens: int | None = None,
    ) -> None:
        expected = {
            "documents": documents,
            "valid_targets": valid_targets,
            "truncated_tokens": truncated_tokens,
        }
        actual = {
            "documents": self.real_sample_count,
            "valid_targets": self.valid_target_count,
            "truncated_tokens": self.truncated_token_count,
        }
        mismatches = [
            f"{name}: expected {value}, got {actual[name]}"
            for name, value in expected.items()
            if value is not None and actual[name] != value
        ]
        if mismatches:
            raise ValueError(
                "OLMo3 PPL document accounting mismatch:\n  " + "\n  ".join(mismatches)
            )

    def summary(self) -> dict[str, Any]:
        """Return stable metadata suitable for rank-zero launch logging."""

        return {
            "manifest": str(self.manifest_path),
            "datasets": list(self.dataset_names),
            "dataset_count": len(self.entries),
            "total_tokens": self.total_token_count,
            "sequence_length": self.sequence_length,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "documents": self.real_sample_count,
            "padded_samples": self.padded_sample_count,
            "dummy_samples": self.dummy_sample_count,
            "eval_iters": self.eval_iters,
            "valid_targets": self.valid_target_count,
            "truncated_tokens": self.truncated_token_count,
            "padding_tokens": self.padding_token_count,
            "source_documents": {
                name: int(count)
                for name, count in zip(
                    self.dataset_names,
                    self.document_index.source_document_counts,
                )
            },
            "source_valid_targets": {
                name: int(count)
                for name, count in zip(
                    self.dataset_names,
                    self.document_index.source_valid_target_counts,
                )
            },
            "index_cache": (
                str(self.document_index.cache_path)
                if self.document_index.cache_path is not None
                else None
            ),
        }


def build_local_source_statistics(
    per_sample_loss_stats: Tensor,
    source_ids: Tensor,
    source_count: int,
) -> Tensor:
    """Accumulate ``[total_sum, lm_sum, z_sum, token_count]`` by source."""

    if per_sample_loss_stats.ndim != 2 or per_sample_loss_stats.shape[1] != 4:
        raise ValueError(
            "PPL loss stats must have shape [batch, 4], got "
            f"{tuple(per_sample_loss_stats.shape)}"
        )
    source_ids = source_ids.reshape(-1).to(
        device=per_sample_loss_stats.device, dtype=torch.long
    )
    if source_ids.numel() != per_sample_loss_stats.shape[0]:
        raise ValueError(
            "PPL source IDs must match the loss batch dimension: "
            f"{source_ids.numel()} != {per_sample_loss_stats.shape[0]}"
        )
    if source_count < 1:
        raise ValueError("source_count must be positive")

    source_stats = torch.zeros(
        (source_count, 4),
        dtype=torch.float32,
        device=per_sample_loss_stats.device,
    )
    valid = source_ids.ge(0) & source_ids.lt(source_count)
    if bool(valid.any()):
        source_stats.index_add_(
            0,
            source_ids[valid],
            per_sample_loss_stats.detach().float()[valid],
        )
    return source_stats


def source_statistics_to_loss_dict(
    source_stats: Tensor, dataset_names: Sequence[str]
) -> dict[str, tuple[Tensor, Tensor]]:
    """Build aggregate and per-source numerator/denominator metrics."""

    if source_stats.shape != (len(dataset_names), 4):
        raise ValueError(
            "PPL source-stat shape mismatch: "
            f"expected {(len(dataset_names), 4)}, got {tuple(source_stats.shape)}"
        )
    aggregate = source_stats.sum(dim=0)
    metrics: dict[str, tuple[Tensor, Tensor]] = {
        "total loss": (aggregate[0], aggregate[3]),
        "lm loss": (aggregate[1], aggregate[3]),
        "z loss": (aggregate[2], aggregate[3]),
    }
    for source_id, name in enumerate(dataset_names):
        stats = source_stats[source_id]
        prefix = f"ppl/{name}"
        metrics[f"{prefix}/total loss"] = (stats[0], stats[3])
        metrics[f"{prefix}/lm loss"] = (stats[1], stats[3])
        metrics[f"{prefix}/z loss"] = (stats[2], stats[3])
    return metrics
