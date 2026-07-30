"""Fail-closed OLMo 3 checkpoint identity and lifecycle contracts.

Megatron's ``torch_dist`` format is topology-independent when the distributed
optimizer is saved in fully-sharded model space.  It does not, however, encode
the project-level distinction between the base and Siamese/Depth architectures.
This module adds a small sidecar contract at the checkpoint root and validates
that contract before a resume, stage transition, or topology reshard.

The sidecar never contains an absolute checkpoint path.  Checkpoint locations
remain launch parameters, so a checkpoint tree can be moved without rewriting
its identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigurationError


SCHEMA = "olmo3.mindspeed.checkpoint/v1"
CONTRACT_NAME = "olmo3_checkpoint_contract.json"
INTENT_DIR = "olmo3_checkpoint_intents"
TRACKER_NAME = "latest_checkpointed_iteration.txt"
_ITERATION_RE = re.compile(r"^iter_(\d+)$")

# Stage 4 has two intentional entry points.  The released workflow runs Think
# first and Instruct second, while keeping direct Stage3 -> Instruct available
# for an independently trained Instruct branch.
_TRANSITIONS = {
    "stage1": frozenset({"stage2"}),
    "stage2": frozenset({"stage3"}),
    "stage3": frozenset({"sft_think", "sft_instruct"}),
    "sft_think": frozenset({"sft_instruct"}),
    "sft_instruct": frozenset(),
}


def _safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ConfigurationError(f"unsafe {label}: {value!r}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def architecture_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that determine persistent parameter key/shape space."""

    model = config["model"]
    variant = config["variant"]
    return {
        "model_size": model["size"],
        "checkpoint_keyspace": variant["checkpoint_keyspace"],
        "variant": variant["name"],
        "model_impl": variant["model_impl"],
        "num_layers": model["num_layers"],
        "hidden_size": model["hidden_size"],
        "ffn_hidden_size": model["ffn_hidden_size"],
        "num_attention_heads": model["num_attention_heads"],
        "num_query_groups": model["num_query_groups"],
        "kv_channels": model["kv_channels"],
        "true_vocab_size": model["true_vocab_size"],
        "padded_vocab_size": model["padded_vocab_size"],
        "tie_word_embeddings": model["tie_word_embeddings"],
        "qk_norm": model["qk_norm"],
        "qk_norm_mode": model["qk_norm_mode"],
        "rms_norm_eps": model["rms_norm_eps"],
        "use_siamese_norm": variant["use_siamese_norm"],
        "siamese_norm_variant": variant["siamese_norm_variant"],
        "use_depth_attention": variant["use_depth_attention"],
        "depth_attention_stride": variant["depth_attention_stride"],
    }


def topology_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    topology = config["topology"]
    hsdp = topology["hsdp"]
    return {
        "world_size": topology["world_size"],
        "tensor_parallel": topology["tensor_parallel"],
        "context_parallel": topology["context_parallel"],
        "pipeline_parallel": topology["pipeline_parallel"],
        "data_parallel": topology["data_parallel"],
        "sequence_parallel": bool(topology["sequence_parallel"]),
        "hsdp_shard_size": hsdp["shard_size"],
        "hsdp_instances": hsdp["num_instances"],
    }


def schedule_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return trainer state that must not change during a same-stage resume."""

    training = config["training"]
    optimization = config["optimization"]
    return {
        "stage": config["stage"]["name"],
        "global_batch_size": training["global_batch_size"],
        "micro_batch_size": training["micro_batch_size"],
        "train_iters": training["train_iters"],
        "peak_lr": optimization["peak_lr"],
        "min_lr": optimization["min_lr"],
        "scheduler": optimization["scheduler"],
        "warmup_steps": optimization["warmup_steps"],
        "warmup_tokens": optimization["warmup_tokens"],
        "optimizer": {
            "name": optimization["optimizer"],
            "weight_decay": optimization["weight_decay"],
            "adam_beta1": optimization["adam_beta1"],
            "adam_beta2": optimization["adam_beta2"],
            "adam_eps": optimization["adam_eps"],
            "z_loss": optimization["z_loss"],
            "grad_clip": optimization["grad_clip"],
        },
    }


def lifecycle_policy(lifecycle: str) -> dict[str, bool]:
    if lifecycle == "fresh":
        return {
            "load_model": False,
            "load_adam_moments": False,
            "load_master_params": False,
            "load_scheduler": False,
            "load_rng": False,
            "load_counters": False,
        }
    if lifecycle == "resume":
        return {
            "load_model": True,
            "load_adam_moments": True,
            "load_master_params": True,
            "load_scheduler": True,
            "load_rng": True,
            "load_counters": True,
        }
    if lifecycle == "transition":
        return {
            "load_model": True,
            "load_adam_moments": True,
            "load_master_params": True,
            "load_scheduler": False,
            "load_rng": False,
            "load_counters": False,
        }
    raise ConfigurationError(f"unknown checkpoint lifecycle {lifecycle!r}")


def effective_lifecycle_policy(
    config: Mapping[str, Any],
    source_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the policy after accounting for MCore's TP/PP RNG restriction."""

    policy: dict[str, Any] = lifecycle_policy(config["runtime"]["lifecycle"])
    if config["runtime"]["lifecycle"] == "resume" and source_contract is not None:
        source_topology = source_contract.get("writer", {}).get("topology", {})
        target_topology = topology_identity(config)
        changed_model_parallel_rng_space = any(
            source_topology.get(name) != target_topology.get(name)
            for name in ("tensor_parallel", "pipeline_parallel")
        )
        if changed_model_parallel_rng_space:
            # This mirrors Megatron checkpointing.py: torch_dist can reshard
            # model/optimizer tensors, but TP/PP RNG tracker state is ignored.
            policy["load_rng"] = False
            policy["rng_reset_reason"] = "tensor_or_pipeline_parallel_changed"
    return policy


def build_contract(
    config: Mapping[str, Any],
    *,
    source_contract: Mapping[str, Any] | None = None,
    adopted: bool = False,
    optimizer_state: bool = True,
) -> dict[str, Any]:
    architecture = architecture_identity(config)
    lifecycle = config["runtime"]["lifecycle"]
    stage = config["stage"]["name"]
    lineage: list[dict[str, Any]] = []
    if source_contract is not None:
        lineage.extend(source_contract.get("lineage", ()))
        source_writer = source_contract.get("writer", {})
        lineage.append(
            {
                "stage": source_writer.get("stage"),
                "run_id": source_writer.get("run_id"),
                "contract_sha256": contract_sha256(source_contract),
            }
        )
    contract = {
        "schema": SCHEMA,
        "architecture": architecture,
        "architecture_sha256": _sha256(architecture),
        "schedule": schedule_identity(config),
        "format": {
            "name": "torch_dist",
            "version": 1,
            "distributed_optimizer": bool(optimizer_state),
            "optimizer": "adamw",
            "optimizer_state": (
                [
                    "master_params_fp32",
                    "adam_exp_avg",
                    "adam_exp_avg_sq",
                ]
                if optimizer_state
                else []
            ),
            # This is Megatron's default in core_v0.12.1.  The pipeline never
            # emits --no-ckpt-fully-parallel-save.
            "fully_sharded_model_space": bool(optimizer_state),
            "topology_reshardable": bool(optimizer_state),
        },
        "writer": {
            "stage": stage,
            "run_id": config["runtime"]["run_id"],
            "topology": topology_identity(config),
            "lifecycle": lifecycle,
            "policy": effective_lifecycle_policy(config, source_contract),
        },
        "lineage": lineage,
        "adopted_legacy_checkpoint": bool(adopted),
    }
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return _sha256(payload)


def contract_path(checkpoint_root: Path | str) -> Path:
    return Path(checkpoint_root).expanduser().resolve() / CONTRACT_NAME


def _validate_contract_value(value: Any, label: Path | str) -> dict[str, Any]:
    path = str(label)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ConfigurationError(
            f"unsupported checkpoint contract schema in {path}: "
            f"{value.get('schema') if isinstance(value, dict) else type(value).__name__}"
        )
    architecture = value.get("architecture")
    if not isinstance(architecture, dict):
        raise ConfigurationError(f"checkpoint contract has no architecture object: {path}")
    actual_digest = _sha256(architecture)
    if value.get("architecture_sha256") != actual_digest:
        raise ConfigurationError(
            f"checkpoint architecture digest mismatch in {path}; "
            "the sidecar may have been edited"
        )
    if value.get("contract_sha256") != contract_sha256(value):
        raise ConfigurationError(
            f"checkpoint contract digest mismatch in {path}; "
            "non-architecture fields may have been edited"
        )
    schedule = value.get("schedule")
    if schedule is not None and not isinstance(schedule, dict):
        raise ConfigurationError(
            f"checkpoint contract schedule is not an object: {path}"
        )
    return value


def load_contract_file(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(
            f"checkpoint contract file is missing: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read checkpoint contract {path}: {exc}") from exc
    return _validate_contract_value(value, path)


def load_contract(checkpoint_root: Path | str) -> dict[str, Any]:
    path = contract_path(checkpoint_root)
    if not path.is_file():
        raise ConfigurationError(
            f"checkpoint identity contract is missing: {path}. "
            "Refusing to guess model size or variant. Use the checkpoint "
            "'adopt' command once for a verified legacy checkpoint."
        )
    return load_contract_file(path)


def write_contract(
    checkpoint_root: Path | str,
    contract: Mapping[str, Any],
    *,
    allow_existing_equal: bool = True,
) -> Path:
    path = contract_path(checkpoint_root)
    if contract.get("contract_sha256") != contract_sha256(contract):
        raise ConfigurationError(
            "refusing to write an unsealed or modified checkpoint contract"
        )
    if path.exists():
        current = load_contract(path.parent)
        if allow_existing_equal and current == dict(contract):
            return path
        raise ConfigurationError(
            f"checkpoint contract already exists and is immutable: {path}"
        )
    _atomic_json(path, dict(contract))
    return path


def _read_tracker(root: Path) -> tuple[int | None, str | None]:
    path = root / TRACKER_NAME
    if not path.is_file():
        return None, None
    value = path.read_text(encoding="utf-8").strip()
    if value == "release":
        return None, "release"
    try:
        iteration = int(value)
    except ValueError:
        return None, f"invalid:{value}"
    if iteration < 0:
        return None, f"invalid:{value}"
    return iteration, None


def _checkpoint_metadata_error(value: Any) -> str | None:
    """Validate the pinned Megatron distributed-checkpoint descriptor."""

    if not isinstance(value, dict):
        return "metadata.json is not an object"
    allowed = {
        "sharded_backend",
        "sharded_backend_version",
        "common_backend",
        "common_backend_version",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        return f"metadata.json has unsupported fields: {unknown}"
    if value.get("sharded_backend") != "torch_dist":
        return "metadata.json sharded_backend must be 'torch_dist'"
    version = value.get("sharded_backend_version")
    if isinstance(version, bool) or version != 1:
        return "metadata.json sharded_backend_version must be 1"
    if value.get("common_backend", "torch") != "torch":
        return "metadata.json common_backend must be 'torch'"
    common_version = value.get("common_backend_version", 1)
    if isinstance(common_version, bool) or common_version != 1:
        return "metadata.json common_backend_version must be 1"
    return None


def _storage_field(storage_info: Any, name: str) -> Any:
    if isinstance(storage_info, Mapping):
        return storage_info.get(name)
    return getattr(storage_info, name, None)


def _inspect_dcp_metadata(
    iteration_dir: Path,
    metadata_path: Path,
) -> tuple[int, list[str]]:
    """Validate DCP storage extents without deserializing tensor payloads."""

    errors: list[str] = []
    try:
        metadata_size = metadata_path.stat().st_size
    except OSError as exc:
        return 0, [f"cannot stat .metadata: {exc}"]
    if metadata_size <= 0:
        return 0, [".metadata is empty"]
    try:
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
    except Exception as exc:
        return 0, [f"cannot decode PyTorch DCP .metadata: {exc}"]

    storage_data = getattr(metadata, "storage_data", None)
    if not isinstance(storage_data, Mapping):
        return 0, [".metadata storage_data is not a mapping"]
    if not storage_data:
        return 0, [".metadata storage_data is empty"]

    base = iteration_dir.resolve()
    reference_count = 0
    for storage_key, storage_info in storage_data.items():
        reference_count += 1
        label = f"storage_data[{storage_key!r}]"
        raw_relative_path = _storage_field(storage_info, "relative_path")
        if (
            not isinstance(raw_relative_path, str)
            or not raw_relative_path
            or "\x00" in raw_relative_path
            or "\\" in raw_relative_path
        ):
            errors.append(f"{label} has unsafe relative_path={raw_relative_path!r}")
            continue
        relative_path = Path(raw_relative_path)
        if (
            relative_path.is_absolute()
            or relative_path == Path(".")
            or any(part == ".." for part in relative_path.parts)
        ):
            errors.append(f"{label} has unsafe relative_path={raw_relative_path!r}")
            continue
        referenced_path = (iteration_dir / relative_path).resolve()
        if not referenced_path.is_relative_to(base):
            errors.append(f"{label} escapes the checkpoint iteration directory")
            continue
        if referenced_path.suffix != ".distcp":
            errors.append(f"{label} does not reference a .distcp file")
            continue

        offset = _storage_field(storage_info, "offset")
        length = _storage_field(storage_info, "length")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            errors.append(f"{label} has invalid offset={offset!r}")
            continue
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            errors.append(f"{label} has invalid length={length!r}")
            continue
        if not referenced_path.is_file():
            errors.append(
                f"{label} references a missing file: {raw_relative_path!r}"
            )
            continue
        try:
            referenced_size = referenced_path.stat().st_size
        except OSError as exc:
            errors.append(f"{label} cannot stat {raw_relative_path!r}: {exc}")
            continue
        required_size = offset + length
        if referenced_size < required_size:
            errors.append(
                f"{label} extent exceeds {raw_relative_path!r}: "
                f"size={referenced_size}, required={required_size}"
            )
    return reference_count, errors


def inspect_checkpoint(checkpoint_root: Path | str) -> dict[str, Any]:
    """Inspect a checkpoint tree without importing torch or loading tensors."""

    root = Path(checkpoint_root).expanduser().resolve()
    tracker_iteration, tracker_state = _read_tracker(root)
    iterations: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            match = _ITERATION_RE.match(child.name)
            if not match or not child.is_dir():
                continue
            metadata_path = child / "metadata.json"
            metadata: dict[str, Any] | None = None
            metadata_error: str | None = None
            integrity_errors: list[str] = []
            if metadata_path.is_file():
                try:
                    decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata = decoded if isinstance(decoded, dict) else None
                    metadata_error = _checkpoint_metadata_error(decoded)
                except (OSError, json.JSONDecodeError) as exc:
                    metadata_error = str(exc)
            else:
                # Older Megatron Core torch_dist checkpoints do not have this
                # informational sidecar.  Completion is proven by the tracker,
                # common state, DCP metadata, and every referenced non-empty
                # shard below; keep missing metadata visible to inspection but
                # do not reject an otherwise complete legacy iteration.
                metadata_error = "metadata.json is absent (legacy torch_dist)"
            if metadata_error is not None:
                if metadata_path.is_file():
                    integrity_errors.append(metadata_error)

            common_path = child / "common.pt"
            try:
                common_size = common_path.stat().st_size if common_path.is_file() else 0
            except OSError as exc:
                common_size = 0
                integrity_errors.append(f"cannot stat common.pt: {exc}")
            if common_size <= 0:
                integrity_errors.append("common.pt is missing or empty")

            distcp_paths = [
                path for path in child.rglob("*.distcp") if path.is_file()
            ]
            distcp_files = len(distcp_paths)
            nonempty_distcp_files = 0
            for distcp_path in distcp_paths:
                try:
                    if distcp_path.stat().st_size > 0:
                        nonempty_distcp_files += 1
                except OSError as exc:
                    integrity_errors.append(
                        f"cannot stat {distcp_path.relative_to(child)}: {exc}"
                    )
            if nonempty_distcp_files < 1:
                integrity_errors.append("no non-empty .distcp file")

            dcp_metadata_path = child / ".metadata"
            storage_references = 0
            dcp_metadata_errors: list[str] = []
            if not dcp_metadata_path.is_file():
                dcp_metadata_errors.append(".metadata is missing")
            else:
                storage_references, dcp_metadata_errors = _inspect_dcp_metadata(
                    child,
                    dcp_metadata_path,
                )
            integrity_errors.extend(dcp_metadata_errors)
            iterations.append(
                {
                    "iteration": int(match.group(1)),
                    "directory": child.name,
                    "metadata_present": metadata_path.is_file(),
                    "metadata": metadata,
                    "metadata_error": metadata_error,
                    "common_pt_present": common_path.is_file(),
                    "common_pt_size": common_size,
                    "dcp_metadata_present": dcp_metadata_path.is_file(),
                    "dcp_metadata_error": (
                        "; ".join(dcp_metadata_errors)
                        if dcp_metadata_errors
                        else None
                    ),
                    "storage_references": storage_references,
                    "distcp_files": distcp_files,
                    "nonempty_distcp_files": nonempty_distcp_files,
                    "integrity_errors": integrity_errors,
                    "complete": not integrity_errors,
                }
            )
    latest = next(
        (
            item
            for item in iterations
            if tracker_iteration is not None
            and item["iteration"] == tracker_iteration
        ),
        None,
    )
    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    if contract_path(root).exists():
        try:
            contract = load_contract(root)
        except ConfigurationError as exc:
            contract_error = str(exc)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "tracker_iteration": tracker_iteration,
        "tracker_state": tracker_state,
        "iterations": iterations,
        "latest": latest,
        "latest_complete": bool(latest and latest["complete"]),
        "contract": contract,
        "contract_sha256": contract_sha256(contract) if contract else None,
        "contract_error": contract_error,
    }


def _validate_source_checkpoint_exists(checkpoint_root: Path | str) -> None:
    inspection = inspect_checkpoint(checkpoint_root)
    if not inspection["exists"]:
        raise ConfigurationError(f"checkpoint root does not exist: {checkpoint_root}")
    if not inspection["latest_complete"]:
        raise ConfigurationError(
            "checkpoint tracker does not point to a complete torch_dist iteration: "
            f"{checkpoint_root}"
        )


def validate_lifecycle_source(
    config: Mapping[str, Any],
    source_contract: Mapping[str, Any],
) -> None:
    lifecycle = config["runtime"]["lifecycle"]
    if lifecycle not in {"resume", "transition"}:
        raise ConfigurationError(
            "source checkpoint validation applies only to resume or transition"
        )
    expected_arch = architecture_identity(config)
    actual_arch = source_contract["architecture"]
    if actual_arch != expected_arch:
        changed = sorted(
            key
            for key in set(actual_arch) | set(expected_arch)
            if actual_arch.get(key) != expected_arch.get(key)
        )
        raise ConfigurationError(
            "checkpoint architecture/keyspace mismatch; refusing a loose load. "
            f"changed_fields={changed}"
        )

    source_stage = source_contract.get("writer", {}).get("stage")
    target_stage = config["stage"]["name"]
    if lifecycle == "resume":
        if source_stage != target_stage:
            raise ConfigurationError(
                f"same-stage resume requires stage={target_stage}, "
                f"but checkpoint contract says {source_stage}"
            )
        source_schedule = source_contract.get("schedule")
        if source_schedule is not None:
            if not isinstance(source_schedule, Mapping):
                raise ConfigurationError(
                    "checkpoint contract schedule is not an object"
                )
            expected_schedule = schedule_identity(config)
            if _canonical_json(source_schedule) != _canonical_json(
                expected_schedule
            ):
                changed = sorted(
                    key
                    for key in set(source_schedule) | set(expected_schedule)
                    if _canonical_json(source_schedule.get(key))
                    != _canonical_json(expected_schedule.get(key))
                )
                raise ConfigurationError(
                    "same-stage resume schedule mismatch; refusing to restore "
                    f"scheduler/counters under changed settings: changed_fields={changed}"
                )
    else:
        allowed = _TRANSITIONS.get(str(source_stage), frozenset())
        if target_stage not in allowed:
            raise ConfigurationError(
                f"unsupported OLMo 3 stage transition {source_stage!r} -> "
                f"{target_stage!r}; allowed={sorted(allowed)}"
            )

    source_format = source_contract.get("format", {})
    if (
        source_format.get("name") != "torch_dist"
        or not source_format.get("distributed_optimizer")
    ):
        raise ConfigurationError(
            "resume/transition requires a torch_dist checkpoint containing the "
            "distributed AdamW state"
        )

    old_topology = source_contract.get("writer", {}).get("topology")
    new_topology = topology_identity(config)
    if old_topology != new_topology and not (
        source_format.get("fully_sharded_model_space")
        and source_format.get("topology_reshardable")
    ):
        raise ConfigurationError(
            "checkpoint topology changed but optimizer state was not saved in "
            "fully-sharded model space"
        )

    policy = lifecycle_policy(lifecycle)
    if not (
        policy["load_model"]
        and policy["load_adam_moments"]
        and policy["load_master_params"]
    ):
        raise ConfigurationError("internal lifecycle policy dropped optimizer state")


@dataclass(frozen=True)
class LifecyclePreparation:
    source_contract: dict[str, Any] | None
    target_contract: dict[str, Any]
    target_contract_path: Path
    intent_path: Path


def _intent_payload(
    config: Mapping[str, Any],
    *,
    source_contract: Mapping[str, Any] | None,
    target_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact immutable launch intent for one resolved run."""

    return {
        "schema": SCHEMA,
        "run_id": config["runtime"]["run_id"],
        "lifecycle": config["runtime"]["lifecycle"],
        "source_contract_sha256": (
            contract_sha256(source_contract) if source_contract else None
        ),
        "target_contract_sha256": contract_sha256(target_contract),
        "target_writer": target_contract["writer"],
    }


def prepare_lifecycle(
    config: Mapping[str, Any],
    *,
    write: bool,
    verify_checkpoint_tree: bool = True,
    source_contract_override: Mapping[str, Any] | None = None,
) -> LifecyclePreparation:
    """Validate the load contract and prepare the immutable save contract."""

    lifecycle = config["runtime"]["lifecycle"]
    load_root = config["runtime"].get("load")
    save_root = Path(config["runtime"]["save"]).expanduser().resolve()
    source_contract: dict[str, Any] | None = None
    if lifecycle in {"resume", "transition"}:
        if not load_root:
            raise ConfigurationError(f"{lifecycle} requires runtime.load")
        if verify_checkpoint_tree:
            _validate_source_checkpoint_exists(load_root)
        source_contract = (
            _validate_contract_value(
                dict(source_contract_override),
                "source_contract_override",
            )
            if source_contract_override is not None
            else load_contract(load_root)
        )
        validate_lifecycle_source(config, source_contract)
        if (
            lifecycle == "resume"
            and Path(load_root).expanduser().resolve() == save_root
            and source_contract.get("writer", {}).get("topology")
            != topology_identity(config)
        ):
            raise ConfigurationError(
                "an in-place resume cannot change TP/CP/PP/DP/HSDP topology "
                "because the root identity is immutable; reshard into a new "
                "checkpoint destination"
            )
    elif lifecycle == "fresh":
        inspection = inspect_checkpoint(save_root)
        if inspection["tracker_iteration"] is not None or inspection["tracker_state"]:
            raise ConfigurationError(
                f"fresh launch refuses an existing checkpoint tree: {save_root}"
            )
    else:
        raise ConfigurationError(f"unknown lifecycle {lifecycle!r}")

    target_contract = build_contract(config, source_contract=source_contract)
    target_path = contract_path(save_root)
    run_id = _safe_component(config["runtime"]["run_id"], "run ID")
    intent_path = save_root / INTENT_DIR / f"{run_id}.json"
    if write:
        save_root.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            existing = load_contract(save_root)
            if existing["architecture"] != target_contract["architecture"]:
                raise ConfigurationError(
                    f"save root belongs to another checkpoint keyspace: {save_root}"
                )
            # A same-stage resume writes back into the same tree.  Its root
            # identity remains immutable; the per-run intent records topology.
            if lifecycle != "resume" or Path(load_root).resolve() != save_root:
                raise ConfigurationError(
                    "checkpoint destination already has an identity contract; "
                    "only an in-place same-stage resume may reuse a save root"
                )
        else:
            write_contract(save_root, target_contract)
        if intent_path.exists():
            raise ConfigurationError(f"checkpoint run intent already exists: {intent_path}")
        _atomic_json(
            intent_path,
            _intent_payload(
                config,
                source_contract=source_contract,
                target_contract=target_contract,
            ),
        )
    return LifecyclePreparation(
        source_contract=source_contract,
        target_contract=target_contract,
        target_contract_path=target_path,
        intent_path=intent_path,
    )


def verify_prepared_lifecycle(
    config: Mapping[str, Any],
    *,
    verify_checkpoint_tree: bool = True,
    source_contract_override: Mapping[str, Any] | None = None,
) -> LifecyclePreparation:
    """Verify a control-plane activation without mutating shared storage.

    ``prepare_lifecycle(..., write=True)`` is deliberately a single-writer
    control-plane action. Every distributed worker calls this read-only check
    before entering ``torchrun``. This prevents thousands of ranks from racing
    to create the same sidecar while still making an unprepared or tampered
    launch fail closed.
    """

    prepared = prepare_lifecycle(
        config,
        write=False,
        verify_checkpoint_tree=verify_checkpoint_tree,
        source_contract_override=source_contract_override,
    )
    save_root = Path(config["runtime"]["save"]).expanduser().resolve()
    existing_target = load_contract(save_root)
    lifecycle = config["runtime"]["lifecycle"]
    load_root = config["runtime"].get("load")
    in_place_resume = (
        lifecycle == "resume"
        and load_root is not None
        and Path(load_root).expanduser().resolve() == save_root
    )
    expected_root_contract = (
        prepared.source_contract
        if in_place_resume
        else prepared.target_contract
    )
    if existing_target != expected_root_contract:
        raise ConfigurationError(
            "checkpoint destination contract does not match this activated run: "
            f"{prepared.target_contract_path}"
        )

    try:
        intent = json.loads(prepared.intent_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "checkpoint lifecycle has not been activated by the control plane; "
            f"missing immutable run intent: {prepared.intent_path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read checkpoint run intent {prepared.intent_path}: {exc}"
        ) from exc
    expected_intent = _intent_payload(
        config,
        source_contract=prepared.source_contract,
        target_contract=prepared.target_contract,
    )
    if intent != expected_intent:
        raise ConfigurationError(
            "checkpoint run intent does not match the resolved launch; "
            f"refusing to start: {prepared.intent_path}"
        )
    return prepared


def adopt_legacy_checkpoint(
    checkpoint_root: Path | str,
    config: Mapping[str, Any],
    *,
    acknowledgement: bool,
    weights_only: bool = False,
) -> Path:
    """Attach an explicit identity to a verified pre-sidecar checkpoint.

    Adoption cannot infer the architecture from storage metadata reliably, so
    it requires a deliberate acknowledgement and a fully resolved config.
    """

    if not acknowledgement:
        raise ConfigurationError(
            "legacy adoption requires --i-verified-model-and-variant"
        )
    _validate_source_checkpoint_exists(checkpoint_root)
    contract = build_contract(
        config,
        adopted=True,
        optimizer_state=not weights_only,
    )
    return write_contract(checkpoint_root, contract)
