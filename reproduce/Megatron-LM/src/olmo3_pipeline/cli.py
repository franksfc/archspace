from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from .command import (
    build_environment,
    build_torchrun_command,
    shell_join,
)
from .checkpoint import prepare_lifecycle
from .config import (
    CONFIG_ROOT,
    PROJECT_ROOT,
    ConfigurationError,
    apply_override,
    compose,
)
from .data_pipeline import (
    DataPipelineError,
    load_runtime_data_manifest,
    sft_packed_work_dir,
    validate_runtime_data_profile,
)
from .model_config import build_model_config
from .provenance import (
    source_manifest,
    upstream_state,
    verify_source_manifest,
    verify_upstream_snapshot,
)
from .resolve import resolve
from .topology import (
    validate_local_listener_ports,
    validate_local_sio_pairs,
    validate_runtime_stack,
)


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


def _atomic_text(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            os.chmod(temporary, 0o755)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _node_rank_value_index(command: list[str]) -> int:
    indices = [
        index + 1
        for index, value in enumerate(command[:-1])
        if value == "--node_rank"
    ]
    if len(indices) != 1:
        raise ConfigurationError(
            "generated torchrun command must contain exactly one --node_rank"
        )
    return indices[0]


def _command_with_node_rank(command: list[str], node_rank: int) -> list[str]:
    result = list(command)
    result[_node_rank_value_index(result)] = str(node_rank)
    return result


def _shell_join_with_worker_node_rank(command: list[str]) -> str:
    rank_index = _node_rank_value_index(command)
    return " ".join(
        '"${worker_node_rank}"' if index == rank_index else shlex.quote(value)
        for index, value in enumerate(command)
    )


def _deployment_node_rank(config: dict[str, Any]) -> int:
    """Resolve a worker-local rank without changing the frozen run contract."""

    raw = os.environ.get("OLMO3_NODE_RANK", os.environ.get("NODE_RANK"))
    if raw is None:
        return int(config["runtime"]["node_rank"])
    try:
        node_rank = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"worker node rank must be an integer, got {raw!r}"
        ) from exc
    nnodes = int(config["runtime"]["nnodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ConfigurationError(
            f"worker node rank={node_rank} must be in [0, {nnodes})"
        )
    return node_rank


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=("1b", "3b", "7b"))
    parser.add_argument(
        "--variant", required=True, choices=("base", "siamese_depth")
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("stage1", "stage2", "stage3", "sft_think", "sft_instruct"),
    )
    parser.add_argument(
        "--data",
        required=True,
        choices=tuple(sorted(p.stem for p in (CONFIG_ROOT / "data").glob("*.json"))),
    )
    parser.add_argument(
        "--topology",
        required=True,
        choices=tuple(
            sorted(p.stem for p in (CONFIG_ROOT / "topology").glob("*.json"))
        ),
    )
    parser.add_argument(
        "--performance",
        action="append",
        default=None,
        choices=tuple(
            sorted(p.stem for p in (CONFIG_ROOT / "performance").glob("*.json"))
        ),
    )
    parser.add_argument("--set", action="append", default=[])

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lifecycle", choices=("fresh", "resume", "transition"))
    parser.add_argument("--data-root")
    parser.add_argument("--data-manifest")
    parser.add_argument("--data-work-dir")
    parser.add_argument("--data-args-path")
    parser.add_argument(
        "--data-cache-path",
        help=(
            "Optional shared Megatron sample-index cache. Defaults to "
            "<output>/data-cache when omitted."
        ),
    )
    parser.add_argument("--tokenizer")
    parser.add_argument("--load")
    parser.add_argument("--save", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--world-size", type=int)
    parser.add_argument("--nproc-per-node", type=int)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--cp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--sp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--hsdp-shard-size", type=int)
    parser.add_argument("--ddp-num-buckets", type=int)

    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--expected-instances", type=int)
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-tokens", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument(
        "--exit-interval",
        type=int,
        help=(
            "Stop cleanly after this iteration while retaining the resolved "
            "full training schedule. Intended for save/resume validation."
        ),
    )
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--valid-manifest")
    parser.add_argument("--valid-cache-dir")

    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--hccl-if-base-port",
        type=int,
        help=(
            "Optional HCCL listener base port, frozen into the run environment. "
            "Use a distinct range for sequential distributed phases."
        ),
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-base-url")
    parser.add_argument(
        "--wandb-log-style",
        choices=("llamafactory", "native"),
        default="llamafactory",
        help=(
            "Freeze W&B metric naming. The default retains the production "
            "OLMo2/LLaMAFactory-compatible train/* and eval/* keys."
        ),
    )


def _set_if(config: dict[str, Any], path: str, value: Any) -> None:
    if value is None:
        return
    apply_override(config, f"{path}={json.dumps(value)}")


def _compose_args(args: argparse.Namespace) -> dict[str, Any]:
    performance = args.performance or ["ordinary_hsdp"]
    config = compose(
        model=args.model,
        variant=args.variant,
        stage=args.stage,
        data=args.data,
        topology=args.topology,
        performance=performance,
        overrides=args.set,
    )
    shortcuts = {
        "runtime.run_id": args.run_id,
        "runtime.lifecycle": args.lifecycle
        or ("fresh" if args.stage == "stage1" else "transition"),
        "runtime.load": args.load,
        "runtime.save": args.save,
        "runtime.output": args.output,
        "runtime.data_cache_path": args.data_cache_path,
        "runtime.python": args.python,
        "runtime.nnodes": args.nnodes,
        "runtime.node_rank": args.node_rank,
        "runtime.master_addr": args.master_addr,
        "runtime.master_port": args.master_port,
        "runtime.hccl_if_base_port": args.hccl_if_base_port,
        "data.root": args.data_root,
        "data.manifest": args.data_manifest,
        "data.work_dir": args.data_work_dir,
        "data.data_args_path": args.data_args_path,
        "data.tokenizer.path": args.tokenizer,
        "topology.world_size": args.world_size,
        "topology.nproc_per_node": args.nproc_per_node,
        "topology.tensor_parallel": args.tp,
        "topology.context_parallel": args.cp,
        "topology.pipeline_parallel": args.pp,
        "topology.sequence_parallel": args.sp,
        "training.micro_batch_size": args.micro_batch_size,
        "training.global_batch_size": args.global_batch_size,
        "topology.hsdp.shard_size": args.hsdp_shard_size,
        "topology.hsdp.ddp_num_buckets": args.ddp_num_buckets,
        "training.train_tokens": args.train_tokens,
        "training.epochs": args.epochs,
        "data.expected_instances": args.expected_instances,
        "data.expected_fingerprint": args.expected_fingerprint,
        "optimization.peak_lr": args.peak_lr,
        "optimization.min_lr": args.min_lr,
        "optimization.warmup_tokens": args.warmup_tokens,
        "training.save_interval": args.save_interval,
        "training.exit_interval": args.exit_interval,
        "training.eval_interval": args.eval_interval,
        "training.eval_iters": args.eval_iters,
        "runtime.validation.manifest": args.valid_manifest,
        "runtime.validation.cache_dir": args.valid_cache_dir,
        "runtime.wandb.project": args.wandb_project,
        "runtime.wandb.entity": args.wandb_entity,
        "runtime.wandb.base_url": args.wandb_base_url,
        "runtime.wandb.log_style": args.wandb_log_style,
    }
    for path, value in shortcuts.items():
        _set_if(config, path, value)
    if args.warmup_tokens is not None:
        # A dedicated token budget intentionally overrides an SFT profile's
        # optional percentage default; never leave two schedule authorities.
        apply_override(config, "optimization.warmup_fraction=null")
    if args.wandb_project:
        _set_if(config, "runtime.wandb.enabled", True)
    return resolve(config)


def _render(config: dict[str, Any]) -> tuple[Path, list[str], dict[str, str]]:
    run_id = config["runtime"]["run_id"]
    if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
        raise ConfigurationError(f"unsafe run ID: {run_id!r}")
    upstream = upstream_state()
    bad_upstream = [
        name
        for name, state in upstream.items()
        if (
            not state["matches_lock"]
            or not state.get("matches_url", True)
            or state["dirty"]
            or (
                state.get("full_history_required", False)
                and (
                    state.get("shallow", False)
                    or not state.get("full_heads_refspec", False)
                    or state.get("generated_python_artifacts", 0) != 0
                )
            )
        )
    ]
    if bad_upstream:
        raise ConfigurationError(
            "third-party repositories do not match the lock or are modified: "
            f"{bad_upstream}"
        )
    run_dir = PROJECT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise ConfigurationError(
            f"immutable run directory already exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    model_config_path = run_dir / "model_config.json"
    model_config = build_model_config(config)
    command = build_torchrun_command(config, model_config_path)
    environment = build_environment(config)

    _atomic_json(run_dir / "resolved.json", config)
    _atomic_json(model_config_path, model_config)
    _atomic_json(run_dir / "upstream.json", upstream)
    _atomic_json(run_dir / "source_manifest.json", source_manifest())
    _atomic_json(run_dir / "environment.json", environment)
    _atomic_json(run_dir / "command.json", command)
    _atomic_text(run_dir / "command.txt", shell_join(command) + "\n")
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        *(f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items())),
    ]
    training_python = Path(command[0])
    if training_python.parent != Path("."):
        # Megatron's dataset helper Makefile invokes ``python3`` and
        # ``python3-config`` instead of the configured interpreter directly.
        # Keep those subprocesses in the same environment as the explicit
        # training Python.  This is intentionally emitted into environment.sh
        # so checkpoint activation, worker preflight, and the actual launch
        # all share one interpreter toolchain.
        env_lines.append(
            "export PATH="
            f"{shlex.quote(str(training_python.parent))}"
            '${PATH:+:${PATH}}'
        )
    _atomic_text(run_dir / "environment.sh", "\n".join(env_lines) + "\n")
    checkpoint_activate = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source {shlex.quote(str(run_dir / 'environment.sh'))}",
        (
            f"exec {shlex.quote(command[0])} -m "
            "olmo3_pipeline.checkpoint_cli preflight "
            f"--resolved {shlex.quote(str(run_dir / 'resolved.json'))} --write"
        ),
    ]
    _atomic_text(
        run_dir / "checkpoint-activate.sh",
        "\n".join(checkpoint_activate) + "\n",
        executable=True,
    )
    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source {shlex.quote(str(run_dir / 'environment.sh'))}",
        (
            "worker_node_rank="
            f"${{OLMO3_NODE_RANK:-${{NODE_RANK:-{config['runtime']['node_rank']}}}}}"
        ),
        (
            'if [[ ! "$worker_node_rank" =~ ^[0-9]+$ ]] || '
            f"(( worker_node_rank >= {config['runtime']['nnodes']} )); then"
        ),
        (
            "  printf 'invalid worker node rank %q; expected [0, "
            f"{config['runtime']['nnodes']})\\n' \"$worker_node_rank\" >&2"
        ),
        "  exit 2",
        "fi",
        "readonly worker_node_rank",
        (
            f"{shlex.quote(command[0])} -m "
            "olmo3_pipeline.checkpoint_cli verify-prepared "
            f"--resolved {shlex.quote(str(run_dir / 'resolved.json'))}"
        ),
        (
            f"{shlex.quote(command[0])} -m "
            "olmo3_pipeline.cli worker-preflight "
            f"--resolved {shlex.quote(str(run_dir / 'resolved.json'))}"
        ),
        f"exec {_shell_join_with_worker_node_rank(command)}",
    ]
    _atomic_text(run_dir / "launch.sh", "\n".join(launch_lines) + "\n", executable=True)
    return run_dir, command, environment


def _print_summary(config: dict[str, Any], run_dir: Path | None = None) -> None:
    summary = {
        "run_id": config["runtime"]["run_id"],
        "model": config["model"]["size"],
        "variant": config["variant"]["name"],
        "stage": config["stage"]["name"],
        "world_size": config["topology"]["world_size"],
        "tp": config["topology"]["tensor_parallel"],
        "cp": config["topology"]["context_parallel"],
        "dp": config["topology"]["data_parallel"],
        "sp": config["topology"]["sequence_parallel"],
        "micro_batch_size": config["training"]["micro_batch_size"],
        "global_batch_size": config["training"]["global_batch_size"],
        "gradient_accumulation": config["training"]["gradient_accumulation"],
        "sequence_length": config["stage"]["sequence_length"],
        "tokens_per_step": config["training"]["tokens_per_step"],
        "train_iters": config["training"]["train_iters"],
        "warmup_steps": config["optimization"]["warmup_steps"],
        "hsdp_instances": config["topology"]["hsdp"]["num_instances"],
        "performance": config["performance"]["profiles"],
    }
    if run_dir is not None:
        summary["run_dir"] = str(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_resolved_for_worker(path: Path) -> dict[str, Any]:
    """Load and revalidate one frozen config before a worker enters torchrun."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read resolved worker config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"resolved worker config must be a JSON object: {path}")
    return resolve(value)


def _validate_worker_data_mount(config: dict[str, Any]) -> None:
    """Perform one metadata/stat sweep per node before distributed ranks start."""

    data = config["data"]
    backend = data["backend"]
    if backend == "olmo3_sft_numpy":
        try:
            from runtime.olmo3_sft_dataset import describe_olmo3_sft_cache

            description = describe_olmo3_sft_cache(
                Path(data["root"]),
                sft_packed_work_dir(data["work_dir"]),
                sequence_length=int(config["stage"]["sequence_length"]),
                eos_token_id=int(data["tokenizer"]["eos_token_id"]),
                pad_token_id=int(data["tokenizer"]["pad_token_id"]),
                dtype="uint32",
                expected_fingerprint=str(data["expected_fingerprint"]),
            )
            expected = {
                "base_instances": int(data["expected_instances"]),
                "fingerprint": str(data["expected_fingerprint"]),
            }
            mismatches = {
                key: (description.get(key), value)
                for key, value in expected.items()
                if description.get(key) != value
            }
            if mismatches:
                raise DataPipelineError(
                    f"prepared SFT cache contract mismatch: {mismatches}"
                )
        except (DataPipelineError, OSError, RuntimeError, ValueError, KeyError) as exc:
            raise ConfigurationError(
                f"worker data-mount preflight failed: {exc}"
            ) from exc
        return
    if backend not in {"olmo3_numpy_fsl", "olmo3_numpy_packed"}:
        return
    try:
        contract = load_runtime_data_manifest(
            Path(data["manifest"]),
            root=Path(data["root"]),
            verify_files=True,
            # Source SHA256 values are mandatory and sealed in the manifest.
            # Re-hashing hundreds of GiB on every node is an explicit offline
            # data-audit action; launch preflight verifies mount presence/size.
            checksums=False,
            expected_stage=config["stage"]["name"],
            expected_backend=backend,
            expected_token_count=int(data["known_token_count"]),
        )
        validate_runtime_data_profile(
            contract,
            config_name=data["name"],
            config_sha256=contract["data_config_sha256"],
            pad_token_id=int(data["tokenizer"]["pad_token_id"]),
        )
        from runtime.olmo3_packed_dataset import (
            describe_olmo3_long_context_cache,
            describe_olmo3_midtraining_cache,
        )

        source_paths = tuple(Path(path) for path in contract["resolved_token_paths"])
        common = {
            "source_contract_sha256": str(contract["contract_sha256"]),
            "global_batch_size": int(config["training"]["global_batch_size"]),
            "data_seed": int(config["training"]["data_seed"]),
            "sequence_length": int(config["stage"]["sequence_length"]),
            "pad_token_id": int(data["tokenizer"]["pad_token_id"]),
            "dtype": "uint32",
        }
        if backend == "olmo3_numpy_fsl":
            description = describe_olmo3_midtraining_cache(
                source_paths,
                Path(data["work_dir"]),
                **common,
            )
        else:
            packing = data["packing"]
            description = describe_olmo3_long_context_cache(
                source_paths,
                Path(data["work_dir"]),
                eos_token_id=int(data["tokenizer"]["eos_token_id"]),
                source_group_size=int(packing["source_group_size"]),
                source_permutation_seed=int(packing["source_permutation_seed"]),
                **common,
            )
        expected = {
            "sequence_length": int(config["stage"]["sequence_length"]),
            "source_count": len(source_paths),
        }
        mismatches = {
            key: (description.get(key), value)
            for key, value in expected.items()
            if description.get(key) != value
        }
        if mismatches:
            raise DataPipelineError(
                f"prepared Stage-2/3 cache contract mismatch: {mismatches}"
            )
    except (DataPipelineError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ConfigurationError(
            f"worker data-mount preflight failed: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="olmo3ctl")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "render", "launch"):
        subparser = subparsers.add_parser(action)
        _add_common(subparser)
    worker_preflight = subparsers.add_parser(
        "worker-preflight",
        help="read-only per-node topology validation for a rendered run",
    )
    worker_preflight.add_argument("--resolved", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "worker-preflight":
            config = _load_resolved_for_worker(args.resolved)
            run_dir = args.resolved.expanduser().resolve().parent
            verify_source_manifest(run_dir / "source_manifest.json")
            verify_upstream_snapshot(run_dir / "upstream.json")
            _validate_worker_data_mount(config)
            validate_local_listener_ports(config)
            validate_local_sio_pairs(config)
            return 0
        config = _compose_args(args)
        if args.action == "validate":
            _print_summary(config)
            return 0
        if args.action == "launch":
            if config["runtime"]["nnodes"] != 1:
                raise ConfigurationError(
                    "--action launch is single-node only. For a multi-node run, "
                    "render and activate once, then invoke the shared launch.sh "
                    "on every node with OLMO3_NODE_RANK or NODE_RANK set."
                )
            # The control process validates source storage before creating an
            # immutable run directory. It writes the destination contract only
            # after runtime/topology checks pass, so distributed workers never
            # race to create checkpoint metadata.
            prepare_lifecycle(config, write=False)
            validate_runtime_stack(PROJECT_ROOT)
            validate_local_listener_ports(config)
            validate_local_sio_pairs(config)
        run_dir, command, environment = _render(config)
        _print_summary(config, run_dir)
        if args.action == "render":
            return 0
        prepare_lifecycle(config, write=True)
        launch_env = os.environ.copy()
        launch_env.update(environment)
        command = _command_with_node_rank(command, _deployment_node_rank(config))
        os.execvpe(command[0], command, launch_env)
        raise AssertionError("os.execvpe unexpectedly returned")
    except ConfigurationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
