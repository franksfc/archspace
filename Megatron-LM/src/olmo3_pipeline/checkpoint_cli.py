"""Command-line checkpoint lifecycle, resharding, and export plans."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import (
    adopt_legacy_checkpoint,
    inspect_checkpoint,
    load_contract_file,
    prepare_lifecycle,
    verify_prepared_lifecycle,
)
from .command import build_environment, build_torchrun_command
from .config import PROJECT_ROOT, ConfigurationError
from .model_config import build_model_config
from .resolve import (
    HCCL_IF_BASE_PORT_MAX,
    HCCL_IF_BASE_PORT_MIN,
    resolve,
    validate_hccl_if_base_port,
)
from .topology import validate_local_listener_ports


_ROUNDTRIP_HCCL_PORT_STRIDE = 256
_SHELL_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return value


def _load_resolved(path: Path) -> dict[str, Any]:
    config = _load_object(path.expanduser().resolve())
    # Re-resolving catches hand-edited/incomplete files and recomputes every
    # topology-dependent derived value.
    return resolve(config)


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


def _atomic_text(path: Path, value: str, *, executable: bool = False) -> None:
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


def _replace_training_entry(command: list[str], replacement: Path) -> list[str]:
    result = list(command)
    candidates = [
        index
        for index, value in enumerate(result)
        if value.endswith("pretrain_olmo3_mindspeed.py")
        or value.endswith("pretrain_olmo3_mindspeed_mc2.py")
    ]
    if len(candidates) != 1:
        raise ConfigurationError(
            f"could not identify one training entry in generated command: {candidates}"
        )
    result[candidates[0]] = str(replacement)
    return result


def _node_rank_value_index(command: list[str]) -> int:
    indices = [
        index + 1
        for index, value in enumerate(command[:-1])
        if value == "--node_rank"
    ]
    if len(indices) != 1:
        raise ConfigurationError(
            "generated checkpoint torchrun command must contain exactly one "
            "--node_rank"
        )
    return indices[0]


def _shell_join_with_worker_node_rank(command: list[str]) -> str:
    rank_index = _node_rank_value_index(command)
    return " ".join(
        '"${worker_node_rank}"' if index == rank_index else shlex.quote(value)
        for index, value in enumerate(command)
    )


def _worker_node_rank_preamble(config: dict[str, Any]) -> list[str]:
    runtime = config["runtime"]
    default_rank = int(runtime["node_rank"])
    nnodes = int(runtime["nnodes"])
    max_rank_digits = len(str(nnodes - 1))
    return [
        (
            "worker_node_rank="
            f"${{OLMO3_NODE_RANK:-${{NODE_RANK:-{default_rank}}}}}"
        ),
        (
            'if [[ ! "$worker_node_rank" =~ ^(0|[1-9][0-9]*)$ ]] || '
            f"(( ${{#worker_node_rank}} > {max_rank_digits} )) || "
            f"(( 10#$worker_node_rank >= {nnodes} )); then"
        ),
        (
            "  printf 'invalid worker node rank %q; expected [0, "
            f"{nnodes})\\n' \"$worker_node_rank\" >&2"
        ),
        "  exit 2",
        "fi",
        "readonly worker_node_rank",
    ]


def _next_roundtrip_master_port(port: int) -> int:
    if port < 1 or port > 65_535:
        raise ConfigurationError(f"master port must be in [1, 65535], got {port}")
    # A distinct rendezvous endpoint prevents pass 2 from attaching to a
    # torchrun store that is still being torn down after pass 1.
    return port + 1 if port < 65_535 else port - 1


def _next_roundtrip_hccl_if_base_port(port: int) -> int:
    """Select a non-overlapping valid HCCL port range for round-trip pass 2."""

    port = validate_hccl_if_base_port(port)
    if port + _ROUNDTRIP_HCCL_PORT_STRIDE <= HCCL_IF_BASE_PORT_MAX:
        return port + _ROUNDTRIP_HCCL_PORT_STRIDE
    if port - _ROUNDTRIP_HCCL_PORT_STRIDE >= HCCL_IF_BASE_PORT_MIN:
        return port - _ROUNDTRIP_HCCL_PORT_STRIDE
    # The validated interval is much wider than one stride; keep this guard so
    # a future range change cannot silently make the two passes overlap.
    raise ConfigurationError(
        "cannot allocate a distinct HCCL_IF_BASE_PORT range for roundtrip "
        f"from {port}"
    )


def _reshard_command(
    config: dict[str, Any],
    model_config_path: Path,
    *,
    destination: Path,
) -> list[str]:
    config = copy.deepcopy(config)
    config["runtime"]["save"] = str(destination)
    command = build_torchrun_command(config, model_config_path)
    command = _replace_training_entry(
        command,
        PROJECT_ROOT / "scripts" / "checkpoint" / "reshard_checkpoint.py",
    )
    if config["performance"].get("tensor_parallel_kernel") == "ascend_mc2":
        entry_index = next(
            index
            for index, value in enumerate(command)
            if value.endswith("reshard_checkpoint.py")
        )
        if "--use-ascend-mc2" not in command:
            # The normal TP2/MC2 entry point injects this before importing the
            # MindSpeed adaptor. Reshard replaces that entry point, so the flag
            # must be present in the new process argv from startup.
            command.insert(entry_index + 1, "--use-ascend-mc2")
    return command


def _shell_preamble(environment: dict[str, str]) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        *(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(environment.items())
        ),
    ]


def _shell_command_with_environment(
    command: list[str],
    environment: dict[str, str],
) -> str:
    assignments: list[str] = []
    for key, value in sorted(environment.items()):
        if _SHELL_ENVIRONMENT_NAME.fullmatch(key) is None:
            raise ConfigurationError(
                f"unsafe per-command environment variable name: {key!r}"
            )
        assignments.append(f"{key}={shlex.quote(str(value))}")
    rendered = _shell_join_with_worker_node_rank(command)
    return " ".join([*assignments, rendered])


def _write_plan(
    plan_dir: Path,
    *,
    config: dict[str, Any],
    commands: Iterable[list[str]],
    description: dict[str, Any],
    command_environments: Iterable[dict[str, str]] | None = None,
) -> Path:
    plan_dir = plan_dir.expanduser().resolve()
    if plan_dir.exists():
        raise ConfigurationError(f"immutable checkpoint plan already exists: {plan_dir}")
    plan_dir.mkdir(parents=True)
    model_config_path = plan_dir / "model_config.json"
    _atomic_json(plan_dir / "resolved.json", config)
    _atomic_json(model_config_path, build_model_config(config))
    environment = build_environment(config)
    command_list = list(commands)
    if command_environments is None:
        command_environment_list = [{} for _ in command_list]
    else:
        command_environment_list = [
            {str(key): str(value) for key, value in item.items()}
            for item in command_environments
        ]
    if len(command_environment_list) != len(command_list):
        raise ConfigurationError(
            "checkpoint plan must provide exactly one per-command environment "
            f"for each command: {len(command_environment_list)} != "
            f"{len(command_list)}"
        )
    _atomic_json(plan_dir / "plan.json", description)
    _atomic_json(plan_dir / "commands.json", command_list)
    _atomic_json(plan_dir / "environment.json", environment)
    _atomic_json(
        plan_dir / "command_environments.json",
        command_environment_list,
    )
    lines = _shell_preamble(environment)
    lines.extend(_worker_node_rank_preamble(config))
    lines.extend(
        _shell_command_with_environment(command, command_environment)
        for command, command_environment in zip(
            command_list, command_environment_list, strict=True
        )
    )
    _atomic_text(
        plan_dir / "launch.sh",
        "\n".join(lines) + "\n",
        executable=True,
    )
    return plan_dir


def _source_contract_override(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    source_contract = getattr(args, "source_contract", None)
    skip_storage_check = bool(getattr(args, "skip_storage_check", False))
    if source_contract is None:
        if (
            skip_storage_check
            and config["runtime"]["lifecycle"] in {"resume", "transition"}
        ):
            raise ConfigurationError(
                "--skip-storage-check requires --source-contract. Copy the "
                "source checkpoint's olmo3_checkpoint_contract.json to a "
                "locally readable path; topology/model identity is never guessed."
            )
        return None
    return load_contract_file(source_contract)


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, sort_keys=True))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    config = _load_resolved(args.resolved)
    prepared = prepare_lifecycle(
        config,
        write=args.write,
        verify_checkpoint_tree=not args.skip_storage_check,
        source_contract_override=_source_contract_override(args, config),
    )
    print(
        json.dumps(
            {
                "source_contract": prepared.source_contract,
                "target_contract": prepared.target_contract,
                "target_contract_path": str(prepared.target_contract_path),
                "intent_path": str(prepared.intent_path),
                "written": args.write,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_verify_prepared(args: argparse.Namespace) -> int:
    """Read-only worker gate for a lifecycle activated by the control plane."""

    config = _load_resolved(args.resolved)
    validate_local_listener_ports(config)
    prepared = verify_prepared_lifecycle(
        config,
        verify_checkpoint_tree=not args.skip_storage_check,
        source_contract_override=_source_contract_override(args, config),
    )
    print(
        json.dumps(
            {
                "state": "prepared",
                "target_contract_path": str(prepared.target_contract_path),
                "intent_path": str(prepared.intent_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_adopt(args: argparse.Namespace) -> int:
    config = _load_resolved(args.resolved)
    path = adopt_legacy_checkpoint(
        args.checkpoint,
        config,
        acknowledgement=args.i_verified_model_and_variant,
        weights_only=args.weights_only,
    )
    print(json.dumps({"contract": str(path)}, indent=2))
    return 0


def _cmd_reshard(args: argparse.Namespace) -> int:
    config = _load_resolved(args.resolved)
    validate_local_listener_ports(config)
    if config["runtime"]["lifecycle"] not in {"resume", "transition"}:
        raise ConfigurationError("reshard requires resume or transition lifecycle")
    destination = args.destination.expanduser().resolve()
    target_config = copy.deepcopy(config)
    target_config["runtime"]["save"] = str(destination)
    prepare_lifecycle(
        target_config,
        write=args.write_contract,
        verify_checkpoint_tree=not args.skip_storage_check,
        source_contract_override=_source_contract_override(args, target_config),
    )
    plan_dir = args.plan_dir.expanduser().resolve()
    model_config_path = plan_dir / "model_config.json"
    command = _reshard_command(
        config,
        model_config_path,
        destination=destination,
    )
    path = _write_plan(
        plan_dir,
        config=target_config,
        commands=[command],
        description={
            "action": "reshard",
            "source": str(Path(config["runtime"]["load"]).expanduser().resolve()),
            "destination": str(destination),
            "lifecycle": config["runtime"]["lifecycle"],
            "preserves": ["model", "adam_moments", "fp32_master_params"],
            "resets": (
                []
                if config["runtime"]["lifecycle"] == "resume"
                else ["trainer", "scheduler", "rng", "sample_counters"]
            ),
            "target_topology": config["topology"],
        },
    )
    print(json.dumps({"plan_dir": str(path), "launch": str(path / "launch.sh")}, indent=2))
    return 0


def _cmd_roundtrip(args: argparse.Namespace) -> int:
    config = _load_resolved(args.resolved)
    requested_hccl_port = getattr(args, "hccl_if_base_port", None)
    if requested_hccl_port is None:
        requested_hccl_port = config["runtime"].get("hccl_if_base_port")
    if requested_hccl_port is None:
        raise ConfigurationError(
            "roundtrip requires --hccl-if-base-port or a frozen "
            "runtime.hccl_if_base_port so its two passes cannot reuse the "
            "same HCCL listener range"
        )
    first_hccl_port = validate_hccl_if_base_port(requested_hccl_port)
    config["runtime"]["hccl_if_base_port"] = first_hccl_port
    if config["runtime"]["lifecycle"] != "resume":
        raise ConfigurationError(
            "roundtrip validates full same-stage state and therefore requires lifecycle=resume"
        )
    prepare_lifecycle(
        config,
        write=False,
        verify_checkpoint_tree=not args.skip_storage_check,
        source_contract_override=_source_contract_override(args, config),
    )
    plan_dir = args.plan_dir.expanduser().resolve()
    first = args.work_dir.expanduser().resolve() / "pass1"
    second = args.work_dir.expanduser().resolve() / "pass2"
    model_config_path = plan_dir / "model_config.json"
    command1 = _reshard_command(config, model_config_path, destination=first)
    second_config = copy.deepcopy(config)
    second_config["runtime"]["load"] = str(first)
    first_master_port = int(config["runtime"]["master_port"])
    second_master_port = _next_roundtrip_master_port(first_master_port)
    second_hccl_port = _next_roundtrip_hccl_if_base_port(first_hccl_port)
    second_config["runtime"]["master_port"] = second_master_port
    second_config["runtime"]["hccl_if_base_port"] = second_hccl_port
    validate_local_listener_ports(config)
    validate_local_listener_ports(second_config)
    command2 = _reshard_command(second_config, model_config_path, destination=second)
    path = _write_plan(
        plan_dir,
        config=config,
        commands=[command1, command2],
        description={
            "action": "roundtrip",
            "source": str(Path(config["runtime"]["load"]).expanduser().resolve()),
            "pass1": str(first),
            "pass2": str(second),
            "master_ports": {
                "pass1": first_master_port,
                "pass2": second_master_port,
            },
            "hccl_if_base_ports": {
                "pass1": first_hccl_port,
                "pass2": second_hccl_port,
            },
            "validation": (
                "both passes use MCore strict torch_dist load and preserve the "
                "distributed optimizer state"
            ),
        },
        command_environments=[
            {"HCCL_IF_BASE_PORT": str(first_hccl_port)},
            {"HCCL_IF_BASE_PORT": str(second_hccl_port)},
        ],
    )
    print(json.dumps({"plan_dir": str(path), "launch": str(path / "launch.sh")}, indent=2))
    return 0


def _add_storage_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-storage-check",
        action="store_true",
        help="Build a plan before the remote checkpoint filesystem is mounted.",
    )
    parser.add_argument(
        "--source-contract",
        type=Path,
        help=(
            "Local copy of the source olmo3_checkpoint_contract.json. Required "
            "with --skip-storage-check so architecture/topology validation "
            "remains fail-closed."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="olmo3ckpt")
    sub = parser.add_subparsers(dest="action", required=True)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--checkpoint", type=Path, required=True)
    inspect_parser.set_defaults(handler=_cmd_inspect)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--resolved", type=Path, required=True)
    preflight.add_argument("--write", action="store_true")
    _add_storage_check(preflight)
    preflight.set_defaults(handler=_cmd_preflight)

    verify_prepared = sub.add_parser("verify-prepared")
    verify_prepared.add_argument("--resolved", type=Path, required=True)
    _add_storage_check(verify_prepared)
    verify_prepared.set_defaults(handler=_cmd_verify_prepared)

    adopt = sub.add_parser("adopt")
    adopt.add_argument("--checkpoint", type=Path, required=True)
    adopt.add_argument("--resolved", type=Path, required=True)
    adopt.add_argument("--i-verified-model-and-variant", action="store_true")
    adopt.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "Mark an externally produced weight-only checkpoint as lacking "
            "resumable Adam state."
        ),
    )
    adopt.set_defaults(handler=_cmd_adopt)

    reshard = sub.add_parser("reshard")
    reshard.add_argument("--resolved", type=Path, required=True)
    reshard.add_argument("--destination", type=Path, required=True)
    reshard.add_argument("--plan-dir", type=Path, required=True)
    reshard.add_argument("--write-contract", action="store_true")
    _add_storage_check(reshard)
    reshard.set_defaults(handler=_cmd_reshard)

    roundtrip = sub.add_parser("roundtrip")
    roundtrip.add_argument("--resolved", type=Path, required=True)
    roundtrip.add_argument("--work-dir", type=Path, required=True)
    roundtrip.add_argument("--plan-dir", type=Path, required=True)
    roundtrip.add_argument(
        "--hccl-if-base-port",
        type=int,
        help=(
            "HCCL listener base for pass 1. Pass 2 receives a separate "
            f"{_ROUNDTRIP_HCCL_PORT_STRIDE}-port-strided range."
        ),
    )
    _add_storage_check(roundtrip)
    roundtrip.set_defaults(handler=_cmd_roundtrip)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigurationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
