from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo3_pipeline.checkpoint import build_contract, write_contract
from olmo3_pipeline.checkpoint_cli import (
    _cmd_reshard,
    _cmd_roundtrip,
    _next_roundtrip_hccl_if_base_port,
    _next_roundtrip_master_port,
    _reshard_command,
)
from olmo3_pipeline.config import ConfigurationError, compose
from olmo3_pipeline.resolve import resolve


def test_internal_reshard_help_does_not_import_npu_runtime() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "checkpoint"
        / "reshard_checkpoint.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "Internal NPU worker entry point" in completed.stdout


def _resolved(tmp_path: Path, *, variant: str = "base") -> dict:
    config = compose(
        model="1b",
        variant=variant,
        stage="stage1",
        data="dolma3_6t",
        topology="stage1_tp1_4096npu",
        performance=["ordinary_hsdp"],
    )
    config["data"]["root"] = str(tmp_path / "data")
    config["data"]["data_args_path"] = str(tmp_path / "data-args.json")
    config["data"]["tokenizer"]["path"] = str(tmp_path / "tokenizer")
    config["training"]["train_tokens"] = 6_000_000_000_000
    config["optimization"]["peak_lr"] = 1e-3
    config["optimization"]["min_lr"] = 1e-4
    config["optimization"]["warmup_tokens"] = 8_388_608_000
    config["runtime"] = {
        "run_id": "checkpoint-command-test",
        "lifecycle": "resume",
        "load": str(tmp_path / "source"),
        "save": str(tmp_path / "save"),
        "output": str(tmp_path / "output"),
        "python": sys.executable,
        "nnodes": 256,
        "node_rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "hccl_if_base_port": 21_000,
    }
    return resolve(config)


def _complete_source_checkpoint(config: dict, *, iteration: int = 7) -> None:
    source = Path(config["runtime"]["load"])
    iteration_dir = source / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        '{"sharded_backend": "torch_dist", "sharded_backend_version": 1}\n',
        encoding="utf-8",
    )
    (iteration_dir / "common.pt").write_bytes(b"common-state")
    (iteration_dir / "__0_0.distcp").write_bytes(b"0123456789")
    with (iteration_dir / ".metadata").open("wb") as stream:
        pickle.dump(
            SimpleNamespace(
                storage_data={
                    "model.weight": SimpleNamespace(
                        relative_path="__0_0.distcp",
                        offset=2,
                        length=5,
                    )
                }
            ),
            stream,
        )
    (source / "latest_checkpointed_iteration.txt").write_text(
        f"{iteration}\n", encoding="utf-8"
    )
    source_config = json.loads(json.dumps(config))
    source_config["runtime"]["save"] = str(source)
    write_contract(source, build_contract(source_config))


def _resolved_tp2_sio(tmp_path: Path) -> dict:
    config = compose(
        model="3b",
        variant="siamese_depth",
        stage="stage1",
        data="dolma3_6t",
        topology="tp2_sio_2048npu",
        performance=["tp2_sio_mc2"],
    )
    config["data"]["root"] = str(tmp_path / "data")
    config["data"]["data_args_path"] = str(tmp_path / "data-args.json")
    config["data"]["tokenizer"]["path"] = str(tmp_path / "tokenizer")
    config["training"]["train_tokens"] = 6_000_000_000_000
    config["optimization"]["peak_lr"] = 7e-4
    config["optimization"]["min_lr"] = 7e-5
    config["optimization"]["warmup_tokens"] = 8_388_608_000
    config["runtime"] = {
        "run_id": "checkpoint-tp2-sio-test",
        "lifecycle": "resume",
        "load": str(tmp_path / "source"),
        "save": str(tmp_path / "save"),
        "output": str(tmp_path / "output"),
        "python": sys.executable,
        "nnodes": 128,
        "node_rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
    }
    return resolve(config)


def test_reshard_command_reuses_native_megatron_loader(tmp_path: Path) -> None:
    config = _resolved(tmp_path)
    command = _reshard_command(
        config,
        tmp_path / "model_config.json",
        destination=tmp_path / "resharded",
    )
    entries = [value for value in command if value.endswith("reshard_checkpoint.py")]
    assert len(entries) == 1
    assert "--use-distributed-optimizer" in command
    assert command[command.index("--ckpt-format") + 1] == "torch_dist"
    assert command[command.index("--load") + 1] == str(tmp_path / "source")
    assert command[command.index("--save") + 1] == str(tmp_path / "resharded")
    assert command[command.index("--dist-ckpt-strictness") + 1] == "raise_all"
    assert "--use-ascend-mc2" not in command


def test_tp2_sio_reshard_preserves_mc2_flag_in_process_argv(
    tmp_path: Path,
) -> None:
    command = _reshard_command(
        _resolved_tp2_sio(tmp_path),
        tmp_path / "model_config.json",
        destination=tmp_path / "resharded",
    )
    entry_index = next(
        index
        for index, value in enumerate(command)
        if value.endswith("reshard_checkpoint.py")
    )
    assert command.count("--use-ascend-mc2") == 1
    assert command[entry_index + 1] == "--use-ascend-mc2"


def test_reshard_cli_writes_immutable_native_plan(tmp_path: Path) -> None:
    config = _resolved(tmp_path)
    _complete_source_checkpoint(config)
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    plan = tmp_path / "plan"
    _cmd_reshard(
        argparse.Namespace(
            resolved=resolved,
            destination=tmp_path / "target",
            plan_dir=plan,
            write_contract=False,
            skip_storage_check=False,
        )
    )
    assert (plan / "launch.sh").is_file()
    assert (plan / "model_config.json").is_file()
    command = json.loads((plan / "commands.json").read_text(encoding="utf-8"))[0]
    assert any(value.endswith("reshard_checkpoint.py") for value in command)
    launcher = (plan / "launch.sh").read_text(encoding="utf-8")
    assert "worker_node_rank=${OLMO3_NODE_RANK:-${NODE_RANK:-0}}" in launcher
    assert "(( ${#worker_node_rank} > 3 ))" in launcher
    assert "(( 10#$worker_node_rank >= 256 ))" in launcher
    assert '"${worker_node_rank}"' in launcher
    assert "--node_rank 0" not in launcher
    subprocess.run(["bash", "-n", str(plan / "launch.sh")], check=True)
    for invalid_rank in (
        "256",
        "-1",
        "not-a-rank",
        "999999999999999999999999999",
    ):
        invalid = subprocess.run(
            ["bash", str(plan / "launch.sh")],
            env={**os.environ, "OLMO3_NODE_RANK": invalid_rank},
            text=True,
            capture_output=True,
            check=False,
        )
        assert invalid.returncode == 2
        assert "expected [0, 256)" in invalid.stderr
    with pytest.raises(ConfigurationError, match="immutable checkpoint plan"):
        _cmd_reshard(
            argparse.Namespace(
                resolved=resolved,
                destination=tmp_path / "target",
                plan_dir=plan,
                write_contract=False,
                skip_storage_check=False,
            )
        )


def test_reshard_rejects_hccl_listener_in_ephemeral_range(
    tmp_path: Path,
) -> None:
    config = _resolved(tmp_path)
    ephemeral_start = int(
        Path("/proc/sys/net/ipv4/ip_local_port_range")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    config["runtime"]["hccl_if_base_port"] = ephemeral_start
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        ConfigurationError,
        match="HCCL listener range .* overlaps kernel ephemeral range",
    ):
        _cmd_reshard(
            argparse.Namespace(
                resolved=resolved,
                destination=tmp_path / "target",
                plan_dir=tmp_path / "plan",
                write_contract=False,
                skip_storage_check=False,
                source_contract=None,
            )
        )


@pytest.mark.parametrize(
    ("first", "second", "first_hccl", "second_hccl"),
    (
        (29_500, 29_501, 21_000, 21_256),
        (65_535, 65_534, 65_520, 65_264),
    ),
)
def test_roundtrip_uses_distinct_master_and_hccl_ports(
    tmp_path: Path,
    first: int,
    second: int,
    first_hccl: int,
    second_hccl: int,
) -> None:
    config = _resolved(tmp_path)
    config["runtime"]["master_port"] = first
    config["runtime"]["hccl_if_base_port"] = first_hccl
    _complete_source_checkpoint(config)
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    plan = tmp_path / "roundtrip-plan"
    _cmd_roundtrip(
        argparse.Namespace(
            resolved=resolved,
            work_dir=tmp_path / "roundtrip-work",
            plan_dir=plan,
            skip_storage_check=False,
            source_contract=None,
        )
    )

    commands = json.loads((plan / "commands.json").read_text(encoding="utf-8"))
    assert len(commands) == 2
    assert commands[0][commands[0].index("--master_port") + 1] == str(first)
    assert commands[1][commands[1].index("--master_port") + 1] == str(second)
    environments = json.loads(
        (plan / "command_environments.json").read_text(encoding="utf-8")
    )
    assert environments == [
        {"HCCL_IF_BASE_PORT": str(first_hccl)},
        {"HCCL_IF_BASE_PORT": str(second_hccl)},
    ]
    frozen_environment = json.loads(
        (plan / "environment.json").read_text(encoding="utf-8")
    )
    assert frozen_environment["HCCL_IF_BASE_PORT"] == str(first_hccl)
    description = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
    assert description["master_ports"] == {"pass1": first, "pass2": second}
    assert description["hccl_if_base_ports"] == {
        "pass1": first_hccl,
        "pass2": second_hccl,
    }
    launcher = (plan / "launch.sh").read_text(encoding="utf-8")
    assert launcher.count('"${worker_node_rank}"') == 2
    assert (
        f"HCCL_IF_BASE_PORT={first_hccl} " in launcher
        and f"HCCL_IF_BASE_PORT={second_hccl} " in launcher
    )
    subprocess.run(["bash", "-n", str(plan / "launch.sh")], check=True)


def test_roundtrip_master_port_rejects_out_of_range_values() -> None:
    with pytest.raises(ConfigurationError, match=r"\[1, 65535\]"):
        _next_roundtrip_master_port(0)


@pytest.mark.parametrize("value", (0, 1_023, 65_521, True))
def test_roundtrip_hccl_port_rejects_out_of_range_values(value: object) -> None:
    with pytest.raises(ConfigurationError, match=r"\[1024, 65520\]"):
        _next_roundtrip_hccl_if_base_port(value)  # type: ignore[arg-type]


def test_roundtrip_requires_an_explicit_or_frozen_hccl_port(
    tmp_path: Path,
) -> None:
    config = _resolved(tmp_path)
    del config["runtime"]["hccl_if_base_port"]
    _complete_source_checkpoint(config)
    resolved = tmp_path / "resolved-without-hccl.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="requires --hccl-if-base-port"):
        _cmd_roundtrip(
            argparse.Namespace(
                resolved=resolved,
                work_dir=tmp_path / "roundtrip-work",
                plan_dir=tmp_path / "roundtrip-plan",
                skip_storage_check=False,
                source_contract=None,
                hccl_if_base_port=None,
            )
        )


def test_roundtrip_cli_hccl_port_override_is_frozen(tmp_path: Path) -> None:
    config = _resolved(tmp_path)
    _complete_source_checkpoint(config)
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    plan = tmp_path / "roundtrip-plan"
    _cmd_roundtrip(
        argparse.Namespace(
            resolved=resolved,
            work_dir=tmp_path / "roundtrip-work",
            plan_dir=plan,
            skip_storage_check=False,
            source_contract=None,
            hccl_if_base_port=22_000,
        )
    )
    frozen = json.loads((plan / "resolved.json").read_text(encoding="utf-8"))
    assert frozen["runtime"]["hccl_if_base_port"] == 22_000
    environments = json.loads(
        (plan / "command_environments.json").read_text(encoding="utf-8")
    )
    assert environments == [
        {"HCCL_IF_BASE_PORT": "22000"},
        {"HCCL_IF_BASE_PORT": "22256"},
    ]


def test_reshard_plan_for_unmounted_storage_requires_local_contract(
    tmp_path: Path,
) -> None:
    config = _resolved(tmp_path)
    source_config = json.loads(json.dumps(config))
    source_config["runtime"]["save"] = config["runtime"]["load"]
    contract_copy = tmp_path / "source-contract.json"
    contract_copy.write_text(
        json.dumps(build_contract(source_config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")

    _cmd_reshard(
        argparse.Namespace(
            resolved=resolved,
            destination=tmp_path / "target",
            plan_dir=tmp_path / "offline-plan",
            write_contract=False,
            skip_storage_check=True,
            source_contract=contract_copy,
        )
    )
    assert (tmp_path / "offline-plan" / "launch.sh").is_file()

    with pytest.raises(ConfigurationError, match="requires --source-contract"):
        _cmd_reshard(
            argparse.Namespace(
                resolved=resolved,
                destination=tmp_path / "target-2",
                plan_dir=tmp_path / "offline-plan-2",
                write_contract=False,
                skip_storage_check=True,
                source_contract=None,
            )
        )
