from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, ConfigurationError
from .provenance import upstream_state


def validate_local_listener_ports(
    config: dict[str, Any],
    *,
    ephemeral_range_path: Path = Path(
        "/proc/sys/net/ipv4/ip_local_port_range"
    ),
) -> None:
    """Reject listener ranges the local kernel may assign to outbound sockets."""

    try:
        fields = ephemeral_range_path.read_text(encoding="utf-8").split()
        if len(fields) != 2:
            raise ValueError(f"expected two integers, got {fields!r}")
        ephemeral_start, ephemeral_end = map(int, fields)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"unable to read kernel ephemeral port range from "
            f"{ephemeral_range_path}: {exc}"
        ) from exc
    if not 1 <= ephemeral_start <= ephemeral_end <= 65535:
        raise ConfigurationError(
            "invalid kernel ephemeral port range: "
            f"{ephemeral_start} {ephemeral_end}"
        )

    runtime = config["runtime"]
    listener_ranges: list[tuple[str, int, int]] = []
    if "master_port" in runtime:
        master_port = int(runtime["master_port"])
        listener_ranges.append(("MASTER", master_port, master_port))
    if "hccl_if_base_port" in runtime:
        nproc_per_node = int(config["topology"]["nproc_per_node"])
        hccl_start = int(runtime["hccl_if_base_port"])
        listener_ranges.append(
            ("HCCL", hccl_start, hccl_start + nproc_per_node - 1)
        )
    if not listener_ranges:
        return

    for label, start, end in listener_ranges:
        if start <= ephemeral_end and end >= ephemeral_start:
            raise ConfigurationError(
                f"{label} listener range [{start}, {end}] overlaps kernel "
                f"ephemeral range [{ephemeral_start}, {ephemeral_end}]; "
                "an outbound socket can steal the listener port after a "
                "successful startup preflight"
            )


def validate_local_sio_pairs(config: dict[str, Any]) -> None:
    """Fail closed unless every local even/odd TP2 pair is a physical SIO pair."""

    perf = config["performance"]
    if not perf.get("require_tp_same_physical_card"):
        return
    if config["topology"]["tensor_parallel"] != 2:
        raise ConfigurationError("SIO validation is only defined for TP2")
    nproc = config["topology"]["nproc_per_node"]
    if nproc != 16:
        raise ConfigurationError(
            "the TP2 SIO mapping is validated only for 16 NPUs per node"
        )

    try:
        mapping = subprocess.check_output(
            ["npu-smi", "info", "-m"], text=True, stderr=subprocess.STDOUT
        )
        topology = subprocess.check_output(
            ["npu-smi", "info", "-t", "topo"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"unable to inspect physical NPU topology: {exc}") from exc

    # Ascend 910C exposes each physical card as two dies. The production
    # launcher used the exact card:die:physical:logical mapping below, with
    # adjacent local ranks assigned to the two dies of one card.
    mapped: dict[int, tuple[int, int, int, int]] = {}
    for line in mapping.splitlines():
        match = re.match(
            r"^\s*([0-7])\s+([01])\s+([0-9]+)\s+([0-9]+)(?:\s|$)",
            line,
        )
        if match is None:
            continue
        card, die, physical, logical = map(int, match.groups())
        if physical in mapped:
            raise ConfigurationError(
                f"npu-smi reports duplicate physical NPU ID {physical}"
            )
        mapped[physical] = (card, die, physical, logical)
    if len(mapped) != nproc:
        raise ConfigurationError(
            "unable to verify the 8-card/16-die NPU map from npu-smi"
        )
    for card in range(8):
        even = 2 * card
        odd = even + 1
        if mapped.get(even) != (card, 0, even, even) or mapped.get(odd) != (
            card,
            1,
            odd,
            odd,
        ):
            raise ConfigurationError(
                f"local ranks {even}/{odd} are not both dies of physical card {card}"
            )

    topology_rows: dict[int, list[str]] = {}
    for line in topology.splitlines():
        columns = line.split()
        if not columns:
            continue
        match = re.fullmatch(r"Phy-ID([0-9]+)", columns[0])
        if match is not None:
            topology_rows[int(match.group(1))] = columns
    if len(topology_rows) != nproc:
        raise ConfigurationError(
            "unable to parse all 16 physical topology rows from npu-smi"
        )
    for row in range(nproc):
        peer = row + 1 if row % 2 == 0 else row - 1
        columns = topology_rows[row]
        relation_index = peer + 1
        if (
            relation_index >= len(columns)
            or columns[relation_index].upper() != "SIO"
        ):
            raise ConfigurationError(
                f"adjacent TP dies {row}/{peer} are not connected by SIO"
            )
    print(
        "OLMO3_RUNTIME_SIO_PREFLIGHT_OK "
        f"local_npus={nproc} tp_pairs={nproc // 2}",
        flush=True,
    )


def validate_runtime_stack(project_root: Path = PROJECT_ROOT) -> None:
    required = (
        project_root / "third_party" / "MindSpeed" / "mindspeed",
        project_root
        / "third_party"
        / "MindSpeed-LLM"
        / "mindspeed_llm",
        project_root / "megatron" / "core",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise ConfigurationError(
            "runtime checkout is incomplete; run scripts/bootstrap_runtime.py "
            "--clone-missing: "
            + ", ".join(missing)
        )
    bad = [
        name
        for name, state in upstream_state().items()
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
    if bad:
        raise ConfigurationError(
            "third-party repositories must match the lock and remain pristine: "
            + ", ".join(bad)
        )
