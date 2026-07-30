from __future__ import annotations

from pathlib import Path

import pytest

from olmo3_pipeline.config import ConfigurationError
from olmo3_pipeline.topology import validate_local_listener_ports


def _config(*, master: int, hccl: int) -> dict:
    return {
        "runtime": {
            "master_port": master,
            "hccl_if_base_port": hccl,
        },
        "topology": {"nproc_per_node": 16},
    }


def test_listener_ports_outside_ephemeral_range_are_accepted(
    tmp_path: Path,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    port_range.write_text("32768 60999\n", encoding="utf-8")
    validate_local_listener_ports(
        _config(master=31_500, hccl=16_000),
        ephemeral_range_path=port_range,
    )


@pytest.mark.parametrize(
    ("master", "hccl", "label"),
    (
        (32_768, 16_000, "MASTER"),
        (31_500, 32_768, "HCCL"),
        (31_500, 32_760, "HCCL"),
    ),
)
def test_listener_overlap_with_ephemeral_range_is_rejected(
    tmp_path: Path,
    master: int,
    hccl: int,
    label: str,
) -> None:
    port_range = tmp_path / "ip_local_port_range"
    port_range.write_text("32768 60999\n", encoding="utf-8")
    with pytest.raises(
        ConfigurationError,
        match=rf"{label} listener range .* overlaps kernel ephemeral range",
    ):
        validate_local_listener_ports(
            _config(master=master, hccl=hccl),
            ephemeral_range_path=port_range,
        )
