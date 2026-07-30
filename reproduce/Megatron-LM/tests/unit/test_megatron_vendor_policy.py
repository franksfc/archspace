from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_post_mindspeed_installer_runs_after_adaptor() -> None:
    source = (
        PROJECT_ROOT / "src" / "runtime" / "mindspeed_runtime.py"
    ).read_text(encoding="utf-8")
    adaptor = source.index('importlib.import_module("mindspeed_llm.tasks.megatron_adaptor_v2")')
    installer = source.index("install_olmo3_mindspeed_compatibility()")
    assert adaptor < installer


def test_runtime_uses_vendored_megatron_before_project_sources() -> None:
    source = (
        PROJECT_ROOT / "src" / "olmo3_pipeline" / "command.py"
    ).read_text(encoding="utf-8")
    assert 'str(PROJECT_ROOT),' in source
    assert 'str(PROJECT_ROOT / "src"),' in source
    assert source.index("str(PROJECT_ROOT),") < source.index(
        'str(PROJECT_ROOT / "src"),'
    )


def test_tp2_runtime_markers_cover_parsed_mc2_and_active_lane_pack() -> None:
    pretrain = (
        PROJECT_ROOT / "src" / "pretrain_olmo3_mindspeed.py"
    ).read_text(encoding="utf-8")
    compatibility = (
        PROJECT_ROOT / "megatron" / "olmo3_mindspeed_compat.py"
    ).read_text(encoding="utf-8")

    assert "OLMO3_RUNTIME_TP2_SP_MC2_ACTIVE" in pretrain
    assert "OLMO3_RUNTIME_HSDP_TP_LANE_PACK2_ACTIVE" in compatibility
    assert pretrain.index("model = build_model(") < pretrain.index(
        "OLMO3_RUNTIME_TP2_SP_MC2_ACTIVE"
    )
    assert compatibility.index(
        "self.grad_reduce_handle = olmo3_hsdp_tp_lane_pack_reduce("
    ) < compatibility.index(
        "OLMO3_RUNTIME_HSDP_TP_LANE_PACK2_ACTIVE"
    )
