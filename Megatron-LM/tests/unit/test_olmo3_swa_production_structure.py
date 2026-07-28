"""Static guardrails for the production OLMo3 SWA/CP implementation."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "modeling"
    / "olmo3_swa.py"
)
BASE_SOURCE = SOURCE.with_name("olmo3_base.py")
SIAMESE_DEPTH_SOURCE = SOURCE.with_name("olmo3_siamese_depth_base.py")


def test_only_production_cp_primitives_remain() -> None:
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {
        "_single_hop_left_halo_exchange",
        "_VariableLeftHaloExchange",
        "_AsyncVariableLeftHaloExchange",
        "_PackedSwaOneHopOverlap",
        "_FusedQkvUlyssesAllToAll",
        "_packed_swa_halo_plan",
        "_pack_swa_halo_tnd",
    }.issubset(definitions)


def test_attention_module_adds_no_checkpointed_halo_state() -> None:
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    attention = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Olmo3DotProductAttention"
    )
    constructor = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    forbidden_calls = {"register_parameter", "register_buffer"}
    called = {
        node.func.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert forbidden_calls.isdisjoint(called)


def _function_source(
    module: ast.Module,
    *,
    function: str,
    class_name: str | None = None,
) -> str:
    body: list[ast.stmt] = module.body
    if class_name is not None:
        owner = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = owner.body
    node = next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    )
    return ast.unparse(node)


def test_single_halo_runtime_markers_guard_the_actual_optimized_branches() -> None:
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    attention_prepare = _function_source(
        module,
        class_name="Olmo3DotProductAttention",
        function="_prepare_ulysses_qkv",
    )
    packed_halo = _function_source(
        module,
        class_name="Olmo3DotProductAttention",
        function="_mindspeed_packed_swa_halo_forward",
    )
    overlap_forward = _function_source(
        module,
        class_name="_PackedSwaOneHopOverlap",
        function="forward",
    )
    overlap_backward = _function_source(
        module,
        class_name="_PackedSwaOneHopOverlap",
        function="backward",
    )

    assert "OLMO3_RUNTIME_FULL_FUSED_QKV_A2A_ACTIVE" in attention_prepare
    assert "OLMO3_RUNTIME_CP_SINGLE_HALO_ACTIVE" in packed_halo
    assert (
        "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_FORWARD_ACTIVE"
        in overlap_forward
    )
    assert (
        "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_BACKWARD_ACTIVE"
        in overlap_backward
    )
    assert overlap_forward.index("_begin_adjacent_halo_p2p(") < (
        overlap_forward.index(
            "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_FORWARD_ACTIVE"
        )
    )
    assert overlap_backward.index("_begin_adjacent_halo_p2p(") < (
        overlap_backward.index(
            "OLMO3_RUNTIME_SINGLE_HALO_ASYNC_BACKWARD_ACTIVE"
        )
    )
    assert attention_prepare.index("_fused_qkv_ulysses_all_to_all(") < (
        attention_prepare.index(
            "OLMO3_RUNTIME_FULL_FUSED_QKV_A2A_ACTIVE"
        )
    )


def test_olmo3_self_attention_removes_only_the_external_ulysses_owner() -> None:
    """Both architectures restore OLMo3 as the one CP communication owner."""

    base_module = ast.parse(BASE_SOURCE.read_text(encoding="utf-8"))
    base_init = _function_source(
        base_module,
        class_name="Olmo3BaseSelfAttention",
        function="__init__",
    )
    depth_module = ast.parse(SIAMESE_DEPTH_SOURCE.read_text(encoding="utf-8"))
    depth_init = _function_source(
        depth_module,
        class_name="Olmo3DepthSelfAttentionBase",
        function="__init__",
    )
    assert "unwrap_external_ulysses_for_olmo3(self.core_attention)" in base_init
    assert "unwrap_external_ulysses_for_olmo3(self.core_attention)" in depth_init
    assert depth_init.index("unwrap_external_ulysses_for_olmo3") < (
        depth_init.index("DepthAwareCoreAttention")
    )


def test_packed_swa_stays_local_while_full_attention_owns_ulysses() -> None:
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    packed = _function_source(
        module,
        class_name="Olmo3DotProductAttention",
        function="_mindspeed_packed_forward",
    )
    prepare = _function_source(
        module,
        class_name="Olmo3DotProductAttention",
        function="_prepare_ulysses_qkv",
    )
    # The sliding branch exits into the CP-local halo before the Full-only
    # fused Ulysses preparation point.
    assert packed.index("_mindspeed_packed_swa_halo_forward") < (
        packed.index("_prepare_ulysses_qkv")
    )
    assert "_fused_qkv_ulysses_all_to_all" in prepare
