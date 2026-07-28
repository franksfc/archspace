from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo3_pipeline.checkpoint import (
    CONTRACT_NAME,
    adopt_legacy_checkpoint,
    architecture_identity,
    build_contract,
    contract_sha256,
    effective_lifecycle_policy,
    inspect_checkpoint,
    lifecycle_policy,
    load_contract,
    load_contract_file,
    prepare_lifecycle,
    schedule_identity,
    validate_lifecycle_source,
    verify_prepared_lifecycle,
    write_contract,
)
from olmo3_pipeline.checkpoint_cli import main as checkpoint_main
from olmo3_pipeline.config import ConfigurationError, compose
from olmo3_pipeline.resolve import resolve


def _config(
    tmp_path: Path,
    *,
    stage: str = "stage1",
    variant: str = "base",
    model: str = "1b",
    lifecycle: str | None = None,
) -> dict:
    if stage == "stage1":
        data = "dolma3_6t"
        topology = "stage1_tp1_4096npu"
        performance = ["ordinary_hsdp"]
    elif stage == "stage2":
        data = "dolmino_100b"
        topology = "stage2_cp2_512npu"
        performance = ["cp_single_halo"]
    else:
        raise AssertionError(stage)
    config = compose(
        model=model,
        variant=variant,
        stage=stage,
        data=data,
        topology=topology,
        performance=performance,
    )
    config["data"]["root"] = str(tmp_path / "data")
    config["data"]["tokenizer"]["path"] = str(tmp_path / "tokenizer")
    if stage == "stage1":
        config["data"]["data_args_path"] = str(tmp_path / "data-args.json")
        config["training"]["train_tokens"] = 6_000_000_000_000
        config["optimization"]["peak_lr"] = 1e-3
        config["optimization"]["min_lr"] = 1e-4
        config["optimization"]["warmup_tokens"] = 8_388_608_000
    else:
        config["data"]["manifest"] = str(
            tmp_path / "runtime.data-manifest.json"
        )
        config["data"]["work_dir"] = str(tmp_path / "work")
        config["optimization"]["peak_lr"] = 2.5e-4
    selected_lifecycle = lifecycle or ("fresh" if stage == "stage1" else "transition")
    config["runtime"] = {
        "run_id": f"test-{stage}-{variant}",
        "lifecycle": selected_lifecycle,
        "load": (
            str(tmp_path / "source-placeholder")
            if selected_lifecycle in {"resume", "transition"}
            else None
        ),
        "save": str(tmp_path / f"{stage}-save"),
        "output": str(tmp_path / f"{stage}-output"),
        "python": "python3",
    }
    return resolve(config)


def _write_dcp_metadata(
    iteration_dir: Path,
    *,
    relative_path: str = "__0_0.distcp",
    offset: int = 2,
    length: int = 5,
) -> None:
    metadata = SimpleNamespace(
        storage_data={
            "model.weight": SimpleNamespace(
                relative_path=relative_path,
                offset=offset,
                length=length,
            )
        }
    )
    with (iteration_dir / ".metadata").open("wb") as stream:
        pickle.dump(metadata, stream)


def _complete_checkpoint(root: Path, iteration: int = 12) -> Path:
    iteration_dir = root / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "metadata.json").write_text(
        '{"sharded_backend": "torch_dist", "sharded_backend_version": 1}\n',
        encoding="utf-8",
    )
    (iteration_dir / "common.pt").write_bytes(b"common-state")
    (iteration_dir / "__0_0.distcp").write_bytes(b"0123456789")
    _write_dcp_metadata(iteration_dir)
    (root / "latest_checkpointed_iteration.txt").write_text(
        f"{iteration}\n", encoding="utf-8"
    )
    return iteration_dir


def test_lifecycle_policy_preserves_optimizer_but_resets_transition_state() -> None:
    resume = lifecycle_policy("resume")
    transition = lifecycle_policy("transition")
    for name in ("load_model", "load_adam_moments", "load_master_params"):
        assert resume[name]
        assert transition[name]
    for name in ("load_scheduler", "load_rng", "load_counters"):
        assert resume[name]
        assert not transition[name]


def test_base_and_siamese_depth_are_distinct_keyspaces(tmp_path: Path) -> None:
    base = _config(tmp_path, variant="base")
    siamese = _config(tmp_path, variant="siamese_depth")
    assert architecture_identity(base) != architecture_identity(siamese)
    siamese["runtime"]["lifecycle"] = "resume"
    with pytest.raises(ConfigurationError, match="keyspace mismatch"):
        validate_lifecycle_source(siamese, build_contract(base))


def test_model_sizes_are_fail_closed(tmp_path: Path) -> None:
    source = _config(tmp_path, model="1b")
    target = _config(tmp_path, model="3b")
    target["runtime"]["lifecycle"] = "resume"
    with pytest.raises(ConfigurationError, match="keyspace mismatch"):
        validate_lifecycle_source(target, build_contract(source))


def test_stage_transition_keeps_optimizer_and_allows_topology_reshard(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _complete_checkpoint(source_root)
    source = _config(tmp_path, stage="stage1")
    source["runtime"]["save"] = str(source_root)
    source_contract = build_contract(source)
    write_contract(source_root, source_contract)

    target = _config(tmp_path, stage="stage2", lifecycle="transition")
    target["runtime"]["load"] = str(source_root)
    prepared = prepare_lifecycle(target, write=False)
    assert prepared.source_contract == source_contract
    assert prepared.target_contract["writer"]["policy"]["load_adam_moments"]
    assert not prepared.target_contract["writer"]["policy"]["load_scheduler"]
    assert (
        prepared.target_contract["writer"]["topology"]["context_parallel"] == 2
    )


def test_non_reshardable_optimizer_rejects_topology_change(tmp_path: Path) -> None:
    source = _config(tmp_path, stage="stage1")
    contract = build_contract(source)
    contract["format"]["fully_sharded_model_space"] = False
    contract["format"]["topology_reshardable"] = False
    target = _config(tmp_path, stage="stage2", lifecycle="transition")
    with pytest.raises(ConfigurationError, match="topology changed"):
        validate_lifecycle_source(target, contract)


def test_same_stage_resume_restores_full_trainer_state(tmp_path: Path) -> None:
    source = _config(tmp_path, stage="stage1")
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["run_id"] = "resume-test"
    contract = build_contract(source)
    assert contract["schedule"] == schedule_identity(source)
    validate_lifecycle_source(target, contract)
    assert all(lifecycle_policy("resume").values())


def test_exit_interval_is_not_part_of_resume_schedule_identity(
    tmp_path: Path,
) -> None:
    source = _config(tmp_path, stage="stage1")
    source["training"]["exit_interval"] = 1
    contract = build_contract(source)
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["run_id"] = "resume-at-next-exit"
    target["training"]["exit_interval"] = 2
    assert "exit_interval" not in contract["schedule"]
    validate_lifecycle_source(target, contract)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("training", "global_batch_size", 4096),
        ("training", "micro_batch_size", 1),
        ("training", "train_iters", 89_408),
        ("optimization", "peak_lr", 9e-4),
        ("optimization", "min_lr", 9e-5),
        ("optimization", "scheduler", "linear"),
        ("optimization", "warmup_steps", 251),
        ("optimization", "warmup_tokens", 8_388_608_001),
        ("optimization", "optimizer", "other"),
        ("optimization", "weight_decay", 0.2),
        ("optimization", "adam_beta1", 0.8),
        ("optimization", "adam_beta2", 0.9),
        ("optimization", "adam_eps", 1e-7),
        ("optimization", "z_loss", 2e-5),
        ("optimization", "grad_clip", 0.5),
    ],
)
def test_same_stage_resume_rejects_changed_schedule(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
) -> None:
    source = _config(tmp_path, stage="stage1")
    contract = build_contract(source)
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["run_id"] = f"changed-{field}"
    target[section][field] = replacement
    with pytest.raises(ConfigurationError, match="resume schedule mismatch"):
        validate_lifecycle_source(target, contract)


def test_legacy_contract_without_schedule_remains_resume_compatible(
    tmp_path: Path,
) -> None:
    source = _config(tmp_path, stage="stage1")
    legacy_contract = build_contract(source)
    legacy_contract.pop("schedule")
    legacy_contract["contract_sha256"] = contract_sha256(legacy_contract)
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["run_id"] = "legacy-schedule-resume"
    target["optimization"]["peak_lr"] = 8e-4
    validate_lifecycle_source(target, legacy_contract)


def test_tp_reshard_preserves_state_but_resets_unmappable_rng(tmp_path: Path) -> None:
    source = _config(tmp_path, stage="stage1")
    contract = build_contract(source)
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["topology"]["tensor_parallel"] = 2
    target["topology"]["data_parallel"] //= 2
    policy = effective_lifecycle_policy(target, contract)
    assert policy["load_model"]
    assert policy["load_adam_moments"]
    assert policy["load_master_params"]
    assert policy["load_scheduler"]
    assert policy["load_counters"]
    assert not policy["load_rng"]
    assert policy["rng_reset_reason"] == "tensor_or_pipeline_parallel_changed"


def test_invalid_stage_skip_is_rejected(tmp_path: Path) -> None:
    source = _config(tmp_path, stage="stage2")
    target = _config(tmp_path, stage="stage1", lifecycle="fresh")
    target["runtime"]["lifecycle"] = "transition"
    with pytest.raises(ConfigurationError, match="unsupported OLMo 3 stage transition"):
        validate_lifecycle_source(target, build_contract(source))


def test_inspect_and_explicit_legacy_adoption(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _complete_checkpoint(root, iteration=42)
    inspection = inspect_checkpoint(root)
    assert inspection["tracker_iteration"] == 42
    assert inspection["latest_complete"]
    assert inspection["contract"] is None

    config = _config(tmp_path)
    with pytest.raises(ConfigurationError, match="i-verified-model-and-variant"):
        adopt_legacy_checkpoint(root, config, acknowledgement=False)
    path = adopt_legacy_checkpoint(root, config, acknowledgement=True)
    assert path.name == CONTRACT_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["adopted_legacy_checkpoint"] is True


def test_inspect_accepts_legacy_torch_dist_without_metadata_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-without-informational-metadata"
    iteration_dir = _complete_checkpoint(root, iteration=42)
    (iteration_dir / "metadata.json").unlink()

    inspection = inspect_checkpoint(root)

    assert inspection["latest_complete"]
    assert inspection["latest"]["metadata_present"] is False
    assert inspection["latest"]["metadata_error"] == (
        "metadata.json is absent (legacy torch_dist)"
    )
    assert inspection["latest"]["integrity_errors"] == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("metadata_json", "sharded_backend"),
        ("common", "common.pt"),
        ("dcp_metadata", ".metadata"),
        ("distcp", "non-empty .distcp"),
        ("extent", "extent exceeds"),
        ("traversal", "unsafe relative_path"),
    ],
)
def test_inspect_rejects_incomplete_or_unsafe_dcp_storage(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    root = tmp_path / mutation
    iteration_dir = _complete_checkpoint(root)
    if mutation == "metadata_json":
        (iteration_dir / "metadata.json").write_text(
            '{"sharded_backend": "zarr", "sharded_backend_version": 1}\n',
            encoding="utf-8",
        )
    elif mutation == "common":
        (iteration_dir / "common.pt").write_bytes(b"")
    elif mutation == "dcp_metadata":
        (iteration_dir / ".metadata").write_bytes(b"")
    elif mutation == "distcp":
        (iteration_dir / "__0_0.distcp").write_bytes(b"")
    elif mutation == "extent":
        _write_dcp_metadata(iteration_dir, offset=8, length=4)
    elif mutation == "traversal":
        _write_dcp_metadata(iteration_dir, relative_path="../escape.distcp")
    else:
        raise AssertionError(mutation)

    inspection = inspect_checkpoint(root)
    assert not inspection["latest_complete"]
    assert expected_error in " ".join(inspection["latest"]["integrity_errors"])


def test_inspect_rejects_dcp_metadata_without_storage_data(tmp_path: Path) -> None:
    root = tmp_path / "no-storage-data"
    iteration_dir = _complete_checkpoint(root)
    with (iteration_dir / ".metadata").open("wb") as stream:
        pickle.dump(SimpleNamespace(storage_data={}), stream)
    inspection = inspect_checkpoint(root)
    assert not inspection["latest_complete"]
    assert ".metadata storage_data is empty" in inspection["latest"][
        "integrity_errors"
    ]


def test_weight_only_hf_import_cannot_masquerade_as_resume(tmp_path: Path) -> None:
    root = tmp_path / "weight-only"
    _complete_checkpoint(root)
    config = _config(tmp_path)
    path = adopt_legacy_checkpoint(
        root,
        config,
        acknowledgement=True,
        weights_only=True,
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["format"]["distributed_optimizer"] is False
    target = copy.deepcopy(config)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["load"] = str(root)
    with pytest.raises(ConfigurationError, match="distributed AdamW state"):
        validate_lifecycle_source(target, contract)


def test_checkpoint_intent_rejects_run_id_path_traversal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["runtime"]["run_id"] = "../escape"
    with pytest.raises(ConfigurationError, match="unsafe run ID"):
        prepare_lifecycle(config, write=False)


def test_contract_digest_detects_format_tampering(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    config = _config(tmp_path)
    path = write_contract(root, build_contract(config))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["format"]["topology_reshardable"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="contract digest mismatch"):
        load_contract(root)


def test_transition_never_reuses_existing_destination(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _complete_checkpoint(source_root)
    source = _config(tmp_path, stage="stage1")
    source["runtime"]["save"] = str(source_root)
    write_contract(source_root, build_contract(source))
    target = _config(tmp_path, stage="stage2", lifecycle="transition")
    target["runtime"]["load"] = str(source_root)
    destination = Path(target["runtime"]["save"])
    write_contract(destination, build_contract(target))
    with pytest.raises(ConfigurationError, match="destination already"):
        prepare_lifecycle(target, write=True)


def test_topology_reshard_requires_new_destination(tmp_path: Path) -> None:
    root = tmp_path / "resume-root"
    _complete_checkpoint(root)
    source = _config(tmp_path)
    source["runtime"]["save"] = str(root)
    write_contract(root, build_contract(source))
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["load"] = str(root)
    target["runtime"]["save"] = str(root)
    target["topology"]["hsdp"]["shard_size"] //= 2
    target["topology"]["hsdp"]["num_instances"] *= 2
    with pytest.raises(ConfigurationError, match="in-place resume cannot change"):
        prepare_lifecycle(target, write=False)


def test_unmounted_source_uses_sealed_local_contract_copy(tmp_path: Path) -> None:
    source = _config(tmp_path, stage="stage1")
    source_contract = build_contract(source)
    local_contract = tmp_path / "copied-source-contract.json"
    local_contract.write_text(
        json.dumps(source_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target = copy.deepcopy(source)
    target["runtime"]["lifecycle"] = "resume"
    target["runtime"]["run_id"] = "unmounted-source-resume"
    target["runtime"]["load"] = str(tmp_path / "not-mounted")
    target["runtime"]["save"] = str(tmp_path / "new-destination")
    prepared = prepare_lifecycle(
        target,
        write=False,
        verify_checkpoint_tree=False,
        source_contract_override=load_contract_file(local_contract),
    )
    assert prepared.source_contract == source_contract

    tampered = json.loads(local_contract.read_text(encoding="utf-8"))
    tampered["writer"]["stage"] = "stage2"
    local_contract.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="contract digest mismatch"):
        load_contract_file(local_contract)


def test_worker_verification_requires_single_writer_activation(tmp_path: Path) -> None:
    config = _config(tmp_path, stage="stage1", lifecycle="fresh")
    with pytest.raises(ConfigurationError, match="identity contract is missing"):
        verify_prepared_lifecycle(config)

    prepared = prepare_lifecycle(config, write=True)
    verified = verify_prepared_lifecycle(config)
    assert verified.target_contract == prepared.target_contract
    assert verified.intent_path == prepared.intent_path


def test_checkpoint_cli_activation_then_worker_gate_roundtrip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, stage="stage1", lifecycle="fresh")
    resolved = tmp_path / "resolved.json"
    resolved.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert checkpoint_main(
        ["preflight", "--resolved", str(resolved), "--write"]
    ) == 0
    activation = json.loads(capsys.readouterr().out)
    assert activation["written"] is True
    assert Path(activation["target_contract_path"]).is_file()
    assert Path(activation["intent_path"]).is_file()

    assert checkpoint_main(
        ["verify-prepared", "--resolved", str(resolved)]
    ) == 0
    worker = json.loads(capsys.readouterr().out)
    assert worker["state"] == "prepared"
    assert worker["intent_path"] == activation["intent_path"]


def test_worker_verification_rejects_tampered_run_intent(tmp_path: Path) -> None:
    config = _config(tmp_path, stage="stage1", lifecycle="fresh")
    prepared = prepare_lifecycle(config, write=True)
    value = json.loads(prepared.intent_path.read_text(encoding="utf-8"))
    value["run_id"] = "another-run"
    prepared.intent_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="intent does not match"):
        verify_prepared_lifecycle(config)


def test_in_place_resume_keeps_root_identity_but_verifies_new_intent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resume-root"
    _complete_checkpoint(root)
    source = _config(tmp_path, stage="stage1", lifecycle="fresh")
    source["runtime"]["save"] = str(root)
    source_contract = build_contract(source)
    write_contract(root, source_contract)

    resume = copy.deepcopy(source)
    resume["runtime"]["run_id"] = "same-stage-resume"
    resume["runtime"]["lifecycle"] = "resume"
    resume["runtime"]["load"] = str(root)
    prepare_lifecycle(resume, write=True)
    verified = verify_prepared_lifecycle(resume)
    assert load_contract(root) == source_contract
    assert verified.target_contract["writer"]["run_id"] == "same-stage-resume"
