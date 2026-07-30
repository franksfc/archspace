from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import olmo3_pipeline.data_pipeline as data_pipeline
from olmo3_pipeline.data_pipeline import (
    DataPipelineError,
    FileEntry,
    balanced_shards,
    build_cache_plan,
    build_runtime_data_manifest,
    build_sft_plan,
    build_tokenization_plan,
    create_inventory,
    download_plan,
    execute_download,
    finalize_stage1_index,
    _sft_packed_completion,
    load_runtime_data_manifest,
    materialize_tokenization_plan,
    run_sft_plan,
    validate_runtime_data_profile,
    verify_inventory,
    write_immutable_json,
)
from olmo3_pipeline.config import PROJECT_ROOT


def _files(root: Path) -> None:
    for name, size in (
        ("data/a.jsonl.zst", 101),
        ("data/b.jsonl.zst", 83),
        ("data/c.jsonl.zst", 47),
        ("data/d.jsonl.zst", 31),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([size % 251]) * size)


def _tokenizer(root: Path) -> Path:
    path = root / "tokenizer"
    path.mkdir()
    (path / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return path


def test_inventory_roundtrip_and_checksum_failure(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _files(root)
    manifest = tmp_path / "inventory.json"
    inventory = create_inventory(
        root=root,
        patterns=("**/*.jsonl.zst",),
        checksums=True,
        workers=2,
    )
    write_immutable_json(manifest, inventory)
    report = verify_inventory(manifest, root=None, checksums=True, workers=2)
    assert report["state"] == "ok"
    assert report["files"] == 4

    (root / "data" / "b.jsonl.zst").write_bytes(b"changed")
    report = verify_inventory(manifest, root=None, checksums=True, workers=2)
    assert report["state"] == "failed"
    assert report["problems"][0]["path"] == "data/b.jsonl.zst"


def test_recursive_pattern_also_matches_root_file(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "root.npy").write_bytes(b"\x00\x00\x00\x00")
    inventory = create_inventory(
        root=root,
        patterns=("**/*.npy",),
        checksums=False,
        workers=1,
    )
    assert [entry["path"] for entry in inventory["entries"]] == ["root.npy"]


def test_balanced_shards_are_deterministic() -> None:
    entries = [
        FileEntry("a", 10),
        FileEntry("b", 9),
        FileEntry("c", 8),
        FileEntry("d", 7),
    ]
    first = balanced_shards(entries, 2)
    second = balanced_shards(tuple(reversed(entries)), 2)
    assert first == second
    assert [sum(item.size for item in bucket) for bucket in first] == [17, 17]


def test_dolma_plan_has_no_embedded_deployment_root(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _files(root)
    manifest = tmp_path / "inventory.json"
    write_immutable_json(
        manifest,
        create_inventory(
            root=root,
            patterns=("**/*.jsonl.zst",),
            checksums=False,
            workers=1,
        ),
    )
    plan_path = tmp_path / "contracts" / "tokenize.json"
    plan = build_tokenization_plan(
        config_name="dolmino_100b",
        source_manifest=manifest,
        source_root=None,
        output_root=tmp_path / "output",
        work_root=tmp_path / "work",
        tokenizer=_tokenizer(tmp_path),
        tokenizer_sha256=None,
        engine="dolma",
        shards=2,
        workers=3,
        patterns=("**/*.jsonl.zst",),
        python="/usr/bin/python3",
        preprocess_script=None,
        preprocess_script_sha256=None,
        sequence_length=None,
    )
    materialize_tokenization_plan(plan, plan_path)
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "work").is_dir()
    loaded = json.loads(plan_path.read_text())
    assert loaded["parts"][0]["config"]["processes"] == 3
    assert "fields" not in loaded["parts"][0]["config"]
    assert loaded["parts"][0]["config"]["tokenizer"]["name_or_path"].endswith(
        "/tokenizer/tokenizer.json"
    )
    assert loaded["sequence_length"] == 8192
    assert str(tmp_path) in loaded["source_root"]
    assert "/private/example-organization/example-user" not in plan_path.read_text()


def test_stage1_finalize_writes_megatron_data_args(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _files(root)
    manifest = tmp_path / "inventory.json"
    write_immutable_json(
        manifest,
        create_inventory(
            root=root,
            patterns=("**/*.jsonl.zst",),
            checksums=False,
            workers=1,
        ),
    )
    plan_path = tmp_path / "plan.json"
    preprocess_script = tmp_path / "preprocess_data.py"
    preprocess_script.write_text("# smoke\n", encoding="utf-8")
    plan = build_tokenization_plan(
        config_name="dolma3_6t",
        source_manifest=manifest,
        source_root=None,
        output_root=tmp_path / "indexed",
        work_root=tmp_path / "work",
        tokenizer=_tokenizer(tmp_path),
        tokenizer_sha256=None,
        engine="megatron",
        shards=2,
        workers=1,
        patterns=("**/*.jsonl.zst",),
        python="/usr/bin/python3",
        preprocess_script=preprocess_script,
        preprocess_script_sha256=None,
        sequence_length=None,
    )
    materialize_tokenization_plan(plan, plan_path)
    for part in plan["parts"]:
        prefix = Path(part["expected_data_prefix"])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.bin").write_bytes(b"\x00" * 16)
        Path(f"{prefix}.idx").write_bytes(b"IDX")
    args_path = tmp_path / "indexed" / "data_args_path.txt"
    report = finalize_stage1_index(
        plan_path=plan_path,
        data_args_path=args_path,
    )
    assert report["state"] == "ok"
    assert len(args_path.read_text().splitlines()) == 2


def test_stage1_index_rejects_document_drop_threshold(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _files(root)
    manifest = tmp_path / "inventory.json"
    write_immutable_json(
        manifest,
        create_inventory(
            root=root,
            patterns=("**/*.jsonl.zst",),
            checksums=False,
            workers=1,
        ),
    )
    with pytest.raises(DataPipelineError, match="must not set --seq-length"):
        build_tokenization_plan(
            config_name="dolma3_6t",
            source_manifest=manifest,
            source_root=None,
            output_root=tmp_path / "indexed",
            work_root=tmp_path / "work",
            tokenizer=_tokenizer(tmp_path),
            tokenizer_sha256=None,
            engine="megatron",
            shards=2,
            workers=1,
            patterns=("**/*.jsonl.zst",),
            python="/usr/bin/python3",
            preprocess_script=tmp_path / "preprocess_data.py",
            preprocess_script_sha256="0" * 64,
            sequence_length=8192,
        )


def test_immutable_manifest_rejects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    write_immutable_json(path, {"a": 1})
    write_immutable_json(path, {"a": 1})
    with pytest.raises(DataPipelineError):
        write_immutable_json(path, {"a": 2})


def test_inventory_contract_detects_manifest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _files(root)
    manifest = tmp_path / "inventory.json"
    inventory = create_inventory(
        root=root,
        patterns=("**/*.jsonl.zst",),
        checksums=False,
        workers=1,
    )
    write_immutable_json(manifest, inventory)
    tampered = json.loads(manifest.read_text())
    tampered["entries"][0]["size"] += 1
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DataPipelineError, match="contract SHA256"):
        verify_inventory(manifest, root=None, checksums=False, workers=1)


def test_download_cli_defaults_to_no_network_dry_run(tmp_path: Path) -> None:
    plan = tmp_path / "download.plan.json"
    manifest = tmp_path / "download.inventory.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data" / "olmo3_data.py"),
            "download",
            "--data",
            "dolmino_100b",
            "--root",
            str(tmp_path / "not-downloaded"),
            "--plan",
            str(plan),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert '"state": "planned"' in completed.stdout
    assert plan.is_file()
    assert not manifest.exists()


def test_download_rejects_nonempty_unclaimed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "stale.jsonl.zst").write_bytes(b"stale")
    manifest = tmp_path / "inventory.json"
    plan = download_plan(
        config_name="dolmino_100b",
        root=root,
        manifest=manifest,
        repository=None,
        revision=None,
        endpoint=None,
        patterns=("**/*.jsonl.zst",),
        workers=1,
        checksums=False,
    )
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **_: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with pytest.raises(DataPipelineError, match="non-empty"):
        execute_download(plan)


def test_download_claim_allows_same_plan_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    manifest = tmp_path / "inventory.json"
    plan = download_plan(
        config_name="dolmino_100b",
        root=root,
        manifest=manifest,
        repository=None,
        revision=None,
        endpoint=None,
        patterns=("**/*.jsonl.zst",),
        workers=1,
        checksums=False,
    )

    def fake_snapshot_download(**_: object) -> None:
        path = root / "data" / "part.jsonl.zst"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    first = execute_download(plan)
    second = execute_download(plan)
    assert first["root_claim_sha256"] == second["root_claim_sha256"]


def test_sft_plan_freezes_raw_inventory_and_assistant_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "part.parquet").write_bytes(b"PAR1")
    raw_manifest = tmp_path / "raw.json"
    write_immutable_json(
        raw_manifest,
        create_inventory(
            root=raw,
            patterns=("**/*.parquet",),
            checksums=False,
            workers=1,
        ),
    )
    monkeypatch.setattr(data_pipeline, "_git_head", lambda _path: "0" * 40)
    monkeypatch.setattr(data_pipeline, "_git_dirty", lambda _path: False)
    plan = build_sft_plan(
        config_name="dolci_think",
        raw_manifest=raw_manifest,
        raw_root=None,
        expected_raw_files=1,
        raw_glob=str(raw / "*.parquet"),
        converted_root=tmp_path / "converted",
        converted_manifest=tmp_path / "converted.json",
        work_root=tmp_path / "work",
        tokenizer=_tokenizer(tmp_path),
        tokenizer_sha256=None,
        converter=tmp_path / "convert.py",
        converter_sha256="0" * 64,
        open_instruct_root=tmp_path / "open-instruct",
        python=sys.executable,
        workers=2,
        shuffle_seed=42,
    )
    assert plan["assistant_only_loss"] is True
    assert plan["sequence_length"] == 32768
    assert plan["raw_files"] == 1


def test_production_sft_expands_pristine_parquet_inventory(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("part-0.parquet", "part-1.parquet"):
        (raw / name).write_bytes(b"PAR1")
    raw_manifest = tmp_path / "raw.json"
    write_immutable_json(
        raw_manifest,
        create_inventory(
            root=raw,
            patterns=("**/*.parquet",),
            checksums=False,
            workers=1,
        ),
    )
    open_instruct = tmp_path / "open-instruct"
    open_instruct.mkdir()
    subprocess.run(["git", "init", "-q", str(open_instruct)], check=True)
    (open_instruct / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(open_instruct), "add", "README.md"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(open_instruct),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    converter = tmp_path / "convert.py"
    converter.write_text("# fixture\n", encoding="utf-8")
    plan = build_sft_plan(
        config_name="dolci_think",
        raw_manifest=raw_manifest,
        raw_root=None,
        expected_raw_files=2,
        raw_glob=str(raw / "*.parquet"),
        converted_root=tmp_path / "converted",
        converted_manifest=tmp_path / "converted.json",
        work_root=tmp_path / "work",
        tokenizer=_tokenizer(tmp_path),
        tokenizer_sha256=None,
        converter=converter,
        converter_sha256=None,
        open_instruct_root=open_instruct,
        python=sys.executable,
        workers=1,
        shuffle_seed=42,
    )
    expected = [
        str((raw / "part-0.parquet").resolve()),
        "1.0",
        str((raw / "part-1.parquet").resolve()),
        "1.0",
    ]
    assert plan["dataset_mixer"] == expected
    command = plan["commands"][0]
    mixer_start = command.index("--dataset_mixer_list") + 1
    split_start = command.index("--dataset_mixer_list_splits")
    assert command[mixer_start:split_start] == expected

    (open_instruct / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(DataPipelineError, match="pristine"):
        build_sft_plan(
            config_name="dolci_think",
            raw_manifest=raw_manifest,
            raw_root=None,
            expected_raw_files=2,
            raw_glob=str(raw / "*.parquet"),
            converted_root=tmp_path / "converted-2",
            converted_manifest=tmp_path / "converted-2.json",
            work_root=tmp_path / "work-2",
            tokenizer=_tokenizer(second),
            tokenizer_sha256=None,
            converter=converter,
            converter_sha256=None,
            open_instruct_root=open_instruct,
            python=sys.executable,
            workers=1,
            shuffle_seed=42,
        )
    assert plan["raw_manifest_sha256"]


def test_run_sft_plan_propagates_packed_completion_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "part.parquet").write_bytes(b"PAR1")
    raw_manifest = tmp_path / "raw.json"
    write_immutable_json(
        raw_manifest,
        create_inventory(
            root=raw,
            patterns=("**/*.parquet",),
            checksums=False,
            workers=1,
        ),
    )
    converter = tmp_path / "convert.py"
    converter.write_text("# fixture\n", encoding="utf-8")
    converted_root = tmp_path / "converted"
    plan_path = tmp_path / "sft.plan.json"
    monkeypatch.setattr(data_pipeline, "_git_head", lambda _path: "0" * 40)
    monkeypatch.setattr(data_pipeline, "_git_dirty", lambda _path: False)
    plan = build_sft_plan(
        config_name="dolci_think",
        raw_manifest=raw_manifest,
        raw_root=None,
        expected_raw_files=1,
        raw_glob=str(raw / "*.parquet"),
        converted_root=converted_root,
        converted_manifest=tmp_path / "converted.json",
        work_root=tmp_path / "work",
        tokenizer=_tokenizer(tmp_path),
        tokenizer_sha256=None,
        converter=converter,
        converter_sha256=None,
        open_instruct_root=tmp_path / "open-instruct",
        python=sys.executable,
        workers=1,
        shuffle_seed=42,
    )
    write_immutable_json(plan_path, plan)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        converted_root.mkdir(exist_ok=True)
        (converted_root / "token_ids_part_0000.npy").write_bytes(bytes(4))
        (converted_root / "labels_mask_part_0000.npy").write_bytes(bytes(1))
        (converted_root / "token_ids_part_0000.csv.gz").write_bytes(b"metadata")
        (converted_root / "dataset_statistics.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(data_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(
        data_pipeline,
        "provenance",
        lambda **_kwargs: {"action": "test-fixture"},
    )
    monkeypatch.setattr(
        data_pipeline,
        "_sft_packed_completion",
        lambda **_kwargs: {
            "packed_complete_manifest": "/contracts/complete.json",
            "packed_complete_manifest_sha256": "a" * 64,
            "instances": 19,
            "fingerprint": "sha256:sft-fixture",
        },
    )

    report = run_sft_plan(plan_path, dry_run=False)

    assert report["state"] == "ok"
    assert report["instances"] == 19
    assert report["fingerprint"] == "sha256:sft-fixture"
    assert report["packed_complete_manifest_sha256"] == "a" * 64


def _fake_stage_data_config(
    tmp_path: Path,
    *,
    stage: str,
    backend: str,
    known_token_count: int | None,
    scope: str = "production",
    token_count_policy: str = "fixed",
) -> tuple[dict, Path]:
    config_path = tmp_path / f"{stage}.json"
    config_path.write_text('{"test": true}\n', encoding="utf-8")
    return (
        {
            "name": f"test_{stage}",
            "stage": stage,
            "backend": backend,
            "scope": scope,
            "token_count_policy": token_count_policy,
            "known_token_count": known_token_count,
            "tokenizer": {
                "eos_token_id": 100257,
                "pad_token_id": 100277,
                "bos_token_id": None,
            },
            "packing": {
                "kind": (
                    "fixed_sequence_length"
                    if stage == "stage2"
                    else "obfd_fsl"
                ),
                "sequence_length": 8192 if stage == "stage2" else 65536,
            },
        },
        config_path,
    )


def test_runtime_manifest_has_variable_ordered_sources_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenized = tmp_path / "tokenized"
    for index, tokens in enumerate((2, 3, 5)):
        token_path = tokenized / f"part-{index:02d}" / "tokens.npy"
        token_path.parent.mkdir(parents=True)
        token_path.write_bytes(bytes(tokens * 4))
        token_path.with_suffix(".csv.gz").write_bytes(b"metadata")
    inventory_path = tmp_path / "tokenized.inventory.json"
    write_immutable_json(
        inventory_path,
        create_inventory(
            root=tokenized,
            patterns=("**/*.npy", "**/*.csv.gz"),
            checksums=True,
            workers=2,
        ),
    )
    fake = _fake_stage_data_config(
        tmp_path,
        stage="stage3",
        backend="olmo3_numpy_packed",
        known_token_count=10,
    )
    monkeypatch.setattr(data_pipeline, "data_config", lambda _name: fake)
    manifest = build_runtime_data_manifest(
        config_name="test_longmino",
        inventory_path=inventory_path,
        runtime_root=tmp_path / "pod-visible-mount",
    )
    manifest_path = tmp_path / "runtime.data-manifest.json"
    write_immutable_json(manifest_path, manifest)
    loaded = load_runtime_data_manifest(
        manifest_path,
        root=tokenized,
        verify_files=True,
        checksums=True,
        expected_stage="stage3",
        expected_backend="olmo3_numpy_packed",
        expected_token_count=10,
    )
    assert loaded["source_count"] == 3
    assert loaded["token_count"] == 10
    assert [source["source_id"] for source in loaded["sources"]] == [0, 1, 2]
    assert loaded["resolved_token_paths"] == [
        str(tokenized / f"part-{index:02d}" / "tokens.npy")
        for index in range(3)
    ]
    assert all(source["metadata_path"].endswith(".csv.gz") for source in loaded["sources"])
    cache_plan = build_cache_plan(
        config_name="test_longmino",
        source_manifest=manifest_path,
        source_root=None,
        work_root=tmp_path / "cache",
        workers=2,
        patterns=("**/*.npy",),
        python=sys.executable,
    )
    assert cache_plan["source_manifest_schema"] == "olmo3-runtime-data-manifest-v1"
    assert cache_plan["source_files"] == 3
    assert cache_plan["source_tokens"] == 10


def test_manifest_count_policy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    (tokenized / "tokens.npy").write_bytes(bytes(4))
    (tokenized / "tokens.csv.gz").write_bytes(b"metadata")
    inventory_path = tmp_path / "tokenized.inventory.json"
    write_immutable_json(
        inventory_path,
        create_inventory(
            root=tokenized,
            patterns=("**/*.npy", "**/*.csv.gz"),
            checksums=True,
            workers=1,
        ),
    )
    fake = _fake_stage_data_config(
        tmp_path,
        stage="stage2",
        backend="olmo3_numpy_fsl",
        known_token_count=None,
        scope="production",
        token_count_policy="manifest",
    )
    monkeypatch.setattr(data_pipeline, "data_config", lambda _name: fake)
    with pytest.raises(DataPipelineError, match="must use token_count_policy='fixed'"):
        build_runtime_data_manifest(
            config_name="bad_production_profile",
            inventory_path=inventory_path,
            runtime_root=None,
        )


def test_runtime_manifest_is_bound_to_exact_frozen_data_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "name": "test_longmino",
        "stage": "stage3",
        "backend": "olmo3_numpy_packed",
        "known_token_count": 10,
        "tokenizer": {
            "eos_token_id": 100257,
            "pad_token_id": 100277,
            "bos_token_id": None,
        },
        "packing": {
            "kind": "obfd_fsl",
            "sequence_length": 65536,
            "forbid_cross_document_attention": True,
        },
    }
    config_path = tmp_path / "test_longmino.json"
    config_path.write_text(
        json.dumps({"data": config}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        data_pipeline,
        "data_config",
        lambda _name: (config, config_path),
    )
    digest = data_pipeline.sha256_file(config_path)
    manifest = {
        "data_config": "test_longmino",
        "data_config_sha256": digest,
        "stage": "stage3",
        "backend": "olmo3_numpy_packed",
        "known_token_count": 10,
        "tokenizer": config["tokenizer"],
        "packing": config["packing"],
    }

    validate_runtime_data_profile(
        manifest,
        config_name="test_longmino",
        config_sha256=digest,
        pad_token_id=100277,
    )

    for field, replacement in (
        ("stage", "stage2"),
        ("backend", "olmo3_numpy_fsl"),
        ("known_token_count", 11),
        ("packing", {"kind": "fixed_sequence_length"}),
    ):
        tampered = dict(manifest)
        tampered[field] = replacement
        with pytest.raises(
            DataPipelineError,
            match="does not match the selected frozen profile",
        ):
            validate_runtime_data_profile(
                tampered,
                config_name="test_longmino",
                config_sha256=digest,
                pad_token_id=100277,
            )

    with pytest.raises(DataPipelineError, match="pad-token ID"):
        validate_runtime_data_profile(
            manifest,
            config_name="test_longmino",
            config_sha256=digest,
            pad_token_id=1,
        )
    with pytest.raises(DataPipelineError, match="SHA256 no longer matches"):
        validate_runtime_data_profile(
            manifest,
            config_name="test_longmino",
            config_sha256="0" * 64,
            pad_token_id=100277,
        )


def test_runtime_manifest_rejects_missing_paired_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    (tokenized / "tokens.npy").write_bytes(bytes(8))
    inventory_path = tmp_path / "tokenized.inventory.json"
    write_immutable_json(
        inventory_path,
        create_inventory(
            root=tokenized,
            patterns=("**/*.npy",),
            checksums=False,
            workers=1,
        ),
    )
    fake = _fake_stage_data_config(
        tmp_path,
        stage="stage2",
        backend="olmo3_numpy_fsl",
        known_token_count=2,
    )
    monkeypatch.setattr(data_pipeline, "data_config", lambda _name: fake)
    with pytest.raises(DataPipelineError, match="missing paired document metadata"):
        build_runtime_data_manifest(
            config_name="test_dolmino",
            inventory_path=inventory_path,
            runtime_root=None,
        )


def test_runtime_manifest_requires_content_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    (tokenized / "tokens.npy").write_bytes(bytes(8))
    (tokenized / "tokens.csv.gz").write_bytes(b"metadata")
    inventory_path = tmp_path / "tokenized.inventory.json"
    write_immutable_json(
        inventory_path,
        create_inventory(
            root=tokenized,
            patterns=("**/*.npy", "**/*.csv.gz"),
            checksums=False,
            workers=1,
        ),
    )
    fake = _fake_stage_data_config(
        tmp_path,
        stage="stage2",
        backend="olmo3_numpy_fsl",
        known_token_count=2,
    )
    monkeypatch.setattr(data_pipeline, "data_config", lambda _name: fake)
    with pytest.raises(DataPipelineError, match="rebuild the inventory with --checksums"):
        build_runtime_data_manifest(
            config_name="test_dolmino",
            inventory_path=inventory_path,
            runtime_root=None,
        )


def test_sft_packed_completion_returns_instances_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "work" / "packed" / "cache-contract"
    cache_root.mkdir(parents=True)
    complete_path = cache_root / "complete.json"
    complete_path.write_text(
        json.dumps(
            {
                "assistant_only_loss": True,
                "fingerprint": "sha256:packed-fixture",
                "instances": 23,
                "sequence_length": 32768,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake_module = types.ModuleType("runtime.olmo3_sft_dataset")
    fake_module.describe_olmo3_sft_cache = lambda *_args, **_kwargs: {
            "base_instances": 23,
            "cache_root": str(cache_root),
            "fingerprint": "sha256:packed-fixture",
            "sequence_length": 32768,
            "source_count": 1,
        }
    monkeypatch.setitem(
        sys.modules, "runtime.olmo3_sft_dataset", fake_module
    )

    result = _sft_packed_completion(
        config_name="dolci_think",
        converted_root=tmp_path / "converted",
        work_root=tmp_path / "work",
    )

    assert result["instances"] == 23
    assert result["fingerprint"] == "sha256:packed-fixture"
    assert result["packed_complete_manifest"] == str(complete_path)
    assert len(result["packed_complete_manifest_sha256"]) == 64
