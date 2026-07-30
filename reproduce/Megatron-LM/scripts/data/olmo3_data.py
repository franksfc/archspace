#!/usr/bin/env python3
"""OLMo 3 Stage 1-4 data download, tokenization, indexing, and packing CLI.

The CLI never embeds deployment paths.  Paths, tokenizer snapshots, source
revisions, worker counts, and shard counts are explicit arguments.  Commands
that can perform expensive work default to planning/dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from olmo3_pipeline.data_pipeline import (
    DataPipelineError,
    build_cache_plan,
    build_runtime_data_manifest,
    build_sft_plan,
    build_tokenization_plan,
    create_inventory,
    download_plan,
    execute_cache_plan,
    execute_download,
    finalize_stage1_index,
    materialize_tokenization_plan,
    run_cache_plan,
    run_download_plan,
    run_sft_plan,
    run_tokenization_part,
    shell_command,
    verify_inventory,
    write_immutable_json,
)


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _patterns(args: argparse.Namespace, fallback: Sequence[str]) -> list[str]:
    return list(args.pattern or fallback)


def _write_report(path: Path | None, value: dict[str, Any]) -> None:
    if path is not None:
        write_immutable_json(path, value)


def cmd_inventory(args: argparse.Namespace) -> None:
    inventory = create_inventory(
        root=args.root,
        patterns=_patterns(args, ()),
        checksums=args.checksums,
        workers=args.workers,
    )
    write_immutable_json(args.manifest, inventory)
    _print(inventory)


def cmd_verify(args: argparse.Namespace) -> None:
    report = verify_inventory(
        args.manifest,
        root=args.root,
        checksums=args.checksums,
        workers=args.workers,
    )
    _write_report(args.report, report)
    _print(report)
    if report["state"] != "ok":
        raise SystemExit(1)


def cmd_download(args: argparse.Namespace) -> None:
    plan = download_plan(
        config_name=args.data,
        root=args.root,
        manifest=args.manifest,
        repository=args.repository,
        revision=args.revision,
        endpoint=args.endpoint,
        patterns=_patterns(args, ("**/*.jsonl.zst", "**/*.jsonl")),
        workers=args.workers,
        checksums=args.checksums,
    )
    write_immutable_json(args.plan, plan)
    if not args.execute:
        _print({**plan, "state": "planned", "execute": False})
        return
    report = execute_download(plan)
    _write_report(args.report, report)
    _print(report)


def cmd_run_download(args: argparse.Namespace) -> None:
    report = run_download_plan(args.plan, dry_run=not args.execute)
    _write_report(args.report, report)
    _print(report)


def cmd_tokenize_plan(args: argparse.Namespace) -> None:
    plan = build_tokenization_plan(
        config_name=args.data,
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        output_root=args.output_root,
        work_root=args.work_root,
        tokenizer=args.tokenizer,
        tokenizer_sha256=args.tokenizer_sha256,
        engine=args.engine,
        shards=args.shards,
        workers=args.workers,
        patterns=_patterns(args, ("**/*.jsonl.zst", "**/*.jsonl")),
        python=args.python,
        preprocess_script=args.preprocess_script,
        preprocess_script_sha256=args.preprocess_script_sha256,
        sequence_length=args.sequence_length,
    )
    materialize_tokenization_plan(plan, args.plan)
    _print(
        {
            "schema": plan["schema"],
            "action": plan["action"],
            "state": "planned",
            "plan": str(args.plan.resolve()),
            "contract_sha256": plan["contract_sha256"],
            "engine": plan["engine"],
            "parts": len(plan["parts"]),
            "commands": [
                shell_command(part["command"]) for part in plan["parts"]
            ],
        }
    )


def cmd_runtime_manifest(args: argparse.Namespace) -> None:
    manifest = build_runtime_data_manifest(
        config_name=args.data,
        inventory_path=args.inventory,
        runtime_root=args.root,
        token_patterns=_patterns(args, ("**/*.npy",)),
        metadata_suffix=args.metadata_suffix,
    )
    write_immutable_json(args.manifest, manifest)
    _print(
        {
            "schema": manifest["schema"],
            "state": "ok",
            "manifest": str(args.manifest.resolve()),
            "contract_sha256": manifest["contract_sha256"],
            "stage": manifest["stage"],
            "backend": manifest["backend"],
            "data_scope": manifest["data_scope"],
            "token_count_policy": manifest["token_count_policy"],
            "root": manifest["root"],
            "source_count": manifest["source_count"],
            "token_count": manifest["token_count"],
            "token_bytes": manifest["token_bytes"],
            "metadata_bytes": manifest["metadata_bytes"],
        }
    )


def cmd_run_tokenize_part(args: argparse.Namespace) -> None:
    report = run_tokenization_part(
        args.plan, part=args.part, dry_run=not args.execute
    )
    _write_report(args.report, report)
    _print(report)


def cmd_finalize_stage1(args: argparse.Namespace) -> None:
    report = finalize_stage1_index(
        plan_path=args.plan,
        data_args_path=args.data_args_path,
    )
    _write_report(args.report, report)
    _print(report)


def cmd_prepare_cache(args: argparse.Namespace) -> None:
    patterns = _patterns(args, ("**/*.npy",))
    plan = build_cache_plan(
        config_name=args.data,
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        work_root=args.work_root,
        workers=args.workers,
        patterns=patterns,
        python=args.python,
    )
    if args.plan is not None:
        write_immutable_json(args.plan, plan)
    if not args.execute:
        _print({**plan, "state": "planned", "execute": False})
        return
    report = execute_cache_plan(
        config_name=args.data,
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        work_root=args.work_root,
        workers=args.workers,
        patterns=patterns,
    )
    _write_report(args.report, report)
    _print(report)


def cmd_run_cache(args: argparse.Namespace) -> None:
    report = run_cache_plan(args.plan, dry_run=not args.execute)
    _write_report(args.report, report)
    _print(report)


def cmd_sft_plan(args: argparse.Namespace) -> None:
    plan = build_sft_plan(
        config_name=args.data,
        raw_manifest=args.raw_manifest,
        raw_root=args.raw_root,
        expected_raw_files=args.expected_files,
        raw_glob=args.raw_glob,
        converted_root=args.converted_root,
        converted_manifest=args.converted_manifest,
        work_root=args.work_root,
        tokenizer=args.tokenizer,
        tokenizer_sha256=args.tokenizer_sha256,
        converter=args.converter,
        converter_sha256=args.converter_sha256,
        open_instruct_root=args.open_instruct_root,
        python=args.python,
        workers=args.workers,
        shuffle_seed=args.shuffle_seed,
    )
    write_immutable_json(args.plan, plan)
    _print(
        {
            **plan,
            "state": "planned",
            "commands_shell": [
                shell_command(command) for command in plan["commands"]
            ],
        }
    )


def cmd_run_sft(args: argparse.Namespace) -> None:
    report = run_sft_plan(args.plan, dry_run=not args.execute)
    _write_report(args.report, report)
    _print(report)


def add_patterns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="POSIX-style relative-path glob; repeat for multiple patterns",
    )


def add_workers(parser: argparse.ArgumentParser, default: int = 16) -> None:
    parser.add_argument("--workers", type=int, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="freeze a local file tree")
    inventory.add_argument("--root", type=Path, required=True)
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--checksums", action="store_true")
    add_patterns(inventory)
    add_workers(inventory)
    inventory.set_defaults(func=cmd_inventory)

    verify = commands.add_parser("verify", help="verify an immutable inventory")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--root", type=Path)
    verify.add_argument("--checksums", action="store_true")
    verify.add_argument("--report", type=Path)
    add_workers(verify)
    verify.set_defaults(func=cmd_verify)

    download = commands.add_parser(
        "download", help="plan or execute a frozen Hugging Face dataset download"
    )
    download.add_argument("--data", required=True)
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--report", type=Path)
    download.add_argument("--repository")
    download.add_argument("--revision")
    download.add_argument("--endpoint")
    download.add_argument("--checksums", action="store_true")
    download.add_argument(
        "--execute",
        action="store_true",
        help="perform network I/O; omitted means dry-run/plan only",
    )
    add_patterns(download)
    add_workers(download, 32)
    download.set_defaults(func=cmd_download)

    run_download = commands.add_parser(
        "run-download", help="resume or rerun one frozen download plan"
    )
    run_download.add_argument("--plan", type=Path, required=True)
    run_download.add_argument("--report", type=Path)
    run_download.add_argument("--execute", action="store_true")
    run_download.set_defaults(func=cmd_run_download)

    tokenize = commands.add_parser(
        "tokenize-plan", help="create deterministic size-balanced tokenizer parts"
    )
    tokenize.add_argument("--data", required=True)
    tokenize.add_argument("--source-manifest", type=Path, required=True)
    tokenize.add_argument("--source-root", type=Path)
    tokenize.add_argument("--output-root", type=Path, required=True)
    tokenize.add_argument("--work-root", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--tokenizer-sha256")
    tokenize.add_argument("--engine", choices=("dolma", "megatron"), required=True)
    tokenize.add_argument("--shards", type=int, required=True)
    tokenize.add_argument("--sequence-length", type=int)
    tokenize.add_argument("--python", default=sys.executable)
    tokenize.add_argument(
        "--preprocess-script",
        type=Path,
        help="MindSpeed-LLM preprocess_data.py; required for engine=megatron",
    )
    tokenize.add_argument("--preprocess-script-sha256")
    tokenize.add_argument("--plan", type=Path, required=True)
    add_patterns(tokenize)
    add_workers(tokenize, 32)
    tokenize.set_defaults(func=cmd_tokenize_plan)

    runtime_manifest = commands.add_parser(
        "runtime-manifest",
        help="convert a tokenized inventory into a Stage-2/3 training manifest",
    )
    runtime_manifest.add_argument(
        "--data",
        choices=(
            "dolmino_100b",
            "longmino_50b",
        ),
        required=True,
    )
    runtime_manifest.add_argument("--inventory", type=Path, required=True)
    runtime_manifest.add_argument(
        "--root",
        type=Path,
        help="training-visible data root; defaults to the inventory root",
    )
    runtime_manifest.add_argument("--manifest", type=Path, required=True)
    runtime_manifest.add_argument("--metadata-suffix", default=".csv.gz")
    runtime_manifest.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="token-array glob; repeat for multiple patterns",
    )
    runtime_manifest.set_defaults(func=cmd_runtime_manifest)

    run_part = commands.add_parser(
        "run-tokenize-part", help="run one frozen tokenizer part"
    )
    run_part.add_argument("--plan", type=Path, required=True)
    run_part.add_argument("--part", type=int, required=True)
    run_part.add_argument("--report", type=Path)
    run_part.add_argument("--execute", action="store_true")
    run_part.set_defaults(func=cmd_run_tokenize_part)

    finalize = commands.add_parser(
        "finalize-stage1-index",
        help="validate mmap pairs and write Megatron data_args_path.txt",
    )
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--data-args-path", type=Path, required=True)
    finalize.add_argument("--report", type=Path)
    finalize.set_defaults(func=cmd_finalize_stage1)

    cache = commands.add_parser(
        "prepare-cache",
        help="plan or build Stage-2 FSL / Stage-3 OBFD packed indices",
    )
    cache.add_argument(
        "--data",
        choices=(
            "dolmino_100b",
            "longmino_50b",
        ),
        required=True,
    )
    cache.add_argument("--source-manifest", type=Path, required=True)
    cache.add_argument("--source-root", type=Path)
    cache.add_argument("--work-root", type=Path, required=True)
    cache.add_argument("--plan", type=Path)
    cache.add_argument("--report", type=Path)
    cache.add_argument("--python", default=sys.executable)
    cache.add_argument("--execute", action="store_true")
    add_patterns(cache)
    add_workers(cache)
    cache.set_defaults(func=cmd_prepare_cache)

    run_cache = commands.add_parser(
        "run-cache", help="run one frozen Stage-2/Stage-3 cache plan"
    )
    run_cache.add_argument("--plan", type=Path, required=True)
    run_cache.add_argument("--report", type=Path)
    run_cache.add_argument("--execute", action="store_true")
    run_cache.set_defaults(func=cmd_run_cache)

    sft = commands.add_parser(
        "sft-plan", help="freeze official Open-Instruct conversion and SFT packing"
    )
    sft.add_argument(
        "--data",
        choices=(
            "dolci_think",
            "dolci_instruct",
        ),
        required=True,
    )
    sft.add_argument("--raw-manifest", type=Path, required=True)
    sft.add_argument(
        "--raw-root",
        type=Path,
        help="execution-visible root overriding the root recorded in raw manifest",
    )
    sft.add_argument("--expected-files", type=int)
    sft.add_argument("--raw-glob", required=True)
    sft.add_argument("--converted-root", type=Path, required=True)
    sft.add_argument("--converted-manifest", type=Path, required=True)
    sft.add_argument("--work-root", type=Path, required=True)
    sft.add_argument("--tokenizer", type=Path, required=True)
    sft.add_argument("--tokenizer-sha256")
    sft.add_argument("--converter", type=Path, required=True)
    sft.add_argument("--converter-sha256")
    sft.add_argument("--open-instruct-root", type=Path, required=True)
    sft.add_argument("--python", default=sys.executable)
    sft.add_argument("--shuffle-seed", type=int, default=42)
    sft.add_argument("--plan", type=Path, required=True)
    add_workers(sft)
    sft.set_defaults(func=cmd_sft_plan)

    run_sft = commands.add_parser(
        "run-sft", help="run a frozen Open-Instruct conversion and SFT packing plan"
    )
    run_sft.add_argument("--plan", type=Path, required=True)
    run_sft.add_argument("--report", type=Path)
    run_sft.add_argument("--execute", action="store_true")
    run_sft.set_defaults(func=cmd_run_sft)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except DataPipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
