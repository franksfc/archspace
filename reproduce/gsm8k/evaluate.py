#!/usr/bin/env python3
"""Evaluate an NCP or OLMo checkpoint on fixed-protocol GSM8K via HF generate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import (
    DATASET_ID,
    DATASET_NAME,
    DATASET_REVISION,
    EXPECTED_EVALUATION_CORPUS_SHA256,
    EXPECTED_PROMPT_CORPUS_SHA256,
    EXPECTED_TEST_SIZE,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    NUM_FEWSHOT,
    NUM_SAMPLES,
    SCORER_NAME,
    SEED,
    STOP_STRINGS,
    TASK_NAME,
    TEMPERATURE,
    TOP_P,
    build_prompt,
    evaluation_corpus_sha256,
    exact_match,
    extract_olmes_answer,
    prompt_corpus_sha256,
    truncate_at_stop,
)

DEFAULT_MODEL_ID = "ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast"


def configure_rank_local_hf_modules_cache(output_dir: Path) -> str:
    """Prevent concurrent ranks from racing while materializing remote code."""

    configured = os.environ.get("HF_MODULES_CACHE")
    if configured:
        return str(Path(configured).expanduser().resolve())
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    cache_path = (
        output_dir.expanduser().resolve()
        / "cache"
        / "hf-modules"
        / f"local-rank-{local_rank:02d}"
    )
    os.environ["HF_MODULES_CACHE"] = str(cache_path)
    return str(cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_model_path(
    model_id: str,
    revision: str,
    cache_dir: str,
) -> Path:
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir or None,
        )
    ).resolve()


def resolved_snapshot_revision(model_path: Path) -> str | None:
    """Return a Hub snapshot commit when the resolved path exposes one."""

    return model_path.name if model_path.parent.name == "snapshots" else None


def load_dataset_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    if args.dataset_file is not None:
        dataset_file = args.dataset_file.expanduser().resolve()
        if not dataset_file.is_file():
            raise FileNotFoundError(dataset_file)
        suffix_to_builder = {".json": "json", ".jsonl": "json", ".parquet": "parquet"}
        try:
            builder = suffix_to_builder[dataset_file.suffix.lower()]
        except KeyError as error:
            raise ValueError(
                f"unsupported dataset file suffix: {dataset_file.suffix}"
            ) from error
        dataset = load_dataset(
            builder,
            data_files={args.split: str(dataset_file)},
            split=args.split,
        )
        source = str(dataset_file)
    else:
        dataset = load_dataset(
            args.dataset_id,
            args.dataset_name,
            split=args.split,
            revision=args.dataset_revision,
        )
        source = f"{args.dataset_id}@{args.dataset_revision}/{args.dataset_name}/{args.split}"

    rows = [dict(row) for row in dataset]
    return rows, source


def artifact_fingerprint(model_path: Path) -> str:
    """Hash metadata and weight-file stats without reading all model weights."""

    digest = hashlib.sha256()
    tracked = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "modeling_ncp_olmo3.py",
        "configuration_ncp_olmo3.py",
        "standalone_manifest.json",
    )
    for name in tracked:
        path = model_path / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    for path in sorted(model_path.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(stat.st_size.to_bytes(8, "little"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "little"))
    return digest.hexdigest()


def load_model_and_tokenizer(
    args: argparse.Namespace,
    model_path: Path,
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_type = str(getattr(config, "model_type", ""))
    is_standalone_ncp = model_type == "ncp_olmo3"
    common_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    if model_type == "conceptlm_v22_vq":
        raise ValueError(
            "legacy Megatron-backed NCP artifacts are unsupported by this "
            "standalone HF evaluator; use the pure-HF NCP model revision"
        )
    if is_standalone_ncp:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **common_kwargs,
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
        forbidden_runtime_modules = sorted(
            name
            for name in sys.modules
            if name == "megatron"
            or name.startswith("megatron.")
            or name == "vllm"
            or name.startswith("vllm.")
        )
        if forbidden_runtime_modules:
            raise RuntimeError(
                "standalone NCP unexpectedly imported external runtime modules: "
                + ", ".join(forbidden_runtime_modules[:20])
            )
        loader_contract = {
            "backend": "hf_auto_model_from_pretrained",
            "standalone_hf_backend": True,
            "trust_remote_code": True,
            "external_runtime_required": False,
            "attention_implementation": "torch_sdpa",
            "flash_decode": False,
            "device_map": {"": str(device)},
        }
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **common_kwargs,
            attn_implementation="sdpa",
        ).to(device)
        loader_contract = {
            "backend": "hf_auto_model_from_pretrained",
            "trust_remote_code": True,
            "attention_implementation": "sdpa",
            "flash_decode": False,
        }
    model.eval()
    return model, tokenizer, loader_contract


def main() -> None:
    args = parse_args()
    if args.num_samples != NUM_SAMPLES:
        raise ValueError(f"standard GSM8K requires --num-samples {NUM_SAMPLES}")
    if args.seed != SEED:
        raise ValueError(f"standard GSM8K requires --seed {SEED}")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_sequence_length != MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"standard GSM8K requires max sequence length {MAX_SEQUENCE_LENGTH}"
        )
    if args.dataset_file is None and (
        args.dataset_id != DATASET_ID
        or args.dataset_name != DATASET_NAME
        or args.dataset_revision != DATASET_REVISION
        or args.split != "test"
    ):
        raise ValueError(
            "standard GSM8K requires the pinned openai/gsm8k main test split"
        )

    hf_modules_cache = configure_rank_local_hf_modules_cache(args.output_dir)

    import torch
    from transformers import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("GSM8K generation requires a CUDA GPU")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rows, dataset_source = load_dataset_rows(args)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("GSM8K dataset is empty")
    prompts = [build_prompt(str(row["question"])) for row in rows]
    prompt_hash = prompt_corpus_sha256(prompts)
    evaluation_corpus_hash = evaluation_corpus_sha256(rows)
    if len(rows) == EXPECTED_TEST_SIZE:
        if prompt_hash != EXPECTED_PROMPT_CORPUS_SHA256:
            raise RuntimeError(
                "standard GSM8K prompt corpus changed: "
                f"{prompt_hash} != {EXPECTED_PROMPT_CORPUS_SHA256}"
            )
        if evaluation_corpus_hash != EXPECTED_EVALUATION_CORPUS_SHA256:
            raise RuntimeError(
                "standard GSM8K question/answer corpus changed: "
                f"{evaluation_corpus_hash} != {EXPECTED_EVALUATION_CORPUS_SHA256}"
            )

    shard_dir = args.output_dir.expanduser().resolve() / f"shard-{rank:02d}"
    predictions_path = shard_dir / "predictions.jsonl"
    result_path = shard_dir / "result.json"
    if predictions_path.exists() or result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing shard output: {shard_dir}"
        )
    shard_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_id, args.revision, args.cache_dir)
    artifact_before = artifact_fingerprint(model_path)
    model_load_started = time.perf_counter()
    model, tokenizer, loader_contract = load_model_and_tokenizer(
        args,
        model_path,
        device,
    )
    loader_contract["hf_modules_cache"] = hf_modules_cache
    model_load_seconds = time.perf_counter() - model_load_started
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    scheduled = list(range(len(rows)))[rank::world_size]
    completed = 0
    correct = 0
    returned_tokens = 0
    evaluation_started = time.perf_counter()
    with predictions_path.open("x", encoding="utf-8") as prediction_file:
        for doc_index in scheduled:
            prompt = prompts[doc_index]
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            if prompt_tokens + MAX_NEW_TOKENS > args.max_sequence_length:
                raise ValueError(
                    f"doc {doc_index} exceeds context contract: "
                    f"{prompt_tokens}+{MAX_NEW_TOKENS}>{args.max_sequence_length}"
                )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            for sample_index in range(args.num_samples):
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                        stop_strings=list(STOP_STRINGS),
                        tokenizer=tokenizer,
                    )
                generated_ids = generated[0, prompt_tokens:]
                output = truncate_at_stop(
                    tokenizer.decode(generated_ids, skip_special_tokens=True)
                )
                gold = str(rows[doc_index]["answer"])
                is_correct = exact_match(output, gold)
                prediction_file.write(
                    json.dumps(
                        {
                            "doc_index": doc_index,
                            "sample_index": sample_index,
                            "question": rows[doc_index]["question"],
                            "gold": extract_olmes_answer(gold),
                            "output": output,
                            "prediction": extract_olmes_answer(output),
                            "correct": is_correct,
                            "prompt_token_count": prompt_tokens,
                            "returned_token_count": int(generated_ids.numel()),
                            "scorer": SCORER_NAME,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                prediction_file.flush()
                completed += 1
                correct += int(is_correct)
                returned_tokens += int(generated_ids.numel())

    torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    artifact_after = artifact_fingerprint(model_path)
    result = {
        "status": "GSM8K_EVAL_OK",
        "task": TASK_NAME,
        "model_id": args.model_id,
        "revision": args.revision,
        "resolved_model_revision": resolved_snapshot_revision(model_path),
        "resolved_model_path": str(model_path),
        "loader_contract": loader_contract,
        "dataset_source": dataset_source,
        "dataset_sample_count": len(rows),
        "num_fewshot": NUM_FEWSHOT,
        "batch_size": 1,
        "num_samples_per_question": args.num_samples,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "stop_strings": list(STOP_STRINGS),
        "seed": args.seed,
        "seed_rule": "transformers.set_seed(seed) once per rank before evaluation",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "schedule": "doc_indices[rank::world_size]",
        "scheduled_question_count": len(scheduled),
        "sample_count": completed,
        "pass_at_1_correct": correct,
        "pass_at_1": correct / completed if completed else 0.0,
        "scorer": SCORER_NAME,
        "prompt_corpus_sha256": prompt_hash,
        "evaluation_corpus_sha256": evaluation_corpus_hash,
        "returned_token_count": returned_tokens,
        "model_load_seconds": model_load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "artifact_fingerprint_before": artifact_before,
        "artifact_fingerprint_after": artifact_after,
        "artifact_mutated": artifact_before != artifact_after,
        "predictions_jsonl": str(predictions_path),
        "completed_at": utc_now(),
    }
    if result["artifact_mutated"]:
        raise RuntimeError("resolved model artifact changed during evaluation")
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
