#!/usr/bin/env python3
"""Validate and aggregate distributed GSM8K shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .protocol import SCORER_NAME, exact_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else output_dir / "summary.json"
    )
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite {output_json}")

    result_paths = sorted(output_dir.glob("shard-*/result.json"))
    prediction_paths = sorted(output_dir.glob("shard-*/predictions.jsonl"))
    if not result_paths or len(result_paths) != len(prediction_paths):
        raise RuntimeError(
            f"incomplete shard artifacts: {len(result_paths)} results, "
            f"{len(prediction_paths)} prediction files"
        )
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    if any(result.get("status") != "GSM8K_EVAL_OK" for result in results):
        raise RuntimeError("one or more shard results are not GSM8K_EVAL_OK")
    if any(result.get("artifact_mutated") is not False for result in results):
        raise RuntimeError("one or more shard results report artifact mutation")

    contract_fields = (
        "task",
        "model_id",
        "revision",
        "resolved_model_revision",
        "dataset_source",
        "dataset_sample_count",
        "num_fewshot",
        "batch_size",
        "num_samples_per_question",
        "temperature",
        "top_p",
        "max_new_tokens",
        "max_sequence_length",
        "stop_strings",
        "seed",
        "scorer",
        "prompt_corpus_sha256",
        "evaluation_corpus_sha256",
        "world_size",
    )
    for field in contract_fields:
        values = {json.dumps(result.get(field), sort_keys=True) for result in results}
        if len(values) != 1:
            raise RuntimeError(f"shards disagree on {field}: {sorted(values)}")

    rows: list[dict[str, Any]] = []
    for path in prediction_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from error
            expected_correct = exact_match(row["output"], row["gold"])
            if row.get("correct") != expected_correct:
                raise RuntimeError(
                    f"stored scorer result drift at {path}:{line_number}"
                )
            if row.get("scorer") != SCORER_NAME:
                raise RuntimeError(f"wrong scorer at {path}:{line_number}")
            rows.append(row)

    keys = [(int(row["doc_index"]), int(row["sample_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate (doc_index, sample_index) prediction keys")
    dataset_sample_count = int(results[0]["dataset_sample_count"])
    num_samples = int(results[0]["num_samples_per_question"])
    expected_keys = {
        (doc_index, sample_index)
        for doc_index in range(dataset_sample_count)
        for sample_index in range(num_samples)
    }
    missing = expected_keys - set(keys)
    extra = set(keys) - expected_keys
    if missing or extra:
        raise RuntimeError(
            f"prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    correct = sum(bool(row["correct"]) for row in rows)
    summary = {
        "status": "GSM8K_AGGREGATE_OK",
        "model_id": results[0]["model_id"],
        "revision": results[0]["revision"],
        "resolved_model_revision": results[0].get("resolved_model_revision"),
        "dataset_source": results[0]["dataset_source"],
        "question_count": dataset_sample_count,
        "num_samples_per_question": num_samples,
        "prediction_count": len(rows),
        "pass_at_1_correct": correct,
        "pass_at_1": correct / len(rows),
        "scorer": SCORER_NAME,
        "prompt_corpus_sha256": results[0]["prompt_corpus_sha256"],
        "evaluation_corpus_sha256": results[0]["evaluation_corpus_sha256"],
        "shard_count": len(results),
        "artifact_mutated": False,
        "result_files": [str(path) for path in result_paths],
        "prediction_files": [str(path) for path in prediction_paths],
    }
    write_json(output_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
