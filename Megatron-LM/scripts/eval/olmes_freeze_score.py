#!/usr/bin/env python3
"""Freeze and objectively score the pinned OLMES downstream suite.

This module deliberately separates prompt construction from model execution:

* ``freeze`` resolves the pinned OLMES task aliases, downloads/opens the exact
  datasets, builds OLMES ``RequestInstance`` objects, applies the official
  OLMo 3 ChatML template, and writes immutable JSONL request records.
* ``score`` joins native-runner response JSONL files back to those records and
  invokes the metric implementations owned by the same pinned OLMES checkout.

AlpacaEval 2 requests are frozen for generation, but are never scored here.
The scorer also refuses every deferred/model/LLM-judge metric.  This makes the
"no judge scoring" contract an executable guard rather than a convention.

The script has no checkpoint-specific constant.  A frozen request directory
can therefore be reused for step 60000 or any other checkpoint as long as the
model runner consumes the same request contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLMES_ROOT = REPO_ROOT / "third_party" / "olmes"
DEFAULT_OLMES_STUBS = REPO_ROOT / "scripts" / "eval" / "stubs"
DEFAULT_NLTK_DATA = REPO_ROOT / "eval_artifacts" / "nltk_data"
PINNED_OLMES_COMMIT = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
SCHEMA_VERSION = "olmes-native-requests-v1"

# These are the exact pinned OLMES aliases for the objective downstream
# evaluation protocol. Suite aliases are recursively
# expanded by OLMES (BBH=27, MATH=7, MMLU=57 concrete tasks).
DEFAULT_TASK_ALIASES: tuple[str, ...] = (
    "alpaca_eval_v2::tulu",
    "bbh:cot-v1::tulu",
    "drop::llama3",
    "gsm8k::tulu",
    "ifeval::tulu",
    "minerva_math::tulu",
    "mmlu:mc::tulu",
    "popqa::tulu",
    "truthfulqa::tulu",
)

NO_SCORE_SUITE_ALIASES = frozenset({"alpaca_eval_v2::tulu"})
NO_SCORE_TASK_NAMES = frozenset({"alpaca_eval"})

# OLMES' recipes intentionally leave most dataset revisions unset.  That is
# convenient for normal development but not acceptable for a checkpoint
# comparison: datasets.load_dataset() could otherwise select a newer snapshot.
# These revisions are the locally materialized Hub snapshots audited for this
# evaluation run.  They are injected into every concrete task *before* request
# construction and become part of the official OLMES task hash.
PINNED_DATASET_REVISIONS: dict[str, str] = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/hellaswag": "55aa56627db098098f50549e650a00be2684f4bf",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "lukaemon/bbh": "982bb89fd79532a8ac676a61fc42eb1aeec63f99",
    "EleutherAI/drop": "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "google-research-datasets/nq_open": "5dd9790a83002ad084ddeb7c420dc716852c6f28",
    "mandarjoshi/trivia_qa": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "tatsu-lab/alpaca_eval": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
    "HuggingFaceH4/ifeval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "akariasai/PopQA": "098765c79ea10a2cb19c828324e33281b8336ec0",
    "truthfulqa/truthful_qa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
}

# AGIEval is vendored by the pinned OLMES checkout instead of loaded from the
# Hugging Face Hub.  Freeze it under a deterministic content-tree hash so it
# receives the same immutability check as Hub-backed datasets.
PINNED_LOCAL_DATASET_TREES: dict[str, str] = {
    "AGIEval/data/v1": "2825606a70fd1b8cf3412270ab0e5e2d44d89f6ff0b28e1a6469994263716452",
}

# Official file at allenai/Olmo-3-1025-7B revision
# 280b8b16ebb71c802aa12c5cd0d3ef431092ceac.  Rendering is implemented
# directly below so it does not depend on a mutable local tokenizer config.
OLMO3_CHAT_TEMPLATE_REVISION = "280b8b16ebb71c802aa12c5cd0d3ef431092ceac"
OLMO3_CHAT_TEMPLATE_ID = (
    "allenai/Olmo-3-1025-7B@"
    + OLMO3_CHAT_TEMPLATE_REVISION
    + ":chat_template.jinja"
)
OLMO3_CHAT_TEMPLATE_SOURCE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)
OLMO3_CHAT_TEMPLATE_SHA256 = hashlib.sha256(
    OLMO3_CHAT_TEMPLATE_SOURCE.encode("utf-8")
).hexdigest()

LOG = logging.getLogger("olmes_freeze_score")


class ContractError(RuntimeError):
    """Raised when frozen requests or model responses violate the contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bootstrap_olmes(olmes_root: Path, stubs_root: Path) -> None:
    """Put the selected checkout and import-only optional-dependency stubs first."""
    for path in (olmes_root, stubs_root):
        if not path.exists():
            raise ContractError(f"Required OLMES path does not exist: {path}")
    # Insert in reverse because sys.path.insert(0, ...) prepends each entry.
    for path in (olmes_root, stubs_root):
        value = str(path.resolve())
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _configure_huggingface(args: argparse.Namespace) -> None:
    if getattr(args, "hf_home", None):
        hf_home = Path(args.hf_home).resolve()
        os.environ["HF_HOME"] = str(hf_home)
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    if getattr(args, "offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        ):
            os.environ.pop(name, None)
    if getattr(args, "nltk_data", None):
        os.environ["NLTK_DATA"] = str(Path(args.nltk_data).resolve())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(payload: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, _json_dumps(payload, pretty=True) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(_json_dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ContractError(f"Expected JSON object at {path}:{line_number}")
            yield row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _git_info(path: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *command],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    return {
        "root": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _require_pinned_olmes(path: Path) -> dict[str, Any]:
    git_info = _git_info(path)
    if git_info["commit"] != PINNED_OLMES_COMMIT:
        raise ContractError(
            "OLMES checkout commit differs from the repository lock: "
            f"expected={PINNED_OLMES_COMMIT}, "
            f"actual={git_info['commit']}, root={path}"
        )
    return git_info


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dataset_snapshot_path(hf_home: Path, dataset_path: str, revision: str) -> Path:
    repository_cache_name = "datasets--" + dataset_path.replace("/", "--")
    return hf_home / "hub" / repository_cache_name / "snapshots" / revision


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(f"{relative}\0{_sha256_file(file_path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _pin_and_verify_task_dataset(
    task: Any,
    hf_home: Path,
    *,
    offline: bool,
) -> dict[str, Any]:
    """Resolve an exact Hub commit and recompute OLMES' official task hash."""
    from oe_eval.default_configs import TASK_DEFAULTS
    from oe_eval.utils import hash_dict

    dataset_path = task.task_config.get("dataset_path")
    local_path = Path(dataset_path).resolve() if isinstance(dataset_path, str) else None
    if local_path is not None:
        for suffix, expected_tree_hash in PINNED_LOCAL_DATASET_TREES.items():
            if local_path.as_posix().endswith(suffix):
                if not local_path.is_dir():
                    raise ContractError(f"Pinned local dataset directory is missing: {local_path}")
                actual_tree_hash = _sha256_tree(local_path)
                if actual_tree_hash != expected_tree_hash:
                    raise ContractError(
                        f"Pinned local dataset tree mismatch for {local_path}: "
                        f"expected {expected_tree_hash}, got {actual_tree_hash}"
                    )
                revision = f"local-tree-sha256:{actual_tree_hash}"
                task.task_config["revision"] = revision
                task._task_hash = hash_dict(task.task_config, TASK_DEFAULTS)
                return {
                    "dataset_path": str(local_path),
                    "dataset_name": task.task_config.get("dataset_name"),
                    "revision": revision,
                    "revision_source": "vendored-tree-lock",
                    "snapshot_path": str(local_path),
                    "snapshot_present": True,
                }
    configured_revision = task.task_config.get("revision")
    if dataset_path in PINNED_DATASET_REVISIONS:
        revision = PINNED_DATASET_REVISIONS[dataset_path]
        revision_source = "built-in-audited-lock"
        if configured_revision not in (None, revision):
            raise ContractError(
                f"Task {task.task_name} config revision {configured_revision} "
                f"conflicts with audited {dataset_path} revision {revision}"
            )
    else:
        if offline:
            raise ContractError(
                f"Task {task.task_name} uses dataset_path {dataset_path!r}, "
                "which has no built-in revision. Run the initial freeze with "
                "--no-offline so the Hub ref is resolved to an immutable commit."
            )
        from huggingface_hub import HfApi

        requested_revision = configured_revision or "main"
        info = HfApi().dataset_info(
            repo_id=dataset_path,
            revision=requested_revision,
        )
        revision = str(info.sha)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ContractError(
                f"Hub returned an invalid commit for {dataset_path!r}: "
                f"{revision!r}"
            )
        revision_source = (
            f"hub-resolved-at-freeze:{requested_revision}"
        )

    snapshot_path = _dataset_snapshot_path(hf_home, dataset_path, revision)
    if not snapshot_path.is_dir():
        if offline:
            raise ContractError(
                f"Pinned dataset snapshot is not materialized: "
                f"{snapshot_path}. Run the initial freeze with --no-offline."
            )
        from huggingface_hub import snapshot_download

        downloaded = Path(
            snapshot_download(
                repo_id=dataset_path,
                repo_type="dataset",
                revision=revision,
                cache_dir=hf_home / "hub",
                local_files_only=False,
            )
        ).resolve()
        if downloaded != snapshot_path.resolve():
            raise ContractError(
                "Hugging Face snapshot path differs from the exact revision "
                f"contract: expected={snapshot_path.resolve()}, "
                f"actual={downloaded}"
            )
    if not snapshot_path.is_dir():
        raise ContractError(
            f"Pinned dataset snapshot is unavailable after download: "
            f"{snapshot_path}"
        )
    task.task_config["revision"] = revision
    task._task_hash = hash_dict(task.task_config, TASK_DEFAULTS)
    return {
        "dataset_path": dataset_path,
        "dataset_name": task.task_config.get("dataset_name"),
        "revision": revision,
        "revision_source": revision_source,
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_present": True,
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:180] or "task"


def _render_olmo3_chat_context(context: Any) -> tuple[str, bool]:
    """Match OLMES ``convert_chat_instance`` using official OLMo 3 ChatML."""
    if isinstance(context, str):
        return context, False
    if not isinstance(context, dict):
        raise ContractError(
            "Chat request context must be a string or an OLMES messages dictionary; "
            f"got {type(context).__name__}"
        )
    messages = context.get("messages")
    if not isinstance(messages, list):
        raise ContractError("Chat request context is missing a list-valued 'messages' field")
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ContractError(f"Chat message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ContractError(f"Chat message {index} role/content must both be strings")
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n\n")
    rendered.append("<|im_start|>assistant\n")
    assistant_prefix = context.get("assistant_prefix", "")
    if assistant_prefix is None:
        assistant_prefix = ""
    if not isinstance(assistant_prefix, str):
        raise ContractError("assistant_prefix must be a string or null")
    rendered.append(assistant_prefix)
    return "".join(rendered), True


def _resolve_aliases(
    requested_aliases: Sequence[str],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(suite_alias, concrete_alias, task_config)`` in OLMES order."""
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS

    resolved: list[tuple[str, str, dict[str, Any]]] = []

    def expand(root_alias: str, alias: str, stack: tuple[str, ...]) -> None:
        if alias in stack:
            raise ContractError(f"Recursive OLMES task suite: {' -> '.join((*stack, alias))}")
        if alias in TASK_SUITE_CONFIGS:
            children = TASK_SUITE_CONFIGS[alias]["tasks"]
            # Several official OLMo 3 suites are declared as sets. Preserve
            # official list order, but canonicalize set order so request IDs
            # do not depend on PYTHONHASHSEED.
            if isinstance(children, (set, frozenset)):
                children = sorted(children)
            for child in children:
                expand(root_alias, child, (*stack, alias))
            return
        if alias not in TASK_CONFIGS:
            raise ContractError(f"No pinned OLMES task config found for alias: {alias}")
        config = copy.deepcopy(TASK_CONFIGS[alias])
        config.setdefault("metadata", {})["alias"] = alias
        resolved.append((root_alias, alias, config))

    for requested in requested_aliases:
        expand(requested, requested, ())

    concrete_aliases = [item[1] for item in resolved]
    duplicates = sorted(alias for alias, count in Counter(concrete_aliases).items() if count > 1)
    if duplicates:
        raise ContractError(
            "Requested task aliases expand to duplicate concrete tasks: " + ", ".join(duplicates)
        )
    return resolved


def _stable_request_id(
    task_index: int,
    task_alias: str,
    task_name: str,
    request_index_in_task: int,
    doc_id: int,
    idx: int,
) -> str:
    identity = {
        "task_index": task_index,
        "task_alias": task_alias,
        "task_name": task_name,
        "request_index_in_task": request_index_in_task,
        "doc_id": doc_id,
        "idx": idx,
    }
    return f"r{task_index:03d}-{request_index_in_task:08d}-{_sha256_json(identity)[:16]}"


def _freeze_instance(
    instance: Any,
    *,
    suite_alias: str,
    task_alias: str,
    task_index: int,
    request_index: int,
    request_index_in_task: int,
    task_hash: str,
    task_metadata: Mapping[str, Any],
    olmes_commit: str,
) -> dict[str, Any]:
    original_request = copy.deepcopy(instance.request.__dict__)
    rendered_request = copy.deepcopy(original_request)
    rendered_context, chat_applied = _render_olmo3_chat_context(rendered_request["context"])
    rendered_request["context"] = rendered_context
    if "perplexity_context" in rendered_request:
        perplexity_context, _ = _render_olmo3_chat_context(
            rendered_request["perplexity_context"]
        )
        rendered_request["perplexity_context"] = perplexity_context

    # OLMES currently resolves this to False through its defaults, but make the
    # AE2 runtime contract explicit in every frozen row.  Truncating an 8192-
    # token generation request to max_length-max_gen_toks would otherwise leave
    # a zero-token context.  The native runner must instead keep the short prompt
    # and cap effective new tokens at (model_max_length - prompt_length).
    if suite_alias == "alpaca_eval_v2::tulu":
        generation_kwargs = rendered_request.get("generation_kwargs")
        if not isinstance(generation_kwargs, dict):
            raise ContractError("AlpacaEval 2 request has no generation_kwargs object")
        generation_kwargs["truncate_context"] = False

    # Match oe_eval.run_eval.convert_chat_instance exactly: a chat continuation
    # loses leading whitespace only when no assistant prefix was supplied.
    if chat_applied and "continuation" in rendered_request:
        original_context = original_request["context"]
        assistant_prefix = original_context.get("assistant_prefix", "")
        if not assistant_prefix:
            rendered_request["continuation"] = rendered_request["continuation"].lstrip()

    request_id = _stable_request_id(
        task_index,
        task_alias,
        instance.task_name,
        request_index_in_task,
        instance.doc_id,
        instance.idx,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "suite_alias": suite_alias,
        "task_alias": task_alias,
        "task_name": instance.task_name,
        "task_index": task_index,
        "request_index": request_index,
        "request_index_in_task": request_index_in_task,
        "request_type": instance.request_type,
        "doc_id": instance.doc_id,
        "idx": instance.idx,
        "native_id": instance.native_id,
        "native_id_description": getattr(instance, "native_id_description", None),
        "label": instance.label,
        "doc": instance.doc,
        "request": rendered_request,
        "original_request": original_request if chat_applied else None,
        "chat_format": {
            "applied": chat_applied,
            "template_id": OLMO3_CHAT_TEMPLATE_ID if chat_applied else None,
            "template_sha256": OLMO3_CHAT_TEMPLATE_SHA256 if chat_applied else None,
        },
        "task_metadata": {
            "suite_alias": suite_alias,
            "task_alias": task_alias,
            "task_hash": task_hash,
            "olmes_commit": olmes_commit,
            "metadata": task_metadata,
        },
    }


def _validate_frozen_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "request_id",
        "suite_alias",
        "task_alias",
        "task_name",
        "task_index",
        "request_index",
        "request_type",
        "doc_id",
        "idx",
        "doc",
        "request",
        "chat_format",
        "task_metadata",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ContractError(f"Frozen request is missing fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError(
            f"Unsupported frozen request schema {record['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    request_type = record["request_type"]
    request = record["request"]
    if not isinstance(request, dict) or not isinstance(request.get("context"), str):
        raise ContractError(f"{record['request_id']}: request.context must be rendered text")
    if request_type == "loglikelihood" and "continuation" not in request:
        raise ContractError(f"{record['request_id']}: loglikelihood continuation is missing")
    if request_type in {"generate_until", "generate_until_and_loglikelihood"}:
        if not isinstance(request.get("stop_sequences"), list):
            raise ContractError(f"{record['request_id']}: stop_sequences must be a list")
        if not isinstance(request.get("generation_kwargs"), dict):
            raise ContractError(f"{record['request_id']}: generation_kwargs must be an object")
    if request_type == "generate_until_and_loglikelihood":
        if not isinstance(request.get("continuation"), str):
            raise ContractError(
                f"{record['request_id']}: combined continuation must be text"
            )
        if not isinstance(request.get("perplexity_context"), str):
            raise ContractError(
                f"{record['request_id']}: combined perplexity_context must be rendered text"
            )


def _write_suite_request_files(
    output_dir: Path,
    combined_path: Path,
    requested_aliases: Sequence[str],
) -> list[dict[str, Any]]:
    """Split the combined stream into one atomic, hash-addressed file per suite."""
    suites_dir = output_dir / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for suite_index, suite_alias in enumerate(requested_aliases):
        path = suites_dir / f"{suite_index:02d}-{_slug(suite_alias)}.jsonl"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        outputs[suite_alias] = {
            "suite_index": suite_index,
            "suite_alias": suite_alias,
            "path": path,
            "temporary": temporary,
            "handle": temporary.open("w", encoding="utf-8"),
            "num_requests": 0,
            "request_type_counts": Counter(),
            "task_aliases": set(),
        }

    succeeded = False
    try:
        with combined_path.open("r", encoding="utf-8") as combined_handle:
            for line_number, line in enumerate(combined_handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ContractError(
                        f"Invalid freshly frozen JSON at {combined_path}:{line_number}: {error}"
                    ) from error
                suite_alias = record.get("suite_alias")
                if suite_alias not in outputs:
                    raise ContractError(
                        f"Frozen request has unrequested suite_alias {suite_alias!r}"
                    )
                output = outputs[suite_alias]
                output["handle"].write(line)
                output["num_requests"] += 1
                output["request_type_counts"][record["request_type"]] += 1
                output["task_aliases"].add(record["task_alias"])
        for output in outputs.values():
            if output["num_requests"] == 0:
                raise ContractError(f"Suite {output['suite_alias']} produced no requests")
            output["handle"].flush()
            os.fsync(output["handle"].fileno())
        succeeded = True
    finally:
        for output in outputs.values():
            output["handle"].close()

    if not succeeded:
        raise ContractError("Failed while constructing per-suite request files")

    manifests: list[dict[str, Any]] = []
    for suite_alias in requested_aliases:
        output = outputs[suite_alias]
        os.replace(output["temporary"], output["path"])
        manifests.append(
            {
                "suite_index": output["suite_index"],
                "suite_alias": suite_alias,
                "request_file": output["path"].relative_to(output_dir).as_posix(),
                "request_file_sha256": _sha256_file(output["path"]),
                "num_requests": output["num_requests"],
                "num_concrete_tasks": len(output["task_aliases"]),
                "task_aliases": sorted(output["task_aliases"]),
                "request_type_counts": dict(sorted(output["request_type_counts"].items())),
            }
        )
    return manifests


def _freeze(args: argparse.Namespace) -> int:
    _configure_huggingface(args)
    _bootstrap_olmes(args.olmes_root, args.olmes_stubs)

    from oe_eval.run_eval import load_task

    git_info = _require_pinned_olmes(args.olmes_root)
    if git_info["dirty"] and not args.allow_dirty_olmes:
        raise ContractError(
            f"Pinned OLMES checkout is dirty: {args.olmes_root}. "
            "Use --allow-dirty-olmes only for an explicitly non-reproducible run."
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ContractError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = output_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_aliases(args.task)
    LOG.info(
        "Resolved %d requested aliases into %d concrete OLMES tasks",
        len(args.task),
        len(resolved),
    )

    combined_path = output_dir / "requests.jsonl"
    temporary_combined = combined_path.with_name(f".{combined_path.name}.tmp-{os.getpid()}")
    task_manifests: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    verified_datasets: dict[str, dict[str, Any]] = {}
    all_request_ids: set[str] = set()
    global_request_index = 0

    with temporary_combined.open("w", encoding="utf-8") as combined_handle:
        for task_index, (suite_alias, task_alias, task_config) in enumerate(resolved):
            if args.limit_per_concrete_task is not None:
                task_config["limit"] = args.limit_per_concrete_task
            task = load_task(task_config, str(output_dir))
            dataset_source = _pin_and_verify_task_dataset(
                task,
                Path(args.hf_home).resolve(),
                offline=bool(args.offline),
            )
            verified_datasets[dataset_source["dataset_path"]] = {
                key: value for key, value in dataset_source.items() if key != "dataset_name"
            }
            LOG.info(
                "[%d/%d] building %s (%s) at dataset revision %s",
                task_index + 1,
                len(resolved),
                task_alias,
                task.task_name,
                dataset_source["revision"],
            )
            task.download()
            task.build_all_requests()
            instances = task._instances or []
            if not instances:
                raise ContractError(f"OLMES built no requests for {task_alias}")

            request_type_counts: Counter[str] = Counter()
            doc_ids: set[int] = set()
            task_path = requests_dir / f"{task_index:03d}-{_slug(task_alias)}.jsonl"
            temporary_task = task_path.with_name(f".{task_path.name}.tmp-{os.getpid()}")
            with temporary_task.open("w", encoding="utf-8") as task_handle:
                for index_in_task, instance in enumerate(instances):
                    record = _freeze_instance(
                        instance,
                        suite_alias=suite_alias,
                        task_alias=task_alias,
                        task_index=task_index,
                        request_index=global_request_index,
                        request_index_in_task=index_in_task,
                        task_hash=task._task_hash["hash"],
                        task_metadata=task.task_config.get("metadata", {}),
                        olmes_commit=git_info["commit"],
                    )
                    _validate_frozen_record(record)
                    request_id = record["request_id"]
                    if request_id in all_request_ids:
                        raise ContractError(f"Duplicate generated request_id: {request_id}")
                    all_request_ids.add(request_id)
                    line = _json_dumps(record) + "\n"
                    task_handle.write(line)
                    combined_handle.write(line)
                    request_type_counts[record["request_type"]] += 1
                    doc_ids.add(record["doc_id"])
                    global_request_index += 1
                task_handle.flush()
                os.fsync(task_handle.fileno())
            os.replace(temporary_task, task_path)

            relative_task_path = task_path.relative_to(output_dir).as_posix()
            task_manifest = {
                "task_index": task_index,
                "suite_alias": suite_alias,
                "task_alias": task_alias,
                "task_name": task.task_name,
                "task_hash": task._task_hash["hash"],
                "task_config": task.task_config,
                "dataset_source": dataset_source,
                "request_file": relative_task_path,
                "request_file_sha256": _sha256_file(task_path),
                "num_requests": len(instances),
                "num_docs": len(doc_ids),
                "request_type_counts": dict(sorted(request_type_counts.items())),
                "chat_format_applied": bool(task.task_config.get("use_chat_format")),
                "score_policy": (
                    "generate_only_no_judge"
                    if suite_alias in NO_SCORE_SUITE_ALIASES
                    or task.task_name in NO_SCORE_TASK_NAMES
                    else "official_objective_metric"
                ),
            }
            task_manifests.append(task_manifest)
            task_rows.append(copy.deepcopy(task_manifest))
            LOG.info(
                "[%d/%d] froze %d requests / %d docs",
                task_index + 1,
                len(resolved),
                len(instances),
                len(doc_ids),
            )
        combined_handle.flush()
        os.fsync(combined_handle.fileno())
    os.replace(temporary_combined, combined_path)

    _atomic_write_jsonl(output_dir / "tasks.jsonl", task_rows)
    suite_manifests = _write_suite_request_files(output_dir, combined_path, args.task)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "requested_task_aliases": list(args.task),
        "olmes": git_info,
        "chat_template": {
            "id": OLMO3_CHAT_TEMPLATE_ID,
            "source": OLMO3_CHAT_TEMPLATE_SOURCE,
            "sha256": OLMO3_CHAT_TEMPLATE_SHA256,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "datasets": _package_version("datasets"),
            "lm_eval": _package_version("lm-eval"),
            "transformers": _package_version("transformers"),
            "hf_home": os.environ.get("HF_HOME"),
            "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
            "nltk_data": os.environ.get("NLTK_DATA"),
            "offline": bool(args.offline),
        },
        "limits": {"per_concrete_task": args.limit_per_concrete_task},
        "pinned_dataset_revisions": dict(sorted(PINNED_DATASET_REVISIONS.items())),
        "pinned_local_dataset_trees": dict(sorted(PINNED_LOCAL_DATASET_TREES.items())),
        "verified_dataset_snapshots": dict(sorted(verified_datasets.items())),
        "num_concrete_tasks": len(task_manifests),
        "num_requests": global_request_index,
        "requests_file": combined_path.name,
        "requests_file_sha256": _sha256_file(combined_path),
        "tasks_file": "tasks.jsonl",
        "tasks_file_sha256": _sha256_file(output_dir / "tasks.jsonl"),
        "suites": suite_manifests,
        "tasks": task_manifests,
        "response_contract": {
            "join_key": "request_id",
            "accepted_payload_fields": ["model_resps", "response"],
            "generate_until": {
                "required": {"continuation": "string"},
                "recommended": {"num_tokens": "integer", "sum_logits": "number"},
            },
            "loglikelihood": {
                "required": {"sum_logits": "finite number", "num_tokens": "integer"},
                "recommended": {"is_greedy": "boolean", "num_tokens_all": "integer"},
            },
            "runtime_stop_token_ids": [100257, 100265],
            "note": (
                "stop_sequences and max_gen_toks are carried in each request; native "
                "generation must additionally stop at OLMo3 EOS <|endoftext|>/<|im_end|>."
            ),
        },
        "scoring_contract": {
            "alpaca_eval_v2": "requests_and_raw_generations_only; no judge score",
            "deferred_or_judge_metrics": "hard error",
            "other_tasks": "pinned OLMES objective/rule-based metrics",
        },
        "model_context_adaptation": {
            "model_max_length": 8192,
            "alpaca_eval_v2": {
                "truncate_context": False,
                "effective_max_new_tokens": (
                    "min(request.max_gen_toks, model_max_length - prompt_token_count)"
                ),
                "reason": (
                    "preserve the full short prompt; max_gen_toks=8192 must not imply "
                    "a zero-token prompt budget"
                ),
            },
        },
    }
    manifest["manifest_content_sha256"] = _sha256_json(manifest)
    _atomic_write_json(output_dir / "manifest.json", manifest)
    LOG.info(
        "Frozen %d requests from %d concrete tasks in %s",
        global_request_index,
        len(task_manifests),
        output_dir,
    )
    return 0


def _response_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("model_resps", row.get("response"))
    if payload is None:
        reserved = {
            "schema_version",
            "request_id",
            "request_type",
            "task_alias",
            "task_name",
            "rank",
            "world_size",
        }
        payload = {key: value for key, value in row.items() if key not in reserved}
    if not isinstance(payload, dict):
        raise ContractError(f"Response {row.get('request_id')} payload must be an object")
    return copy.deepcopy(payload)


def _validate_model_response(record: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    request_id = record["request_id"]
    request_type = record["request_type"]
    if request_type in {"generate_until", "generate_until_and_loglikelihood"}:
        if not isinstance(response.get("continuation"), str):
            raise ContractError(f"{request_id}: generation response needs string continuation")
    if request_type in {"loglikelihood", "generate_until_and_loglikelihood"}:
        sum_logits = response.get("sum_logits")
        if not isinstance(sum_logits, (int, float)) or not math.isfinite(float(sum_logits)):
            raise ContractError(f"{request_id}: likelihood response needs finite sum_logits")
        num_tokens = response.get("num_tokens")
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 0:
            raise ContractError(f"{request_id}: likelihood response needs nonnegative num_tokens")


def _merge_metric_predictions(metrics: Sequence[Any]) -> list[dict[str, Any]]:
    if not metrics:
        return []
    predictions = [copy.deepcopy(metric._scores_for_docs) for metric in metrics]
    expected_length = len(predictions[0])
    if any(len(rows) != expected_length for rows in predictions):
        raise ContractError("OLMES metrics returned different document counts")
    merged: list[dict[str, Any]] = []
    for rows in zip(*predictions):
        entry = copy.deepcopy(rows[0])
        combined_metrics: dict[str, Any] = {}
        for row in rows:
            for name, value in row.get("metrics", {}).items():
                if name in combined_metrics:
                    raise ContractError(f"Duplicate OLMES metric name while merging: {name}")
                combined_metrics[name] = value
        entry["metrics"] = combined_metrics
        merged.append(entry)
    return merged


def _is_no_score_task(task_manifest: Mapping[str, Any]) -> bool:
    return (
        task_manifest["suite_alias"] in NO_SCORE_SUITE_ALIASES
        or task_manifest["task_name"] in NO_SCORE_TASK_NAMES
        or task_manifest.get("score_policy") == "generate_only_no_judge"
    )


def _score(args: argparse.Namespace) -> int:
    _configure_huggingface(args)
    _bootstrap_olmes(args.olmes_root, args.olmes_stubs)

    from oe_eval.run_eval import load_task
    from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

    freeze_dir = args.freeze_dir.resolve()
    manifest_path = freeze_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            f"Unsupported manifest schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    current_commit = _require_pinned_olmes(args.olmes_root)["commit"]
    if current_commit != manifest["olmes"]["commit"]:
        raise ContractError(
            f"OLMES commit mismatch: freeze={manifest['olmes']['commit']} score={current_commit}"
        )
    requests_path = freeze_dir / manifest["requests_file"]
    requests_sha256 = _sha256_file(requests_path)
    if requests_sha256 != manifest["requests_file_sha256"]:
        raise ContractError(
            f"Frozen requests hash mismatch: {requests_sha256} != "
            f"{manifest['requests_file_sha256']}"
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    records_by_task: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(requests_path):
        _validate_frozen_record(record)
        request_id = record["request_id"]
        if request_id in records_by_id:
            raise ContractError(f"Duplicate request_id in frozen requests: {request_id}")
        records_by_id[request_id] = record
        records_by_task[int(record["task_index"])].append(record)
    if len(records_by_id) != manifest["num_requests"]:
        raise ContractError(
            f"Manifest says {manifest['num_requests']} requests, found {len(records_by_id)}"
        )

    responses: dict[str, dict[str, Any]] = {}
    response_sources: dict[str, str] = {}
    for response_path in args.responses:
        response_path = response_path.resolve()
        for row in _read_jsonl(response_path):
            request_id = row.get("request_id")
            if not isinstance(request_id, str):
                raise ContractError(f"Response in {response_path} has no string request_id")
            if request_id not in records_by_id:
                raise ContractError(f"Unknown response request_id {request_id} in {response_path}")
            payload = _response_payload(row)
            if request_id in responses:
                if args.allow_identical_duplicates and responses[request_id] == payload:
                    continue
                raise ContractError(
                    f"Duplicate response for {request_id}: "
                    f"{response_sources[request_id]} and {response_path}"
                )
            _validate_model_response(records_by_id[request_id], payload)
            responses[request_id] = payload
            response_sources[request_id] = str(response_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    scored_tasks: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    now = _utc_now()
    model_hash = args.model_hash or "native-response-jsonl"
    model_config = {"model": args.model_name, "model_hash": model_hash}
    compute_config = {
        "freeze_dir": str(freeze_dir),
        "response_files": [str(path.resolve()) for path in args.responses],
        "judge_scoring": False,
    }

    for task_manifest in manifest["tasks"]:
        task_index = int(task_manifest["task_index"])
        task_records = records_by_task[task_index]
        missing = [record["request_id"] for record in task_records if record["request_id"] not in responses]
        if _is_no_score_task(task_manifest):
            skipped_tasks.append(
                {
                    "task_index": task_index,
                    "suite_alias": task_manifest["suite_alias"],
                    "task_alias": task_manifest["task_alias"],
                    "task_name": task_manifest["task_name"],
                    "num_requests": len(task_records),
                    "num_responses": len(task_records) - len(missing),
                    "reason": "judge scoring disabled by contract; raw generations only",
                }
            )
            continue
        if missing:
            if args.allow_incomplete_tasks:
                skipped_tasks.append(
                    {
                        "task_index": task_index,
                        "suite_alias": task_manifest["suite_alias"],
                        "task_alias": task_manifest["task_alias"],
                        "task_name": task_manifest["task_name"],
                        "num_requests": len(task_records),
                        "num_responses": len(task_records) - len(missing),
                        "reason": f"incomplete responses ({len(missing)} missing); task not partially scored",
                    }
                )
                continue
            preview = ", ".join(missing[:5])
            raise ContractError(
                f"Task {task_manifest['task_alias']} is missing {len(missing)} responses: {preview}"
            )

        task = load_task(copy.deepcopy(task_manifest["task_config"]), str(output_dir))
        task.make_metrics()
        metrics = task._metrics or []
        if not metrics:
            raise ContractError(f"No OLMES metrics found for {task_manifest['task_alias']}")
        forbidden = [
            metric
            for metric in metrics
            if metric.deferred_metric
            or "judge" in metric.__class__.__name__.lower()
            or "judge" in metric.__class__.__module__.lower()
        ]
        if forbidden:
            names = ", ".join(
                f"{metric.__class__.__module__}.{metric.__class__.__name__}" for metric in forbidden
            )
            raise ContractError(
                f"Judge/deferred metric refused for {task_manifest['task_alias']}: {names}"
            )

        results_for_requests: list[dict[str, Any]] = []
        for response_index, record in enumerate(task_records):
            result = {
                "res_id": response_index,
                "request_type": record["request_type"],
                "doc": record["doc"],
                "request": record["request"],
                "idx": record["idx"],
                "task_name": record["task_name"],
                "doc_id": record["doc_id"],
                "native_id": record.get("native_id"),
                "label": record.get("label"),
                "model_resps": copy.deepcopy(responses[record["request_id"]]),
            }
            results_for_requests.append(result)

        metric_outputs: list[dict[str, Any]] = []
        for metric in metrics:
            metric.compute_for_docs(copy.deepcopy(results_for_requests))
            metric_outputs.append(
                metric.aggregate_to_task(primary_metric=task.task_config.get("primary_metric"))
            )
        merged_metrics: dict[str, Any] = {}
        for metric_output in metric_outputs:
            for name, value in metric_output.items():
                if name in merged_metrics:
                    raise ContractError(
                        f"Duplicate aggregate metric {name} for {task_manifest['task_alias']}"
                    )
                merged_metrics[name] = value

        predictions = _merge_metric_predictions(metrics)
        prediction_path = predictions_dir / f"{task_index:03d}-{_slug(task_manifest['task_alias'])}.jsonl"
        _atomic_write_jsonl(prediction_path, predictions)
        scored_tasks.append(
            {
                "task_name": task.task_name,
                "task_hash": task_manifest["task_hash"],
                "model_hash": model_hash,
                "model_config": model_config,
                "task_config": task.task_config,
                "compute_config": compute_config,
                "processing_time": 0.0,
                "current_date": now,
                "num_instances": len(metrics[0]._scores_for_docs),
                "beaker_info": {},
                "metrics": merged_metrics,
                "task_idx": task_index,
                "prediction_file": prediction_path.relative_to(output_dir).as_posix(),
                "prediction_file_sha256": _sha256_file(prediction_path),
            }
        )
        LOG.info("Scored %s: %s", task_manifest["task_alias"], merged_metrics)

    aggregate_tasks = add_aggregate_tasks(scored_tasks)
    output = {
        "schema_version": "olmes-objective-scores-v1",
        "created_at": _utc_now(),
        "freeze_manifest": str(manifest_path),
        "freeze_manifest_sha256": _sha256_file(manifest_path),
        "olmes_commit": current_commit,
        "model": model_config,
        "num_frozen_requests": len(records_by_id),
        "num_responses": len(responses),
        "judge_scoring": False,
        "task_metrics": scored_tasks,
        "aggregate_metrics": aggregate_tasks,
        "skipped_tasks": skipped_tasks,
    }
    _atomic_write_jsonl(output_dir / "metrics-all.jsonl", [*aggregate_tasks, *scored_tasks])
    _atomic_write_jsonl(output_dir / "skipped-tasks.jsonl", skipped_tasks)
    _atomic_write_json(output_dir / "metrics.json", output)
    LOG.info(
        "Scored %d concrete tasks; skipped %d; no judge metrics invoked",
        len(scored_tasks),
        len(skipped_tasks),
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--olmes-root",
        type=Path,
        default=DEFAULT_OLMES_ROOT,
        help="Pinned official OLMES checkout",
    )
    parser.add_argument(
        "--olmes-stubs",
        type=Path,
        default=DEFAULT_OLMES_STUBS,
        help="Import-only stubs for unused optional OLMES dependencies",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=REPO_ROOT / "eval_artifacts" / "hf_home",
    )
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=DEFAULT_NLTK_DATA,
        help="Pinned local NLTK resources required by OLMES IFEval scoring",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require cached Hugging Face datasets (default: true)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Build and freeze OLMES request JSONL")
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument(
        "--task",
        action="append",
        default=None,
        help="OLMES task/suite alias (repeatable; default: pinned objective suite)",
    )
    freeze.add_argument(
        "--limit-per-concrete-task",
        type=int,
        default=None,
        help="Testing-only limit applied independently to every expanded concrete task",
    )
    freeze.add_argument("--overwrite", action="store_true")
    freeze.add_argument("--allow-dirty-olmes", action="store_true")
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser(
        "score", help="Join response JSONL and run only official objective/rule-based metrics"
    )
    score.add_argument("--freeze-dir", type=Path, required=True)
    score.add_argument("--responses", type=Path, nargs="+", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--model-name", default="olmo3-siamese-depth-native")
    score.add_argument("--model-hash", default=None)
    score.add_argument("--allow-identical-duplicates", action="store_true")
    score.add_argument(
        "--allow-incomplete-tasks",
        action="store_true",
        help="Skip incomplete tasks; never compute misleading partial-task scores",
    )
    score.set_defaults(handler=_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if args.task is None:
            args.task = list(DEFAULT_TASK_ALIASES)
        if args.limit_per_concrete_task is not None and args.limit_per_concrete_task <= 0:
            parser.error("--limit-per-concrete-task must be positive")
    try:
        return int(args.handler(args))
    except ContractError as error:
        LOG.error("Contract violation: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
