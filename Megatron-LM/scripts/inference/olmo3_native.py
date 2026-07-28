#!/usr/bin/env python3
"""Run frozen OLMES-style requests against a native MindSpeed OLMo3 checkpoint.

The input is a JSONL file whose records are immutable evaluation requests.  A
record minimally contains ``request_id``, ``request_type`` and a nested
``request`` object.  Supported request payloads are::

    {"request_type": "loglikelihood",
     "request": {"context": "...", "continuation": "..."}}

    {"request_type": "generate_until",
     "request": {"context": "...", "stop_sequences": ["..."],
                 "generation_kwargs": {"max_gen_toks": 256}}}

Every input field is copied to the corresponding rank JSONL result.  The
runner only appends ``model_resps`` and ``_frozen_eval_runner``.  This makes it
possible for an offline scorer to join by ``request_id`` without importing the
task library into the NPU runtime.

The model topology keeps PP=CP=1. TP is configurable so a long-context request
can use one node's tensor-parallel group, while DP replicas own independent
requests. Generation uses the architecture's explicit static KV-cache API. It
is greedy by default; an explicitly sampled request uses reproducible,
request-seeded top-k/top-p sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# pretrain_olmo3_mindspeed must remain the first project import in the runtime
# path.  Project and torch imports are intentionally lazy so ``--self-test``
# can validate the JSONL/partition contract on a CPU-only control host.
os.environ.setdefault("USE_TF", "FALSE")
os.environ.setdefault("MODEL_USE_NULL_ATTENTION_MASK", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


SCHEMA_VERSION = 1
OLMO3_BASE_CONTEXT_LENGTH = 8192
OLMO3_LONG_CONTEXT_LENGTH = 65536
MAX_SUPPORTED_CONTEXT_LENGTH = OLMO3_LONG_CONTEXT_LENGTH
EXPECTED_PADDED_VOCAB_SIZE = 100352
EXPECTED_TOKENIZER_VOCAB_SIZE = 100278
EXPECTED_EOS_TOKEN_IDS = (100257, 100265)
EXPECTED_EOT_TOKEN_ID = 100257
EXPECTED_IM_END_TOKEN_ID = 100265
EXPECTED_PAD_TOKEN_ID = 100277
EXPECTED_IM_END_TOKEN = "<|im_end|>"
EXPECTED_PAD_TOKEN = "<|pad|>"
DEFAULT_MAX_GEN_TOKENS = 256


class FrozenEvalError(RuntimeError):
    """A frozen request, checkpoint, or inference result broke its contract."""


@dataclass(frozen=True)
class GenerationSettings:
    """Normalized deterministic generation settings for one request."""

    max_gen_toks: int
    truncate_context: bool
    min_acceptable_gen_toks: int
    stop_sequences: tuple[str, ...]
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    ignored_deterministic_kwargs: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _strict_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenEvalError(f"{label} must be an integer, got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise FrozenEvalError(f"{label} must be >= {minimum}, got {result}.")
    return result


def _deduplicate_strings(values: Iterable[Any], *, label: str) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise FrozenEvalError(f"{label} entries must be strings, got {value!r}.")
        # An empty stop sequence would match before any generated token and is
        # almost always a serialization bug.  Reject it rather than silently
        # turning every completion into an empty string.
        if value == "":
            raise FrozenEvalError(f"{label} cannot contain an empty string.")
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def _normalize_generation_settings(
    payload: dict[str, Any],
    *,
    default_max_gen_toks: int = DEFAULT_MAX_GEN_TOKENS,
) -> GenerationSettings:
    """Normalize OLMES/HF generation kwargs and reject stochastic decoding."""
    kwargs_value = payload.get("generation_kwargs")
    if kwargs_value is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(kwargs_value, dict):
        kwargs = dict(kwargs_value)
    else:
        raise FrozenEvalError("request.generation_kwargs must be an object or null.")

    stop_values: list[Any] = []
    direct_stops = payload.get("stop_sequences", [])
    if direct_stops is None:
        direct_stops = []
    if isinstance(direct_stops, str):
        stop_values.append(direct_stops)
    elif isinstance(direct_stops, list):
        stop_values.extend(direct_stops)
    else:
        raise FrozenEvalError("request.stop_sequences must be a string, list, or null.")
    for alias in ("stop_sequences", "until"):
        nested = kwargs.pop(alias, None)
        if nested is None:
            continue
        if isinstance(nested, str):
            stop_values.append(nested)
        elif isinstance(nested, list):
            stop_values.extend(nested)
        else:
            raise FrozenEvalError(
                f"request.generation_kwargs.{alias} must be a string or list."
            )

    max_gen_toks_value = kwargs.pop("max_gen_toks", None)
    max_new_tokens_value = kwargs.pop("max_new_tokens", None)
    if max_gen_toks_value is not None and max_new_tokens_value is not None:
        if max_gen_toks_value != max_new_tokens_value:
            raise FrozenEvalError(
                "max_gen_toks and max_new_tokens disagree in one request."
            )
    if max_gen_toks_value is None:
        max_gen_toks_value = max_new_tokens_value
    if max_gen_toks_value is None:
        max_gen_toks_value = default_max_gen_toks
    max_gen_toks = _strict_int(
        max_gen_toks_value, label="max_gen_toks", minimum=1
    )

    truncate_context = kwargs.pop("truncate_context", True)
    if not isinstance(truncate_context, bool):
        raise FrozenEvalError("truncate_context must be boolean.")
    min_acceptable = kwargs.pop(
        "min_acceptable_gen_toks", kwargs.pop("min_new_tokens", 0)
    )
    min_acceptable_gen_toks = _strict_int(
        min_acceptable,
        label="min_acceptable_gen_toks",
        minimum=0,
    )

    do_sample_value = kwargs.pop("do_sample", False)
    if do_sample_value is None:
        do_sample = False
    elif isinstance(do_sample_value, bool):
        do_sample = do_sample_value
    else:
        raise FrozenEvalError("do_sample must be boolean or null.")
    num_beams = kwargs.pop("num_beams", 1)
    if num_beams not in (None, 1):
        raise FrozenEvalError("Beam search is unsupported; num_beams must be 1.")
    num_return_sequences = kwargs.pop("num_return_sequences", 1)
    if num_return_sequences not in (None, 1):
        raise FrozenEvalError("num_return_sequences must be 1 in frozen JSONL.")

    temperature_value = kwargs.pop("temperature", 1.0)
    if isinstance(temperature_value, bool) or not isinstance(
        temperature_value, (int, float)
    ):
        raise FrozenEvalError("temperature must be numeric.")
    temperature = float(temperature_value)
    top_p_value = kwargs.pop("top_p", 1.0)
    if isinstance(top_p_value, bool) or not isinstance(top_p_value, (int, float)):
        raise FrozenEvalError("top_p must be numeric.")
    top_p = float(top_p_value)
    top_k = _strict_int(kwargs.pop("top_k", 0), label="top_k", minimum=0)
    seed_value = kwargs.pop("seed", None)
    seed = None
    if seed_value is not None:
        seed = _strict_int(seed_value, label="seed", minimum=0)
    if do_sample:
        if not math.isfinite(temperature) or temperature <= 0:
            raise FrozenEvalError(
                "Sampled generation requires a finite temperature > 0."
            )
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise FrozenEvalError("Sampled generation requires 0 < top_p <= 1.")
        if top_k >= EXPECTED_TOKENIZER_VOCAB_SIZE:
            raise FrozenEvalError("top_k must be smaller than the tokenizer vocabulary.")

    # These parameters do not affect the implemented token selection.  We
    # accept and record them so exact OLMES configs can be frozen without a
    # lossy preprocessing rewrite.
    deterministic_noops = {
        "use_cache",
        "return_dict_in_generate",
        "output_scores",
        "max_length",
    }
    ignored = tuple(sorted(key for key in kwargs if key in deterministic_noops))
    unsupported = sorted(set(kwargs) - deterministic_noops)
    if unsupported:
        raise FrozenEvalError(
            "Unsupported generation kwargs would otherwise be silently ignored: "
            + ", ".join(unsupported)
        )

    return GenerationSettings(
        max_gen_toks=max_gen_toks,
        truncate_context=truncate_context,
        min_acceptable_gen_toks=min_acceptable_gen_toks,
        stop_sequences=_deduplicate_strings(stop_values, label="stop_sequences"),
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        ignored_deterministic_kwargs=ignored,
    )


def _stable_request_seed(request_id: str) -> int:
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _stateless_uniform(seed: int, generation_index: int) -> float:
    """Stable U[0,1) independent of rank and NPU RNG implementation."""
    digest = hashlib.sha256(f"{seed}:{generation_index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _find_earliest_stop(text: str, stop_sequences: Sequence[str]) -> tuple[int, str] | None:
    """Return the earliest stop match; input order breaks equal-position ties."""
    best: tuple[int, int, str] | None = None
    for order, stop in enumerate(stop_sequences):
        position = text.find(stop)
        if position < 0:
            continue
        candidate = (position, order, stop)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[0], best[2]


def _stop_search_window_tokens(stop_sequences: Sequence[str]) -> int:
    """Conservative token suffix size for incremental stop candidate checks.

    A byte-level token contributes at least one byte.  The UTF-8 byte length
    plus a 16-token boundary margin therefore covers a complete stop string;
    the 64-token floor also protects tokenizer-specific boundary behavior.
    Any suffix candidate is confirmed against one exact full decode.
    """
    if not stop_sequences:
        return 0
    return max(64, max(len(stop.encode("utf-8")) for stop in stop_sequences) + 16)


def _prefill_logits_sequence_contract(actual: int, prompt_length: int) -> bool:
    """Validate prefill logit materialization and return whether it was reduced.

    The optimized path honors ``materialize_only_last_token_logits`` and
    returns one position. Accepting an exact prompt-length result here keeps
    older frozen inference artifacts readable, while long-cache-smoke requires
    the optimized one-position contract explicitly. No other length is valid.
    """
    _strict_int(actual, label="prefill logits sequence length", minimum=1)
    _strict_int(prompt_length, label="prompt_length", minimum=1)
    if actual == 1:
        return True
    if actual == prompt_length:
        return False
    raise FrozenEvalError(
        "Prefill logits must contain either only the last position or the exact "
        f"prompt length; got {actual} for prompt length {prompt_length}."
    )


def _encode_pair_contract(
    encode: Any,
    *,
    prefix_token_id: int,
    context: str,
    continuation: str,
) -> tuple[list[int], list[int]]:
    """Tokenize a causal LM pair without breaking a whitespace/BPE boundary.

    This is the same contract used by the established native lm-eval adapter:
    trailing context whitespace belongs to the continuation, then the whole
    string is encoded and split only at a verified context-token prefix.
    """
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    if context == "":
        return [int(prefix_token_id)], [int(token) for token in encode(continuation)]
    whole_tokens = [int(token) for token in encode(context + continuation)]
    context_tokens = [int(token) for token in encode(context)]
    if whole_tokens[: len(context_tokens)] != context_tokens:
        raise FrozenEvalError(
            "Tokenizer context is not a prefix of context+continuation after "
            "moving trailing whitespace to the continuation."
        )
    return context_tokens, whole_tokens[len(context_tokens) :]


def _partition_global_indices(
    total: int,
    *,
    partition_index: int,
    partition_count: int,
    strategy: str,
) -> list[int]:
    """Partition global records before DP rank sharding."""
    _strict_int(total, label="total", minimum=0)
    _strict_int(partition_count, label="partition_count", minimum=1)
    _strict_int(partition_index, label="partition_index", minimum=0)
    if partition_index >= partition_count:
        raise FrozenEvalError(
            f"partition_index={partition_index} must be < partition_count={partition_count}."
        )
    if strategy == "modulo":
        return [
            index for index in range(total) if index % partition_count == partition_index
        ]
    if strategy == "contiguous":
        start = total * partition_index // partition_count
        end = total * (partition_index + 1) // partition_count
        return list(range(start, end))
    raise FrozenEvalError(f"Unknown partition strategy: {strategy!r}.")


def _filtered_global_indices(
    records: Sequence[dict[str, Any]],
    *,
    suite_aliases: Sequence[str],
    task_aliases: Sequence[str],
) -> list[int]:
    """Select aliases exactly, preserving the combined JSONL's global order."""
    suite_filter = set(suite_aliases)
    task_filter = set(task_aliases)
    selected: list[int] = []
    for global_index, record in enumerate(records):
        if suite_filter and record.get("suite_alias") not in suite_filter:
            continue
        if task_filter and record.get("task_alias") not in task_filter:
            continue
        selected.append(global_index)
    if not selected:
        details = []
        if suite_filter:
            details.append(f"suite_alias in {sorted(suite_filter)!r}")
        if task_filter:
            details.append(f"task_alias in {sorted(task_filter)!r}")
        raise FrozenEvalError(
            "Alias filters selected no frozen requests: " + " and ".join(details)
        )
    return selected


def _request_payload(record: dict[str, Any], *, line_number: int) -> tuple[str, dict[str, Any]]:
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise FrozenEvalError(f"line {line_number}: request_id must be a non-empty string.")
    request_type = record.get("request_type")
    if not isinstance(request_type, str):
        raise FrozenEvalError(f"line {line_number}: request_type must be a string.")
    request_type = request_type.lower()
    if request_type not in {
        "loglikelihood",
        "generate_until",
        "generate_until_and_loglikelihood",
    }:
        raise FrozenEvalError(
            f"line {line_number}: unsupported request_type={request_type!r}."
        )
    payload = record.get("request")
    if not isinstance(payload, dict):
        raise FrozenEvalError(f"line {line_number}: request must be an object.")
    context = payload.get("context")
    if not isinstance(context, str):
        raise FrozenEvalError(
            f"line {line_number}: request.context must already be rendered to a string."
        )
    if request_type in {
        "loglikelihood",
        "generate_until_and_loglikelihood",
    }:
        continuation = payload.get("continuation")
        if not isinstance(continuation, str):
            raise FrozenEvalError(
                f"line {line_number}: loglikelihood continuation must be a string."
            )
    if request_type == "generate_until_and_loglikelihood":
        perplexity_context = payload.get("perplexity_context")
        if not isinstance(perplexity_context, str):
            raise FrozenEvalError(
                f"line {line_number}: combined request perplexity_context "
                "must already be rendered to a string."
            )
    if request_type in {
        "generate_until",
        "generate_until_and_loglikelihood",
    }:
        _normalize_generation_settings(payload)
    return request_type, payload


def _load_frozen_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FrozenEvalError(f"Frozen request file does not exist: {path}")
    records: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise FrozenEvalError(
                    f"line {line_number}: blank lines are forbidden in frozen JSONL."
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FrozenEvalError(
                    f"line {line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise FrozenEvalError(f"line {line_number}: JSON value must be an object.")
            _request_payload(record, line_number=line_number)
            request_id = str(record["request_id"])
            if request_id in request_ids:
                raise FrozenEvalError(
                    f"line {line_number}: duplicate request_id={request_id!r}."
                )
            request_ids.add(request_id)
            records.append(record)
    if not records:
        raise FrozenEvalError("Frozen request JSONL is empty.")
    return records


def _unwrap_huggingface_tokenizer(tokenizer: Any) -> Any:
    current = tokenizer
    visited: set[int] = set()
    wrapper_types: list[str] = []
    for _ in range(8):
        wrapper_types.append(type(current).__qualname__)
        if callable(getattr(current, "convert_tokens_to_ids", None)) and callable(
            getattr(current, "encode", None)
        ):
            return current
        object_id = id(current)
        if object_id in visited:
            break
        visited.add(object_id)
        nested = getattr(current, "_tokenizer", None)
        if nested is None or nested is current:
            nested = getattr(current, "tokenizer", None)
        if nested is None or nested is current:
            break
        current = nested
    raise FrozenEvalError(
        "Could not resolve the HuggingFace tokenizer through wrappers: "
        + " -> ".join(wrapper_types)
    )


class NativeFrozenEvaluator:
    """One PP1/CP1 native OLMo3 replica serving its DP shard."""

    def __init__(self, model: Any, megatron_tokenizer: Any, args: argparse.Namespace, runtime: dict[str, Any]):
        self.model = model
        self.cache_owner = model
        for _ in range(8):
            if callable(getattr(self.cache_owner, "build_inference_context", None)):
                break
            nested = getattr(self.cache_owner, "module", None)
            if nested is None or nested is self.cache_owner:
                break
            self.cache_owner = nested
        if not callable(getattr(self.cache_owner, "build_inference_context", None)):
            raise FrozenEvalError(
                "Could not resolve build_inference_context through model wrappers."
            )
        self.megatron_tokenizer = megatron_tokenizer
        self.tokenizer = _unwrap_huggingface_tokenizer(megatron_tokenizer)
        self.args = args
        self.torch = runtime["torch"]
        self.dist = runtime["dist"]
        self.parallel_state = runtime["parallel_state"]
        self.device = self.torch.device(f"npu:{self.torch.npu.current_device()}")
        self.rank = int(self.parallel_state.get_data_parallel_rank())
        self.world_size = int(self.parallel_state.get_data_parallel_world_size())
        self.tp_rank = int(self.parallel_state.get_tensor_model_parallel_rank())
        self.tp_world_size = int(
            self.parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_group = self.parallel_state.get_tensor_model_parallel_group()
        self.is_replica_writer = self.tp_rank == 0
        self.dp_group = self.parallel_state.get_data_parallel_group()
        self.dp_gloo_group = self.parallel_state.get_data_parallel_group_gloo()
        self.max_length = int(args.eval_max_length)
        self.padded_vocab_size = int(args.expected_padded_vocab_size)
        model_max_length = int(
            getattr(self.cache_owner, "max_sequence_length", 0) or 0
        )
        if model_max_length < self.max_length:
            raise FrozenEvalError(
                "Loaded model context is smaller than --eval-max-length: "
                f"{model_max_length} < {self.max_length}."
            )
        if self.max_length > OLMO3_BASE_CONTEXT_LENGTH:
            config = getattr(self.cache_owner, "config", None)
            yarn = getattr(config, "olmo3_yarn", None)
            expected_yarn = {
                "factor": 8.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "old_context_len": OLMO3_BASE_CONTEXT_LENGTH,
                "theta": 500000.0,
            }
            actual_yarn = {
                name: getattr(yarn, name, None) for name in expected_yarn
            }
            if actual_yarn != expected_yarn:
                raise FrozenEvalError(
                    "Long-context inference requires the exact OLMo3 Stage-3 "
                    f"Full-attention YaRN contract; got {actual_yarn}."
                )
            if int(getattr(config, "olmo3_sliding_window", 0) or 0) != 4096:
                raise FrozenEvalError(
                    "Long-context inference requires the Stage-3 4096-token SWA window."
                )
            if not bool(getattr(config, "rope_full_precision", False)):
                raise FrozenEvalError(
                    "Long-context inference requires FP32 Q/K RoPE application."
                )
            rotary = getattr(
                getattr(self.cache_owner, "backbone", None),
                "rotary_pos_emb",
                None,
            )
            if (
                rotary is None
                or getattr(rotary, "yarn", None) is not yarn
                or not hasattr(rotary, "base_rotary")
            ):
                raise FrozenEvalError(
                    "Long-context model did not install separate Full/SWA rotary paths."
                )

        tokenizer_size = len(self.tokenizer)
        if tokenizer_size != int(args.expected_tokenizer_vocab_size):
            raise FrozenEvalError(
                f"Tokenizer size mismatch: expected {args.expected_tokenizer_vocab_size}, "
                f"got {tokenizer_size}."
            )
        self.tokenizer_vocab_size = tokenizer_size
        if int(megatron_tokenizer.eod) != EXPECTED_EOT_TOKEN_ID:
            raise FrozenEvalError(
                f"Unexpected Megatron EOD token: {megatron_tokenizer.eod}."
            )
        im_end_id = self.tokenizer.convert_tokens_to_ids(EXPECTED_IM_END_TOKEN)
        if int(im_end_id) != EXPECTED_IM_END_TOKEN_ID:
            raise FrozenEvalError(
                f"Unexpected {EXPECTED_IM_END_TOKEN} ID: {im_end_id}."
            )
        pad_id = self.tokenizer.convert_tokens_to_ids(EXPECTED_PAD_TOKEN)
        if int(pad_id) != EXPECTED_PAD_TOKEN_ID:
            raise FrozenEvalError(f"Unexpected {EXPECTED_PAD_TOKEN} ID: {pad_id}.")
        self.eos_token_ids = frozenset(EXPECTED_EOS_TOKEN_IDS)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        result = [int(token) for token in tokens]
        if any(token < 0 or token >= self.tokenizer_vocab_size for token in result):
            raise FrozenEvalError("Tokenizer emitted an out-of-vocabulary token ID.")
        return result

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        kwargs = {
            "skip_special_tokens": skip_special_tokens,
            "clean_up_tokenization_spaces": False,
        }
        try:
            return str(self.tokenizer.decode(list(token_ids), **kwargs))
        except TypeError:
            kwargs.pop("clean_up_tokenization_spaces")
            return str(self.tokenizer.decode(list(token_ids), **kwargs))

    def _prefix_token_id(self) -> int:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        return int(bos) if bos is not None else EXPECTED_EOT_TOKEN_ID

    def _check_logits(self, logits: Any, *, expected_sequence: int) -> None:
        expected = (1, expected_sequence, self.padded_vocab_size)
        if tuple(logits.shape) != expected:
            raise FrozenEvalError(
                f"Unexpected logits shape {tuple(logits.shape)}; expected {expected}."
            )
        if not bool(self.torch.isfinite(logits).all().item()):
            raise FrozenEvalError("Model returned NaN or Inf logits.")

    def _check_kv_cache_populated(
        self,
        inference_context: Any,
        *,
        expected_offset: int,
    ) -> None:
        """Prove that direct model forwarding populated, but did not advance, KV state."""
        actual_offset = int(inference_context.sequence_len_offset)
        if actual_offset != expected_offset:
            raise FrozenEvalError(
                "Direct model forward unexpectedly changed sequence_len_offset: "
                f"expected {expected_offset}, got {actual_offset}."
            )
        cache = getattr(inference_context, "key_value_memory_dict", None)
        if not isinstance(cache, dict) or not cache:
            raise FrozenEvalError(
                "use_cache=True returned logits without populating key_value_memory_dict."
            )
        expected_layers = int(self.args.num_layers)
        if len(cache) != expected_layers:
            raise FrozenEvalError(
                f"KV cache has {len(cache)} layers, expected {expected_layers}."
            )
        for layer_number, key_value in cache.items():
            if not isinstance(key_value, tuple) or len(key_value) != 2:
                raise FrozenEvalError(
                    f"Malformed KV cache entry for layer {layer_number!r}."
                )
            for cache_tensor in key_value:
                if cache_tensor.ndim < 2:
                    raise FrozenEvalError(
                        f"Malformed KV tensor for layer {layer_number!r}."
                    )
                if int(cache_tensor.shape[0]) != int(
                    inference_context.max_sequence_length
                ):
                    raise FrozenEvalError(
                        f"KV tensor capacity mismatch for layer {layer_number!r}."
                    )
                if int(cache_tensor.shape[1]) < 1:
                    raise FrozenEvalError(
                        f"KV tensor has no batch slot for layer {layer_number!r}."
                    )

    def _check_prefill_logits(
        self,
        logits: Any,
        *,
        prompt_length: int,
        inference_context: Any,
    ) -> tuple[Any, bool]:
        expected_base = (1, self.padded_vocab_size)
        if logits.ndim != 3 or (
            int(logits.shape[0]), int(logits.shape[2])
        ) != expected_base:
            raise FrozenEvalError(
                "Unexpected prefill logits shape "
                f"{tuple(logits.shape)}; expected [1, 1|{prompt_length}, "
                f"{self.padded_vocab_size}]."
            )
        materialized_last_only = _prefill_logits_sequence_contract(
            int(logits.shape[1]), prompt_length
        )
        if not bool(self.torch.isfinite(logits).all().item()):
            raise FrozenEvalError("Prefill returned NaN or Inf logits.")
        self._check_kv_cache_populated(inference_context, expected_offset=0)
        return logits[0, -1, :], materialized_last_only

    def _check_decode_logits(
        self,
        logits: Any,
        *,
        inference_context: Any,
        expected_offset: int,
    ) -> Any:
        # Unlike prefill, accepting more than one position here would conceal a
        # broken incremental path and make decoding quadratic.
        self._check_logits(logits, expected_sequence=1)
        self._check_kv_cache_populated(
            inference_context, expected_offset=expected_offset
        )
        return logits[0, -1, :]

    def _token_logprob_and_greedy(self, last_logits: Any) -> tuple[int, float, int]:
        float_logits = last_logits.float()
        log_probs = self.torch.log_softmax(float_logits, dim=-1)
        # Padded rows are part of the trained softmax normalization, but are not
        # legal tokenizer IDs and therefore cannot be emitted as text.
        greedy_id = int(
            self.torch.argmax(float_logits[: self.tokenizer_vocab_size]).item()
        )
        padded_argmax = int(self.torch.argmax(float_logits).item())
        logprob = float(log_probs[greedy_id].item())
        if not math.isfinite(logprob):
            raise FrozenEvalError("Generated token has a non-finite log probability.")
        return greedy_id, logprob, padded_argmax

    def _token_logprob_and_sample(
        self,
        last_logits: Any,
        *,
        settings: GenerationSettings,
        seed: int,
        generation_index: int,
    ) -> tuple[int, float, int]:
        """Stateless reproducible temperature/top-k/top-p sampling on NPU."""
        float_logits = last_logits.float()
        raw_log_probs = self.torch.log_softmax(float_logits, dim=-1)
        valid_logits = (
            float_logits[: self.tokenizer_vocab_size] / settings.temperature
        )
        if settings.top_k > 0:
            top_values, top_indices = self.torch.topk(valid_logits, settings.top_k)
            filtered = self.torch.full_like(valid_logits, float("-inf"))
            filtered.scatter_(0, top_indices, top_values)
            valid_logits = filtered

        sorted_logits, sorted_indices = self.torch.sort(
            valid_logits, descending=True
        )
        sorted_probs = self.torch.softmax(sorted_logits, dim=-1)
        if settings.top_p < 1.0:
            cumulative = sorted_probs.cumsum(dim=-1)
            # Keep the first token crossing top_p, matching the standard shifted
            # nucleus mask, and always retain at least one token.
            keep = cumulative - sorted_probs < settings.top_p
            keep[0] = True
            sorted_probs = self.torch.where(
                keep, sorted_probs, self.torch.zeros_like(sorted_probs)
            )
            sorted_probs = sorted_probs / sorted_probs.sum()
        cumulative = sorted_probs.cumsum(dim=-1)
        cumulative[-1] = 1.0
        uniform = _stateless_uniform(seed, generation_index)
        sorted_position = int((cumulative < uniform).sum().item())
        sorted_position = min(sorted_position, self.tokenizer_vocab_size - 1)
        token_id = int(sorted_indices[sorted_position].item())
        logprob = float(raw_log_probs[token_id].item())
        padded_argmax = int(self.torch.argmax(float_logits).item())
        if not math.isfinite(logprob):
            raise FrozenEvalError("Sampled token has a non-finite log probability.")
        return token_id, logprob, padded_argmax

    def loglikelihood(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = str(payload["context"])
        continuation = str(payload["continuation"])
        context_tokens, continuation_tokens = _encode_pair_contract(
            self.encode,
            prefix_token_id=self._prefix_token_id(),
            context=context,
            continuation=continuation,
        )
        if not continuation_tokens:
            raise FrozenEvalError("A loglikelihood continuation has no tokens.")
        if len(continuation_tokens) > self.max_length:
            raise FrozenEvalError(
                f"Continuation has {len(continuation_tokens)} tokens, exceeding "
                f"the {self.max_length}-token context."
            )

        original_total = len(context_tokens) + len(continuation_tokens)
        combined = (context_tokens + continuation_tokens)[-(self.max_length + 1) :]
        model_tokens = combined[:-1]
        if not model_tokens or len(model_tokens) > self.max_length:
            raise FrozenEvalError("Invalid tokenized loglikelihood input length.")
        targets = continuation_tokens
        continuation_start = len(model_tokens) - len(targets)
        if continuation_start < 0:
            raise FrozenEvalError("Left truncation removed continuation tokens.")

        input_ids = self.torch.tensor(
            [model_tokens], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            logits = self.model(
                tokens=input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                runtime_gather_output=True,
            )
        self._check_logits(logits, expected_sequence=len(model_tokens))
        continuation_logits = logits[
            0, continuation_start : continuation_start + len(targets), :
        ].float()
        target_tensor = self.torch.tensor(
            targets, dtype=self.torch.long, device=self.device
        )
        log_probs = self.torch.log_softmax(continuation_logits, dim=-1)
        token_log_probs_tensor = log_probs.gather(
            1, target_tensor.unsqueeze(1)
        ).squeeze(1)
        token_log_probs = [float(value) for value in token_log_probs_tensor.tolist()]
        if not all(math.isfinite(value) for value in token_log_probs):
            raise FrozenEvalError("Model returned a non-finite token log probability.")
        is_greedy = bool(
            log_probs[:, : self.tokenizer_vocab_size]
            .argmax(dim=-1)
            .eq(target_tensor)
            .all()
            .item()
        )
        sum_logits = float(math.fsum(token_log_probs))
        result = {
            "sum_logits": sum_logits,
            "num_tokens": len(targets),
            # Match OLMES' verbose response: this is the untruncated pair size.
            "num_tokens_all": original_total,
            "is_greedy": is_greedy,
            "token_ids": targets,
            "tokens": [
                self.decode([token], skip_special_tokens=False) for token in targets
            ],
            # OLMES calls token log probabilities "logits" for historical reasons.
            "logits": token_log_probs,
            "token_logprobs": token_log_probs,
            "model_input_tokens": len(model_tokens),
            "truncated_context_tokens": max(
                0,
                original_total - len(combined),
            ),
        }
        del logits, continuation_logits, log_probs, token_log_probs_tensor
        return result

    def _effective_generation(
        self, payload: dict[str, Any]
    ) -> tuple[list[int], GenerationSettings, int, int]:
        settings = _normalize_generation_settings(
            payload,
            default_max_gen_toks=int(self.args.eval_default_max_gen_toks),
        )
        prompt_tokens = self.encode(str(payload["context"]))
        if not prompt_tokens:
            prompt_tokens = [self._prefix_token_id()]
        original_prompt_length = len(prompt_tokens)

        if settings.truncate_context:
            max_context = self.max_length - settings.max_gen_toks
            if max_context < 1:
                raise FrozenEvalError(
                    f"max_gen_toks={settings.max_gen_toks} leaves no prompt token "
                    f"inside the {self.max_length}-token model window."
                )
            prompt_tokens = prompt_tokens[-max_context:]
            effective_max_gen_toks = settings.max_gen_toks
        else:
            if len(prompt_tokens) > self.max_length:
                raise FrozenEvalError(
                    "truncate_context=False but the prompt itself exceeds the model window."
                )
            effective_max_gen_toks = min(
                settings.max_gen_toks,
                self.max_length - len(prompt_tokens),
            )
            if effective_max_gen_toks == 0:
                minimum = settings.min_acceptable_gen_toks
                if minimum < 1:
                    raise FrozenEvalError(
                        "The prompt fills the model window and no generation token fits."
                    )
                if minimum >= self.max_length:
                    raise FrozenEvalError(
                        "min_acceptable_gen_toks leaves no room for a prompt token."
                    )
                prompt_tokens = prompt_tokens[-(self.max_length - minimum) :]
                effective_max_gen_toks = minimum
        if len(prompt_tokens) + effective_max_gen_toks > self.max_length:
            raise FrozenEvalError("Effective generation would exceed the OLMo3 context limit.")
        return (
            prompt_tokens,
            settings,
            effective_max_gen_toks,
            original_prompt_length,
        )

    def generate_until(
        self, payload: dict[str, Any], *, request_id: str
    ) -> dict[str, Any]:
        (
            prompt_tokens,
            settings,
            effective_max_gen_toks,
            original_prompt_length,
        ) = self._effective_generation(payload)
        cache_length = len(prompt_tokens) + effective_max_gen_toks
        inference_context = self.cache_owner.build_inference_context(
            max_batch_size=1,
            max_sequence_length=cache_length,
            materialize_only_last_token_logits=True,
        )
        inference_context.enable_prefill_mode()
        prompt_tensor = self.torch.tensor(
            [prompt_tokens], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            logits = self.model(
                tokens=prompt_tensor,
                position_ids=None,
                attention_mask=None,
                inference_context=inference_context,
                use_cache=True,
                runtime_gather_output=True,
            )
        # Direct model calls do not advance offsets.  Advance only after the
        # forward succeeds so a failed kernel cannot leave a partially-valid
        # cache state.
        last_logits, prefill_materialized_last_only = self._check_prefill_logits(
            logits,
            prompt_length=len(prompt_tokens),
            inference_context=inference_context,
        )
        inference_context.sequence_len_offset += len(prompt_tokens)
        inference_context.enable_decode_mode()

        generated_ids: list[int] = []
        token_logprobs: list[float] = []
        padded_argmax_ids: list[int] = []
        sampling_seed = (
            settings.seed
            if settings.seed is not None
            else _stable_request_seed(request_id)
        )
        stop_window_tokens = _stop_search_window_tokens(settings.stop_sequences)
        finish_reason = "length"
        matched_stop_sequence: str | None = None
        continuation_raw = ""
        continuation = ""

        for generation_index in range(effective_max_gen_toks):
            if settings.do_sample:
                token_id, token_logprob, padded_argmax_id = (
                    self._token_logprob_and_sample(
                        last_logits,
                        settings=settings,
                        seed=sampling_seed,
                        generation_index=generation_index,
                    )
                )
            else:
                token_id, token_logprob, padded_argmax_id = (
                    self._token_logprob_and_greedy(last_logits)
                )
            generated_ids.append(token_id)
            token_logprobs.append(token_logprob)
            padded_argmax_ids.append(padded_argmax_id)

            if token_id in self.eos_token_ids:
                finish_reason = "eos_token"
                matched_stop_sequence = self.decode(
                    [token_id], skip_special_tokens=False
                )
                break
            if stop_window_tokens:
                # Decoding the full completion at every token is O(n^2).  A
                # bounded suffix only detects a candidate; exact full decode
                # below confirms the match before stopping/cropping.
                suffix = self.decode(
                    generated_ids[-stop_window_tokens:],
                    skip_special_tokens=True,
                )
                if _find_earliest_stop(suffix, settings.stop_sequences) is not None:
                    candidate_raw = self.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    stop_match = _find_earliest_stop(
                        candidate_raw, settings.stop_sequences
                    )
                    if stop_match is not None:
                        stop_position, matched_stop_sequence = stop_match
                        continuation_raw = candidate_raw
                        continuation = candidate_raw[:stop_position]
                        finish_reason = "stop_sequence"
                        break
            if generation_index + 1 == effective_max_gen_toks:
                break

            decode_tensor = self.torch.tensor(
                [[token_id]], dtype=self.torch.long, device=self.device
            )
            with self.torch.no_grad():
                next_logits = self.model(
                    tokens=decode_tensor,
                    position_ids=None,
                    attention_mask=None,
                    inference_context=inference_context,
                    use_cache=True,
                    runtime_gather_output=True,
                )
            last_logits = self._check_decode_logits(
                next_logits,
                inference_context=inference_context,
                expected_offset=len(prompt_tokens) + generation_index,
            )
            inference_context.sequence_len_offset += 1
            logits = next_logits

        if continuation_raw == "":
            continuation_raw = self.decode(
                generated_ids, skip_special_tokens=True
            )
        if finish_reason != "stop_sequence":
            continuation = continuation_raw

        result: dict[str, Any] = {
            "continuation": continuation,
            "sum_logits": float(math.fsum(token_logprobs)),
            "num_tokens": len(generated_ids),
            "token_ids": generated_ids,
            "tokens": [
                self.decode([token], skip_special_tokens=False)
                for token in generated_ids
            ],
            "logits": token_logprobs,
            "token_logprobs": token_logprobs,
            "finish_reason": finish_reason,
            "matched_stop_sequence": matched_stop_sequence,
            "requested_max_gen_toks": settings.max_gen_toks,
            "effective_max_gen_toks": effective_max_gen_toks,
            "original_prompt_tokens": original_prompt_length,
            "model_prompt_tokens": len(prompt_tokens),
            "truncated_context_tokens": original_prompt_length - len(prompt_tokens),
            "eos_token_ids": sorted(self.eos_token_ids),
            "padded_vocab_argmax_ids": padded_argmax_ids,
            "generation_mode": "sample" if settings.do_sample else "greedy",
            "sampling_seed": sampling_seed if settings.do_sample else None,
            "temperature": settings.temperature if settings.do_sample else None,
            "top_p": settings.top_p if settings.do_sample else None,
            "top_k": settings.top_k if settings.do_sample else None,
            "stop_search_window_tokens": stop_window_tokens,
            "prefill_materialized_last_token_logits": (
                prefill_materialized_last_only
            ),
            "ignored_deterministic_generation_kwargs": list(
                settings.ignored_deterministic_kwargs
            ),
        }
        if continuation_raw != continuation:
            result["continuation_raw"] = continuation_raw
        if not math.isfinite(result["sum_logits"]):
            raise FrozenEvalError("Generated sequence has a non-finite log probability.")
        return result

    def evaluate_record(
        self,
        record: dict[str, Any],
        *,
        global_index: int,
        partition_ordinal: int,
    ) -> dict[str, Any]:
        request_type, payload = _request_payload(
            record, line_number=global_index + 1
        )
        if "model_resps" in record:
            raise FrozenEvalError(
                f"Frozen request {record['request_id']!r} already has model_resps."
            )
        started = time.monotonic()
        if request_type == "loglikelihood":
            model_response = self.loglikelihood(payload)
        elif request_type == "generate_until":
            model_response = self.generate_until(
                payload, request_id=str(record["request_id"])
            )
        else:
            generation_response = self.generate_until(
                payload, request_id=str(record["request_id"])
            )
            likelihood_response = self.loglikelihood(
                {
                    "context": payload["perplexity_context"],
                    "continuation": payload["continuation"],
                }
            )
            # This matches OLMES'
            # generate_until_and_loglikelihood_verbose() response schema.
            model_response = {
                "continuation": generation_response["continuation"],
                "sum_logits": likelihood_response["sum_logits"],
                "num_tokens": likelihood_response["num_tokens"],
                "num_tokens_all": likelihood_response["num_tokens_all"],
            }
        output = dict(record)
        output["model_resps"] = model_response
        output["_frozen_eval_runner"] = {
            "schema_version": SCHEMA_VERSION,
            "global_index": global_index,
            "partition_ordinal": partition_ordinal,
            "partition_index": int(self.args.partition_index),
            "partition_count": int(self.args.partition_count),
            "partition_strategy": str(self.args.partition_strategy),
            "dp_rank": self.rank,
            "dp_world_size": self.world_size,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "checkpoint_step": int(self.args.expected_checkpoint_step),
            "wall_seconds": time.monotonic() - started,
            "completed_at": _utc_now(),
        }
        return output

    def _full_last_logits(self, token_ids: list[int]) -> Any:
        input_ids = self.torch.tensor(
            [token_ids], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            full_logits = self.model(
                tokens=input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                runtime_gather_output=True,
            )
        self._check_logits(full_logits, expected_sequence=len(token_ids))
        last = full_logits[0, -1, :].float().clone()
        del full_logits
        return last

    def _synthetic_prompt(self, length: int) -> list[int]:
        seed_tokens = self.encode(
            "OLMo 3 deterministic KV cache verification across sliding-window boundaries. "
        )
        seed_tokens = [
            token
            for token in seed_tokens
            if token not in self.eos_token_ids and token != EXPECTED_PAD_TOKEN_ID
        ]
        if not seed_tokens:
            raise FrozenEvalError("Tokenizer could not create cache-smoke seed tokens.")
        repeats = (length + len(seed_tokens) - 1) // len(seed_tokens)
        return (seed_tokens * repeats)[:length]

    def _compare_cache_logits(
        self,
        cached: Any,
        full: Any,
        *,
        prompt_length: int,
        decode_index: int,
    ) -> dict[str, Any]:
        if tuple(cached.shape) != tuple(full.shape):
            raise FrozenEvalError("Cache/full last-logit shapes differ.")
        if not bool(self.torch.isfinite(cached).all().item()) or not bool(
            self.torch.isfinite(full).all().item()
        ):
            raise FrozenEvalError("Cache smoke produced non-finite logits.")
        difference = (cached.float() - full.float()).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        reference_max = float(full.float().abs().max().item())
        allowed = float(self.args.cache_smoke_atol) + float(
            self.args.cache_smoke_rtol
        ) * reference_max
        cached_argmax = int(
            cached[: self.tokenizer_vocab_size].argmax().item()
        )
        full_argmax = int(full[: self.tokenizer_vocab_size].argmax().item())
        if cached_argmax != full_argmax:
            raise FrozenEvalError(
                "KV-cache greedy token differs from full recomputation at "
                f"prompt={prompt_length}, decode={decode_index}: "
                f"cache={cached_argmax}, full={full_argmax}, max_abs={max_abs}."
            )
        if max_abs > allowed:
            raise FrozenEvalError(
                "KV-cache logits exceed configured cache/full tolerance at "
                f"prompt={prompt_length}, decode={decode_index}: "
                f"max_abs={max_abs}, allowed={allowed}, reference_max={reference_max}."
            )
        return {
            "decode_index": decode_index,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "reference_max_abs": reference_max,
            "allowed_max_abs": allowed,
            "greedy_token_id": full_argmax,
        }

    def _cache_smoke_case(self, prompt_length: int, decode_tokens: int) -> dict[str, Any]:
        prompt = self._synthetic_prompt(prompt_length)
        capacity = prompt_length + decode_tokens
        if capacity > self.max_length:
            raise FrozenEvalError(
                f"Cache smoke case {prompt_length}+{decode_tokens} exceeds {self.max_length}."
            )
        context = self.cache_owner.build_inference_context(
            max_batch_size=1,
            max_sequence_length=capacity,
            materialize_only_last_token_logits=True,
        )
        context.enable_prefill_mode()
        input_ids = self.torch.tensor(
            [prompt], dtype=self.torch.long, device=self.device
        )
        offsets = [int(context.sequence_len_offset)]
        with self.torch.no_grad():
            cached_logits = self.model(
                tokens=input_ids,
                position_ids=None,
                attention_mask=None,
                inference_context=context,
                use_cache=True,
                runtime_gather_output=True,
            )
        cached_last, prefill_materialized_last_only = self._check_prefill_logits(
            cached_logits,
            prompt_length=prompt_length,
            inference_context=context,
        )
        context.sequence_len_offset += prompt_length
        context.enable_decode_mode()
        offsets.append(int(context.sequence_len_offset))
        cached_last = cached_last.float().clone()
        full_last = self._full_last_logits(prompt)

        generated: list[int] = []
        comparisons: list[dict[str, Any]] = []
        for decode_index in range(decode_tokens + 1):
            comparisons.append(
                self._compare_cache_logits(
                    cached_last,
                    full_last,
                    prompt_length=prompt_length,
                    decode_index=decode_index,
                )
            )
            if decode_index == decode_tokens:
                break
            token_id = int(
                full_last[: self.tokenizer_vocab_size].argmax().item()
            )
            generated.append(token_id)
            decode_input = self.torch.tensor(
                [[token_id]], dtype=self.torch.long, device=self.device
            )
            with self.torch.no_grad():
                cached_logits = self.model(
                    tokens=decode_input,
                    position_ids=None,
                    attention_mask=None,
                    inference_context=context,
                    use_cache=True,
                    runtime_gather_output=True,
                )
            cached_last = self._check_decode_logits(
                cached_logits,
                inference_context=context,
                expected_offset=prompt_length + decode_index,
            )
            context.sequence_len_offset += 1
            offsets.append(int(context.sequence_len_offset))
            cached_last = cached_last.float().clone()
            full_last = self._full_last_logits(prompt + generated)

        if offsets != [0, prompt_length] + [
            prompt_length + index for index in range(1, decode_tokens + 1)
        ]:
            raise FrozenEvalError(
                f"Cache smoke offset sequence is wrong: {offsets}."
            )
        repeat_generated = self._cached_greedy_tokens(prompt, decode_tokens)
        if repeat_generated != generated:
            raise FrozenEvalError(
                f"KV-cache greedy decode is non-deterministic for length {prompt_length}."
            )
        return {
            "prompt_length": prompt_length,
            "decode_tokens": decode_tokens,
            "generated_token_ids": generated,
            "offsets": offsets,
            "comparisons": comparisons,
            "deterministic_repeat": True,
            "prefill_materialized_last_token_logits": (
                prefill_materialized_last_only
            ),
        }

    def _cached_greedy_tokens(self, prompt: list[int], count: int) -> list[int]:
        context = self.cache_owner.build_inference_context(
            max_batch_size=1,
            max_sequence_length=len(prompt) + count,
            materialize_only_last_token_logits=True,
        )
        context.enable_prefill_mode()
        input_ids = self.torch.tensor(
            [prompt], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            logits = self.model(
                tokens=input_ids,
                position_ids=None,
                attention_mask=None,
                inference_context=context,
                use_cache=True,
                runtime_gather_output=True,
            )
        last, _ = self._check_prefill_logits(
            logits,
            prompt_length=len(prompt),
            inference_context=context,
        )
        context.sequence_len_offset += len(prompt)
        context.enable_decode_mode()
        output: list[int] = []
        for index in range(count):
            token_id = int(last[: self.tokenizer_vocab_size].argmax().item())
            output.append(token_id)
            decode_input = self.torch.tensor(
                [[token_id]], dtype=self.torch.long, device=self.device
            )
            with self.torch.no_grad():
                logits = self.model(
                    tokens=decode_input,
                    position_ids=None,
                    attention_mask=None,
                    inference_context=context,
                    use_cache=True,
                    runtime_gather_output=True,
                )
            last = self._check_decode_logits(
                logits,
                inference_context=context,
                expected_offset=len(prompt) + index,
            )
            context.sequence_len_offset += 1
        return output

    def run_cache_smoke(self) -> dict[str, Any]:
        lengths = _parse_smoke_lengths(
            self.args.cache_smoke_prompt_lengths,
            max_length=self.max_length,
        )
        decode_tokens = int(self.args.cache_smoke_decode_tokens)
        local = {
            "rank": self.rank,
            "cases": [
                self._cache_smoke_case(length, decode_tokens) for length in lengths
            ],
        }
        gathered: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(
            gathered, local, group=self.dp_gloo_group
        )
        reference = [
            case["generated_token_ids"] for case in gathered[0]["cases"]
        ]
        for rank_payload in gathered:
            rank_tokens = [
                case["generated_token_ids"] for case in rank_payload["cases"]
            ]
            if rank_tokens != reference:
                raise FrozenEvalError(
                    "Cache-smoke greedy tokens differ across DP ranks: "
                    f"rank {rank_payload['rank']}."
                )
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_step": int(self.args.expected_checkpoint_step),
            "world_size": self.world_size,
            "tensor_parallel_size": self.tp_world_size,
            "prompt_lengths": lengths,
            "decode_tokens": decode_tokens,
            "eos_token_ids": sorted(self.eos_token_ids),
            "atol": float(self.args.cache_smoke_atol),
            "rtol": float(self.args.cache_smoke_rtol),
            "rank_results": gathered,
            "cross_rank_greedy_consistent": True,
            "completed_at": _utc_now(),
        }

    def run_long_cache_boundary_smoke(self) -> dict[str, Any]:
        """Fill the Stage-3 cache to 65K without quadratic full recomputation."""

        if self.max_length != OLMO3_LONG_CONTEXT_LENGTH:
            raise FrozenEvalError(
                "long-cache-smoke requires --eval-max-length=65536."
            )
        decode_tokens = int(self.args.cache_smoke_decode_tokens)
        if not 1 <= decode_tokens < self.max_length:
            raise FrozenEvalError(
                "long-cache-smoke decode token count must be inside the model window."
            )
        prompt_length = self.max_length - decode_tokens
        prompt = self._synthetic_prompt(prompt_length)
        context = self.cache_owner.build_inference_context(
            max_batch_size=1,
            max_sequence_length=self.max_length,
            materialize_only_last_token_logits=True,
        )
        context.enable_prefill_mode()
        checkpoint_keys_before = tuple(self.cache_owner.state_dict())
        input_ids = self.torch.tensor(
            [prompt], dtype=self.torch.long, device=self.device
        )
        started = time.monotonic()
        with self.torch.no_grad():
            logits = self.model(
                tokens=input_ids,
                position_ids=None,
                attention_mask=None,
                inference_context=context,
                use_cache=True,
                runtime_gather_output=True,
            )
        last_logits, materialized_last_only = self._check_prefill_logits(
            logits,
            prompt_length=prompt_length,
            inference_context=context,
        )
        if not materialized_last_only:
            raise FrozenEvalError(
                "65K prefill projected every token instead of only the final token."
            )
        prefill_shape = [int(value) for value in logits.shape]
        context.sequence_len_offset += prompt_length
        context.enable_decode_mode()

        generated: list[int] = []
        decode_shapes: list[list[int]] = []
        for decode_index in range(decode_tokens):
            token_id = int(
                last_logits[: self.tokenizer_vocab_size].argmax().item()
            )
            generated.append(token_id)
            decode_input = self.torch.tensor(
                [[token_id]], dtype=self.torch.long, device=self.device
            )
            with self.torch.no_grad():
                logits = self.model(
                    tokens=decode_input,
                    position_ids=None,
                    attention_mask=None,
                    inference_context=context,
                    use_cache=True,
                    runtime_gather_output=True,
                )
            last_logits = self._check_decode_logits(
                logits,
                inference_context=context,
                expected_offset=prompt_length + decode_index,
            )
            decode_shapes.append([int(value) for value in logits.shape])
            context.sequence_len_offset += 1

        if int(context.sequence_len_offset) != self.max_length:
            raise FrozenEvalError(
                "Long cache did not finish at the 65K boundary: "
                f"{context.sequence_len_offset} != {self.max_length}."
            )
        try:
            self.model(
                tokens=self.torch.tensor(
                    [[generated[-1]]],
                    dtype=self.torch.long,
                    device=self.device,
                ),
                position_ids=None,
                attention_mask=None,
                inference_context=context,
                use_cache=True,
                runtime_gather_output=True,
            )
        except ValueError as error:
            if "exceeds max_sequence_length" not in str(error):
                raise
        else:
            raise FrozenEvalError("KV cache accepted token 65,537.")

        checkpoint_keys_after = tuple(self.cache_owner.state_dict())
        if checkpoint_keys_before != checkpoint_keys_after:
            raise FrozenEvalError("KV-cache state changed checkpoint keys.")
        cache_bytes = 0
        cache_shapes: dict[str, list[list[int]]] = {}
        for layer_number, key_value in sorted(
            context.key_value_memory_dict.items()
        ):
            layer_shapes = []
            for tensor in key_value:
                cache_bytes += tensor.numel() * tensor.element_size()
                layer_shapes.append([int(value) for value in tensor.shape])
            cache_shapes[str(layer_number)] = layer_shapes

        local = {
            "dp_rank": self.rank,
            "prompt_length": prompt_length,
            "decode_tokens": decode_tokens,
            "final_offset": int(context.sequence_len_offset),
            "generated_token_ids": generated,
            "prefill_logits_shape": prefill_shape,
            "decode_logits_shapes": decode_shapes,
            "cache_bytes_per_tp_rank": cache_bytes,
            "cache_shapes": cache_shapes,
            "checkpoint_keys_invariant": True,
            "overflow_token_rejected": True,
            "prefill_materialized_last_token_logits": True,
            "wall_seconds": time.monotonic() - started,
        }
        gathered: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(
            gathered,
            local,
            group=self.dp_gloo_group,
        )
        reference_tokens = gathered[0]["generated_token_ids"]
        if any(
            payload["generated_token_ids"] != reference_tokens
            for payload in gathered
        ):
            raise FrozenEvalError(
                "Long-cache generated tokens differ across DP replicas."
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_step": int(self.args.expected_checkpoint_step),
            "max_sequence_length": self.max_length,
            "tensor_parallel_size": self.tp_world_size,
            "data_parallel_size": self.world_size,
            "rank_results": gathered,
            "cross_rank_greedy_consistent": True,
            "completed_at": _utc_now(),
        }


def _parse_smoke_lengths(
    value: str,
    *,
    max_length: int = MAX_SUPPORTED_CONTEXT_LENGTH,
) -> list[int]:
    max_length = _strict_int(max_length, label="max_length", minimum=1)
    if max_length > MAX_SUPPORTED_CONTEXT_LENGTH:
        raise FrozenEvalError(
            "Cache smoke max_length exceeds the implemented OLMo3 context: "
            f"{max_length} > {MAX_SUPPORTED_CONTEXT_LENGTH}."
        )
    try:
        lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise FrozenEvalError(f"Invalid cache smoke lengths: {value!r}.") from error
    if not lengths or len(set(lengths)) != len(lengths):
        raise FrozenEvalError("Cache smoke lengths must be non-empty and unique.")
    if any(length < 1 or length > max_length for length in lengths):
        raise FrozenEvalError(
            f"Cache smoke prompt lengths are outside 1..{max_length}."
        )
    return lengths


def _add_runner_arguments(
    parser: argparse.ArgumentParser, pretrain_extra_args_provider: Any
) -> argparse.ArgumentParser:
    parser = pretrain_extra_args_provider(parser)
    group = parser.add_argument_group("frozen OLMo3 evaluation")
    group.add_argument("--frozen-requests", type=Path)
    group.add_argument("--eval-output-dir", type=Path, required=True)
    group.add_argument(
        "--eval-mode",
        choices=("run", "smoke", "cache-smoke", "long-cache-smoke"),
        default="run",
    )
    group.add_argument("--run-cache-smoke-first", action="store_true")
    group.add_argument("--expected-checkpoint-step", type=int, required=True)
    group.add_argument("--expected-world-size", type=int, default=0)
    group.add_argument("--expected-requests-sha256")
    group.add_argument(
        "--eval-max-length",
        type=int,
        default=OLMO3_BASE_CONTEXT_LENGTH,
        help=(
            "Maximum prompt-plus-generation tokens. Stage 1/2 use 8192; "
            "Stage 3 may use up to 65536 with the long-context model config."
        ),
    )
    group.add_argument(
        "--expected-padded-vocab-size", type=int, default=EXPECTED_PADDED_VOCAB_SIZE
    )
    group.add_argument(
        "--expected-tokenizer-vocab-size",
        type=int,
        default=EXPECTED_TOKENIZER_VOCAB_SIZE,
    )
    group.add_argument(
        "--eval-default-max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKENS
    )
    group.add_argument("--partition-index", type=int, default=0)
    group.add_argument("--partition-count", type=int, default=1)
    group.add_argument(
        "--partition-strategy",
        choices=("modulo", "contiguous"),
        default="modulo",
    )
    group.add_argument(
        "--suite-alias",
        action="append",
        default=[],
        help="Exact suite_alias filter; repeat to select multiple suites.",
    )
    group.add_argument(
        "--task-alias",
        action="append",
        default=[],
        help="Exact task_alias filter; repeat to select multiple tasks.",
    )
    group.add_argument("--eval-resume", action="store_true")
    group.add_argument("--eval-fsync-interval", type=int, default=25)
    group.add_argument("--eval-log-interval", type=int, default=25)
    group.add_argument(
        "--cache-smoke-prompt-lengths", default="7,4095,4096,4097"
    )
    group.add_argument("--cache-smoke-decode-tokens", type=int, default=4)
    group.add_argument("--cache-smoke-atol", type=float, default=0.5)
    group.add_argument("--cache-smoke-rtol", type=float, default=0.02)
    return parser


def _validate_runtime_args(args: argparse.Namespace, dist: Any, parallel_state: Any) -> dict[str, int]:
    actual_world = int(dist.get_world_size())
    topology = {
        "world": actual_world,
        "tp": int(parallel_state.get_tensor_model_parallel_world_size()),
        "pp": int(parallel_state.get_pipeline_model_parallel_world_size()),
        "cp": int(parallel_state.get_context_parallel_world_size()),
        "dp": int(parallel_state.get_data_parallel_world_size()),
    }
    if (
        topology["tp"] < 1
        or topology["dp"] < 1
        or topology["pp"] != 1
        or topology["cp"] != 1
        or topology["tp"] * topology["dp"] != actual_world
    ):
        raise FrozenEvalError(
            "Frozen evaluation requires PP=CP=1 and TP*DP=world; "
            f"got {topology}."
        )
    if args.expected_world_size and actual_world != int(args.expected_world_size):
        raise FrozenEvalError(
            f"Expected world size {args.expected_world_size}, got {actual_world}."
        )
    eval_max_length = _strict_int(
        args.eval_max_length,
        label="eval_max_length",
        minimum=1,
    )
    if eval_max_length > MAX_SUPPORTED_CONTEXT_LENGTH:
        raise FrozenEvalError(
            "eval_max_length exceeds the implemented OLMo3 context: "
            f"{eval_max_length} > {MAX_SUPPORTED_CONTEXT_LENGTH}."
        )
    if int(args.model_max_position_embeddings) < eval_max_length:
        raise FrozenEvalError(
            "Model max_position_embeddings is smaller than the evaluation context."
        )
    if int(args.seq_length) < eval_max_length:
        raise FrozenEvalError(
            "Runtime seq_length is smaller than the evaluation context."
        )
    if int(args.ckpt_step) != int(args.expected_checkpoint_step):
        raise FrozenEvalError(
            f"--ckpt-step={args.ckpt_step} does not match expected step "
            f"{args.expected_checkpoint_step}."
        )
    _partition_global_indices(
        0,
        partition_index=int(args.partition_index),
        partition_count=int(args.partition_count),
        strategy=str(args.partition_strategy),
    )
    _strict_int(args.eval_fsync_interval, label="eval_fsync_interval", minimum=1)
    _strict_int(args.eval_log_interval, label="eval_log_interval", minimum=1)
    _strict_int(
        args.eval_default_max_gen_toks,
        label="eval_default_max_gen_toks",
        minimum=1,
    )
    _strict_int(
        args.cache_smoke_decode_tokens,
        label="cache_smoke_decode_tokens",
        minimum=1,
    )
    _parse_smoke_lengths(
        args.cache_smoke_prompt_lengths,
        max_length=eval_max_length,
    )
    if args.cache_smoke_atol < 0 or args.cache_smoke_rtol < 0:
        raise FrozenEvalError("Cache smoke tolerances must be non-negative.")
    return topology


def _rank_output_paths(args: argparse.Namespace, rank: int) -> tuple[Path, Path, Path]:
    partition_dir = Path(args.eval_output_dir) / (
        f"partition_{int(args.partition_index):05d}_of_{int(args.partition_count):05d}"
    )
    final_path = partition_dir / f"rank_{rank:05d}.jsonl"
    partial_path = partition_dir / f"rank_{rank:05d}.jsonl.partial"
    status_path = partition_dir / f"rank_{rank:05d}.status.json"
    return final_path, partial_path, status_path


def _read_completed_ids(path: Path) -> list[str]:
    request_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise FrozenEvalError(f"Blank line in partial output {path}:{line_number}.")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise FrozenEvalError(
                    f"Invalid partial JSON {path}:{line_number}: {error}."
                ) from error
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            if not isinstance(request_id, str):
                raise FrozenEvalError(
                    f"Partial output lacks request_id at {path}:{line_number}."
                )
            request_ids.append(request_id)
    return request_ids


def _run_requests(
    evaluator: NativeFrozenEvaluator,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    eligible_global_indices = _filtered_global_indices(
        records,
        suite_aliases=args.suite_alias,
        task_aliases=args.task_alias,
    )
    filtered_partition_indices = _partition_global_indices(
        len(eligible_global_indices),
        partition_index=int(args.partition_index),
        partition_count=int(args.partition_count),
        strategy=str(args.partition_strategy),
    )
    partition_pairs = [
        (partition_ordinal, eligible_global_indices[filtered_ordinal])
        for partition_ordinal, filtered_ordinal in enumerate(
            filtered_partition_indices
        )
    ]
    local_pairs = [
        pair
        for pair in partition_pairs
        if pair[0] % evaluator.world_size == evaluator.rank
    ]
    expected_ids = [records[global_index]["request_id"] for _, global_index in local_pairs]
    final_path, partial_path, status_path = _rank_output_paths(args, evaluator.rank)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.dist.barrier(group=evaluator.tp_group)

    completed = 0
    if final_path.exists():
        if not args.eval_resume:
            raise FrozenEvalError(f"Rank output already exists: {final_path}")
        completed_ids = _read_completed_ids(final_path)
        if completed_ids != expected_ids:
            raise FrozenEvalError(
                f"Existing final output does not match rank {evaluator.rank}'s frozen shard."
            )
        completed = len(completed_ids)
    else:
        if partial_path.exists():
            if not args.eval_resume:
                raise FrozenEvalError(
                    f"Partial output exists; pass --eval-resume to continue: {partial_path}"
                )
            completed_ids = _read_completed_ids(partial_path)
            if completed_ids != expected_ids[: len(completed_ids)]:
                raise FrozenEvalError(
                    f"Partial output is not a prefix of rank {evaluator.rank}'s shard."
                )
            completed = len(completed_ids)
        mode = "a" if completed else "x"
        started = time.monotonic()
        handle = (
            partial_path.open(mode, encoding="utf-8")
            if evaluator.is_replica_writer
            else None
        )
        try:
            for local_position in range(completed, len(local_pairs)):
                partition_ordinal, global_index = local_pairs[local_position]
                result = evaluator.evaluate_record(
                    records[global_index],
                    global_index=global_index,
                    partition_ordinal=partition_ordinal,
                )
                if handle is not None:
                    handle.write(_json_line(result))
                completed = local_position + 1
                if (
                    handle is not None
                    and completed % int(args.eval_fsync_interval) == 0
                ):
                    handle.flush()
                    os.fsync(handle.fileno())
                if evaluator.is_replica_writer and (
                    completed % int(args.eval_log_interval) == 0
                    or completed == len(local_pairs)
                ):
                    print(
                        "FROZEN_EVAL_PROGRESS "
                        f"rank={evaluator.rank} completed={completed}/{len(local_pairs)} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if handle is not None:
                handle.close()
        if evaluator.is_replica_writer:
            os.replace(partial_path, final_path)

    # Only TP rank zero owns files, but all TP ranks execute identical model
    # collectives. Do not inspect or publish the completed output until its
    # writer has atomically renamed it.
    evaluator.dist.barrier(group=evaluator.tp_group)

    status = {
        "schema_version": SCHEMA_VERSION,
        "rank": evaluator.rank,
        "world_size": evaluator.world_size,
        "partition_index": int(args.partition_index),
        "partition_count": int(args.partition_count),
        "partition_strategy": str(args.partition_strategy),
        "suite_aliases": list(args.suite_alias),
        "task_aliases": list(args.task_alias),
        "records": completed,
        "first_request_id": expected_ids[0] if expected_ids else None,
        "last_request_id": expected_ids[-1] if expected_ids else None,
        "output": str(final_path),
        "output_sha256": _sha256(final_path),
        "completed_at": _utc_now(),
    }
    if evaluator.is_replica_writer:
        _atomic_write_json(status_path, status)
    return status


def _runtime_imports() -> dict[str, Any]:
    # Do not move another project import above this import.  It installs the
    # MindSpeed/torch_npu compatibility patches required by Megatron.
    from pretrain_olmo3_mindspeed import (  # noqa: PLC0415
        extra_args_provider as pretrain_extra_args_provider,
        model_provider,
    )

    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415
    from megatron.core import parallel_state  # noqa: PLC0415
    from megatron.core.enums import ModelType  # noqa: PLC0415
    from megatron.training import get_args, get_tokenizer  # noqa: PLC0415
    from megatron.training.checkpointing import load_checkpoint  # noqa: PLC0415
    from megatron.training.initialize import initialize_megatron  # noqa: PLC0415
    from megatron.training.training import get_model  # noqa: PLC0415

    return {
        "ModelType": ModelType,
        "dist": dist,
        "get_args": get_args,
        "get_model": get_model,
        "get_tokenizer": get_tokenizer,
        "initialize_megatron": initialize_megatron,
        "load_checkpoint": load_checkpoint,
        "model_provider": model_provider,
        "parallel_state": parallel_state,
        "pretrain_extra_args_provider": pretrain_extra_args_provider,
        "torch": torch,
    }


def _run_self_tests() -> None:
    assert _find_earliest_stop("xxENDyySTOP", ["STOP", "END"]) == (2, "END")
    assert _find_earliest_stop("abc", ["x"]) is None
    assert _parse_smoke_lengths(
        "7,4096,65534",
        max_length=OLMO3_LONG_CONTEXT_LENGTH,
    ) == [7, 4096, 65534]
    try:
        _parse_smoke_lengths(
            "65537",
            max_length=OLMO3_LONG_CONTEXT_LENGTH,
        )
    except FrozenEvalError as error:
        assert "outside 1..65536" in str(error)
    else:
        raise AssertionError("65,537-token cache smoke length was not rejected")
    assert _partition_global_indices(
        10, partition_index=1, partition_count=3, strategy="modulo"
    ) == [1, 4, 7]
    assert _partition_global_indices(
        10, partition_index=1, partition_count=3, strategy="contiguous"
    ) == [3, 4, 5]
    assert _filtered_global_indices(
        [
            {"suite_alias": "a", "task_alias": "x"},
            {"suite_alias": "b", "task_alias": "x"},
            {"suite_alias": "a", "task_alias": "y"},
        ],
        suite_aliases=["a"],
        task_aliases=["y"],
    ) == [2]
    settings = _normalize_generation_settings(
        {
            "stop_sequences": ["END", "END"],
            "generation_kwargs": {
                "until": ["STOP"],
                "max_gen_toks": 17,
                "do_sample": False,
                "temperature": 0.0,
                "truncate_context": False,
            },
        }
    )
    assert settings.max_gen_toks == 17
    assert settings.stop_sequences == ("END", "STOP")
    assert not settings.truncate_context
    assert not settings.do_sample
    sampled = _normalize_generation_settings(
        {
            "generation_kwargs": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 2048,
            }
        }
    )
    assert sampled.do_sample and sampled.max_gen_toks == 2048
    assert sampled.temperature == 0.7 and sampled.top_p == 0.95
    assert _stable_request_seed("abc") == _stable_request_seed("abc")
    assert _stateless_uniform(42, 3) == _stateless_uniform(42, 3)
    assert 0.0 <= _stateless_uniform(42, 3) < 1.0
    assert _stop_search_window_tokens(["END"]) == 64
    assert _prefill_logits_sequence_contract(1, 7)
    assert not _prefill_logits_sequence_contract(7, 7)
    try:
        _prefill_logits_sequence_contract(6, 7)
    except FrozenEvalError:
        pass
    else:
        raise AssertionError("invalid prefill materialization was accepted")
    fake_vocab = {"Q": [1], " A": [3], "Q A": [1, 3]}
    fake_pair = _encode_pair_contract(
        lambda text: fake_vocab[text],
        prefix_token_id=9,
        context="Q ",
        continuation="A",
    )
    assert fake_pair == ([1], [3])
    good = {
        "request_id": "task#doc=0#idx=0",
        "request_type": "loglikelihood",
        "request": {"context": "Q", "continuation": " A"},
    }
    assert _request_payload(good, line_number=1)[0] == "loglikelihood"
    combined = {
        "request_id": "task#doc=1#idx=0",
        "request_type": "generate_until_and_loglikelihood",
        "request": {
            "context": "Q",
            "perplexity_context": "Q",
            "continuation": " A",
            "stop_sequences": [],
            "generation_kwargs": {
                "max_gen_toks": 4,
                "do_sample": False,
            },
        },
    }
    assert (
        _request_payload(combined, line_number=2)[0]
        == "generate_until_and_loglikelihood"
    )
    print("FROZEN_EVAL_SELF_TEST_OK", flush=True)


def _run_tokenizer_contract_test(tokenizer_path: Path) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    cases = [
        ("Question: choose the correct option. ", "A"),
        ("<|im_start|>assistant\n", " A"),
        ("A factual question:\t", "true"),
    ]
    for context, continuation in cases:
        context_tokens, continuation_tokens = _encode_pair_contract(
            encode,
            prefix_token_id=EXPECTED_EOT_TOKEN_ID,
            context=context,
            continuation=continuation,
        )
        whole = [int(token) for token in encode(context + continuation)]
        if context_tokens + continuation_tokens != whole:
            raise AssertionError(
                f"Tokenizer pair contract failed for {context!r} + {continuation!r}."
            )
    if tokenizer.convert_tokens_to_ids(EXPECTED_IM_END_TOKEN) != EXPECTED_IM_END_TOKEN_ID:
        raise AssertionError("Tokenizer has the wrong <|im_end|> ID.")
    if len(tokenizer) != EXPECTED_TOKENIZER_VOCAB_SIZE:
        raise AssertionError(f"Unexpected tokenizer size: {len(tokenizer)}")
    print(
        f"FROZEN_EVAL_TOKENIZER_CONTRACT_OK path={tokenizer_path}",
        flush=True,
    )


def main() -> None:
    if "--self-test" in sys.argv:
        if sys.argv != [sys.argv[0], "--self-test"]:
            raise SystemExit("--self-test cannot be combined with runtime arguments")
        _run_self_tests()
        return
    if "--self-test-tokenizer" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--self-test-tokenizer", type=Path, required=True)
        test_args = parser.parse_args()
        _run_tokenizer_contract_test(test_args.self_test_tokenizer)
        return

    runtime = _runtime_imports()
    initialize_megatron = runtime["initialize_megatron"]
    pretrain_extra_args_provider = runtime["pretrain_extra_args_provider"]
    initialize_megatron(
        extra_args_provider=lambda parser: _add_runner_arguments(
            parser, pretrain_extra_args_provider
        ),
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
    args = runtime["get_args"]()
    topology = _validate_runtime_args(
        args, runtime["dist"], runtime["parallel_state"]
    )
    rank = int(runtime["dist"].get_rank())
    Path(args.eval_output_dir).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] | None = None
    requests_sha256: str | None = None
    if args.eval_mode == "run":
        if args.frozen_requests is None:
            raise FrozenEvalError("--frozen-requests is required in run mode.")
        records = _load_frozen_records(Path(args.frozen_requests))
        requests_sha256 = _sha256(Path(args.frozen_requests))
        if (
            args.expected_requests_sha256
            and requests_sha256 != args.expected_requests_sha256
        ):
            raise FrozenEvalError(
                "Frozen request SHA256 mismatch: "
                f"expected {args.expected_requests_sha256}, got {requests_sha256}."
            )

    model_list = runtime["get_model"](
        runtime["model_provider"],
        runtime["ModelType"].encoder_or_decoder,
        wrap_with_ddp=False,
    )
    iteration, checkpoint_flops = runtime["load_checkpoint"](
        model_list, None, None, strict=True
    )
    if int(iteration) != int(args.expected_checkpoint_step):
        raise FrozenEvalError(
            f"Loaded checkpoint iteration {iteration}, expected "
            f"{args.expected_checkpoint_step}."
        )
    if len(model_list) != 1:
        raise FrozenEvalError(f"Expected one local model chunk, got {len(model_list)}.")
    model = model_list[0]
    model.eval()
    evaluator = NativeFrozenEvaluator(
        model, runtime["get_tokenizer"](), args, runtime
    )

    mode = "cache-smoke" if args.eval_mode == "smoke" else args.eval_mode
    if mode in ("cache-smoke", "long-cache-smoke") or args.run_cache_smoke_first:
        if mode == "long-cache-smoke":
            smoke_result = evaluator.run_long_cache_boundary_smoke()
            smoke_filename = "long_cache_smoke.json"
            smoke_marker = "LONG_CACHE_SMOKE_COMPLETE"
            smoke_signature = "OLMO3_FROZEN_LONG_CACHE_SMOKE_COMPLETE"
        else:
            smoke_result = evaluator.run_cache_smoke()
            smoke_filename = "cache_smoke.json"
            smoke_marker = "CACHE_SMOKE_COMPLETE"
            smoke_signature = "OLMO3_FROZEN_CACHE_SMOKE_COMPLETE"
        if rank == 0:
            _atomic_write_json(
                Path(args.eval_output_dir) / smoke_filename, smoke_result
            )
            _atomic_write_text(
                Path(args.eval_output_dir) / smoke_marker,
                _utc_now() + "\n",
            )
            print(
                f"{smoke_signature} "
                f"checkpoint_step={iteration} world={topology['world']}",
                flush=True,
            )
        runtime["dist"].barrier(group=evaluator.dp_group)

    if mode == "run":
        assert records is not None
        status = _run_requests(evaluator, records, args)
        gathered_status: list[Any] = [None] * evaluator.world_size
        runtime["dist"].all_gather_object(
            gathered_status, status, group=evaluator.dp_gloo_group
        )
        if rank == 0:
            eligible_global_indices = _filtered_global_indices(
                records,
                suite_aliases=args.suite_alias,
                task_aliases=args.task_alias,
            )
            partition_size = len(
                _partition_global_indices(
                    len(eligible_global_indices),
                    partition_index=int(args.partition_index),
                    partition_count=int(args.partition_count),
                    strategy=str(args.partition_strategy),
                )
            )
            actual_records = sum(int(item["records"]) for item in gathered_status)
            if actual_records != partition_size:
                raise FrozenEvalError(
                    f"Rank outputs contain {actual_records} records, expected {partition_size}."
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "completed_at": _utc_now(),
                "checkpoint": {
                    "root": str(args.load),
                    "iteration": int(iteration),
                    "num_floating_point_operations_so_far": checkpoint_flops,
                },
                "frozen_requests": {
                    "path": str(args.frozen_requests),
                    "sha256": requests_sha256,
                    "total_records": len(records),
                },
                "partition": {
                    "index": int(args.partition_index),
                    "count": int(args.partition_count),
                    "strategy": str(args.partition_strategy),
                    "records": partition_size,
                    "eligible_records_before_partition": len(
                        eligible_global_indices
                    ),
                    "suite_aliases": list(args.suite_alias),
                    "task_aliases": list(args.task_alias),
                },
                "topology": topology,
                "model": {
                    "implementation": str(args.model_impl),
                    "max_sequence_length": int(args.eval_max_length),
                    "padded_vocab_size": int(args.expected_padded_vocab_size),
                    "tokenizer_vocab_size": int(args.expected_tokenizer_vocab_size),
                    "runtime_gather_output": True,
                    "generation_use_cache": True,
                    "generation": "greedy_or_request_seeded_top_k_top_p",
                    "eos_token_ids": list(EXPECTED_EOS_TOKEN_IDS),
                },
                "rank_outputs": gathered_status,
                "command": sys.argv,
            }
            partition_dir = Path(gathered_status[0]["output"]).parent
            _atomic_write_json(partition_dir / "run_manifest.json", manifest)
            _atomic_write_text(partition_dir / "COMPLETE", _utc_now() + "\n")
            print(
                "OLMO3_FROZEN_EVAL_COMPLETE "
                f"checkpoint_step={iteration} partition={args.partition_index}/"
                f"{args.partition_count} records={partition_size}",
                flush=True,
            )
        runtime["dist"].barrier(group=evaluator.dp_group)

    runtime["parallel_state"].destroy_model_parallel()
    runtime["dist"].destroy_process_group()


if __name__ == "__main__":
    main()
