"""Immutable native-inference and PPL evaluation plans for OLMo 3.

The control plane consumes a frozen training ``resolved.json`` and the native
Megatron ``torch_dist`` checkpoint root.  It does not guess model geometry,
variant, rotary configuration, checkpoint iteration, or tokenizer location.
Generated launch plans contain no training-data dependency and can be moved
between clusters by rendering them again with different explicit paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoint import (
    architecture_identity,
    inspect_checkpoint,
    load_contract,
)
from .config import ConfigurationError, PROJECT_ROOT
from .model_config import build_model_config
from .resolve import resolve


SCHEMA = "olmo3.mindspeed.evaluation-plan/v1"
SUPPORTED_STAGES = frozenset(
    {"stage1", "stage2", "stage3", "sft_think", "sft_instruct"}
)
SUPPORTED_SIZES = frozenset({"1b", "3b", "7b"})
SUPPORTED_VARIANTS = frozenset({"base", "siamese_depth"})
INFERENCE_MODES = frozenset({"run", "cache-smoke", "long-cache-smoke"})


def _atomic_text(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            os.chmod(temporary, 0o755)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must contain a JSON object: {path}")
    return value


def _load_source_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, "resolved training config")
    for key in (
        "model",
        "variant",
        "stage",
        "optimization",
        "training",
        "topology",
        "runtime",
    ):
        if not isinstance(config.get(key), dict):
            raise ConfigurationError(
                f"resolved training config has no {key!r} object: {path}"
            )
    model = config["model"]
    variant = config["variant"]
    stage = config["stage"]
    if model.get("size") not in SUPPORTED_SIZES:
        raise ConfigurationError(
            f"evaluation supports model sizes {sorted(SUPPORTED_SIZES)}, "
            f"got {model.get('size')!r}"
        )
    if variant.get("name") not in SUPPORTED_VARIANTS:
        raise ConfigurationError(
            f"evaluation supports variants {sorted(SUPPORTED_VARIANTS)}, "
            f"got {variant.get('name')!r}"
        )
    if stage.get("name") not in SUPPORTED_STAGES:
        raise ConfigurationError(
            f"evaluation supports stages {sorted(SUPPORTED_STAGES)}, "
            f"got {stage.get('name')!r}"
        )
    if int(stage.get("model_max_sequence_length", 0)) not in (8192, 65536):
        raise ConfigurationError(
            "resolved model_max_sequence_length must be 8192 or 65536"
        )
    # Evaluation must not trust a hand-edited resolved file. Re-run the same
    # fail-closed model, stage, topology, optimizer, and data contracts used by
    # training workers before deriving any numerical inference configuration.
    return resolve(config)


def _select_iteration(
    checkpoint_root: Path,
    requested_iteration: int | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    inspection = inspect_checkpoint(checkpoint_root)
    if not inspection["exists"]:
        raise ConfigurationError(
            f"checkpoint root does not exist: {checkpoint_root}"
        )
    if inspection["contract_error"]:
        raise ConfigurationError(
            f"checkpoint identity contract is invalid: "
            f"{inspection['contract_error']}"
        )
    if inspection["contract"] is None:
        raise ConfigurationError(
            "checkpoint identity contract is missing; evaluation refuses to "
            "guess model size or variant"
        )
    iteration = (
        int(inspection["tracker_iteration"])
        if requested_iteration is None
        and inspection["tracker_iteration"] is not None
        else requested_iteration
    )
    if iteration is None:
        raise ConfigurationError(
            "checkpoint has no latest iteration tracker; pass --iteration"
        )
    if isinstance(iteration, bool) or int(iteration) < 0:
        raise ConfigurationError("checkpoint iteration must be non-negative")
    selected = next(
        (
            item
            for item in inspection["iterations"]
            if int(item["iteration"]) == int(iteration)
        ),
        None,
    )
    if selected is None:
        raise ConfigurationError(
            f"checkpoint iteration {iteration} is absent from {checkpoint_root}"
        )
    if not selected["complete"]:
        raise ConfigurationError(
            f"checkpoint iteration {iteration} is incomplete: "
            + "; ".join(selected["integrity_errors"])
        )
    return int(iteration), selected, inspection


def _verify_checkpoint_identity(
    config: Mapping[str, Any],
    checkpoint_root: Path,
    contract: Mapping[str, Any],
) -> None:
    expected = architecture_identity(config)
    actual = contract.get("architecture")
    if actual != expected:
        fields = sorted(set(expected) | set(actual or {}))
        mismatches = [
            f"{field}: resolved={expected.get(field)!r}, "
            f"checkpoint={(actual or {}).get(field)!r}"
            for field in fields
            if expected.get(field) != (actual or {}).get(field)
        ]
        raise ConfigurationError(
            "resolved model and checkpoint architecture differ:\n  "
            + "\n  ".join(mismatches)
        )
    source_stage = str(config["stage"]["name"])
    writer_stage = contract.get("writer", {}).get("stage")
    if writer_stage != source_stage:
        raise ConfigurationError(
            "resolved config must describe the checkpoint writer stage so "
            "the 8K/65K rotary contract cannot be selected incorrectly: "
            f"resolved={source_stage!r}, checkpoint={writer_stage!r}, "
            f"root={checkpoint_root}"
        )


def _validate_tokenizer(path: Path) -> Path:
    path = path.expanduser().resolve()
    required = ("tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ConfigurationError(
            f"tokenizer directory {path} is missing {missing}"
        )
    return path


def _validate_parallel(
    config: Mapping[str, Any],
    *,
    nnodes: int,
    nproc_per_node: int,
    tensor_parallel: int,
) -> dict[str, int]:
    values = {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "tensor_parallel": tensor_parallel,
    }
    for label, value in values.items():
        if isinstance(value, bool) or value < 1:
            raise ConfigurationError(f"{label} must be a positive integer")
    world_size = nnodes * nproc_per_node
    if world_size % tensor_parallel:
        raise ConfigurationError(
            f"tensor_parallel={tensor_parallel} must divide world_size={world_size}"
        )
    model = config["model"]
    divisibility = {
        "num_attention_heads": int(model["num_attention_heads"]),
        "num_query_groups": int(model["num_query_groups"]),
        "hidden_size": int(model["hidden_size"]),
        "ffn_hidden_size": int(model["ffn_hidden_size"]),
        "padded_vocab_size": int(model["padded_vocab_size"]),
    }
    errors = [
        f"{name}={value} is not divisible by TP={tensor_parallel}"
        for name, value in divisibility.items()
        if value % tensor_parallel
    ]
    if errors:
        raise ConfigurationError(
            "invalid native inference TP topology:\n  " + "\n  ".join(errors)
        )
    return {
        "world_size": world_size,
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": 1,
        "context_parallel": 1,
        "data_parallel": world_size // tensor_parallel,
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
    }


def _common_model_args(
    config: Mapping[str, Any],
    *,
    model_config_path: Path,
    tokenizer: Path,
    checkpoint_root: Path,
    iteration: int,
    topology: Mapping[str, int],
    sequence_length: int,
) -> list[str]:
    model = config["model"]
    variant = config["variant"]
    stage = config["stage"]
    optim = config["optimization"]
    model_max_length = int(stage["model_max_sequence_length"])
    if sequence_length < 1 or sequence_length > model_max_length:
        raise ConfigurationError(
            f"sequence length must be in [1, {model_max_length}], "
            f"got {sequence_length}"
        )
    return [
        "--use-mcore-models",
        "--model-impl",
        str(variant["model_impl"]),
        "--model-config",
        str(model_config_path),
        "--num-layers",
        str(model["num_layers"]),
        "--hidden-size",
        str(model["hidden_size"]),
        "--ffn-hidden-size",
        str(model["ffn_hidden_size"]),
        "--num-attention-heads",
        str(model["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups",
        str(model["num_query_groups"]),
        "--kv-channels",
        str(model["kv_channels"]),
        "--seq-length",
        str(sequence_length),
        "--max-position-embeddings",
        str(model_max_length),
        "--model-max-position-embeddings",
        str(model_max_length),
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        str(model["rope_theta"]),
        "--rotary-percent",
        "1.0",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        str(model["rms_norm_eps"]),
        "--qk-layernorm",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--attention-dropout",
        str(model["attention_dropout"]),
        "--hidden-dropout",
        str(model["hidden_dropout"]),
        "--init-method-std",
        str(model["initializer_range"]),
        "--vocab-size",
        str(model["true_vocab_size"]),
        "--make-vocab-size-divisible-by",
        "128",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(topology["data_parallel"]),
        "--train-iters",
        "1",
        "--lr-decay-iters",
        "1",
        "--lr-warmup-iters",
        "0",
        "--optimizer",
        "adam",
        "--lr",
        str(optim["peak_lr"]),
        "--min-lr",
        str(optim["min_lr"]),
        "--lr-decay-style",
        str(optim["scheduler"]),
        "--weight-decay",
        str(optim["weight_decay"]),
        "--adam-beta1",
        str(optim["adam_beta1"]),
        "--adam-beta2",
        str(optim["adam_beta2"]),
        "--adam-eps",
        str(optim["adam_eps"]),
        "--clip-grad",
        str(optim["grad_clip"]),
        "--vocab-z-loss-coeff",
        str(optim["z_loss"]),
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        "flash",
        "--use-flash-attn",
        "--use-fused-rmsnorm",
        "--use-fused-swiglu",
        "--use-fused-rotary-pos-emb",
        "--tensor-model-parallel-size",
        str(topology["tensor_parallel"]),
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--context-parallel-algo",
        "ulysses_cp_algo",
        "--distributed-timeout-minutes",
        "240",
        "--dataset-backend",
        "megatron_indexed",
        "--no-create-attention-mask-in-dataloader",
        "--pad-token-id",
        "100277",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer),
        "--dataloader-type",
        "single",
        "--num-workers",
        "0",
        "--attn-implementation",
        "flash_attention_2",
        "--no-model-bf16-autocast",
        "--mcore-native",
        "--seed",
        str(config["training"]["seed"]),
        "--log-interval",
        "10",
        "--eval-iters",
        "0",
        "--eval-interval",
        "1",
        "--load",
        str(checkpoint_root),
        "--ckpt-step",
        str(iteration),
        "--ckpt-format",
        "torch_dist",
        "--dist-ckpt-strictness",
        "raise_all",
        "--no-load-optim",
        "--no-load-rng",
        "--exit-on-missing-checkpoint",
    ]


def _torchrun(
    *,
    python: Path,
    entry: Path,
    topology: Mapping[str, int],
    master_addr: str,
    master_port: int,
    arguments: Iterable[str],
) -> list[str]:
    if not master_addr:
        raise ConfigurationError("master_addr must be non-empty")
    if isinstance(master_port, bool) or not 1 <= master_port <= 65535:
        raise ConfigurationError("master_port must be in [1, 65535]")
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(topology["nnodes"]),
        "--node_rank",
        "__OLMO3_NODE_RANK__",
        "--nproc_per_node",
        str(topology["nproc_per_node"]),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        "--rdzv_backend",
        "static",
        str(entry),
        *arguments,
    ]


def _environment(output_root: Path) -> dict[str, str]:
    pythonpath = [
        PROJECT_ROOT / "third_party" / "MindSpeed",
        PROJECT_ROOT / "third_party" / "MindSpeed-LLM",
        PROJECT_ROOT,
        PROJECT_ROOT / "src",
    ]
    return {
        "PYTHONPATH": ":".join(str(path) for path in pythonpath),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "USE_TF": "FALSE",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "MODEL_USE_NULL_ATTENTION_MASK": "1",
        "MCORE_DIST_CKPT_NO_FORK": "1",
        "LM_LOSS_UPCAST": "0",
        "MEGATRON_OLMO3_WEIGHT_DECAY": "1",
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
        "OLMO3_MC2_ALL_GATHER_RECOMPUTATION": "0",
        "OLMO3_SWA_CP_MODE": "none",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OLMO3_EVAL_OUTPUT_ROOT": str(output_root),
    }


def _render_command(command: Sequence[str]) -> str:
    return " ".join(
        '"${worker_node_rank}"'
        if value == "__OLMO3_NODE_RANK__"
        else shlex.quote(value)
        for value in command
    )


def _write_plan(
    *,
    plan_dir: Path,
    plan: Mapping[str, Any],
    source_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    command: Sequence[str],
    environment: Mapping[str, str],
) -> Path:
    plan_dir = plan_dir.expanduser().resolve()
    if plan_dir.exists():
        raise ConfigurationError(
            f"immutable evaluation plan already exists: {plan_dir}"
        )
    plan_dir.mkdir(parents=True)
    _atomic_json(plan_dir / "plan.json", plan)
    _atomic_json(plan_dir / "source_resolved.json", source_config)
    _atomic_json(plan_dir / "model_config.json", model_config)
    _atomic_json(plan_dir / "command.json", list(command))
    _atomic_json(plan_dir / "environment.json", dict(environment))
    _atomic_text(
        plan_dir / "command.txt",
        shlex.join(command).replace(
            shlex.quote("__OLMO3_NODE_RANK__"), "${worker_node_rank}"
        )
        + "\n",
    )
    environment_lines = [
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(environment.items())
    ]
    _atomic_text(
        plan_dir / "environment.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(environment_lines)
        + "\n",
        executable=True,
    )
    nnodes = int(plan["topology"]["nnodes"])
    launch = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'readonly PLAN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)',
        'source "${PLAN_DIR}/environment.sh"',
        "worker_node_rank=${OLMO3_NODE_RANK:-${NODE_RANK:-0}}",
        (
            'if [[ ! "$worker_node_rank" =~ ^(0|[1-9][0-9]*)$ ]] || '
            f"(( 10#$worker_node_rank >= {nnodes} )); then"
        ),
        (
            "  printf 'invalid worker node rank %q; expected [0, "
            f"{nnodes})\\n' \"$worker_node_rank\" >&2"
        ),
        "  exit 2",
        "fi",
        "readonly worker_node_rank",
        _render_command(command),
    ]
    _atomic_text(
        plan_dir / "launch.sh",
        "\n".join(launch) + "\n",
        executable=True,
    )
    return plan_dir


def _prepare(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    Path,
    int,
    dict[str, Any],
    dict[str, int],
    Path,
    Path,
]:
    source_resolved = args.resolved.expanduser().resolve()
    config = _load_source_config(source_resolved)
    checkpoint_root = args.checkpoint.expanduser().resolve()
    iteration, selected, inspection = _select_iteration(
        checkpoint_root, args.iteration
    )
    contract = load_contract(checkpoint_root)
    _verify_checkpoint_identity(config, checkpoint_root, contract)
    topology = _validate_parallel(
        config,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        tensor_parallel=args.tensor_parallel,
    )
    tokenizer = _validate_tokenizer(args.tokenizer)
    python = args.python.expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ConfigurationError(f"Python executable is unavailable: {python}")
    checkpoint = {
        "root": str(checkpoint_root),
        "iteration": iteration,
        "iteration_directory": selected["directory"],
        "contract_sha256": inspection["contract_sha256"],
        "storage_references": selected["storage_references"],
        "distcp_files": selected["distcp_files"],
    }
    return (
        config,
        source_resolved,
        iteration,
        checkpoint,
        topology,
        tokenizer,
        python,
    )


def _cmd_inference_plan(args: argparse.Namespace) -> int:
    (
        config,
        source_resolved,
        iteration,
        checkpoint,
        topology,
        tokenizer,
        python,
    ) = _prepare(args)
    if args.mode not in INFERENCE_MODES:
        raise ConfigurationError(
            f"unsupported inference mode {args.mode!r}"
        )
    if (
        isinstance(args.partition_count, bool)
        or args.partition_count < 1
        or isinstance(args.partition_index, bool)
        or args.partition_index < 0
        or args.partition_index >= args.partition_count
    ):
        raise ConfigurationError(
            "partition index/count must satisfy "
            "0 <= partition_index < partition_count"
        )
    if args.mode != "run" and (
        args.partition_index != 0
        or args.partition_count != 1
        or args.resume
        or args.run_cache_smoke_first
        or args.suite_alias
        or args.task_alias
    ):
        raise ConfigurationError(
            "cache-smoke modes require partition 0/1 and do not accept "
            "resume, filters, or --run-cache-smoke-first"
        )
    model_max_length = int(config["stage"]["model_max_sequence_length"])
    eval_max_length = args.max_length or model_max_length
    if eval_max_length < 1 or eval_max_length > model_max_length:
        raise ConfigurationError(
            f"--max-length must be in [1, {model_max_length}]"
        )
    if args.mode == "long-cache-smoke" and eval_max_length != 65536:
        raise ConfigurationError(
            "long-cache-smoke requires a Stage-3/4 65536-token config "
            "and --max-length=65536"
        )
    frozen_requests: Path | None = None
    requests_sha256: str | None = None
    if args.mode == "run":
        if args.frozen_requests is None:
            raise ConfigurationError(
                "--frozen-requests is required for inference mode 'run'"
            )
        frozen_requests = args.frozen_requests.expanduser().resolve()
        if not frozen_requests.is_file() or frozen_requests.stat().st_size == 0:
            raise ConfigurationError(
                f"frozen request JSONL is unavailable or empty: "
                f"{frozen_requests}"
            )
        requests_sha256 = _sha256(frozen_requests)
    elif args.frozen_requests is not None:
        raise ConfigurationError(
            "--frozen-requests is accepted only in inference mode 'run'"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ConfigurationError(
            f"immutable inference output identity already exists: {output}"
        )
    plan_dir = args.plan_dir.expanduser().resolve()
    model_config = build_model_config(config)
    model_config_path = plan_dir / "model_config.json"
    common = _common_model_args(
        config,
        model_config_path=model_config_path,
        tokenizer=tokenizer,
        checkpoint_root=Path(checkpoint["root"]),
        iteration=iteration,
        topology=topology,
        sequence_length=eval_max_length,
    )
    runner_args = [
        "--eval-output-dir",
        str(output),
        "--eval-mode",
        args.mode,
        "--expected-checkpoint-step",
        str(iteration),
        "--expected-world-size",
        str(topology["world_size"]),
        "--eval-max-length",
        str(eval_max_length),
        "--expected-padded-vocab-size",
        str(config["model"]["padded_vocab_size"]),
        "--expected-tokenizer-vocab-size",
        str(config["model"]["true_vocab_size"]),
        "--partition-index",
        str(args.partition_index),
        "--partition-count",
        str(args.partition_count),
        "--partition-strategy",
        args.partition_strategy,
        "--cache-smoke-prompt-lengths",
        args.cache_smoke_prompt_lengths,
        "--cache-smoke-decode-tokens",
        str(args.cache_smoke_decode_tokens),
        "--cache-smoke-atol",
        str(args.cache_smoke_atol),
        "--cache-smoke-rtol",
        str(args.cache_smoke_rtol),
    ]
    if args.resume:
        runner_args.append("--eval-resume")
    if args.run_cache_smoke_first:
        runner_args.append("--run-cache-smoke-first")
    for alias in args.suite_alias:
        runner_args.extend(["--suite-alias", alias])
    for alias in args.task_alias:
        runner_args.extend(["--task-alias", alias])
    if frozen_requests is not None:
        runner_args.extend(
            [
                "--frozen-requests",
                str(frozen_requests),
                "--expected-requests-sha256",
                str(requests_sha256),
            ]
        )
    command = _torchrun(
        python=python,
        entry=PROJECT_ROOT / "scripts" / "inference" / "olmo3_native.py",
        topology=topology,
        master_addr=args.master_addr,
        master_port=args.master_port,
        arguments=[*common, *runner_args],
    )
    environment = _environment(output)
    plan = {
        "schema": SCHEMA,
        "kind": "native-inference",
        "source_resolved": str(source_resolved),
        "source_resolved_sha256": _sha256(source_resolved),
        "checkpoint": checkpoint,
        "model": {
            "size": config["model"]["size"],
            "variant": config["variant"]["name"],
            "model_impl": config["variant"]["model_impl"],
            "writer_stage": config["stage"]["name"],
            "max_sequence_length": eval_max_length,
        },
        "topology": topology,
        "mode": args.mode,
        "output": str(output),
        "frozen_requests": (
            None
            if frozen_requests is None
            else {
                "path": str(frozen_requests),
                "sha256": requests_sha256,
            }
        ),
        "checkpoint_load": {
            "format": "torch_dist",
            "strict": "raise_all",
            "optimizer": False,
            "rng": False,
        },
    }
    path = _write_plan(
        plan_dir=plan_dir,
        plan=plan,
        source_config=config,
        model_config=model_config,
        command=command,
        environment=environment,
    )
    print(
        json.dumps(
            {
                "plan_dir": str(path),
                "launch": str(path / "launch.sh"),
                "iteration": iteration,
                "model_size": config["model"]["size"],
                "variant": config["variant"]["name"],
                "mode": args.mode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _manifest_paths(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"PPL manifest is missing: {path}")
    result: list[Path] = []
    names: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split(",", 1)]
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ConfigurationError(
                f"invalid PPL manifest row {path}:{line_number}"
            )
        name, relative = fields
        if name in names:
            raise ConfigurationError(
                f"duplicate PPL source {name!r} in {path}"
            )
        names.add(name)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ConfigurationError(
                f"PPL paths must be relative and confined: {path}:{line_number}"
            )
        token_path = (path.parent / relative_path).resolve()
        if not token_path.is_relative_to(path.parent.resolve()):
            raise ConfigurationError(
                f"PPL path escapes manifest root: {path}:{line_number}"
            )
        if not token_path.is_file() or token_path.stat().st_size == 0:
            raise ConfigurationError(
                f"PPL token stream is unavailable or empty: {token_path}"
            )
        if token_path.stat().st_size % 4:
            raise ConfigurationError(
                f"PPL token stream is not little-endian uint32 aligned: "
                f"{token_path}"
            )
        result.append(token_path)
    if not result:
        raise ConfigurationError(f"PPL manifest contains no datasets: {path}")
    return result


def _cmd_ppl_plan(args: argparse.Namespace) -> int:
    (
        config,
        source_resolved,
        iteration,
        checkpoint,
        topology,
        tokenizer,
        python,
    ) = _prepare(args)
    manifest = args.manifest.expanduser().resolve()
    token_paths = _manifest_paths(manifest)
    sequence_length = args.sequence_length
    if sequence_length is None:
        sequence_length = min(
            8192, int(config["stage"]["model_max_sequence_length"])
        )
    if sequence_length < 2:
        raise ConfigurationError("PPL sequence length must be at least 2")
    model_max_length = int(config["stage"]["model_max_sequence_length"])
    if sequence_length > model_max_length:
        raise ConfigurationError(
            f"PPL sequence length {sequence_length} exceeds model context "
            f"{model_max_length}"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ConfigurationError(
            f"immutable PPL result already exists: {output}"
        )
    cache_dir = args.cache_dir.expanduser().resolve()
    plan_dir = args.plan_dir.expanduser().resolve()
    model_config = build_model_config(config)
    model_config_path = plan_dir / "model_config.json"
    common = _common_model_args(
        config,
        model_config_path=model_config_path,
        tokenizer=tokenizer,
        checkpoint_root=Path(checkpoint["root"]),
        iteration=iteration,
        topology=topology,
        sequence_length=sequence_length,
    )
    # PPL uses exact token-weighted per-document sums. The provider determines
    # the complete number of evaluation iterations after reading the manifest.
    ppl_args = [
        "--calculate-per-token-loss",
        "--skip-train",
        "--valid-ppl-manifest",
        str(manifest),
        "--valid-ppl-cache-dir",
        str(cache_dir),
        "--valid-ppl-eos-token-id",
        str(args.eos_token_id),
        "--valid-ppl-pad-token-id",
        str(args.pad_token_id),
        "--ppl-output-json",
        str(output),
    ]
    expected = {
        "documents": args.expected_documents,
        "valid-targets": args.expected_valid_targets,
        "truncated-tokens": args.expected_truncated_tokens,
    }
    for label, value in expected.items():
        if value is not None:
            if isinstance(value, bool) or value < 0:
                raise ConfigurationError(
                    f"expected {label} must be a non-negative integer"
                )
            ppl_args.extend(
                [
                    "--valid-ppl-expected-" + label,
                    str(value),
                ]
            )
    command = _torchrun(
        python=python,
        entry=PROJECT_ROOT / "scripts" / "eval" / "olmo3_ppl.py",
        topology=topology,
        master_addr=args.master_addr,
        master_port=args.master_port,
        arguments=[*common, *ppl_args],
    )
    environment = _environment(output.parent)
    plan = {
        "schema": SCHEMA,
        "kind": "document-ppl",
        "source_resolved": str(source_resolved),
        "source_resolved_sha256": _sha256(source_resolved),
        "checkpoint": checkpoint,
        "model": {
            "size": config["model"]["size"],
            "variant": config["variant"]["name"],
            "model_impl": config["variant"]["model_impl"],
            "writer_stage": config["stage"]["name"],
            "model_max_sequence_length": model_max_length,
            "eval_sequence_length": sequence_length,
        },
        "topology": topology,
        "manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "token_streams": [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                }
                for path in token_paths
            ],
        },
        "cache_dir": str(cache_dir),
        "output": str(output),
        "checkpoint_load": {
            "format": "torch_dist",
            "strict": "raise_all",
            "optimizer": False,
            "rng": False,
        },
    }
    path = _write_plan(
        plan_dir=plan_dir,
        plan=plan,
        source_config=config,
        model_config=model_config,
        command=command,
        environment=environment,
    )
    print(
        json.dumps(
            {
                "plan_dir": str(path),
                "launch": str(path / "launch.sh"),
                "iteration": iteration,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_common_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resolved",
        type=Path,
        required=True,
        help="Resolved JSON from the exact checkpoint writer run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Native Megatron torch_dist checkpoint root.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Complete checkpoint iteration; default: tracker-selected latest.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable in the Ascend training environment.",
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=16)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="olmo3eval")
    subparsers = parser.add_subparsers(dest="action", required=True)

    inference = subparsers.add_parser(
        "inference-plan",
        help="Freeze a native generation/log-likelihood/KV-cache plan.",
    )
    _add_common_plan_args(inference)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument(
        "--mode",
        choices=sorted(INFERENCE_MODES),
        default="run",
    )
    inference.add_argument("--frozen-requests", type=Path)
    inference.add_argument("--max-length", type=int)
    inference.add_argument("--partition-index", type=int, default=0)
    inference.add_argument("--partition-count", type=int, default=1)
    inference.add_argument(
        "--partition-strategy",
        choices=("modulo", "contiguous"),
        default="modulo",
    )
    inference.add_argument("--suite-alias", action="append", default=[])
    inference.add_argument("--task-alias", action="append", default=[])
    inference.add_argument("--resume", action="store_true")
    inference.add_argument("--run-cache-smoke-first", action="store_true")
    inference.add_argument(
        "--cache-smoke-prompt-lengths",
        default="7,4095,4096,4097",
    )
    inference.add_argument("--cache-smoke-decode-tokens", type=int, default=4)
    inference.add_argument("--cache-smoke-atol", type=float, default=0.5)
    inference.add_argument("--cache-smoke-rtol", type=float, default=0.02)
    inference.set_defaults(handler=_cmd_inference_plan)

    ppl = subparsers.add_parser(
        "ppl-plan",
        help="Freeze an exact independent document-PPL evaluation plan.",
    )
    _add_common_plan_args(ppl)
    ppl.add_argument("--manifest", type=Path, required=True)
    ppl.add_argument("--cache-dir", type=Path, required=True)
    ppl.add_argument("--output", type=Path, required=True)
    ppl.add_argument("--sequence-length", type=int)
    ppl.add_argument("--eos-token-id", type=int, default=100257)
    ppl.add_argument("--pad-token-id", type=int, default=100277)
    ppl.add_argument("--expected-documents", type=int)
    ppl.add_argument("--expected-valid-targets", type=int)
    ppl.add_argument("--expected-truncated-tokens", type=int)
    ppl.set_defaults(handler=_cmd_ppl_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigurationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
