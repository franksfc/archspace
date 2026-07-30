"""Project-owned runtime compatibility for the pristine MindSpeed-LLM backend."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch


def install_mindspeed_cross_entropy_patches() -> None:
    """Install MindSpeed-LLM's NPU-friendly vocab-parallel CE helpers."""

    from megatron.core.tensor_parallel.cross_entropy import (
        VocabParallelCrossEntropy,
    )
    from mindspeed_llm.core.tensor_parallel.cross_entropy import (
        calculate_logits_max,
        calculate_predicted_logits,
    )

    VocabParallelCrossEntropy.calculate_logits_max = staticmethod(
        calculate_logits_max
    )
    VocabParallelCrossEntropy.calculate_predicted_logits = staticmethod(
        calculate_predicted_logits
    )


def install_mtp_feature_guard() -> None:
    """Disable MindSpeed-LLM MTP patch registration unless MTP is enabled."""

    from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature

    original_register_patches = MultiTokenPredictionFeature.register_patches
    if getattr(original_register_patches, "_olmo3_mtp_guard", False):
        return

    def register_patches(self: Any, patch_manager: Any, args: Any) -> Any:
        if not getattr(args, "mtp_num_layers", None):
            return None
        return original_register_patches(self, patch_manager, args)

    register_patches._olmo3_mtp_guard = True  # type: ignore[attr-defined]
    MultiTokenPredictionFeature.register_patches = register_patches


def _use_llamafactory_wandb(training_module: ModuleType, args: Any, wandb_writer: Any) -> bool:
    if wandb_writer is None or not training_module.is_last_rank():
        return False
    style = os.getenv("OLMO3_WANDB_LOG_STYLE", "").strip().lower()
    if style in ("llamafactory", "hf", "trainer"):
        return True
    if style in ("", "0", "false", "off", "none", "native"):
        return False
    return bool(getattr(args, "tokenized_path", None))


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


def _perplexity(lm_loss: float) -> float:
    """Exponentiate token-mean CE without silently clipping the metric."""
    try:
        return math.exp(lm_loss)
    except OverflowError:
        return math.inf


def _write_document_ppl_result(
    args: Any,
    *,
    iteration: int,
    eval_losses: dict[str, float],
) -> None:
    """Write the explicit document-PPL result contract on the last rank."""

    raw_output = getattr(args, "ppl_output_json", None)
    if raw_output is None:
        return
    output = Path(raw_output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"immutable document-PPL result already exists: {output}"
        )
    if "lm loss" not in eval_losses:
        raise RuntimeError(
            "document-PPL evaluation did not produce aggregate 'lm loss'"
        )

    metrics: dict[str, dict[str, float]] = {}
    for key, loss in sorted(eval_losses.items()):
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"document-PPL metric {key!r} is non-finite: {loss}"
            )
        metric: dict[str, float] = {"loss": loss}
        if key == "lm loss" or (
            key.startswith("ppl/") and key.endswith("/lm loss")
        ):
            perplexity = _perplexity(loss)
            if not math.isfinite(perplexity):
                raise FloatingPointError(
                    f"document-PPL metric {key!r} overflowed: {perplexity}"
                )
            metric["perplexity"] = perplexity
        metrics[key] = metric

    payload = {
        "schema": "olmo3.document-ppl-result/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(iteration),
        "checkpoint": {
            "root": str(Path(args.load).expanduser().resolve()),
            "requested_step": int(args.ckpt_step),
        },
        "manifest": str(Path(args.valid_ppl_manifest).expanduser().resolve()),
        "dataset": getattr(args, "ppl_validation_summary", None),
        "sequence_length": int(args.seq_length),
        "global_batch_size": int(args.global_batch_size),
        "eval_iters": int(args.eval_iters),
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _wandb_global_step(iteration: int) -> int:
    offset = int(os.getenv("WANDB_GLOBAL_STEP_OFFSET", os.getenv("GLOBAL_STEP_OFFSET", "0")) or "0")
    return int(iteration) + offset


def _olmo3_epoch(args: Any) -> float:
    train_samples = int(getattr(args, "dataset_train_len", 0) or 0)
    if train_samples <= 0:
        train_samples = int(getattr(args, "train_samples", 0) or 0)
    if train_samples <= 0:
        train_samples = int(os.getenv("OLMO3_TRAIN_SAMPLES", "0") or "0")
    if train_samples <= 0:
        train_iters = int(getattr(args, "train_iters", 0) or 0)
        global_batch_size = int(getattr(args, "global_batch_size", 0) or 0)
        train_samples = train_iters * global_batch_size
    if train_samples <= 0:
        return 0.0
    return float(args.consumed_train_samples) / float(train_samples)


def _preview_llamafactory_losses(
    loss_dict: dict[str, Any], total_loss_dict: dict[str, Any], skipped_iter: int
) -> dict[str, float]:
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    shadow = dict(total_loss_dict)
    if not skipped_iter:
        shadow[advanced_iters_key] = shadow.get(advanced_iters_key, 0) + 1
    elif advanced_iters_key not in shadow:
        shadow[advanced_iters_key] = 0
    shadow[skipped_iters_key] = shadow.get(skipped_iters_key, 0) + skipped_iter

    got_nan = False
    for key, value in loss_dict.items():
        if not skipped_iter:
            shadow[key] = (
                shadow.get(key, torch.tensor([0.0], dtype=torch.float, device=value.device)) + value
            )
        else:
            float_value = value.float().sum().item()
            is_nan = (
                float_value == float("inf")
                or float_value == -float("inf")
                or float_value != float_value
            )
            got_nan = got_nan or is_nan
    shadow[nan_iters_key] = shadow.get(nan_iters_key, 0) + int(got_nan)

    denominator = float(max(1, shadow[advanced_iters_key]))
    losses = {}
    for key, value in shadow.items():
        if key in (advanced_iters_key, skipped_iters_key, nan_iters_key):
            continue
        losses[key] = value.item() / denominator
    return losses


def _add_llamafactory_loss_metrics(
    metrics: dict[str, float], prefix: str, losses: dict[str, Any]
) -> None:
    """Map pure LM loss to the standard key and keep auxiliary z-loss separate."""
    lm_loss = _scalar(losses.get("lm loss"))
    if lm_loss is None and losses:
        lm_loss = _scalar(next(iter(losses.values())))
    if lm_loss is not None:
        metrics[f"{prefix}/loss"] = lm_loss

    z_loss = _scalar(losses.get("z loss"))
    if z_loss is not None:
        metrics[f"{prefix}/z_loss"] = z_loss

    total_loss = _scalar(losses.get("total loss"))
    if total_loss is not None:
        metrics[f"{prefix}/total_loss"] = total_loss


def _log_llamafactory_train_wandb(
    wandb_writer: Any,
    args: Any,
    iteration: int,
    losses: dict[str, Any],
    grad_norm: Any,
    learning_rate: Any,
) -> None:
    metrics = {
        "train/epoch": _olmo3_epoch(args),
        "train/global_step": _wandb_global_step(iteration),
    }
    _add_llamafactory_loss_metrics(metrics, "train", losses)
    grad_norm_value = _scalar(grad_norm)
    if grad_norm_value is not None:
        metrics["train/grad_norm"] = grad_norm_value
    learning_rate_value = _scalar(learning_rate)
    if learning_rate_value is not None:
        metrics["train/learning_rate"] = learning_rate_value
    wandb_writer.log(metrics)


def _log_llamafactory_eval_wandb(
    wandb_writer: Any, args: Any, iteration: int, losses: dict[str, Any]
) -> None:
    metrics = {
        "eval/epoch": _olmo3_epoch(args),
        "eval/global_step": _wandb_global_step(iteration),
    }
    _add_llamafactory_loss_metrics(metrics, "eval", losses)
    lm_loss = _scalar(losses.get("lm loss"))
    if lm_loss is not None:
        metrics["eval/perplexity"] = _perplexity(lm_loss)
    per_source: dict[str, dict[str, Any]] = {}
    for key, value in losses.items():
        if not key.startswith("ppl/"):
            continue
        source_and_metric = key[len("ppl/") :]
        source, separator, metric = source_and_metric.rpartition("/")
        if not separator or not source:
            continue
        per_source.setdefault(source, {})[metric] = value
    for source, source_losses in per_source.items():
        prefix = f"eval/{source}"
        _add_llamafactory_loss_metrics(metrics, prefix, source_losses)
        source_lm_loss = _scalar(source_losses.get("lm loss"))
        if source_lm_loss is not None:
            metrics[f"{prefix}/perplexity"] = _perplexity(source_lm_loss)
    wandb_writer.log(metrics)


def install_llamafactory_wandb_training_log(training_module: ModuleType) -> None:
    """Make MindSpeed-LLM trainer emit LLaMA-Factory-shaped train W&B metrics."""

    original_training_log = training_module.training_log
    if getattr(original_training_log, "_olmo3_llamafactory_wandb", False):
        return

    def training_log(
        loss_dict: dict[str, Any],
        total_loss_dict: dict[str, Any],
        learning_rate: Any,
        decoupled_learning_rate: Any,
        iteration: int,
        loss_scale: float,
        report_memory_flag: bool,
        skipped_iter: int,
        grad_norm: Any,
        params_norm: Any,
        num_zeros_in_grad: Any,
    ) -> bool:
        args = training_module.get_args()
        raw_wandb_writer = training_module.get_wandb_writer()
        llamafactory_wandb_writer = (
            raw_wandb_writer
            if _use_llamafactory_wandb(training_module, args, raw_wandb_writer)
            else None
        )
        llamafactory_losses = {}
        if llamafactory_wandb_writer and iteration % args.log_interval == 0:
            llamafactory_losses = _preview_llamafactory_losses(
                loss_dict, total_loss_dict, skipped_iter
            )

        if not llamafactory_wandb_writer:
            return original_training_log(
                loss_dict,
                total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                loss_scale,
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )

        original_get_wandb_writer: Callable[[], Any] = training_module.get_wandb_writer
        training_module.get_wandb_writer = lambda: None
        try:
            result = original_training_log(
                loss_dict,
                total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                loss_scale,
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )
        finally:
            training_module.get_wandb_writer = original_get_wandb_writer

        if iteration % args.log_interval == 0:
            _log_llamafactory_train_wandb(
                llamafactory_wandb_writer,
                args,
                iteration,
                llamafactory_losses,
                grad_norm,
                learning_rate,
            )
        return result

    training_log._olmo3_llamafactory_wandb = True  # type: ignore[attr-defined]
    training_module.training_log = training_log


def install_llamafactory_wandb_eval_log(training_module: ModuleType) -> None:
    """Make MindSpeed-LLM eval emit LLaMA-Factory-shaped W&B metrics."""

    original_eval_print = training_module.evaluate_and_print_results
    if getattr(original_eval_print, "_olmo3_llamafactory_eval_wandb", False):
        return

    globals_ = original_eval_print.__globals__
    evaluate = globals_["evaluate"]
    get_args = globals_["get_args"]
    get_tensorboard_writer = globals_["get_tensorboard_writer"]
    get_wandb_writer = globals_["get_wandb_writer"]
    is_last_rank = globals_["is_last_rank"]
    print_rank_last = globals_["print_rank_last"]

    def evaluate_and_print_results(
        prefix: str,
        forward_step_func: Any,
        data_iterator: Any,
        model: Any,
        iteration: int,
        process_non_loss_data_func: Any,
        config: Any,
        verbose: bool = False,
        write_to_tensorboard: bool = True,
        non_loss_data_func: Any = None,
    ) -> None:
        args = get_args()
        writer = get_tensorboard_writer() if write_to_tensorboard else None
        raw_wandb_writer = get_wandb_writer()
        llamafactory_wandb_writer = (
            raw_wandb_writer
            if _use_llamafactory_wandb(training_module, args, raw_wandb_writer)
            else None
        )

        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            forward_step_func,
            data_iterator,
            model,
            process_non_loss_data_func,
            config,
            verbose,
            non_loss_data_func,
        )
        if timelimit:
            return

        string = f" validation loss at {prefix} | "
        eval_losses = {}
        for key in total_loss_dict:
            loss_value = total_loss_dict[key].item()
            eval_losses[key] = loss_value
            string += "{} value: {:.6E} | ".format(key, loss_value)
            ppl = None
            if key == "lm loss" or (
                key.startswith("ppl/") and key.endswith("/lm loss")
            ):
                ppl = _perplexity(loss_value)
                string += "{} PPL: {:.6E} | ".format(key, ppl)
            if writer:
                writer.add_scalar("{} validation".format(key), loss_value, iteration)
                writer.add_scalar(
                    "{} validation vs samples".format(key), loss_value, args.consumed_train_samples
                )
                if args.log_validation_ppl_to_tensorboard and ppl is not None:
                    writer.add_scalar("{} validation ppl".format(key), ppl, iteration)
                    writer.add_scalar(
                        "{} validation ppl vs samples".format(key), ppl, args.consumed_train_samples
                    )

        if is_last_rank():
            _write_document_ppl_result(
                args,
                iteration=iteration,
                eval_losses=eval_losses,
            )

        if llamafactory_wandb_writer and eval_losses and is_last_rank():
            _log_llamafactory_eval_wandb(llamafactory_wandb_writer, args, iteration, eval_losses)

        if process_non_loss_data_func is not None and writer and is_last_rank():
            process_non_loss_data_func(collected_non_loss_data, iteration, writer)

        length = len(string) + 1
        print_rank_last("-" * length)
        print_rank_last(string)
        print_rank_last("-" * length)

    evaluate_and_print_results._olmo3_llamafactory_eval_wandb = True  # type: ignore[attr-defined]
    training_module.evaluate_and_print_results = evaluate_and_print_results
