# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import VocabUtility


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""

        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        target,
        label_smoothing=0.0,
        z_loss_coeff=0.0,
        return_loss_components=False,
    ):
        """Vocab parallel cross entropy forward function."""

        # The OLMo3 dense z-loss path recomputes softmax in backward from the original
        # (normally BF16) logits and the global log-partition. This halves its saved-tensor
        # footprint versus retaining an FP32 softmax, and is opt-in so the legacy zero-z-loss
        # path remains byte-for-byte unchanged. Avoid the MindSpeed helper here because its
        # memory optimization releases the original BF16 logits storage after the FP32 cast.
        recompute_softmax = z_loss_coeff > 0.0
        if recompute_softmax:
            # ``Tensor.float()`` returns the input tensor unchanged when it is already FP32.
            # The forward softmax below mutates that FP32 buffer in place, so retaining the
            # un-cloned input would save probabilities rather than the original logits and
            # produce an incorrect recomputed-softmax gradient. BF16/FP16 casts allocate a
            # distinct FP32 buffer and therefore do not need this memory-expensive clone.
            logits_for_backward = (
                vocab_parallel_logits.clone()
                if vocab_parallel_logits.dtype == torch.float32
                else vocab_parallel_logits
            )
            vocab_parallel_logits = vocab_parallel_logits.float()
            logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        else:
            logits_for_backward = None
            vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
                vocab_parallel_logits
            )
        world_size = get_tensor_model_parallel_world_size()
        if world_size > 1:
            torch.distributed.all_reduce(
                logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
            )

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All-reduce is only needed to combine vocabulary shards. With TP=1,
        # these tensors already contain the complete vocabulary result; avoiding
        # the no-op collectives also avoids lazily creating thousands of
        # single-rank HCCL communicators at large data-parallel scale.
        if world_size > 1:
            torch.distributed.all_reduce(
                predicted_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )

            torch.distributed.all_reduce(
                sum_exp_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """  # pylint: disable=line-too-long
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities.
            # So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        lm_loss = loss
        if z_loss_coeff > 0.0:
            # `sum_exp_logits` is the full-vocabulary sum after the TP all-reduce. Reuse it and
            # the global maximum from the numerically stable softmax instead of evaluating another
            # distributed logsumexp. OLMo3 applies this dense z-loss independently to every token.
            log_partition = torch.log(sum_exp_logits) + logits_max
            z_loss = z_loss_coeff * log_partition.square()
            loss = lm_loss + z_loss
        else:
            z_loss = torch.zeros_like(lm_loss)

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.z_loss_coeff = z_loss_coeff
        ctx.recompute_softmax = recompute_softmax

        # Store softmax, target-mask and masked-target for backward pass.
        if recompute_softmax:
            ctx.save_for_backward(logits_for_backward, target_mask, masked_target_1d, log_partition)
        else:
            # Preserve the pre-z-loss saved-tensor footprint for the default path.
            ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        if return_loss_components:
            # Reporting components must never create an alternative optimization path. The first
            # output remains the only differentiable loss; the other two are exact per-token
            # metrics for logging and evaluation.
            lm_loss_metric = lm_loss.detach()
            z_loss_metric = z_loss.detach()
            ctx.mark_non_differentiable(lm_loss_metric, z_loss_metric)
            return loss, lm_loss_metric, z_loss_metric
        return loss

    @staticmethod
    def backward(ctx, grad_output, *unused_component_grad_outputs):
        """Vocab parallel cross entropy backward function."""
        del unused_component_grad_outputs

        # Retreive tensors from the forward path.
        if ctx.recompute_softmax:
            logits, target_mask, masked_target_1d, log_partition = ctx.saved_tensors
            softmax = torch.exp(logits.float() - log_partition.unsqueeze(dim=-1))
        else:
            softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
        )

        if ctx.z_loss_coeff > 0.0:
            # d(z_coeff * log(Z)^2) / d(logit_i) =
            # 2 * z_coeff * log(Z) * softmax_i. Apply that factor before subtracting the
            # target/label-smoothing terms so the ordinary CE gradient remains unchanged.
            grad_input.mul_(1.0 + 2.0 * ctx.z_loss_coeff * log_partition.unsqueeze(dim=-1))

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad

            # Finally elementwise multiplication with the output gradients.
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
            )

        return grad_input, None, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, label_smoothing=0.0, z_loss_coeff=0.0
):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, vocab_size/num_parallel_ranks]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        label_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)

        z_loss_coeff: coefficient for the dense per-token
                      ``logsumexp(logits, vocab) ** 2`` auxiliary loss.
                      Defaults to 0.0, preserving the existing loss and gradients.
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing, z_loss_coeff, False
    )


def vocab_parallel_cross_entropy_with_loss_components(
    vocab_parallel_logits, target, label_smoothing=0.0, z_loss_coeff=0.0
):
    """Return total, language-modeling, and scaled z-loss tensors per token.

    Only the total loss is differentiable. The language-modeling and z-loss outputs are detached
    reporting metrics, so callers cannot accidentally remove or double-count the z-loss gradient.
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing, z_loss_coeff, True
    )
