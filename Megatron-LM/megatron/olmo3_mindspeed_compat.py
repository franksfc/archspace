"""Post-MindSpeed compatibility for the OLMo3 production paths.

MindSpeed and MindSpeed-LLM are intentionally pristine third-party checkouts.
Their adaptor replaces a few Megatron Core callables at import time.  This
module is installed *after* that adaptor and reinstalls only the contracts that
OLMo3 needs:

* full-precision RoPE while retaining the configured fused backend;
* the ordinary partial-DistributedOptimizer RS -> AR pipeline;
* inter-backward overlap and the validated TP2 lane pack.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

from megatron.olmo3_runtime_markers import emit_rank0_runtime_marker_once


def install_npu_fused_infer_attention_v2_fallback(
    torch_npu_module: object | None = None,
) -> bool:
    """Bridge the MindSpeed v2 KV-cache API to installed CANN v1 kernels.

    The pinned MindSpeed-LLM checkout calls
    ``npu_fused_infer_attention_score_v2`` while the production torch-npu
    build exposes the equivalent prompt and incremental kernels separately.
    Install a narrow process-local adapter only when the native v2 symbol is
    absent. The fallback deliberately accepts only BSH self-attention with
    equal query/KV head counts, which is the exact OLMo3 path; unsupported
    layouts or GQA fail loudly instead of silently changing attention math.
    """

    if torch_npu_module is None:
        try:
            import torch_npu as torch_npu_module
        except (ImportError, OSError):
            # CPU-only tooling (linting, checkpoint inspection, and unit tests)
            # intentionally runs without the Ascend runtime libraries.
            return False

    if getattr(torch_npu_module, "npu_fused_infer_attention_score_v2", None) is not None:
        return False

    prompt_kernel = getattr(
        torch_npu_module, "npu_prompt_flash_attention", None
    )
    incremental_kernel = getattr(
        torch_npu_module, "npu_incre_flash_attention", None
    )
    if not callable(prompt_kernel) or not callable(incremental_kernel):
        return False

    def _fallback(
        query,
        key,
        value,
        *,
        num_query_heads: int,
        num_key_value_heads: int,
        input_layout: str,
        pse_shift=None,
        softmax_scale: float = 1.0,
        sparse_mode: int = 0,
        atten_mask=None,
        pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647,
    ):
        if input_layout != "BSH":
            raise NotImplementedError(
                "OLMo3's torch-npu v2 fallback supports only BSH layout"
            )
        if int(num_query_heads) != int(num_key_value_heads):
            raise NotImplementedError(
                "OLMo3's torch-npu v2 fallback requires equal query/KV heads"
            )

        if int(query.shape[1]) == 1 and int(key.shape[1]) != 1:
            output = incremental_kernel(
                query,
                key,
                value,
                num_heads=int(num_query_heads),
                input_layout=input_layout,
                pse_shift=pse_shift,
                padding_mask=None,
                scale_value=float(softmax_scale),
            )
        else:
            output = prompt_kernel(
                query,
                key,
                value,
                num_heads=int(num_query_heads),
                input_layout=input_layout,
                pse_shift=pse_shift,
                sparse_mode=int(sparse_mode),
                padding_mask=None,
                atten_mask=atten_mask,
                scale_value=float(softmax_scale),
                pre_tokens=int(pre_tokens),
                next_tokens=int(next_tokens),
            )
        # MindSpeed-LLM's v2 call site consumes result[0].
        return (output,)

    setattr(torch_npu_module, "npu_fused_infer_attention_score_v2", _fallback)
    return True


class _HandleList:
    """A minimal ``torch.distributed.Work``-compatible handle collection."""

    def __init__(self, handles: Iterable[object]):
        self.handles = tuple(handle for handle in handles if handle is not None)
        if not self.handles:
            raise RuntimeError("OLMo3 expected at least one asynchronous collective handle")
        self.waited = False

    def wait(self):
        if not self.waited:
            for handle in self.handles:
                handle.wait()
            self.waited = True
        return True


def _npu_start_grad_sync(self) -> None:
    """NPU-safe direct-collective version of OLMo3's first HSDP stage."""

    from megatron.core.distributed.param_and_grad_buffer import (
        dist_reduce_scatter_func,
        shard_buffer,
    )

    assert (
        self.grad_reduce_handle is None and self.intra_grad_reduce_handle is None
    ), "Should not have multiple communication calls outstanding at once"

    if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
        self.check_grads(
            check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
            check_for_large=self.ddp_config.check_for_large_grads,
        )

    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    reduce_op = (
        torch.distributed.ReduceOp.AVG
        if self.ddp_config.average_in_collective
        else torch.distributed.ReduceOp.SUM
    )
    async_op = self.ddp_config.overlap_grad_reduce
    replicated_distopt_instance = (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
        and self.intra_distributed_optimizer_instance_size == 1
    )
    partial_distopt = (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
        and self.intra_distributed_optimizer_instance_size > 1
    )

    if self.ddp_config.num_distributed_optimizer_instances > 1 and async_op:
        stream_context = torch.cuda.stream(self.communication_stream)
        self.communication_stream.wait_stream(torch.cuda.default_stream())
    else:
        stream_context = nullcontext()

    if replicated_distopt_instance:
        assert self.inter_distributed_optimizer_instance_group is not None
        communication_group = self.inter_distributed_optimizer_instance_group
    elif self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    handles = []
    with stream_context:
        for bucket in self.buckets:
            if self.ddp_config.use_distributed_optimizer and not replicated_distopt_instance:
                local_data_view = shard_buffer(
                    bucket.grad_data,
                    self.intra_distributed_optimizer_instance_size,
                )[self.intra_distributed_optimizer_instance_rank]
                handle = dist_reduce_scatter_func(
                    local_data_view,
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
            else:
                handle = torch.distributed.all_reduce(
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
            if handle is not None:
                handles.append(handle)

    if partial_distopt and not async_op:
        assert self.inter_distributed_optimizer_instance_group is not None
        for bucket in self.buckets:
            local_data_view = shard_buffer(
                bucket.grad_data,
                self.intra_distributed_optimizer_instance_size,
            )[self.intra_distributed_optimizer_instance_rank]
            torch.distributed.all_reduce(
                local_data_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=False,
            )

    if async_op and partial_distopt:
        self.intra_grad_reduce_handle = _HandleList(handles)
    elif async_op:
        self.grad_reduce_handle = _HandleList(handles)
    else:
        self.grad_reduce_handle = None


def _npu_start_inter_grad_sync(self) -> None:
    """NPU-safe direct-collective version of OLMo3's second HSDP stage."""

    from megatron.core.distributed.param_and_grad_buffer import (
        olmo3_hsdp_inter_overlap_backward_enabled,
        olmo3_hsdp_inter_tp_lane_pack,
        olmo3_hsdp_tp_lane_pack_reduce,
        shard_buffer,
    )

    assert self.ddp_config.overlap_grad_reduce
    assert self.ddp_config.use_distributed_optimizer
    assert self.ddp_config.num_distributed_optimizer_instances > 1
    assert self.intra_distributed_optimizer_instance_size > 1
    assert self.inter_distributed_optimizer_instance_group is not None
    assert self.intra_grad_reduce_handle is not None, (
        f"Intra-instance communication has not been issued "
        f"({len(self.params_with_grad)}/{len(self.params)} params ready)"
    )
    assert self.grad_reduce_handle is None

    stream_context = torch.cuda.stream(self.communication_stream)
    with stream_context:
        self.intra_grad_reduce_handle.wait()
    self.intra_grad_reduce_handle = None

    reduce_op = (
        torch.distributed.ReduceOp.AVG
        if self.ddp_config.average_in_collective
        else torch.distributed.ReduceOp.SUM
    )
    if olmo3_hsdp_inter_tp_lane_pack() == 1:
        handles = []
        with stream_context:
            for bucket in self.buckets:
                local_data_view = shard_buffer(
                    bucket.grad_data,
                    self.intra_distributed_optimizer_instance_size,
                )[self.intra_distributed_optimizer_instance_rank]
                handles.append(
                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=True,
                    )
                )
        self.grad_reduce_handle = _HandleList(handles)
    else:
        if len(self.buckets) != 1:
            raise RuntimeError(
                "OLMo3 HSDP TP-lane pack2 requires one bucket per bucket group"
            )
        bucket = self.buckets[0]
        local_data_view = shard_buffer(
            bucket.grad_data,
            self.intra_distributed_optimizer_instance_size,
        )[self.intra_distributed_optimizer_instance_rank]
        with stream_context:
            self.grad_reduce_handle = olmo3_hsdp_tp_lane_pack_reduce(
                self,
                local_data_view,
                reduce_op,
            )
        emit_rank0_runtime_marker_once(
            "OLMO3_RUNTIME_HSDP_TP_LANE_PACK2_ACTIVE",
            inter_backward_overlap=int(
                olmo3_hsdp_inter_overlap_backward_enabled()
            ),
            pack=2,
        )


def _npu_finish_grad_sync(self) -> None:
    """Finish the direct NPU collectives without a coalescing-manager contract."""

    self.param_gather_dispatched = False
    if not self.ddp_config.overlap_grad_reduce:
        self.start_grad_sync()
        return
    if (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
        and self.intra_distributed_optimizer_instance_size > 1
    ):
        if self.grad_reduce_handle is None:
            self.start_inter_grad_sync()
        self.grad_reduce_handle.wait()
        torch.cuda.default_stream().wait_stream(self.communication_stream)
        self.grad_reduce_handle = None
        return
    assert self.grad_reduce_handle is not None, (
        f"Communication call has not been issued "
        f"({len(self.params_with_grad)}/{len(self.params)} params ready)"
    )
    self.grad_reduce_handle.wait()
    self.grad_reduce_handle = None


def install_olmo3_mindspeed_compatibility() -> None:
    """Install OLMo3 contracts after the pristine MindSpeed adaptor."""

    install_npu_fused_infer_attention_v2_fallback()

    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup,
    )
    from megatron.core.models.common.embeddings import rope_utils
    import megatron.core.transformer.attention as attention_module

    if getattr(
        _ParamAndGradBucketGroup.start_grad_sync,
        "_olmo3_post_mindspeed_compat",
        False,
    ):
        return

    _npu_start_grad_sync._olmo3_post_mindspeed_compat = True
    _npu_start_inter_grad_sync._olmo3_post_mindspeed_compat = True
    _npu_finish_grad_sync._olmo3_post_mindspeed_compat = True
    _ParamAndGradBucketGroup.start_grad_sync = _npu_start_grad_sync
    _ParamAndGradBucketGroup.start_inter_grad_sync = _npu_start_inter_grad_sync
    _ParamAndGradBucketGroup.finish_grad_sync = _npu_finish_grad_sync

    # Restore the full-precision OLMo3 implementations that the generic
    # MindSpeed rotary feature replaces. Update Attention's imported aliases
    # as well because that module may already have been imported by the adaptor.
    rope_utils._apply_rotary_pos_emb_bshd = (
        rope_utils.olmo3_apply_rotary_pos_emb_bshd
    )
    rope_utils.apply_rotary_pos_emb = rope_utils.olmo3_apply_rotary_pos_emb
    attention_module._apply_rotary_pos_emb_bshd = (
        rope_utils.olmo3_apply_rotary_pos_emb_bshd
    )
    attention_module.apply_rotary_pos_emb = rope_utils.olmo3_apply_rotary_pos_emb
