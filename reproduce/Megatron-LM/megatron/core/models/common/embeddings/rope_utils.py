# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.utils import is_te_min_version

logger = logging.getLogger(__name__)

# Prefer fused RoPE from Apex as we need the `transpose_output_memory` argument for the bshd trick.
# See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2469.
try:
    # pylint: disable=unused-import
    from apex.transformer.functional import fused_apply_rotary_pos_emb
except ImportError:
    try:
        from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
    except ImportError:
        fused_apply_rotary_pos_emb = None


try:
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
except ImportError:
    try:
        from apex.transformer.functional import fused_apply_rotary_pos_emb_thd
    except ImportError:
        fused_apply_rotary_pos_emb_thd = None


try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
except ImportError:
    apply_rotary_emb_flash = None


__all__ = ['apply_rotary_emb_flash']


NpuRotaryBackend = Callable[[Tensor, Tensor, Tensor, int], Tensor]


def _resolve_npu_rotary_backend(t: Tensor) -> Optional[NpuRotaryBackend]:
    """Return MindSpeed's fused Ascend RoPE op for an NPU tensor.

    The import is intentionally lazy.  Megatron is also imported by CPU-only
    configuration, conversion, and unit-test processes where importing the NPU
    extension either is impossible or would eagerly initialise device state.
    Keeping this resolver in the project-owned Megatron tree also means the
    pristine MindSpeed repositories do not need to be patched.

    Tests may replace this function with a CPU fake that implements the same
    four-argument ABI.
    """

    if t.device.type != "npu":
        return None
    try:
        from mindspeed.ops.npu_rotary_position_embedding import (
            npu_rotary_position_embedding,
        )
    except (ImportError, OSError):
        return None
    return npu_rotary_position_embedding


def _runtime_requests_npu_rope_fusion() -> bool:
    """Whether MindSpeed's explicit fused-RoPE switch is enabled.

    Direct BSHD calls are used for packed document positions and YaRN m-scale,
    so they do not receive ``TransformerConfig.apply_rope_fusion``.  Consulting
    the already-parsed runtime argument preserves the meaning of
    ``--use-fused-rotary-pos-emb`` for those paths.  Import/configuration tools
    have no global training args and correctly return ``False``.
    """

    try:
        from megatron.training import get_args

        args = get_args()
    except (AssertionError, ImportError, RuntimeError):
        return False
    return bool(getattr(args, "use_fused_rotary_pos_emb", False))


def _warn_missing_npu_rope_backend_once() -> None:
    """Report an unavailable requested NPU kernel without changing config."""

    if getattr(_warn_missing_npu_rope_backend_once, "_warned", False):
        return
    logger.warning(
        "Ascend fused RoPE was requested but the MindSpeed NPU rotary backend "
        "is unavailable; using the mathematically exact PyTorch equation for "
        "this process. TransformerConfig.apply_rope_fusion is left unchanged."
    )
    _warn_missing_npu_rope_backend_once._warned = True


def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    full_precision: bool = False,
    _force_npu_fusion: bool = False,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]
    input_dtype = t.dtype

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # OLMo 3 applies RoPE to FP32 Q/K and computes the trigonometric functions
    # in FP32, then returns Q/K in their original dtype.  In particular, do
    # not compute cos/sin in BF16 and cast them afterwards: that is not the
    # official rope_full_precision contract.
    if full_precision:
        t = t.float()
        freqs = freqs.float()

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    npu_fusion_requested = (
        _force_npu_fusion or _runtime_requests_npu_rope_fusion()
    )
    npu_backend = (
        _resolve_npu_rotary_backend(t) if npu_fusion_requested else None
    )
    if npu_backend is not None:
        if t.ndim != 4:
            raise ValueError(
                "Ascend fused RoPE requires a 4-D SBHD/BSHD-compatible tensor; "
                f"got shape {tuple(t.shape)}."
            )
        mode = 1 if rotary_interleaved else 0
        try:
            # The CANN op has a strict contiguous ABI.  Under the OLMo3
            # contract t/cos/sin are all FP32 here; the cast back happens only
            # after the fused op has completed.
            t = npu_backend(
                t.contiguous(), cos_.contiguous(), sin_.contiguous(), mode
            )
        except Exception as exc:
            raise RuntimeError(
                "Ascend fused RoPE backend failed for "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, mode={mode}."
            ) from exc
    else:
        if npu_fusion_requested and t.device.type == "npu":
            _warn_missing_npu_rope_backend_once()
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    if full_precision:
        t = t.to(input_dtype)
    return torch.cat((t, t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    else:
        return freqs[: x.size(0)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    full_precision: bool = False,
    _force_npu_fusion: bool = False,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = cu_seqlens // cp_size
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    return torch.cat(
        [
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs),
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                full_precision=full_precision,
                _force_npu_fusion=_force_npu_fusion,
            )
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    global fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_thd
    full_precision = bool(getattr(config, "rope_full_precision", False))

    if config.apply_rope_fusion:
        npu_backend_available = _resolve_npu_rotary_backend(t) is not None
        if cu_seqlens is None:
            # NOTE: TE backends do not support mRoPE in bshd format when bs > 1
            needs_project_equation = (
                (config.mrope_section is not None and freqs.shape[1] > 1)
                or config.multi_latent_attention
                or mscale != 1.0
            )
            if npu_backend_available or needs_project_equation:
                return _apply_rotary_pos_emb_bshd(
                    t,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                    full_precision=full_precision,
                    _force_npu_fusion=npu_backend_available,
                )
            input_dtype = t.dtype
            fused_t = t.float() if full_precision else t
            fused_freqs = freqs.float() if full_precision else freqs
            if fused_apply_rotary_pos_emb is not None:
                if config.rotary_interleaved:
                    output = fused_apply_rotary_pos_emb(
                        fused_t, fused_freqs, interleaved=True
                    )
                else:
                    output = fused_apply_rotary_pos_emb(
                        fused_t, fused_freqs, transpose_output_memory=True
                    )
                return output.to(input_dtype) if full_precision else output

            # CPU conversion/tests and installations without TE/Apex still
            # require correct RoPE.  Do not mutate apply_rope_fusion (which
            # previously disabled the NPU path globally); use the exact
            # equation only for this call when no fused backend exists.
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
                full_precision=full_precision,
            )
        else:
            if npu_backend_available:
                return _apply_rotary_pos_emb_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                    full_precision=full_precision,
                    _force_npu_fusion=True,
                )
            if (
                fused_apply_rotary_pos_emb_thd is None
                or config.multi_latent_attention
                or mscale != 1.0
            ):
                return _apply_rotary_pos_emb_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                    full_precision=full_precision,
                )
            input_dtype = t.dtype
            fused_t = t.float() if full_precision else t
            fused_freqs = freqs.float() if full_precision else freqs
            cp_size = parallel_state.get_context_parallel_world_size()
            if cp_size > 1:
                if not is_te_min_version("1.11.0", check_equality=False):
                    raise ValueError("Only TE >= 1.12 supports RoPE fusion for THD format with CP.")
                output = fused_apply_rotary_pos_emb_thd(
                    fused_t,
                    cu_seqlens,
                    fused_freqs,
                    cp_size=cp_size,
                    cp_rank=parallel_state.get_context_parallel_rank(),
                )
                return output.to(input_dtype) if full_precision else output
            else:
                output = fused_apply_rotary_pos_emb_thd(
                    fused_t, cu_seqlens, fused_freqs
                )
                return output.to(input_dtype) if full_precision else output
    else:
        if cu_seqlens is None:
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
                full_precision=full_precision,
            )
        else:
            return _apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
                full_precision=full_precision,
            )


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_interleaved: bool = False,
    full_precision: bool = False,
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    input_dtype = t.dtype
    t_work = t.float() if full_precision else t
    cos = cos.to(t_work.dtype)
    sin = sin.to(t_work.dtype)

    npu_fusion_requested = _runtime_requests_npu_rope_fusion()
    npu_backend = (
        _resolve_npu_rotary_backend(t_work) if npu_fusion_requested else None
    )
    if full_precision or apply_rotary_emb_flash is None or npu_backend is not None:
        # FlashAttention rotary kernels do not consistently accept FP32 on all
        # supported backends.  The Ascend op does, so the OLMo3 inference-cache
        # path uses that fused FP32 kernel when available and the exact equation
        # otherwise.
        if rotary_interleaved:
            cos_full = torch.repeat_interleave(cos, 2, dim=-1)
            sin_full = torch.repeat_interleave(sin, 2, dim=-1)
        else:
            cos_full = torch.cat((cos, cos), dim=-1)
            sin_full = torch.cat((sin, sin), dim=-1)
        rot_dim = cos_full.shape[-1]
        t_rot, t_pass = t_work[..., :rot_dim], t_work[..., rot_dim:]
        while cos_full.dim() < t_rot.dim():
            cos_full = cos_full.unsqueeze(1)
            sin_full = sin_full.unsqueeze(1)
        if npu_backend is not None:
            if t_rot.ndim != 4:
                raise ValueError(
                    "Ascend fused RoPE requires a 4-D inference-cache tensor; "
                    f"got shape {tuple(t_rot.shape)}."
                )
            mode = 1 if rotary_interleaved else 0
            try:
                y_rot = npu_backend(
                    t_rot.contiguous(),
                    cos_full.contiguous(),
                    sin_full.contiguous(),
                    mode,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Ascend fused inference-cache RoPE backend failed for "
                    f"shape={tuple(t_rot.shape)}, dtype={t_rot.dtype}, mode={mode}."
                ) from exc
        else:
            if npu_fusion_requested and t_work.device.type == "npu":
                _warn_missing_npu_rope_backend_once()
            y_rot = (t_rot * cos_full) + (
                _rotate_half(t_rot, rotary_interleaved) * sin_full
            )
        y = torch.cat((y_rot, t_pass), dim=-1)
    else:
        # Use Flash Attention's optimized kernel for rotary embedding
        t_work = t_work.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t_work, cos, sin, rotary_interleaved)
        y = y.permute(1, 0, 2, 3)

    return y.to(input_dtype) if full_precision else y


# MindSpeed installs backend shims by replacing the public module attributes
# after import. Keep immutable references to the OLMo3-aware implementations
# so ``megatron.olmo3_mindspeed_compat`` can reinstall them after that patch
# phase without modifying either pristine third-party repository.
olmo3_apply_rotary_pos_emb_bshd = _apply_rotary_pos_emb_bshd
olmo3_apply_rotary_pos_emb = apply_rotary_pos_emb
