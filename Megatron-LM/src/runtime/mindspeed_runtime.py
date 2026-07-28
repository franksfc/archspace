"""MindSpeed runtime loading and patches for OLMo-family training."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from megatron.olmo3_mindspeed_patches import (
    install_mindspeed_cross_entropy_patches,  # noqa: F401
    install_mtp_feature_guard,
    install_llamafactory_wandb_eval_log,
    install_llamafactory_wandb_training_log,
)


def load_mindspeed_runtime() -> tuple[Any, Any]:
    """Install runtime guards and return MindSpeed CP batch helper + pretrain."""

    os.environ.setdefault("TRAINING_BACKEND", "mindspeed")
    install_mtp_feature_guard()

    importlib.import_module("mindspeed_llm.tasks.megatron_adaptor_v2")
    mc2_recomputation = os.environ.get(
        "OLMO3_MC2_ALL_GATHER_RECOMPUTATION", "0"
    ).strip()
    if mc2_recomputation not in {"0", "1"}:
        raise RuntimeError(
            "OLMO3_MC2_ALL_GATHER_RECOMPUTATION must be exactly 0 or 1"
        )
    mc2_enabled = "--use-ascend-mc2" in sys.argv
    if mc2_enabled:
        if mc2_recomputation != "1":
            raise RuntimeError(
                "the production Ascend MC2 path requires all-gather recomputation"
            )
        from mindspeed.core.tensor_parallel.ascend_turbo.ascend_turbo_cfg import (
            ascend_turbo_cfg,
        )

        if not bool(ascend_turbo_cfg.all_gather_recomputation):
            raise RuntimeError(
                "the pinned MindSpeed MC2 runtime disabled all-gather recomputation"
            )
    elif mc2_recomputation != "0":
        raise RuntimeError(
            "MC2 all-gather recomputation was requested without --use-ascend-mc2"
        )
    # MindSpeed and MindSpeed-LLM stay pristine. Their adaptor intentionally
    # replaces several Megatron callables; restore the narrow OLMo3 production
    # contracts only after that replacement phase has completed.
    from megatron.olmo3_mindspeed_compat import (
        install_olmo3_mindspeed_compatibility,
    )

    install_olmo3_mindspeed_compatibility()
    get_batch_utils = importlib.import_module("mindspeed_llm.core.context_parallel.get_batch_utils")
    training_module = importlib.import_module("mindspeed_llm.training.training")
    install_llamafactory_wandb_training_log(training_module)
    install_llamafactory_wandb_eval_log(training_module)
    return get_batch_utils.get_batch_on_this_cp_rank, training_module.pretrain
