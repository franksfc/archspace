#!/usr/bin/env python3
"""Load and immediately re-save an OLMo 3 checkpoint in the target topology.

The heavy lifting is intentionally delegated to Megatron Core's native
``torch_dist`` loader/saver.  In particular, this preserves distributed AdamW
moments and FP32 master parameters and uses MCore's model-space optimizer
resharding instead of inventing a second checkpoint implementation.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_TF", "FALSE")


def main() -> None:
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(
            "usage: reshard_checkpoint.py [generated Megatron arguments]\n\n"
            "Internal NPU worker entry point. Generate a complete, immutable "
            "command with `olmo3ckpt reshard`; direct invocation is unsupported."
        )
        return

    from runtime.mindspeed_runtime import (
        install_mindspeed_cross_entropy_patches,
        load_mindspeed_runtime,
    )

    # MindSpeed patches Megatron modules during import. Apply it before
    # importing any MCore training implementation, exactly like production.
    load_mindspeed_runtime()

    import torch.distributed as dist
    from megatron.core.enums import ModelType
    from megatron.training import get_args, print_rank_0
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import (
        preprocess_common_state_dict,
        setup_model_and_optimizer,
    )
    from pretrain_olmo3_mindspeed import extra_args_provider, model_provider

    install_mindspeed_cross_entropy_patches()
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
    args = get_args()
    if not args.load:
        raise ValueError("checkpoint reshard requires --load")
    if not args.save:
        raise ValueError("checkpoint reshard requires --save")
    if os.path.realpath(args.load) == os.path.realpath(args.save):
        raise ValueError("checkpoint reshard requires distinct --load and --save roots")
    if args.ckpt_format != "torch_dist":
        raise ValueError(
            "native topology reshard accepts torch_dist input only. HF/legacy "
            "conversion is intentionally disabled until an audited OLMo 3 "
            "tensor converter exists."
        )
    if not args.use_distributed_optimizer:
        raise ValueError("OLMo 3 reshard requires the distributed AdamW optimizer")
    if not args.ckpt_fully_parallel_save:
        raise ValueError(
            "resharded output must use fully-sharded model-space optimizer state"
        )

    model, optimizer, scheduler = setup_model_and_optimizer(
        model_provider,
        ModelType.encoder_or_decoder,
    )
    # setup_model_and_optimizer performs the strict load.  Same-stage resume
    # preserves iteration/scheduler/RNG/counters; --olmo3-stage-transition
    # preserves model+Adam+master params and resets the new-stage trainer state.
    iteration = int(args.iteration)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        scheduler,
        int(args.num_floating_point_operations_so_far),
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
    )
    dist.barrier()
    print_rank_0(
        "OLMO3_CHECKPOINT_RESHARD_OK "
        f"iteration={iteration} tp={args.tensor_model_parallel_size} "
        f"cp={args.context_parallel_size} pp={args.pipeline_model_parallel_size} "
        f"distopt_instances={args.num_distributed_optimizer_instances}"
    )


if __name__ == "__main__":
    main()
