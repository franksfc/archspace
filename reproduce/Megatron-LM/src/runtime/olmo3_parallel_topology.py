"""Pure topology validation for OLMo3 fixed-length training stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage2ParallelTopology:
    """Derived parallel sizes for one valid Stage-2 runtime."""

    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    data_parallel_size: int
    micro_batch_size: int
    global_batch_size: int
    gradient_accumulation: int
    sequence_parallel: bool


@dataclass(frozen=True)
class Stage3ParallelTopology:
    """Derived parallel sizes for one valid packed Stage-3 runtime."""

    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    data_parallel_size: int
    micro_batch_size: int
    global_batch_size: int
    gradient_accumulation: int
    sequence_parallel: bool
    packed_tokens_per_rank_microbatch: int


def _validate_positive_integer_values(**values: int) -> None:
    invalid = [
        f"{name} must be a positive integer"
        for name, value in values.items()
        if isinstance(value, bool) or not isinstance(value, int) or value < 1
    ]
    if invalid:
        raise ValueError("; ".join(invalid))


def _derive_data_parallel_and_accumulation(
    *,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    micro_batch_size: int,
    global_batch_size: int,
) -> tuple[int, int]:
    model_parallel_size = (
        tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    )
    if world_size % model_parallel_size:
        raise ValueError(
            "world_size must be divisible by TP*PP*CP "
            f"(world={world_size}, TP*PP*CP={model_parallel_size})"
        )
    data_parallel_size = world_size // model_parallel_size

    samples_per_microbatch_wave = data_parallel_size * micro_batch_size
    if global_batch_size % samples_per_microbatch_wave:
        raise ValueError(
            "global_batch_size must be divisible by DP*micro_batch_size "
            f"(GBS={global_batch_size}, DP={data_parallel_size}, "
            f"micro={micro_batch_size})"
        )
    gradient_accumulation = global_batch_size // samples_per_microbatch_wave
    if gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be at least one")
    return data_parallel_size, gradient_accumulation


def _validate_attention_partitioning(
    *,
    tensor_parallel_size: int,
    context_parallel_size: int,
    context_parallel_algo: str,
    sequence_length: int,
    sequence_parallel: bool,
    num_attention_heads: int,
    num_query_groups: int,
) -> None:
    parallel_head_shards = tensor_parallel_size * context_parallel_size
    if (
        num_attention_heads % parallel_head_shards
        or num_query_groups % parallel_head_shards
    ):
        raise ValueError(
            "attention heads and KV heads must be divisible by TP*CP "
            f"(heads={num_attention_heads}, kv_heads={num_query_groups}, "
            f"TP*CP={parallel_head_shards})"
        )
    if context_parallel_size > 1 and context_parallel_algo != "ulysses_cp_algo":
        raise ValueError("CP>1 requires context_parallel_algo=ulysses_cp_algo")
    if sequence_length % context_parallel_size:
        raise ValueError(
            "sequence_length must be divisible by CP "
            f"(sequence={sequence_length}, CP={context_parallel_size})"
        )
    if sequence_parallel:
        if tensor_parallel_size < 2:
            raise ValueError("sequence parallelism requires TP>=2")
        if sequence_length % parallel_head_shards:
            raise ValueError(
                "sequence_length must be divisible by TP*CP with sequence parallelism "
                f"(sequence={sequence_length}, TP*CP={parallel_head_shards})"
            )


def validate_stage2_parallel_topology(
    *,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    micro_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    sequence_parallel: bool,
    context_parallel_algo: str,
    num_attention_heads: int,
    num_query_groups: int,
) -> Stage2ParallelTopology:
    """Validate semantics-preserving TP/CP/SP/micro layouts for Stage 2.

    Stage 2 is fixed-length, unpacked 8192-token training. Its model graph does
    not require a frozen TP/CP/micro profile, so the valid topology is derived
    from world size and global batch size instead. This deliberately does not
    inspect data or filesystem paths.
    """

    _validate_positive_integer_values(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
    )
    data_parallel_size, gradient_accumulation = (
        _derive_data_parallel_and_accumulation(
            world_size=world_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
        )
    )
    _validate_attention_partitioning(
        tensor_parallel_size=tensor_parallel_size,
        context_parallel_size=context_parallel_size,
        context_parallel_algo=context_parallel_algo,
        sequence_length=sequence_length,
        sequence_parallel=sequence_parallel,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
    )

    return Stage2ParallelTopology(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        data_parallel_size=data_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        gradient_accumulation=gradient_accumulation,
        sequence_parallel=sequence_parallel,
    )


def validate_stage3_parallel_topology(
    *,
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    micro_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    sequence_parallel: bool,
    context_parallel_algo: str,
    num_attention_heads: int,
    num_query_groups: int,
) -> Stage3ParallelTopology:
    """Validate a packed Stage-3 TP/CP/SP/micro layout.

    Packed document boundaries and TND metadata are batch-aware, so no fixed
    TP, CP, SP, or micro profile is required. The remaining restrictions are
    mathematical: PP stays one for the forward-local Siamese/Depth state,
    model-parallel groups must tile the world, attention heads must tile TP*CP,
    sequence shards must be integral, and one optimizer step must contain an
    integral number of microbatch waves.
    """

    _validate_positive_integer_values(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
    )
    if pipeline_parallel_size != 1:
        raise ValueError(
            "packed Stage 3 requires PP=1 because Siamese/Depth state is forward-local"
        )
    data_parallel_size, gradient_accumulation = (
        _derive_data_parallel_and_accumulation(
            world_size=world_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
        )
    )
    _validate_attention_partitioning(
        tensor_parallel_size=tensor_parallel_size,
        context_parallel_size=context_parallel_size,
        context_parallel_algo=context_parallel_algo,
        sequence_length=sequence_length,
        sequence_parallel=sequence_parallel,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
    )
    packed_tokens = micro_batch_size * sequence_length
    if packed_tokens > 2**31 - 1:
        raise ValueError(
            "packed TND token offsets must fit signed int32 "
            f"(micro*sequence={packed_tokens})"
        )

    return Stage3ParallelTopology(
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
        data_parallel_size=data_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        gradient_accumulation=gradient_accumulation,
        sequence_parallel=sequence_parallel,
        packed_tokens_per_rank_microbatch=packed_tokens,
    )
