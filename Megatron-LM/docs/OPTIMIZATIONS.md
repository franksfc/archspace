# Production optimization profiles

Each exposed profile has an explicit topology and mathematical contract.

## TP2, SP, SIO and MC2

`tp2_sio_mc2` enables:

- TP2 with sequence parallelism;
- Ascend MC2 tensor-parallel kernels;
- MC2 all-gather recomputation during backward;
- physical even/odd TP pairing on an SIO link within a 16-NPU node;
- ordinary distributed-optimizer HSDP;
- FP32 gradient accumulation/reduction;
- overlap-grad-reduce and overlap-param-gather;
- HSDP inter-backward overlap;
- TP-lane pack2 after its rank/pair contract is validated.

The reference topology `tp2_sio_2048npu` is:

```text
TP2 × CP1 × DP1024, SP on, MBS1, GBS1024, GA1
HSDP shard8, 128 instances, bucket8
```

The SIO check is a physical launch precondition, not a model flag. Validation
fails if MC2 is requested without TP2, sequence parallel, FP32 gradients or
TP-lane pack2. The frozen multi-node `launch.sh` runs the read-only SIO
preflight on every worker before `torchrun`, so a correct pair on the control
host cannot mask an incorrect worker mapping.

## One-hop SWA halo and Full-attention Ulysses

`cp_single_halo` enables:

- document-safe, ragged, exact left K/V halo for SWA;
- local SWA Q and output;
- asynchronous forward halo overlap;
- exact backward halo overlap;
- fused-QKV all-to-all packing for Full-attention Ulysses;
- packed `cu_seqlens`, document-relative positions and no cross-document
  attention;
- Full-only YaRN, with original RoPE on SWA layers;
- ordinary HSDP, FP32 gradients and DP communication overlap.

It supports CP only when:

```text
attention_heads % (TP × CP) == 0
TP × CP <= NPUs per node
sequence_length / CP >= SWA_window - 1
```

For OLMo 3's 4,096-token SWA window, the minimum local sequence is 4,095.
The `stage2_cp2_512npu`, `stage3_cp16_1024npu`, and `sft_cp8_256npu`
reference topologies use a 4,096-token local shard.

## Combining TP2 and CP

The profiles are composable in configuration, but composition is not a promise
that an arbitrary topology is valid. The same divisibility, node-local group
and one-hop constraints apply. For example, a 16-head model cannot use
`TP2 × CP16` with Ulysses because 16 heads are not divisible by 32.

The combined configuration is validated before run materialization. No
alternate algorithm is selected implicitly.

Profile composition is order-independent. When TP2 and CP profiles are
combined, the resolver canonicalizes their names and deterministically uses
the CP-safe Gloo scalar-reporting backend; reversing the command-line profile
order cannot change the resolved training contract.

## Supported scope

The production profiles expose ordinary HSDP, TP2/SIO/MC2 and CP single-halo.
The resolver rejects unknown optimizer, performance and communication fields
instead of selecting an implicit fallback.

## What does not change mathematical semantics

The profiles change communication layout, overlap and recomputation,
not the requested global batch or optimizer mathematics:

- gradients remain FP32;
- ordinary AdamW/distributed optimizer state is retained;
- SWA sees exactly the same document-safe 4,095-token left context;
- Full attention remains global within each packed document;
- recomputation does not reuse stale forward activations;
- every rank makes the same update decision.
