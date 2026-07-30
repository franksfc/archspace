# Checkpoint lifecycle

The production format is Megatron Core `torch_dist` with the distributed
AdamW state saved in fully-sharded model space. This is the format used by
Stage 1, Stage 2, Stage 3, and SFT.

`olmo3ckpt` is a control-plane wrapper. Tensor loading, optimizer resharding,
and storage serialization remain upstream Megatron Core operations.

## Identity contract

Each checkpoint root has an `olmo3_checkpoint_contract.json` sidecar. It
records:

- the exact model size and persistent architecture shape;
- either the `olmo3_base_v1` or `olmo3_siamese_depth_v1` keyspace;
- the last writer's stage and TP/CP/PP/DP/HSDP topology;
- the resume-critical schedule: stage, global/micro batch size, train
  iterations, peak/minimum LR, scheduler, warmup steps/tokens, and the fixed
  AdamW hyperparameters (weight decay, betas, epsilon, z-loss, and grad clip);
- that the checkpoint contains model weights, FP32 master parameters, and both
  Adam moments;
- whether optimizer state is topology-reshardable.

Absolute checkpoint paths are never stored in the contract. The checkpoint
tree can therefore be moved or mounted at another path.

The loader is fail-closed. A 1B checkpoint cannot be labeled 3B, and a Base
checkpoint cannot be loaded into Siamese/Depth (or vice versa). For a
checkpoint created before the sidecar existed, inspect it and explicitly adopt
it once:

```bash
olmo3ckpt inspect --checkpoint "$CHECKPOINT_ROOT"

olmo3ckpt adopt \
  --checkpoint "$CHECKPOINT_ROOT" \
  --resolved "$RUN_DIR/resolved.json" \
  --i-verified-model-and-variant
```

The acknowledgement is intentional: storage metadata alone cannot prove which
custom keyspace produced a legacy checkpoint.

`inspect` does not treat `metadata.json` alone as proof that an iteration
finished. A complete `torch_dist` iteration must also contain a non-empty
`common.pt`, a non-empty PyTorch DCP `.metadata`, and at least one non-empty
`.distcp` shard. The inspector decodes only the DCP metadata—not tensor
payloads—and checks every `storage_data` reference for a safe relative path,
an existing file, and an in-bounds `offset + length`.

## Fresh, resume, and transition

The three lifecycle modes have different state semantics:

| lifecycle | model | Adam moments | FP32 master params | scheduler | RNG | iteration/sample counters |
| --- | --- | --- | --- | --- | --- | --- |
| `fresh` | initialize | initialize | initialize | initialize | initialize | reset |
| `resume` | load | load | load | load | load | load |
| `transition` | load | load | load | reset to target stage | reset | reset |

A same-stage `resume` restores scheduler and counters, so its source contract's
schedule must match the resolved target schedule exactly. This prevents a
nominal resume from silently changing GBS/MBS, total iterations, LR/warmup, or
AdamW constants. A stage `transition` intentionally does not compare schedules
because it starts the destination stage's schedule. Contracts created before
the optional schedule field was introduced remain loadable for compatibility.

For a same-stage TP/PP reshard, Megatron can reshard model and optimizer
tensors but cannot map the old model-parallel RNG tracker exactly; the contract
therefore records `load_rng: false` for that one case. CP/HSDP-only resharding
does not trigger this exception. Scheduler and sample/iteration counters still
resume.

Stage transitions are restricted to:

```text
Stage1 -> Stage2 -> Stage3 -> Think SFT -> Instruct SFT
                          \-> Instruct SFT
```

The `--olmo3-stage-transition` implementation in the vendored root
`megatron/` package performs
the transition policy. In particular, a batch-size or topology change does not
discard Adam state. The target stage's LR/warmup schedule replaces the previous
schedule, while per-parameter moments and FP32 master weights are retained.

A rendered run has the following storage preflight:

```bash
olmo3ckpt preflight --resolved "$RUN_DIR/resolved.json"
```

`--write` creates the target root contract and immutable run intent. When
the checkpoint filesystem is mounted only inside workers, copy only its
`olmo3_checkpoint_contract.json` to the submission host and use:

```bash
olmo3ckpt preflight \
  --resolved "$RUN_DIR/resolved.json" \
  --skip-storage-check \
  --source-contract "$LOCAL_SOURCE_CONTRACT"
```

`--skip-storage-check` never skips model-size, variant, stage-transition, or
topology compatibility checks. For a `resume` or `transition`, it is rejected
unless `--source-contract` supplies a sealed local copy. The normal storage
preflight follows after the volume is mounted.

The normal rendered bundle wires this into launch safely:

```bash
# One control process after checkpoint storage is mounted.
"$RUN_DIR/checkpoint-activate.sh"

# Every distributed worker with its deployment coordinate.
OLMO3_NODE_RANK="$NODE_RANK" "$RUN_DIR/launch.sh"
```

Activation is the only writer of the destination sidecar and per-run intent.
`launch.sh` calls `olmo3ckpt verify-prepared` read-only before `torchrun`; it
checks the complete source iteration, sealed source/destination identities,
stage transition, topology reshard contract, and exact run intent. Concurrent
execution of `checkpoint-activate.sh` on every node is prohibited. `launch.sh` also accepts
`NODE_RANK` as a fallback and range-checks the selected rank.
`olmo3ctl launch` performs the same activation for a single-node run; it is
rejected for `nnodes>1` because independent nodes would otherwise race while
rendering the same immutable run ID.

## TP/CP/HSDP reshard

`torch_dist` restores tensors into the target model's sharded state dict.
Changing CP or HSDP therefore does not require an intermediate checkpoint
format. TP changes, and distributed-optimizer changes across model-parallel
topologies, require the source to have been saved in fully-sharded model space.
This pipeline keeps Megatron Core's default enabled and never emits
`--no-ckpt-fully-parallel-save`.

Create a one-shot load/re-save plan:

```bash
olmo3ckpt reshard \
  --resolved "$TARGET_RUN_DIR/resolved.json" \
  --destination "$RESHARDED_CHECKPOINT_ROOT" \
  --plan-dir "$PLAN_DIR" \
  --write-contract

# The same immutable plan on every node.
OLMO3_NODE_RANK="$NODE_RANK" "$PLAN_DIR/launch.sh"
```

The resolved config supplies every source path and target TP/CP/PP/HSDP value.
Topology resharding must use a new destination root; an in-place resume is
allowed only when topology is unchanged. This keeps the root identity truthful
after a failed or successful conversion.
The generated launcher calls `scripts/checkpoint/reshard_checkpoint.py`, which
uses Megatron's `setup_model_and_optimizer()`, strict checkpoint loader, and
`save_checkpoint()`. There is no second optimizer serialization
implementation.

Like the training launcher, a multi-node checkpoint plan resolves the deployment
coordinate at execution time. `OLMO3_NODE_RANK` takes precedence over
`NODE_RANK`, with the resolved `runtime.node_rank` used only as a fallback. The
launcher rejects non-integer ranks and ranks outside
`[0, runtime.nnodes)`. Thus every worker executes the same immutable plan
without freezing all workers to rank zero.

The TP2/SIO/MC2 topology remains an MC2 process during resharding. Replacing
the training entry point does not drop `--use-ascend-mc2`; the generated
reshard argv contains the flag before the checkpoint script imports the
MindSpeed adaptor.

## Round-trip validation

The round-trip plan performs two strict load/re-save passes and includes the
distributed optimizer in both:

```bash
olmo3ckpt roundtrip \
  --resolved "$RESUME_RUN_DIR/resolved.json" \
  --work-dir "$ROUNDTRIP_WORK_DIR" \
  --plan-dir "$PLAN_DIR"

OLMO3_NODE_RANK="$NODE_RANK" "$PLAN_DIR/launch.sh"
```

The target production topology provides end-to-end storage validation; a
small TP/CP/HSDP topology provides smoke coverage. Real NPU execution is
required to prove tensor and optimizer restoration; CPU unit tests validate
contracts and command construction only.

Pass 2 automatically uses a different valid rendezvous port from pass 1
(`master_port + 1`, or `65534` when pass 1 uses `65535`). This prevents the
second `torchrun` from attaching to a rendezvous store that is still being
torn down. `plan.json` records both ports.

`torch_dist` is the supported format for checkpoint movement, stage
transitions, TP/CP/HSDP resharding, and round-trip validation. These native
operations preserve Adam moments and FP32 master parameters.
