# Configuration reference

The JSON files under `configs/` describe reusable facts. Cluster-specific
values are CLI arguments or `--set dotted.path=value` overrides.

## Models

The architecture values come from the pinned OLMo-core
`TransformerConfig.olmo3_1B`, `olmo3_3B` and `olmo3_7B` constructors.

| Model | Layers | Hidden | FFN | Q heads | KV heads | Head dim |
|---|---:|---:|---:|---:|---:|---:|
| 1B | 16 | 2,048 | 8,192 | 16 | 16 | 128 |
| 3B | 16 | 3,328 | 13,312 | 16 | 16 | 208 |
| 7B | 32 | 4,096 | 11,008 | 32 | 32 | 128 |

All sizes use:

- true vocabulary 100,278 and padded vocabulary 100,352;
- untied embedding/output weights;
- Q/K RMS normalization and RMS epsilon `1e-6`;
- RoPE theta 500,000 applied in FP32;
- three 4,096-token SWA layers followed by one Full-attention layer;
- zero attention/hidden dropout.

Architecture dimensions are official. LR, min LR, warmup tokens, train tokens
and deployment topology are experiment inputs. In particular, the repository
does not label a locally chosen 1B or 3B optimizer schedule as an official
released recipe.

## Variants

| Name | Model implementation | Siamese Norm | Depth Attention | Checkpoint keyspace |
|---|---|---|---|---|
| `base` | `olmo3` | off | off | `olmo3_base_v1` |
| `siamese_depth` | `olmo3_siamese_depth` | `hybrid_pre` | stride 8 | `olmo3_siamese_depth_v1` |

The resolver validates the complete variant contract; individual feature flags
are not intended as ad-hoc launch toggles.

## Stages and data must match

| Stage | Required data profile | Loader |
|---|---|---|
| `stage1` | `dolma3_6t` | `megatron_indexed` |
| `stage2` | `dolmino_100b` | `olmo3_numpy_fsl` |
| `stage3` | `longmino_50b` | `olmo3_numpy_packed` |
| `sft_think` | `dolci_think` | `olmo3_sft_numpy` |
| `sft_instruct` | `dolci_instruct` | `olmo3_sft_numpy` |

A packed data profile cannot be composed with an unpacked stage or vice versa.

## Topology arithmetic

The resolver computes:

```text
MP = TP × CP × PP
DP = world_size / MP
GA = GBS / (DP × MBS)
tokens_per_step = GBS × sequence_length
train_iters = ceil(train_tokens / tokens_per_step)
warmup_steps = ceil(warmup_tokens / tokens_per_step)
HSDP instances = (DP × CP) / HSDP shard_size
```

All divisions that represent group sizes or gradient accumulation must be
integral. Pipeline parallelism currently fails closed at PP1.

`--world-size`, `--tp`, `--cp`, `--micro-batch-size`,
`--global-batch-size`, `--hsdp-shard-size` and `--ddp-num-buckets` override a
topology preset without editing it.

## Hyperparameters

Direct hyperparameter flags:

```text
--train-tokens
--epochs
--peak-lr
--min-lr
--warmup-tokens
--save-interval
--eval-interval
--eval-iters
```

Warmup is stored in tokens and converted to steps only after the final GBS and
sequence length are known. This keeps warmup semantics stable when topology or
batch size changes. SFT uses a warmup fraction instead; its dataset instance
count determines the exact number of steps.

Every path is an input:

```text
--data-root
--data-manifest
--data-work-dir
--data-args-path
--data-cache-path
--tokenizer
--load
--save
--output
--valid-manifest
--valid-cache-dir
--python
```

`--data-manifest` is required only for the Stage 2/3 NumPy backends.
Stage 1 uses `--data-args-path`. Its optional `--data-cache-path` selects a
shared Megatron sample-index cache; when omitted, the cache remains isolated at
`<output>/data-cache`. Populate an explicit shared cache with one job before
starting unrelated consumers; independent cold starts are not mutually
synchronized by Megatron. Stage 4 uses its prepared data root, work directory,
expected instance count, and content fingerprint.

Stage 1 PPL validation is opt-in and fail-closed: `--valid-manifest` and a
positive `--eval-iters` must be supplied together. Omitting both disables
validation. Stage 2, Stage 3, and SFT follow the official recipes with
`eval_iters=0` and do not attach the Stage 1 PPL evaluator
without taking a split from the training mix.

No example path is a project default.

## W&B and identity

`--run-id` is required and becomes both the immutable run-contract directory
name and W&B run ID. A previously rendered ID is never overwritten.

W&B is enabled only when `--wandb-project` is supplied. Entity and base URL are
optional parameters. `--wandb-log-style llamafactory` is the frozen default and
retains the production OLMo2-compatible `train/*` and `eval/*` metric keys;
select `native` explicitly to use unmodified Megatron names. `resume` uses W&B
resume mode `allow`; fresh and transition runs use a new identity.

## Overrides

Any JSON field can be set with a dotted override:

```bash
--set optimization.peak_lr=0.00025 \
--set optimization.warmup_tokens=838860800 \
--set training.train_tokens=50000000000
```

Dedicated CLI flags are the primary auditable interface. Dotted overrides use
JSON syntax for booleans, lists and strings containing punctuation.
