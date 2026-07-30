# Inference and evaluation contract

## Runtime separation

Two independent environments are used.

| Runtime | Purpose | Torch contract |
|---|---|---|
| Ascend training environment | Native checkpoint load, generation, log-likelihood, KV-cache smoke, document PPL | `torch==2.6.0`, `torch_npu==2.6.0`, locked CANN/HCCL |
| OLMES control environment | Dataset download, immutable request construction, objective scoring | CPU `torch==2.8.0`, pinned OLMES dependencies |

The OLMES environment is created from an empty Conda prefix:

```bash
scripts/eval/bootstrap_eval_env.sh \
  --target-env olmo3-olmes-eval
```

The command installs the official checkout at
`third_party/olmes`, verifies commit
`5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`, and downloads the
required NLTK resources into `eval_artifacts/nltk_data`.

## Inputs

Every native evaluation plan has four explicit inputs:

```text
resolved.json from the checkpoint writer run
native Megatron torch_dist checkpoint root
OLMo 3 tokenizer directory
new immutable plan/output identity
```

The plan builder rejects:

- an absent or incomplete checkpoint iteration;
- a checkpoint architecture different from `resolved.json`;
- a checkpoint writer stage different from `resolved.json`;
- a missing tokenizer contract;
- a TP value incompatible with model heads, hidden size, FFN size, or padded
  vocabulary;
- an existing plan or result identity.

Checkpoint loading is model-only, strict, and topology-reshardable:

```text
--ckpt-format torch_dist
--dist-ckpt-strictness raise_all
--no-load-optim
--no-load-rng
```

No training data path enters an inference or PPL command.
The PPL runner does not construct an optimizer, FP32 master parameters,
optimizer moments, a scheduler, or DDP wrappers.

## OLMES request freeze

The first freeze populates the pinned Hugging Face snapshots:

```bash
conda run -n olmo3-olmes-eval \
  python scripts/eval/olmes_freeze_score.py \
  --no-offline \
  freeze \
  --output-dir eval_artifacts/frozen/olmo3-objective-v1
```

Subsequent reconstruction uses the same snapshots in offline mode:

```bash
conda run -n olmo3-olmes-eval \
  python scripts/eval/olmes_freeze_score.py \
  freeze \
  --output-dir eval_artifacts/frozen/olmo3-objective-v2
```

The default suite contains:

```text
AlpacaEval 2 LC request generation
BBH
DROP
GSM8K
IFEval
MATH
MMLU
PopQA
TruthfulQA
```

AlpacaEval requests are generated but not scored. Deferred metrics,
model-based metrics, and LLM-judge metrics are rejected by the scorer.
The pinned `ifeval::tulu` configuration uses
`prompt_level_loose_acc` as its primary metric and also records the strict,
loose, prompt-level, and instruction-level objective metrics.

The official OLMo 3 Base main and held-out suites use the aliases owned by the
pinned OLMES checkout:

```bash
conda run -n olmo3-olmes-eval \
  python scripts/eval/olmes_freeze_score.py \
  --no-offline \
  freeze \
  --output-dir eval_artifacts/frozen/olmo3-base-official-v1 \
  --task olmo3:base:stem_qa_mc \
  --task olmo3:base:nonstem_qa_mc \
  --task olmo3:base:gen \
  --task olmo3:base:math \
  --task olmo3:base:code \
  --task olmo3:base:code_fim \
  --task olmo3:heldout
```

Hub branches and tags without a built-in audited lock are resolved to exact
40-character dataset commits during the online freeze. Those commits, local
snapshot paths, task hashes, and request hashes are stored in the immutable
freeze manifest. Code-task objective metrics execute generated code and
therefore require an isolated code-execution environment.

## Native downstream inference

The native request runner implements all request types used by the frozen
suite:

```text
loglikelihood
generate_until
generate_until_and_loglikelihood
```

The combined request returns the generated text and the OLMES-compatible
gold-continuation log-likelihood fields from one immutable request record.

The following plan evaluates a Stage-4 checkpoint. The same command shape
applies to Stage 1, Stage 2, and Stage 3 by changing `--resolved`,
`--checkpoint`, and the new output identities.

```bash
EVAL_RUN_ID=olmo3-instruct-objective-001
CHECKPOINT_ITERATION=12345
TRAIN_PYTHON=/opt/conda/envs/olmo3-mindspeed-pipeline/bin/python

conda run -n olmo3-mindspeed-pipeline \
  olmo3eval inference-plan \
  --resolved "runs/$INSTRUCT_RUN_ID/resolved.json" \
  --checkpoint /checkpoint/olmo3/sft_instruct \
  --iteration "$CHECKPOINT_ITERATION" \
  --tokenizer "$TOKENIZER_DIR" \
  --frozen-requests \
    eval_artifacts/frozen/olmo3-objective-v1/requests.jsonl \
  --plan-dir "runs/eval/$EVAL_RUN_ID/plan" \
  --output "runs/eval/$EVAL_RUN_ID/responses" \
  --mode run \
  --python "$TRAIN_PYTHON" \
  --nnodes 1 \
  --nproc-per-node 16 \
  --tensor-parallel 16 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --run-cache-smoke-first
```

One-node launch:

```bash
OLMO3_NODE_RANK=0 "runs/eval/$EVAL_RUN_ID/plan/launch.sh"
```

Multi-node launch executes the same immutable `launch.sh` on every node with
distinct `OLMO3_NODE_RANK` values in `[0, nnodes)`.

Request-level parallelism has two independent dimensions:

```text
--partition-index / --partition-count
data-parallel replicas inside one native launch
```

Every TP rank participates in model collectives. Only TP rank zero writes the
response file for its DP replica.

## Objective scoring

All rank response files for one frozen request identity are passed together:

```bash
MODEL_NAME=olmo3-instruct-objective-001
MODEL_HASH=sha256-or-run-contract-digest

conda run -n olmo3-olmes-eval \
  python scripts/eval/olmes_freeze_score.py \
  score \
  --freeze-dir eval_artifacts/frozen/olmo3-objective-v1 \
  --responses "runs/eval/$EVAL_RUN_ID"/responses/partition_*_of_*/rank_*.jsonl \
  --output-dir "runs/eval/$EVAL_RUN_ID/scores" \
  --model-name "$MODEL_NAME" \
  --model-hash "$MODEL_HASH"
```

The scorer verifies request IDs, request hashes, response completeness,
OLMES commit identity, task completeness, and objective metric policy before
writing `metrics.json`.

## Independent document PPL

The manifest format is:

```text
dataset-name,relative/path/to/little-endian-uint32-token-stream
```

Each stream contains EOS-delimited documents. Documents are evaluated
independently, truncated to the requested sequence length, and right padded.
Padding transitions and dummy samples have zero loss weight. The dataset is
padded to a complete global batch; no real document is repeated or dropped.

Plan construction:

```bash
PPL_RUN_ID=olmo3-stage3-ppl-001
STAGE3_RUN_ID=olmo3-stage3-train-001
CHECKPOINT_ITERATION=11921
TRAIN_PYTHON=/opt/conda/envs/olmo3-mindspeed-pipeline/bin/python

conda run -n olmo3-mindspeed-pipeline \
  olmo3eval ppl-plan \
  --resolved "runs/$STAGE3_RUN_ID/resolved.json" \
  --checkpoint /checkpoint/olmo3/stage3 \
  --iteration "$CHECKPOINT_ITERATION" \
  --tokenizer "$TOKENIZER_DIR" \
  --manifest "$PPL_MANIFEST" \
  --cache-dir "$PPL_CACHE_DIR" \
  --output "runs/ppl/$PPL_RUN_ID/ppl.json" \
  --plan-dir "runs/ppl/$PPL_RUN_ID/plan" \
  --sequence-length 8192 \
  --python "$TRAIN_PYTHON" \
  --nnodes 1 \
  --nproc-per-node 16 \
  --tensor-parallel 16 \
  --master-addr 127.0.0.1 \
  --master-port 29501
```

Launch:

```bash
OLMO3_NODE_RANK=0 "runs/ppl/$PPL_RUN_ID/plan/launch.sh"
```

The final JSON records:

```text
aggregate lm loss and PPL
aggregate total loss and z-loss
per-source lm loss and PPL
per-source total loss and z-loss
document, target, truncation, padding, and dummy-sample accounting
checkpoint and manifest identity
```

PPL is `exp(token-weighted mean LM cross entropy)`. Z-loss is reported
separately and is not included in PPL.

## KV-cache validation

Short-context cache equivalence:

```bash
CACHE_RUN_ID=olmo3-stage1-cache-001
STAGE1_RUN_ID=olmo3-stage1-train-001
TRAIN_PYTHON=/opt/conda/envs/olmo3-mindspeed-pipeline/bin/python

conda run -n olmo3-mindspeed-pipeline \
  olmo3eval inference-plan \
  --resolved "runs/$STAGE1_RUN_ID/resolved.json" \
  --checkpoint /checkpoint/olmo3/stage1 \
  --tokenizer "$TOKENIZER_DIR" \
  --plan-dir "runs/cache/$CACHE_RUN_ID/plan" \
  --output "runs/cache/$CACHE_RUN_ID/result" \
  --mode cache-smoke \
  --max-length 8192 \
  --python "$TRAIN_PYTHON" \
  --nnodes 1 \
  --nproc-per-node 16 \
  --tensor-parallel 16
```

Stage-3/4 65K boundary validation:

```bash
LONG_CACHE_RUN_ID=olmo3-stage3-long-cache-001
STAGE3_RUN_ID=olmo3-stage3-train-001
TRAIN_PYTHON=/opt/conda/envs/olmo3-mindspeed-pipeline/bin/python

conda run -n olmo3-mindspeed-pipeline \
  olmo3eval inference-plan \
  --resolved "runs/$STAGE3_RUN_ID/resolved.json" \
  --checkpoint /checkpoint/olmo3/stage3 \
  --tokenizer "$TOKENIZER_DIR" \
  --plan-dir "runs/cache/$LONG_CACHE_RUN_ID/plan" \
  --output "runs/cache/$LONG_CACHE_RUN_ID/result" \
  --mode long-cache-smoke \
  --max-length 65536 \
  --python "$TRAIN_PYTHON" \
  --nnodes 1 \
  --nproc-per-node 16 \
  --tensor-parallel 16
```

The Stage-3/4 inference contract uses:

```text
4 Full Attention layers with factor-8 YaRN
12 SWA layers with original RoPE
4096-token SWA compute window
65,536-token model limit
last-token-only vocabulary projection after long prefill
```

The current cache is static. All 16 layers reserve the configured maximum
length. SWA layers read only the rightmost 4096 tokens during cached
attention, but they do not yet use a 4096-slot ring cache.

## Verification state

| Component | Implemented | CPU-tested | NPU validation required |
|---|---:|---:|---:|
| Immutable inference/PPL plan generation | yes | yes | no |
| Checkpoint identity and complete-iteration checks | yes | yes | real storage mount |
| 1B/3B/7B, base/siamese-depth config matrix | yes | yes | real checkpoint load |
| Native generation and log-likelihood | yes | runner self-test | yes |
| TP inference and runtime output gather | yes | static contract | yes |
| Short KV-cache equivalence | yes | control logic | yes |
| 65K prefill and incremental decode | yes | static contract | yes |
| Static Full/SWA KV-cache policy | yes | static contract | yes |
| Ring/paged SWA cache | no | no | not applicable |
| Independent document PPL | yes | plan and dataset accounting | yes |
| Pinned OLMES freeze/score | yes | paths, locks, policy | dataset download and full run |

Native inference currently requires PP=1, CP=1, SP disabled, static equal-length
batches, and no padding. TP is supported. Packed cache, dynamic batching,
continuous batching, flash decode, and ring/paged cache are outside the
current contract.
