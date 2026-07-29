# Next Concept Prediction (NCP)

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/ArchSpace-Collection/models)

Next Concept Prediction (NCP) investigates autoregressive language models that
learn an explicit representation of upcoming concepts while retaining the
standard next-token prediction interface. The current research implementation
is **ConceptLM V2.2-VQ**, built on an OLMo 3 7B backbone.

> **Project status (2026-07-29):** Large-scale pretraining and continuation
> experiments have been run internally. This repository includes the standalone
> Hugging Face inference path and fixed-protocol evaluation scripts. The
> corresponding pure-HF model revision will be published at
> [`ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast`](https://huggingface.co/ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast).
> Training code and sanitized reproduction configs will be added in later
> updates.

## 1. Basic Information

| Item | Details |
| --- | --- |
| Architecture name | Next Concept Prediction (NCP) |
| Experimental implementation | ConceptLM V2.2-VQ |
| Backbone | OLMo 3 7B |
| Measured parameter count | 8,938,363,793 |
| Current phase | Release models and training logs |

## 2. Architecture Overview

The current model separates token processing into three stages:

1. A 16-layer token encoder processes the input sequence.
2. Tokens are grouped into chunks of four and mean-pooled. An 8-layer
   high-level module models the resulting concept sequence.
3. A 16-layer decoder combines token and concept representations and produces
   the normal causal language-model output.

The V2.2-VQ implementation uses 32 codebooks with 128 entries per codebook and
a head dimension of 128. The current configuration uses shifted concept
features, detached high-level targets, MSE for the high-level prediction loss,
and learned residual routes between encoder, concept, and decoder paths.

The training objective contains the standard language-model loss plus VQ,
high-level prediction, auxiliary, and optional route-regularization terms. In
schematic form:

```text
L_total = L_LM + w_vq L_VQ + w_hlm L_HLM + L_aux + L_route
```

The exact normalization and effective weights will be documented with the
released implementation. The deployed interface remains causal text generation;
the released inference implementation uses the standard Hugging Face
`AutoModelForCausalLM.from_pretrained` interface with decoder-cache generation
and does not require the Megatron training runtime.

## 3. Experimental Status

The following summary was reconstructed from existing launch manifests, logs,
checkpoint trackers, monitoring artifacts, and evaluation outputs. It reports
only facts visible in those artifacts. Internal filesystem paths and training
source code are intentionally not copied into this repository.

### 3.1 Training Stages

| Stage | Recorded status |
| --- | --- |
| Stage 1 | Main large-scale pretraining reached diagnostics at approximately step 1,385,000. The latest retained full-state checkpoint is step 1,382,310, against a planned 1,430,512 steps. |
| Stage 2 | Continuation artifacts and full-state checkpoints exist. A single authoritative public result summary is still being prepared. |

Training was recorded with TensorBoard, Weights & Biases, and additional
diagnostics for VQ usage, representation rank, routing, gradients, and
parameter updates.

## 4. Evaluation Results

[![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-Experiment_Report-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/archspace/ncp-olmo3/reports/NCP-Olmo3--VmlldzoxNzYwMjkzMw)

### 4.1 Stage 1 checkpoint comparison

All values are percentages. Improvements in parentheses are absolute percentage
points over the OLMo 3 Stage 1 checkpoint.

| Model | GSM8K (pass@1) | HumanEval (pass@1) | MMLU (accuracy) |
| --- | ---: | ---: | ---: |
| OLMo 3 Stage 1 | 39.88 | 26.94 | 62.25 |
| NCP OLMo3 Stage 1 StepLast | **46.78 (+6.90)** | **31.27 (+4.33)** | **64.77 (+2.51)** |

The deltas are computed from the underlying unrounded metrics; subtracting the
displayed two-decimal MMLU values can therefore differ by 0.01.

### 4.2 Reproducible GSM8K evaluation

The repository includes a fixed GSM8K evaluation under
[`reproduce/gsm8k`](reproduce/gsm8k). Both NCP and OLMo are loaded exclusively
through `AutoModelForCausalLM.from_pretrained`; this path does not use vLLM or a
direct Megatron checkpoint loader.

The default model is
[`ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast`](https://huggingface.co/ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast).
The comparison baseline is the public OLMo 3 Stage 1 revision:

| Model | `--model-id` | `--revision` |
| --- | --- | --- |
| NCP (ours) | `ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast` | `main` |
| OLMo 3 Stage 1 | `allenai/Olmo-3-1025-7B` | `stage1-step1413814` |

The fixed protocol is:

- GSM8K test split pinned to dataset revision
  `740312add88f781978c0658806c59bc2815b9866`;
- eight fixed, human-written first-N demonstrations;
- prompt format `Question: ...\nAnswer: ...`, verified by an ordered prompt
  corpus hash;
- batch size one per process, one sample per question, seed `1234`, BF16,
  `use_cache=True`, and a 4,096-token context;
- sampling with temperature `0.6`, top-p `0.6`, and at most 512 new tokens;
- stop strings `Question:`, `</s>`, and `<|im_end|>`;
- OLMES-compatible scoring: remove digit-grouping commas, take the last numeric
  span, and compare it exactly with the extracted gold answer.

Install the evaluation dependencies in an isolated environment:

```bash
python -m pip install -r reproduce/gsm8k/requirements.txt
```

The NCP release is a standalone Transformers model. Loading and generation use
only the model repository plus the dependencies listed above; no ConceptLM,
Megatron, or vLLM checkout is required.

Run one independent batch-one process per GPU. The evaluator fixes
`--num-samples 1` and `--seed 1234` to prevent silent protocol drift.

```bash
torchrun --standalone --nproc-per-node=8 \
  -m reproduce.gsm8k.evaluate \
  --model-id ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast \
  --revision main \
  --num-samples 1 \
  --output-dir results/gsm8k/ncp-stage1-last

python -m reproduce.gsm8k.aggregate \
  --output-dir results/gsm8k/ncp-stage1-last
```

On an already allocated eight-GPU node, the equivalent wrapper is:

```bash
bash reproduce/gsm8k/launch_8gpu.sh
```

For OLMo Stage 1, use the same command with the public baseline model and pinned
revision:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m reproduce.gsm8k.evaluate \
  --model-id allenai/Olmo-3-1025-7B \
  --revision stage1-step1413814 \
  --num-samples 1 \
  --output-dir results/gsm8k/olmo3-stage1

python -m reproduce.gsm8k.aggregate \
  --output-dir results/gsm8k/olmo3-stage1
```

Each process writes an independent shard. Aggregation fails closed on missing
or duplicate `(doc_index, sample_index)` keys, scorer drift, protocol mismatch,
or model-artifact mutation.

## 5. Code and Reproduction Plan

Coming Soon

## 6. Remaining Work

### 6.1 Training Roadmap

- [ ] Complete Stage 2 mid-training and consolidate its evaluation results.
- [ ] Complete Stage 3 long-context training and evaluate long-context behavior.
- [ ] Run supervised fine-tuning (SFT) on the selected NCP checkpoint.
- [ ] Compare Stage 1, Stage 2, Stage 3, and SFT checkpoints using a consistent
  evaluation pipeline.

### 6.2 Release Roadmap

- [x] publish selected checkpoints.
- [ ] Add sanitized training logs.
- [ ] Add sanitized training and evaluation code.
- [ ] Publish the model card, evaluation results, limitations, and data
  documentation.
- [ ] Publish experiment logs or summarized training curves where licensing and
  privacy constraints allow.
- [ ] Create a versioned release with a changelog and migration notes.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
