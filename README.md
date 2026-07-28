# Next Concept Prediction (NCP)

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/ArchSpace-Collection/models)

Next Concept Prediction (NCP) investigates autoregressive language models that
learn an explicit representation of upcoming concepts while retaining the
standard next-token prediction interface. The current research implementation
is **ConceptLM V2.2-VQ**, built on an OLMo 3 7B backbone.

> **Project status (2026-07-28):** Large-scale pretraining and continuation
> experiments have been run internally. This repository currently documents the
> architecture and observed results only. Training code, sanitized reproduction
> configs, model weights, and the Hugging Face implementation will be added in
> later updates.

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
the current Megatron implementation has also been exercised with decoder-cache
generation.

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

Coming Soon

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
