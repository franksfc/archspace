# SiameseNorm and DepthAttention for OLMo 3

This branch implements SiameseNorm and DepthAttention on top of the OLMo 3 architecture.

## Repository Layout

| Path | Description |
|---|---|
| [`archs/SiameseNorm-DepthAttention`](archs/SiameseNorm-DepthAttention) | Hugging Face Transformers-compatible implementation for inference and generation |
| [`reproduce/Megatron-LM`](reproduce/Megatron-LM) | Ascend/MindSpeed training, checkpoint conversion, inference, and evaluation pipeline |

The reproduction pipeline covers:

- Stage 1: 8K pretraining
- Stage 2: 8K mid-training
- Stage 3: 65K long-context training
- Stage 4: Think SFT and Instruct SFT
- Checkpoint save, resume, transition, and Hugging Face conversion
- Native inference, PPL validation, and OLMES evaluation

Detailed setup and execution commands are available in the
[reproduction runbook](reproduce/Megatron-LM/docs/RUNBOOK.md).

## Released Checkpoints and Logs

- [Hugging Face checkpoints](https://huggingface.co/ArchSpace-Collection/SiameseNorm-DepthAttention)
- [Weights & Biases report](https://wandb.ai/archspace/SiameseNormDepthAttention/reports/Siamese-Norm-and-Depth-Attention-in-OLMo-3-1B--VmlldzoxNzYwMzAwNw)

The released checkpoints currently cover the complete four-stage OLMo 3 1B
SiameseNorm–DepthAttention pipeline. Matched baseline training and additional
model-scale validation are ongoing.
