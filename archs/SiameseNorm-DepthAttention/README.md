---
library_name: transformers
pipeline_tag: text-generation
tags:
  - olmo3
  - custom-code
  - siamese-norm
  - depth-attention
  - sliding-window-attention
---

# OLMo 3 with SiameseNorm and DepthAttention

This repository contains the 1B checkpoints from the four-stage OLMo 3
training pipeline with SiameseNorm and DepthAttention.

## Checkpoints

| Checkpoint | Hub subfolder | Context length |
|---|---|---:|
| Stage 1 pretraining | `olmo3/1b/stage1` | 8,192 |
| Stage 2 mid-training | `olmo3/1b/stage2` | 8,192 |
| Stage 3 long-context training | `olmo3/1b/stage3` | 65,536 |
| Stage 4 Think SFT | `olmo3/1b/stage4/think` | 65,536 |
| Stage 4 Instruct SFT | `olmo3/1b/stage4/instruct` | 65,536 |

Stage 3 and Stage 4 apply YaRN only to Full-attention layers. Sliding-window
attention layers retain the original RoPE and a 4,096-token window.

## Loading

Select one checkpoint through `subfolder`. SDPA is the recommended and
release-validated BF16 inference backend:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "ArchSpace-Collection/SiameseNorm-DepthAttention"
subfolder = "olmo3/1b/stage4/instruct"

tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    subfolder=subfolder,
    trust_remote_code=True,
    fix_mistral_regex=False,
)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
```

`fix_mistral_regex=False` is intentional and preserves the tokenizer behavior
used during training.

The eager backend can also load these checkpoints. For BF16 generation, SDPA
is recommended because eager cache partitioning can introduce small rounding
differences when the leading logits are nearly tied; in that narrow case,
greedy generation can select a different token. This is a numerical
backend/cache-partition effect, not a checkpoint conversion or weight-integrity
problem.

The repository root contains the shared custom modeling code required by
Transformers remote-code loading. Each checkpoint subfolder also contains a
self-contained copy of its configuration, tokenizer, modeling code, and
weights.

## Architecture

- 16 transformer layers
- hidden size 2,048
- intermediate size 8,192
- 16 query heads and 16 key/value heads
- 128-dimensional attention heads
- 3:1 sliding-window/full-attention pattern
- 4,096-token sliding window
- reordered RMSNorm, SiameseNorm, and DepthAttention

The Hugging Face implementation is intended for inference and generation.
Exact continuation of the native distributed training objective should use
the accompanying MindSpeed/Megatron training pipeline.
