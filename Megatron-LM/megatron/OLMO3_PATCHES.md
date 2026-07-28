# OLMo3 Megatron delta

This directory is a complete Megatron Core `core_v0.12.1` package based on
commit `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`. Project changes are kept here;
`third_party/MindSpeed` and `third_party/MindSpeed-LLM` must remain pristine.

The retained production changes are intentionally narrow:

- dense vocabulary z-loss with separate LM/z/total reporting;
- OLMo3 full-precision RoPE, packed THD RoPE, Full-only YaRN support, and
  inference-cache RoPE;
- OLMo3 attention/cache integration needed by Full/SWA and packed CP;
- official OLMo3 weight-decay parameter grouping;
- exact packed-token accounting and deferred loss reporting;
- ordinary partial DistributedOptimizer/HSDP with correct global gradient
  statistics;
- inter-backward overlap and the validated TP2 lane-pack transport;
- same-stage checkpoint resume, distributed optimizer reshard, and explicit
  Stage 1 -> 2 -> 3 transition semantics;
- NPU-safe no-fork distributed-checkpoint writing;
- deterministic Stage 4 sampling and CP-loader ownership.
- post-import MindSpeed-LLM feature guards, W&B metric adaptation, and
  NPU-friendly cross-entropy glue, all implemented in the root `megatron/`
  package without editing either third-party checkout.

MindSpeed applies runtime monkey patches after importing Megatron. The project
therefore calls `install_olmo3_mindspeed_compatibility()` from
`megatron/olmo3_mindspeed_compat.py` immediately after importing
`mindspeed_llm.tasks.megatron_adaptor_v2`. That installer restores only
full-precision RoPE and the validated ordinary HSDP scheduling on top of the
pristine MindSpeed runtime.
