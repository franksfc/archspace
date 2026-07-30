# OLMo 3 Stage 1–4 Runbook

本文描述 Stage 1–4 生产主链。
所有路径、实验身份和局部 optimizer schedule 均来自外部输入，不由仓库猜测。

## 1. 执行顺序

```text
全新 Conda 环境
→ 固定源码与 Ascend 运行时校验
→ Base/SFT tokenizer 快照
→ Dolma 3 / Dolmino / Longmino / Think / Instruct 数据快照
→ Stage 1 mmap index
→ Stage 2 FSL cache
→ Stage 3 OBFD cache
→ Think / Instruct assistant-only SFT cache
→ Stage 1 fresh 或 resume
→ Stage 1→2 checkpoint transition
→ Stage 2 resume/train
→ Stage 2→3 checkpoint transition
→ Stage 3 resume/train
→ Stage 3→Think checkpoint transition
→ Think resume/train
→ Think→Instruct checkpoint transition
→ Instruct resume/train
→ native inference
→ PPL / OLMES objective evaluation
```

Checkpoint transition 由两个动作组成：

```text
目标阶段 transition 配置
→ Megatron torch_dist load/re-save
→ 目标阶段 resume 配置
```

这一结构保留模型、Adam moments 和 FP32 master parameters；目标阶段 scheduler、
RNG 和 sample counters 从零开始。同阶段 `resume` 恢复全部训练状态。

### 1.1 入口矩阵

| Phase | Immutable input | Entry point | Output |
|---|---|---|---|
| 环境 | runtime locks、wheelhouse | `scripts/bootstrap_conda_env.sh` | 空前缀构建的训练环境 |
| 上游源码 | `UPSTREAM_LOCK.json` | `scripts/bootstrap_runtime.py` | pristine 固定 revision |
| 下载 | `configs/data/*.json` | `olmo3_data.py download` | 原始 snapshot inventory |
| Stage 1 数据 | Dolma 3 inventory | `tokenize-plan` / `run-tokenize-part` / `finalize-stage1-index` | mmap index |
| Stage 2 数据 | Dolmino inventory | `tokenize-plan` / `runtime-manifest` / `prepare-cache` | FSL cache |
| Stage 3 数据 | Longmino inventory | `tokenize-plan` / `runtime-manifest` / `prepare-cache` | OBFD cache |
| Stage 4 数据 | Dolci inventories | `sft-plan` / `run-sft` | assistant-only packed cache |
| 超参数 | `production.env` + config profiles | `olmo3ctl validate` / train wrapper | immutable run bundle |
| Stage 1 | fresh 或同阶段 checkpoint | `olmo3_train.sh --model 1b\|3b\|7b` | Stage 1 checkpoint |
| Stage 2–4 | 前一阶段 checkpoint | transition plan + resume bundle | 目标阶段 checkpoint |
| 推理 | native `torch_dist` checkpoint | `olmo3eval inference-plan` | frozen responses/cache report |
| PPL | 独立 PPL manifest | `olmo3eval ppl-plan` | token-weighted PPL |
| 下游评测 | frozen OLMES requests | `olmes_freeze_score.py` | objective metrics |

## 2. 外部二进制前提

| 组件 | 合同 |
|---|---|
| OS/架构 | openEuler/AArch64 部署 |
| Python | 3.10.20 |
| PyTorch | 2.6.0 |
| TorchNPU | 2.6.0 |
| CANN/HCCL | `ASCEND_RUNTIME_LOCK.json` 中的 8.3.RC2 合同 |
| NPU | Ascend 910C，训练 Pod 内可见 |

CANN、HCCL、驱动和固件属于宿主机。Conda 只承载 Python 包和显式安装的
Torch/TorchNPU wheels。

`UPSTREAM_LOCK.json.upstream_declared_stack` 记录固定 MindSpeed-LLM revision
声明的 PyTorch 2.7.1。实际部署合同是
`ASCEND_RUNTIME_LOCK.json` 中经过项目验证的
PyTorch/TorchNPU 2.6.0 与 CANN/HCCL 8.3.RC2。该项目级验证偏差不扩展为
MindSpeed 全功能兼容声明。

## 3. 全新 Conda 环境

环境创建命令：

```bash
cd /path/to/olmo3_mindspeed_pipeline

scripts/bootstrap_conda_env.sh \
  --target-env olmo3-mindspeed-pipeline \
  --ascend-wheelhouse /path/to/cp310-aarch64-wheels \
  --ascend-home /usr/local/Ascend/ascend-toolkit/latest \
  --cann-env-script /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --strict-ascend
```

内部顺序：

```text
host CANN/HCCL contract
→ conda create <staging> python=3.10.20
→ locked public Python dependencies
→ torch/torch_npu from local wheelhouse
→ non-editable project install
→ pristine upstream verification
→ TorchNPU tensor probe
→ one-rank HCCL probe
→ staging environment rename
```

该脚本没有 `--source-env`，没有 `conda --clone`，也不读取已有环境的
site-packages。

无 NPU 的控制节点使用非严格创建；训练 Pod 内仍有独立严格检查：

```bash
conda run -n olmo3-mindspeed-pipeline \
  python scripts/verify_ascend_env.py --strict
```

## 4. 生产输入文件

模板：

```bash
cp configs/pipeline/production.env.example \
  /secure/deployment/olmo3-production.env
```

加载：

```bash
set -a
source /secure/deployment/olmo3-production.env
set +a

eval "$(conda shell.bash hook)"
conda activate "$OLMO3_CONDA_ENV"
if [[ -z ${TRAIN_PYTHON:-} ]]; then
  TRAIN_PYTHON="$CONDA_PREFIX/bin/python"
fi

export PYTHONPATH="$OLMO3_ROOT/src"
DATA_CLI="$OLMO3_ROOT/scripts/data/olmo3_data.py"
CKPT_CLI="$OLMO3_ROOT/scripts/checkpoint/olmo3_checkpoint.py"
```

`OLMO3_ROOT` 必须保持导出状态。Conda 中的 Python 包使用它定位未复制进
`site-packages` 的配置、脚本、Megatron overlay 和固定第三方仓库；路径错误时
命令会 fail closed。非 editable 安装中的控制面、模型和 runtime Python
文件必须与该 checkout 逐文件一致；修改源码后需重新运行环境构建脚本。

Python 依赖合同可独立复验：

```bash
"$TRAIN_PYTHON" "$OLMO3_ROOT/scripts/verify_python_dependencies.py"
```

该检查验证完整精确锁及其活动依赖边，只放行文档中固定的
Dolma 1.1.2 / tokenizers 0.20.3 元数据偏差。

输入分类：

| 分类 | 位置 |
|---|---|
| 固定架构 | `configs/models/{1b,3b,7b}.json` |
| 模型扩展 | `configs/variants/{base,siamese_depth}.json` |
| stage 语义 | `configs/stages/*.json` |
| 数据 revision 与 packing | `configs/data/*.json` |
| LR/min LR/warmup/train tokens | `production.env` |
| TP/CP/DP/MBS/GBS/HSDP | topology profile 或训练 CLI override |
| 路径、W&B、端口、run ID | `production.env` |

路径布局：

```bash
BASE_TOKENIZER_DIR="$TOKENIZER_ROOT/dolma2-tokenizer-$BASE_TOKENIZER_REV"
SFT_TOKENIZER_DIR="$TOKENIZER_ROOT/olmo3-instruct-$SFT_TOKENIZER_REV"
OPEN_INSTRUCT_ROOT="$DATA_WORK_ROOT/open-instruct-$OPEN_INSTRUCT_REV"

STAGE1_CKPT="$CHECKPOINT_ROOT/$MODEL-$VARIANT/stage1"
STAGE2_CKPT="$CHECKPOINT_ROOT/$MODEL-$VARIANT/stage2"
STAGE3_CKPT="$CHECKPOINT_ROOT/$MODEL-$VARIANT/stage3"
THINK_CKPT="$CHECKPOINT_ROOT/$MODEL-$VARIANT/think-sft"
INSTRUCT_CKPT="$CHECKPOINT_ROOT/$MODEL-$VARIANT/instruct-sft"
```

可选 W&B 参数：

```bash
WANDB_ARGS=()
if [[ -n ${WANDB_PROJECT:-} ]]; then
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
  [[ -z ${WANDB_ENTITY:-} ]] || WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  [[ -z ${WANDB_BASE_URL:-} ]] || WANDB_ARGS+=(--wandb-base-url "$WANDB_BASE_URL")
fi
```

Stage 4 默认使用 stage 配置中的 3% warmup。只有显式填写 token budget 时才
生成覆盖参数：

```bash
THINK_WARMUP_ARGS=()
if [[ -n ${THINK_WARMUP_TOKENS:-} ]]; then
  THINK_WARMUP_ARGS+=(--warmup-tokens "$THINK_WARMUP_TOKENS")
fi

INSTRUCT_WARMUP_ARGS=()
if [[ -n ${INSTRUCT_WARMUP_TOKENS:-} ]]; then
  INSTRUCT_WARMUP_ARGS+=(--warmup-tokens "$INSTRUCT_WARMUP_TOKENS")
fi

STAGE1_DATA_CACHE_ARGS=()
if [[ -n ${STAGE1_DATA_CACHE_PATH:-} ]]; then
  STAGE1_DATA_CACHE_ARGS+=(--data-cache-path "$STAGE1_DATA_CACHE_PATH")
fi
```

## 5. 源码与 tokenizer

固定第三方源码校验：

```bash
"$TRAIN_PYTHON" "$OLMO3_ROOT/scripts/bootstrap_runtime.py" --clone-missing
"$TRAIN_PYTHON" "$OLMO3_ROOT/scripts/bootstrap_runtime.py"
```

MindSpeed、MindSpeed-LLM 和 OLMo-core 保持 pristine。项目修改只位于根目录
`megatron/`、`src/` 和 `scripts/`。

Tokenizer 快照：

```bash
"$TRAIN_PYTHON" -m huggingface_hub.commands.huggingface_cli download \
  "$BASE_TOKENIZER_REPO" \
  --revision "$BASE_TOKENIZER_REV" \
  --local-dir "$BASE_TOKENIZER_DIR"

"$TRAIN_PYTHON" -m huggingface_hub.commands.huggingface_cli download \
  "$SFT_TOKENIZER_REPO" \
  --revision "$SFT_TOKENIZER_REV" \
  --local-dir "$SFT_TOKENIZER_DIR"
```

固定 tokenizer revision：

| 用途 | repository | revision |
|---|---|---|
| Stage 1–3 | `allenai/dolma2-tokenizer` | `5292e5d6c0f40b67cc765fe41bec991cf4345b5c` |
| Stage 4 | `allenai/dolma-2-tokenizer-olmo-3-instruct-final` | `55f211dfda3974963b869e490617447045069a64` |

Open-Instruct converter：

```bash
git clone "$OPEN_INSTRUCT_REPO" "$OPEN_INSTRUCT_ROOT"
git -C "$OPEN_INSTRUCT_ROOT" checkout --detach "$OPEN_INSTRUCT_REV"
test -z "$(git -C "$OPEN_INSTRUCT_ROOT" status --porcelain)"
```

固定 Open-Instruct revision：

```text
78d1e5aa3cf80a73ce56fd0775e7ad959faf2660
```

## 6. 数据下载

下载根由 `.olmo3-download-root.json` 绑定到单一 repository/revision/plan。
非空且没有相同 claim 的目录会被拒绝。相同 plan 的中断下载通过
`run-download` 恢复。

### 6.1 Stage 1：Dolma 3 Mix 6T

```bash
"$TRAIN_PYTHON" "$DATA_CLI" download \
  --data dolma3_6t \
  --root "$RAW_ROOT/dolma3_6t" \
  --plan "$DATA_CONTRACT_ROOT/dolma3-download.plan.json" \
  --manifest "$DATA_CONTRACT_ROOT/dolma3-raw.inventory.json" \
  --pattern '**/*.jsonl.zst' \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolma3-download.complete.json"
```

固定 revision：

```text
allenai/dolma3_mix-6T
689a3ea2d8217e64d73a5058913fa43ad15e81aa
```

### 6.2 Stage 2：Dolmino 100B

```bash
"$TRAIN_PYTHON" "$DATA_CLI" download \
  --data dolmino_100b \
  --root "$RAW_ROOT/dolmino_100b" \
  --plan "$DATA_CONTRACT_ROOT/dolmino-download.plan.json" \
  --manifest "$DATA_CONTRACT_ROOT/dolmino-raw.inventory.json" \
  --pattern '**/*.jsonl.zst' \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolmino-download.complete.json"
```

固定 revision：

```text
allenai/dolma3_dolmino_mix-100B-1025
f23942ae8a8114af6e992efe8188ce8c531acd16
```

### 6.3 Stage 3：Longmino 50B

```bash
"$TRAIN_PYTHON" "$DATA_CLI" download \
  --data longmino_50b \
  --root "$RAW_ROOT/longmino_50b" \
  --plan "$DATA_CONTRACT_ROOT/longmino-download.plan.json" \
  --manifest "$DATA_CONTRACT_ROOT/longmino-raw.inventory.json" \
  --pattern '**/*.jsonl.zst' \
  --execute \
  --report "$DATA_CONTRACT_ROOT/longmino-download.complete.json"
```

固定 revision：

```text
allenai/dolma3_longmino_mix-50B-1025
8c0b3b265f95514c0f1b643c95da518e261a32a7
```

### 6.4 Stage 4：Think / Instruct

```bash
"$TRAIN_PYTHON" "$DATA_CLI" download \
  --data dolci_think \
  --root "$RAW_ROOT/dolci_think" \
  --plan "$DATA_CONTRACT_ROOT/dolci-think-download.plan.json" \
  --manifest "$DATA_CONTRACT_ROOT/dolci-think-raw.inventory.json" \
  --pattern '**/*.parquet' --checksums \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolci-think-download.complete.json"

"$TRAIN_PYTHON" "$DATA_CLI" download \
  --data dolci_instruct \
  --root "$RAW_ROOT/dolci_instruct" \
  --plan "$DATA_CONTRACT_ROOT/dolci-instruct-download.plan.json" \
  --manifest "$DATA_CONTRACT_ROOT/dolci-instruct-raw.inventory.json" \
  --pattern '**/*.parquet' --checksums \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolci-instruct-download.complete.json"
```

固定数据：

| profile | repository | revision |
|---|---|---|
| `dolci_think` | `allenai/Dolci-Think-SFT-7B` | `72ec0fe32428bada1ec686a9168a0681eecd2094` |
| `dolci_instruct` | `allenai/Dolci-Instruct-SFT` | `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` |

该 Stage 4 数据合同表示上述两个单独快照，不表示旧版
“Dolci 1x + 多个 tool-use 数据 3x”的复合 mixer。

## 7. Tokenization 与 cache

### 7.1 Stage 1 mmap

```bash
PREPROCESS="$OLMO3_ROOT/third_party/MindSpeed-LLM/preprocess_data.py"

"$TRAIN_PYTHON" "$DATA_CLI" tokenize-plan \
  --data dolma3_6t \
  --source-manifest "$DATA_CONTRACT_ROOT/dolma3-raw.inventory.json" \
  --output-root "$TOKENIZED_ROOT/dolma3_6t" \
  --work-root "$DATA_WORK_ROOT/dolma3-tokenize" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --engine megatron --shards 64 --workers 32 \
  --preprocess-script "$PREPROCESS" \
  --plan "$DATA_CONTRACT_ROOT/dolma3-index.plan.json"
```

64 个 part 的数组任务命令：

```bash
PART_ID=${PART_ID:?}
"$TRAIN_PYTHON" "$DATA_CLI" run-tokenize-part \
  --plan "$DATA_CONTRACT_ROOT/dolma3-index.plan.json" \
  --part "$PART_ID" --execute \
  --report "$DATA_CONTRACT_ROOT/dolma3-index.part-$PART_ID.report.json"
```

所有 part 完成后的 index：

```bash
"$TRAIN_PYTHON" "$DATA_CLI" finalize-stage1-index \
  --plan "$DATA_CONTRACT_ROOT/dolma3-index.plan.json" \
  --data-args-path "$TOKENIZED_ROOT/dolma3_6t/data_args_path.txt" \
  --report "$DATA_CONTRACT_ROOT/dolma3-index.complete.json"
```

Stage 1 context length只存在于训练配置；Megatron 预处理命令不携带
`--sequence-length`。

### 7.2 Stage 2 FSL

```bash
"$TRAIN_PYTHON" "$DATA_CLI" tokenize-plan \
  --data dolmino_100b \
  --source-manifest "$DATA_CONTRACT_ROOT/dolmino-raw.inventory.json" \
  --output-root "$TOKENIZED_ROOT/dolmino_100b" \
  --work-root "$DATA_WORK_ROOT/dolmino-tokenize" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --engine dolma --shards 16 --workers 32 \
  --plan "$DATA_CONTRACT_ROOT/dolmino-tokenize.plan.json"
```

16 个 part 的数组任务命令：

```bash
PART_ID=${PART_ID:?}
"$TRAIN_PYTHON" "$DATA_CLI" run-tokenize-part \
  --plan "$DATA_CONTRACT_ROOT/dolmino-tokenize.plan.json" \
  --part "$PART_ID" --execute \
  --report "$DATA_CONTRACT_ROOT/dolmino-tokenize.part-$PART_ID.report.json"
```

Runtime manifest 与 FSL cache：

```bash
"$TRAIN_PYTHON" "$DATA_CLI" inventory \
  --root "$TOKENIZED_ROOT/dolmino_100b" \
  --manifest "$DATA_CONTRACT_ROOT/dolmino-tokenized.inventory.json" \
  --pattern '**/*.npy' --pattern '**/*.csv.gz' \
  --checksums --workers 32

"$TRAIN_PYTHON" "$DATA_CLI" runtime-manifest \
  --data dolmino_100b \
  --inventory "$DATA_CONTRACT_ROOT/dolmino-tokenized.inventory.json" \
  --root "$TOKENIZED_ROOT/dolmino_100b" \
  --manifest "$DATA_CONTRACT_ROOT/dolmino-runtime.data-manifest.json"

"$TRAIN_PYTHON" "$DATA_CLI" prepare-cache \
  --data dolmino_100b \
  --source-manifest "$DATA_CONTRACT_ROOT/dolmino-runtime.data-manifest.json" \
  --work-root "$DATA_WORK_ROOT/dolmino-fsl" \
  --workers 16 \
  --plan "$DATA_CONTRACT_ROOT/dolmino-fsl.plan.json" \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolmino-fsl.complete.json"
```

### 7.3 Stage 3 OBFD

```bash
"$TRAIN_PYTHON" "$DATA_CLI" tokenize-plan \
  --data longmino_50b \
  --source-manifest "$DATA_CONTRACT_ROOT/longmino-raw.inventory.json" \
  --output-root "$TOKENIZED_ROOT/longmino_50b" \
  --work-root "$DATA_WORK_ROOT/longmino-tokenize" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --engine dolma --shards 16 --workers 32 \
  --plan "$DATA_CONTRACT_ROOT/longmino-tokenize.plan.json"
```

16 个 part 的数组任务命令：

```bash
PART_ID=${PART_ID:?}
"$TRAIN_PYTHON" "$DATA_CLI" run-tokenize-part \
  --plan "$DATA_CONTRACT_ROOT/longmino-tokenize.plan.json" \
  --part "$PART_ID" --execute \
  --report "$DATA_CONTRACT_ROOT/longmino-tokenize.part-$PART_ID.report.json"
```

Runtime manifest 与 OBFD cache：

```bash
"$TRAIN_PYTHON" "$DATA_CLI" inventory \
  --root "$TOKENIZED_ROOT/longmino_50b" \
  --manifest "$DATA_CONTRACT_ROOT/longmino-tokenized.inventory.json" \
  --pattern '**/*.npy' --pattern '**/*.csv.gz' \
  --checksums --workers 32

"$TRAIN_PYTHON" "$DATA_CLI" runtime-manifest \
  --data longmino_50b \
  --inventory "$DATA_CONTRACT_ROOT/longmino-tokenized.inventory.json" \
  --root "$TOKENIZED_ROOT/longmino_50b" \
  --manifest "$DATA_CONTRACT_ROOT/longmino-runtime.data-manifest.json"

"$TRAIN_PYTHON" "$DATA_CLI" prepare-cache \
  --data longmino_50b \
  --source-manifest "$DATA_CONTRACT_ROOT/longmino-runtime.data-manifest.json" \
  --work-root "$DATA_WORK_ROOT/longmino-obfd" \
  --workers 16 \
  --plan "$DATA_CONTRACT_ROOT/longmino-obfd.plan.json" \
  --execute \
  --report "$DATA_CONTRACT_ROOT/longmino-obfd.complete.json"
```

OBFD 输出包含文档边界、document-relative positions、terminal-target mask
和 CP slicing 元数据。跨文档 attention 被屏蔽。

### 7.4 Stage 4 assistant-only cache

Converter 合同：

```bash
SFT_CONVERTER="$OPEN_INSTRUCT_ROOT/scripts/data/convert_sft_data_for_olmocore.py"

"$TRAIN_PYTHON" "$DATA_CLI" sft-plan \
  --data dolci_think \
  --raw-manifest "$DATA_CONTRACT_ROOT/dolci-think-raw.inventory.json" \
  --raw-root "$RAW_ROOT/dolci_think" \
  --expected-files 156 \
  --raw-glob "$RAW_ROOT/dolci_think/data/*.parquet" \
  --converted-root "$TOKENIZED_ROOT/dolci_think" \
  --converted-manifest "$DATA_CONTRACT_ROOT/dolci-think-converted.inventory.json" \
  --work-root "$DATA_WORK_ROOT/dolci-think-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --converter "$SFT_CONVERTER" \
  --open-instruct-root "$OPEN_INSTRUCT_ROOT" \
  --workers 16 \
  --plan "$DATA_CONTRACT_ROOT/dolci-think-sft.plan.json"

"$TRAIN_PYTHON" "$DATA_CLI" run-sft \
  --plan "$DATA_CONTRACT_ROOT/dolci-think-sft.plan.json" \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolci-think-sft.complete.json"
```

```bash
"$TRAIN_PYTHON" "$DATA_CLI" sft-plan \
  --data dolci_instruct \
  --raw-manifest "$DATA_CONTRACT_ROOT/dolci-instruct-raw.inventory.json" \
  --raw-root "$RAW_ROOT/dolci_instruct" \
  --expected-files 15 \
  --raw-glob "$RAW_ROOT/dolci_instruct/data/*.parquet" \
  --converted-root "$TOKENIZED_ROOT/dolci_instruct" \
  --converted-manifest "$DATA_CONTRACT_ROOT/dolci-instruct-converted.inventory.json" \
  --work-root "$DATA_WORK_ROOT/dolci-instruct-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --converter "$SFT_CONVERTER" \
  --open-instruct-root "$OPEN_INSTRUCT_ROOT" \
  --workers 16 \
  --plan "$DATA_CONTRACT_ROOT/dolci-instruct-sft.plan.json"

"$TRAIN_PYTHON" "$DATA_CLI" run-sft \
  --plan "$DATA_CONTRACT_ROOT/dolci-instruct-sft.plan.json" \
  --execute \
  --report "$DATA_CONTRACT_ROOT/dolci-instruct-sft.complete.json"
```

训练断言来自完成报告：

```bash
json_field() {
  "$TRAIN_PYTHON" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$1" "$2"
}

THINK_EXPECTED_INSTANCES=$(
  json_field "$DATA_CONTRACT_ROOT/dolci-think-sft.complete.json" instances
)
THINK_EXPECTED_FINGERPRINT=$(
  json_field "$DATA_CONTRACT_ROOT/dolci-think-sft.complete.json" fingerprint
)
INSTRUCT_EXPECTED_INSTANCES=$(
  json_field "$DATA_CONTRACT_ROOT/dolci-instruct-sft.complete.json" instances
)
INSTRUCT_EXPECTED_FINGERPRINT=$(
  json_field "$DATA_CONTRACT_ROOT/dolci-instruct-sft.complete.json" fingerprint
)
```

## 8. 模型与训练超参数

### 8.1 固定模型结构

| size | layers | hidden | FFN | Q/KV heads | head dim |
|---|---:|---:|---:|---:|---:|
| 1B | 16 | 2048 | 8192 | 16/16 | 128 |
| 3B | 16 | 3328 | 13312 | 16/16 | 208 |
| 7B | 32 | 4096 | 11008 | 32/32 | 128 |

共同结构：

```text
vocab 100278, padded 100352
untied embeddings
reordered RMSNorm, epsilon 1e-6
Q/K RMSNorm
RoPE theta 500000, FP32 application
SWA pattern [4096, 4096, 4096, Full]
attention/hidden dropout 0
```

`siamese_depth` 在同一 OLMo 3 结构上增加 Siamese Norm 和 Depth Attention。
Variant 不在 checkpoint 链中切换。

### 8.2 Stage 合同

| stage | sequence | data | schedule | z-loss |
|---|---:|---|---|---:|
| Stage 1 | 8192 | Dolma 3 Mix 6T | cosine | 1e-5 |
| Stage 2 | 8192 | Dolmino 100B | linear, warmup 0 | 1e-5 |
| Stage 3 | 65536 | Longmino 50B | linear, Full-only YaRN | 1e-5 |
| Think SFT | 32768 | Dolci Think | linear, 2 epochs | 0 |
| Instruct SFT | 32768 | Dolci Instruct | linear, 2 epochs | 0 |

Stage 3 的 12 个 SWA 层使用原始 RoPE；4 个 Full-attention 层使用 YaRN
factor 8、beta-fast 32、beta-slow 1、old context 8192。

### 8.3 显式 optimizer 输入

以下值在 `production.env` 中解析：

```text
STAGE1_TRAIN_TOKENS
STAGE1_PEAK_LR
STAGE1_MIN_LR
STAGE1_WARMUP_TOKENS
STAGE2_PEAK_LR
STAGE2_MIN_LR
STAGE3_PEAK_LR
STAGE3_MIN_LR
STAGE3_WARMUP_TOKENS
THINK_PEAK_LR
THINK_WARMUP_TOKENS
INSTRUCT_PEAK_LR
INSTRUCT_WARMUP_TOKENS
```

7B 官方源码中的参考值：

| stage | peak LR | min LR | warmup | GBS |
|---|---:|---:|---:|---:|
| Stage 1 | 3e-4 | 3e-5 | 2000 steps at GBS512 | 512 sequences |
| Stage 2 | 2.0712352850360292e-4 | 0 | 0 | 256 sequences |
| Stage 3 | 2.0712352850360292e-4 | 0 | 200 steps at GBS64 | 64 sequences |

1B/3B 没有官方发布的完整 optimizer recipe；其 LR 与 warmup 保持显式输入。
Warmup 在本仓库以 token 数存储，最终 step 数由
`GBS × sequence_length` 计算。

配置边界：

| 范围 | 修改位置 | 语义 |
|---|---|---|
| 官方模型尺寸、RMSNorm、QK norm、RoPE、SWA pattern | `configs/models/*.json` | 固定结构；正常实验不覆盖 |
| Base / Siamese-Depth | `MODEL`、`VARIANT` | 整条 checkpoint 链固定 |
| LR、min LR、warmup、训练 token/epoch | `production.env` | 每阶段实验输入 |
| GBS、MBS、TP、CP、HSDP、bucket | topology/performance profile | 只改变合法拓扑与性能合同 |
| 路径、端口、run ID、W&B identity | `production.env` | 每次提交独立 |

## 9. Stage 1

训练入口由 `MODEL` 选择：

```bash
TRAIN="$OLMO3_ROOT/scripts/train/olmo3_train.sh"
```

### 9.1 Fresh render

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage1 --data dolma3_6t \
  --topology "$STAGE1_TOPOLOGY" \
  --performance "$STAGE1_PERFORMANCE" \
  --run-id "$STAGE1_RUN_ID" \
  --lifecycle fresh \
  --data-root "$TOKENIZED_ROOT/dolma3_6t" \
  --data-args-path "$TOKENIZED_ROOT/dolma3_6t/data_args_path.txt" \
  "${STAGE1_DATA_CACHE_ARGS[@]}" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --save "$STAGE1_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE1_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE1_TRAIN_TOKENS" \
  --peak-lr "$STAGE1_PEAK_LR" \
  --min-lr "$STAGE1_MIN_LR" \
  --warmup-tokens "$STAGE1_WARMUP_TOKENS" \
  --world-size "$STAGE1_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --nnodes "$STAGE1_NNODES" \
  --node-rank 0 \
  --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE1_MASTER_PORT" \
  --hccl-if-base-port "$STAGE1_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"
```

控制进程：

```bash
STAGE1_RUN_DIR="$OLMO3_ROOT/runs/$STAGE1_RUN_ID"
"$STAGE1_RUN_DIR/checkpoint-activate.sh"
```

每个节点：

```bash
OLMO3_NODE_RANK="$NODE_RANK" "$STAGE1_RUN_DIR/launch.sh"
```

### 9.2 Same-stage resume

新的 `STAGE1_RESUME_RUN_ID`、端口和输出目录对应一次恢复提交：

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage1 --data dolma3_6t \
  --topology "$STAGE1_TOPOLOGY" \
  --performance "$STAGE1_PERFORMANCE" \
  --run-id "$STAGE1_RESUME_RUN_ID" \
  --lifecycle resume \
  --data-root "$TOKENIZED_ROOT/dolma3_6t" \
  --data-args-path "$TOKENIZED_ROOT/dolma3_6t/data_args_path.txt" \
  "${STAGE1_DATA_CACHE_ARGS[@]}" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --load "$STAGE1_CKPT" --save "$STAGE1_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE1_RESUME_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE1_TRAIN_TOKENS" \
  --peak-lr "$STAGE1_PEAK_LR" \
  --min-lr "$STAGE1_MIN_LR" \
  --warmup-tokens "$STAGE1_WARMUP_TOKENS" \
  --world-size "$STAGE1_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --nnodes "$STAGE1_NNODES" \
  --node-rank 0 \
  --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE1_RESUME_MASTER_PORT" \
  --hccl-if-base-port "$STAGE1_RESUME_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"
```

同一 `checkpoint-activate.sh → launch.sh` 顺序适用于该 resume bundle。

## 10. 通用 stage transition

下列函数只负责 `torch_dist` transition；训练由目标 stage 的 resume bundle
负责：

```bash
transition_checkpoint() {
  local transition_run_id=$1
  local destination=$2
  local plan_dir=$3
  local run_dir="$OLMO3_ROOT/runs/$transition_run_id"

  "$TRAIN_PYTHON" "$CKPT_CLI" reshard \
    --resolved "$run_dir/resolved.json" \
    --destination "$destination" \
    --plan-dir "$plan_dir" \
    --write-contract
}
```

每个 transition plan 在全部目标拓扑节点执行：

```bash
OLMO3_NODE_RANK="$NODE_RANK" "$TRANSITION_PLAN_DIR/launch.sh"
```

## 11. Stage 2

### 11.1 Transition render

```bash
STAGE2_TRANSITION_RUN_ID="${STAGE2_RUN_ID}-transition"

"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage2 --data dolmino_100b \
  --topology "$STAGE2_TOPOLOGY" \
  --performance "$STAGE2_PERFORMANCE" \
  --run-id "$STAGE2_TRANSITION_RUN_ID" \
  --lifecycle transition \
  --data-root "$TOKENIZED_ROOT/dolmino_100b" \
  --data-manifest "$DATA_CONTRACT_ROOT/dolmino-runtime.data-manifest.json" \
  --data-work-dir "$DATA_WORK_ROOT/dolmino-fsl" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --load "$STAGE1_CKPT" --save "$STAGE2_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE2_TRANSITION_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE2_TRAIN_TOKENS" \
  --peak-lr "$STAGE2_PEAK_LR" --min-lr "$STAGE2_MIN_LR" \
  --warmup-tokens "$STAGE2_WARMUP_TOKENS" \
  --world-size "$STAGE2_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$STAGE2_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE2_TRANSITION_MASTER_PORT" \
  --hccl-if-base-port "$STAGE2_TRANSITION_HCCL_IF_BASE_PORT"

STAGE2_TRANSITION_PLAN="$OUTPUT_ROOT/$STAGE2_TRANSITION_RUN_ID-reshard"
transition_checkpoint \
  "$STAGE2_TRANSITION_RUN_ID" "$STAGE2_CKPT" "$STAGE2_TRANSITION_PLAN"

OLMO3_NODE_RANK="$NODE_RANK" "$STAGE2_TRANSITION_PLAN/launch.sh"
```

### 11.2 Resume/train render

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage2 --data dolmino_100b \
  --topology "$STAGE2_TOPOLOGY" \
  --performance "$STAGE2_PERFORMANCE" \
  --run-id "$STAGE2_RUN_ID" \
  --lifecycle resume \
  --data-root "$TOKENIZED_ROOT/dolmino_100b" \
  --data-manifest "$DATA_CONTRACT_ROOT/dolmino-runtime.data-manifest.json" \
  --data-work-dir "$DATA_WORK_ROOT/dolmino-fsl" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --load "$STAGE2_CKPT" --save "$STAGE2_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE2_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE2_TRAIN_TOKENS" \
  --peak-lr "$STAGE2_PEAK_LR" --min-lr "$STAGE2_MIN_LR" \
  --warmup-tokens "$STAGE2_WARMUP_TOKENS" \
  --world-size "$STAGE2_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$STAGE2_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE2_MASTER_PORT" \
  --hccl-if-base-port "$STAGE2_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"

STAGE2_RUN_DIR="$OLMO3_ROOT/runs/$STAGE2_RUN_ID"
"$STAGE2_RUN_DIR/checkpoint-activate.sh"
OLMO3_NODE_RANK="$NODE_RANK" "$STAGE2_RUN_DIR/launch.sh"
```

## 12. Stage 3

Stage 3 使用相同 transition/resume 结构，数据与 schedule 参数替换为：

```bash
STAGE3_TRANSITION_RUN_ID="${STAGE3_RUN_ID}-transition"

"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage3 --data longmino_50b \
  --topology "$STAGE3_TOPOLOGY" \
  --performance "$STAGE3_PERFORMANCE" \
  --run-id "$STAGE3_TRANSITION_RUN_ID" \
  --lifecycle transition \
  --data-root "$TOKENIZED_ROOT/longmino_50b" \
  --data-manifest "$DATA_CONTRACT_ROOT/longmino-runtime.data-manifest.json" \
  --data-work-dir "$DATA_WORK_ROOT/longmino-obfd" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --load "$STAGE2_CKPT" --save "$STAGE3_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE3_TRANSITION_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE3_TRAIN_TOKENS" \
  --peak-lr "$STAGE3_PEAK_LR" --min-lr "$STAGE3_MIN_LR" \
  --warmup-tokens "$STAGE3_WARMUP_TOKENS" \
  --world-size "$STAGE3_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$STAGE3_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE3_TRANSITION_MASTER_PORT" \
  --hccl-if-base-port "$STAGE3_TRANSITION_HCCL_IF_BASE_PORT"

STAGE3_TRANSITION_PLAN="$OUTPUT_ROOT/$STAGE3_TRANSITION_RUN_ID-reshard"
transition_checkpoint \
  "$STAGE3_TRANSITION_RUN_ID" "$STAGE3_CKPT" "$STAGE3_TRANSITION_PLAN"

OLMO3_NODE_RANK="$NODE_RANK" "$STAGE3_TRANSITION_PLAN/launch.sh"
```

Transition plan 完成后的训练 bundle：

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage stage3 --data longmino_50b \
  --topology "$STAGE3_TOPOLOGY" \
  --performance "$STAGE3_PERFORMANCE" \
  --run-id "$STAGE3_RUN_ID" \
  --lifecycle resume \
  --data-root "$TOKENIZED_ROOT/longmino_50b" \
  --data-manifest "$DATA_CONTRACT_ROOT/longmino-runtime.data-manifest.json" \
  --data-work-dir "$DATA_WORK_ROOT/longmino-obfd" \
  --tokenizer "$BASE_TOKENIZER_DIR" \
  --load "$STAGE3_CKPT" --save "$STAGE3_CKPT" \
  --output "$OUTPUT_ROOT/$STAGE3_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --train-tokens "$STAGE3_TRAIN_TOKENS" \
  --peak-lr "$STAGE3_PEAK_LR" --min-lr "$STAGE3_MIN_LR" \
  --warmup-tokens "$STAGE3_WARMUP_TOKENS" \
  --world-size "$STAGE3_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$STAGE3_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$STAGE3_MASTER_PORT" \
  --hccl-if-base-port "$STAGE3_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"

STAGE3_RUN_DIR="$OLMO3_ROOT/runs/$STAGE3_RUN_ID"
"$STAGE3_RUN_DIR/checkpoint-activate.sh"
OLMO3_NODE_RANK="$NODE_RANK" "$STAGE3_RUN_DIR/launch.sh"
```

## 13. Stage 4

主链：

```text
Stage 3 final → Think SFT final → Instruct SFT
```

代码也允许 `Stage 3 → Instruct` 独立分支；该分支不是本主链。

### 13.1 Think transition

```bash
THINK_TRANSITION_RUN_ID="${THINK_RUN_ID}-transition"

"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage sft_think --data dolci_think \
  --topology "$SFT_TOPOLOGY" --performance "$SFT_PERFORMANCE" \
  --run-id "$THINK_TRANSITION_RUN_ID" \
  --lifecycle transition \
  --data-root "$TOKENIZED_ROOT/dolci_think" \
  --data-work-dir "$DATA_WORK_ROOT/dolci-think-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --expected-instances "$THINK_EXPECTED_INSTANCES" \
  --expected-fingerprint "$THINK_EXPECTED_FINGERPRINT" \
  --load "$STAGE3_CKPT" --save "$THINK_CKPT" \
  --output "$OUTPUT_ROOT/$THINK_TRANSITION_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --epochs "$THINK_EPOCHS" \
  --peak-lr "$THINK_PEAK_LR" --min-lr "$THINK_MIN_LR" \
  "${THINK_WARMUP_ARGS[@]}" \
  --world-size "$SFT_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$SFT_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$THINK_TRANSITION_MASTER_PORT" \
  --hccl-if-base-port "$THINK_TRANSITION_HCCL_IF_BASE_PORT"

THINK_TRANSITION_PLAN="$OUTPUT_ROOT/$THINK_TRANSITION_RUN_ID-reshard"
transition_checkpoint \
  "$THINK_TRANSITION_RUN_ID" "$THINK_CKPT" "$THINK_TRANSITION_PLAN"

OLMO3_NODE_RANK="$NODE_RANK" "$THINK_TRANSITION_PLAN/launch.sh"
```

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage sft_think --data dolci_think \
  --topology "$SFT_TOPOLOGY" --performance "$SFT_PERFORMANCE" \
  --run-id "$THINK_RUN_ID" --lifecycle resume \
  --data-root "$TOKENIZED_ROOT/dolci_think" \
  --data-work-dir "$DATA_WORK_ROOT/dolci-think-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --expected-instances "$THINK_EXPECTED_INSTANCES" \
  --expected-fingerprint "$THINK_EXPECTED_FINGERPRINT" \
  --load "$THINK_CKPT" --save "$THINK_CKPT" \
  --output "$OUTPUT_ROOT/$THINK_RUN_ID" --python "$TRAIN_PYTHON" \
  --epochs "$THINK_EPOCHS" \
  --peak-lr "$THINK_PEAK_LR" --min-lr "$THINK_MIN_LR" \
  "${THINK_WARMUP_ARGS[@]}" \
  --world-size "$SFT_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$SFT_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$THINK_MASTER_PORT" \
  --hccl-if-base-port "$THINK_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"

THINK_RUN_DIR="$OLMO3_ROOT/runs/$THINK_RUN_ID"
"$THINK_RUN_DIR/checkpoint-activate.sh"
OLMO3_NODE_RANK="$NODE_RANK" "$THINK_RUN_DIR/launch.sh"
```

### 13.2 Instruct transition

```bash
INSTRUCT_TRANSITION_RUN_ID="${INSTRUCT_RUN_ID}-transition"

"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage sft_instruct --data dolci_instruct \
  --topology "$SFT_TOPOLOGY" --performance "$SFT_PERFORMANCE" \
  --run-id "$INSTRUCT_TRANSITION_RUN_ID" \
  --lifecycle transition \
  --data-root "$TOKENIZED_ROOT/dolci_instruct" \
  --data-work-dir "$DATA_WORK_ROOT/dolci-instruct-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --expected-instances "$INSTRUCT_EXPECTED_INSTANCES" \
  --expected-fingerprint "$INSTRUCT_EXPECTED_FINGERPRINT" \
  --load "$THINK_CKPT" --save "$INSTRUCT_CKPT" \
  --output "$OUTPUT_ROOT/$INSTRUCT_TRANSITION_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --epochs "$INSTRUCT_EPOCHS" \
  --peak-lr "$INSTRUCT_PEAK_LR" --min-lr "$INSTRUCT_MIN_LR" \
  "${INSTRUCT_WARMUP_ARGS[@]}" \
  --world-size "$SFT_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$SFT_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$INSTRUCT_TRANSITION_MASTER_PORT" \
  --hccl-if-base-port "$INSTRUCT_TRANSITION_HCCL_IF_BASE_PORT"

INSTRUCT_TRANSITION_PLAN="$OUTPUT_ROOT/$INSTRUCT_TRANSITION_RUN_ID-reshard"
transition_checkpoint \
  "$INSTRUCT_TRANSITION_RUN_ID" "$INSTRUCT_CKPT" \
  "$INSTRUCT_TRANSITION_PLAN"

OLMO3_NODE_RANK="$NODE_RANK" "$INSTRUCT_TRANSITION_PLAN/launch.sh"
```

```bash
"$TRAIN" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --stage sft_instruct --data dolci_instruct \
  --topology "$SFT_TOPOLOGY" --performance "$SFT_PERFORMANCE" \
  --run-id "$INSTRUCT_RUN_ID" \
  --lifecycle resume \
  --data-root "$TOKENIZED_ROOT/dolci_instruct" \
  --data-work-dir "$DATA_WORK_ROOT/dolci-instruct-sft" \
  --tokenizer "$SFT_TOKENIZER_DIR" \
  --expected-instances "$INSTRUCT_EXPECTED_INSTANCES" \
  --expected-fingerprint "$INSTRUCT_EXPECTED_FINGERPRINT" \
  --load "$INSTRUCT_CKPT" --save "$INSTRUCT_CKPT" \
  --output "$OUTPUT_ROOT/$INSTRUCT_RUN_ID" \
  --python "$TRAIN_PYTHON" \
  --epochs "$INSTRUCT_EPOCHS" \
  --peak-lr "$INSTRUCT_PEAK_LR" --min-lr "$INSTRUCT_MIN_LR" \
  "${INSTRUCT_WARMUP_ARGS[@]}" \
  --world-size "$SFT_WORLD_SIZE" \
  --nproc-per-node "$NPROC_PER_NODE" --nnodes "$SFT_NNODES" \
  --node-rank 0 --master-addr "$MASTER_ADDR" \
  --master-port "$INSTRUCT_MASTER_PORT" \
  --hccl-if-base-port "$INSTRUCT_HCCL_IF_BASE_PORT" \
  "${WANDB_ARGS[@]}"

INSTRUCT_RUN_DIR="$OLMO3_ROOT/runs/$INSTRUCT_RUN_ID"
"$INSTRUCT_RUN_DIR/checkpoint-activate.sh"
OLMO3_NODE_RANK="$NODE_RANK" "$INSTRUCT_RUN_DIR/launch.sh"
```

## 14. 推理

原生入口：

```text
scripts/inference/olmo3_native.py
```

能力边界：

```text
native torch_dist checkpoint
TP inference
greedy or explicitly seeded sampling
loglikelihood
generate_until
Stage 3 maximum prompt + generation = 65536
SWA compute window = 4096
```

CPU 合同自测：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH="$OLMO3_ROOT/src:$OLMO3_ROOT" \
"$TRAIN_PYTHON" "$OLMO3_ROOT/scripts/inference/olmo3_native.py" --self-test
```

真实 checkpoint 命令、65K cache smoke 参数和结果格式位于
`docs/INFERENCE_EVALUATION.md`。静态 cache 路径仍会为所有层分配最大长度；
SWA ring-cache 显存优化不属于当前已验收合同。

## 15. 评测

### 15.1 独立 PPL

PPL 使用独立 validation manifest，不从训练 mix 切分。入口：

```text
scripts/eval/olmo3_ppl.py
```

输出包含 token-weighted loss、PPL、有效 token 数、checkpoint identity 和
validation manifest digest。

### 15.2 OLMES objective suite

冻结/评分入口：

```text
scripts/eval/olmes_freeze_score.py
```

执行结构：

```text
freeze pinned task requests
→ native inference JSONL
→ objective score join
```

客观评分覆盖 BBH、DROP、GSM8K、IFEval、MATH、MMLU、PopQA 和
TruthfulQA。AlpacaEval 2 只冻结生成请求，不执行裁判评分。数据 snapshot revision
和 OLMES commit 均进入评测合同。

完整命令位于 `docs/INFERENCE_EVALUATION.md`。

## 16. 验收

控制节点：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH="$OLMO3_ROOT/src:$OLMO3_ROOT" \
"$TRAIN_PYTHON" -m pytest -q
```

生产链验收项：

```text
1. strict Ascend environment
2. data manifest and cache fingerprints
3. Stage 1 fresh checkpoint
4. Stage 1 same-stage resume
5. Stage 1→2 transition plus Stage 2 resume
6. Stage 2→3 transition plus Stage 3 resume
7. Stage 3→Think transition plus Think resume
8. Think→Instruct transition plus Instruct resume
9. model/Adam/FP32-master checkpoint roundtrip
10. finite lm loss, z-loss, total loss and grad norm
11. exact iteration and consumed-sample continuity on resume
12. 65K cache smoke on a Stage 3 checkpoint
13. independent PPL
14. frozen OLMES objective scoring
```

每个模型规模、variant、目标拓扑和推理/评测入口都需要在对应 NPU 环境复验；
CPU 合同测试不能替代真实分布式执行。
