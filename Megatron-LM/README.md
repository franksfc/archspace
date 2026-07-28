# OLMo 3 MindSpeed Pipeline

该仓库包含 OLMo 3 在 Ascend/MindSpeed 上的数据处理、Stage 1–4 训练、
checkpoint 迁移与恢复、原生推理、PPL 验证和 OLMES 客观评测入口。

主训练链：

```text
Dolma 3 Mix 6T
  → Stage 1 pretraining, 8K
  → Dolmino 100B Stage 2, 8K
  → Longmino 50B Stage 3, 65K
  → Think SFT, 32K
  → Instruct SFT, 32K
  → native inference
  → PPL / OLMES objective evaluation
```

## 状态边界

| 范围 | 状态 |
|---|---|
| Base 与 Siamese/Depth 两种模型定义 | 已实现，静态与单元测试覆盖 |
| 1B/3B/7B 架构解析 | 已实现，来源固定到 OLMo-core 构造器 |
| Stage 1–4 数据与训练命令生成 | 已实现 |
| `torch_dist` checkpoint save/resume/transition | 已实现 |
| 原生推理与 65K KV-cache smoke 入口 | 已实现；需要在目标 NPU 环境验证 |
| PPL 与 OLMES 客观评测 | 已实现；需要使用目标 checkpoint 验证 |

“已实现”不等同于“所有模型规模和拓扑均已在真实 NPU 上完成端到端训练”。

## 目录

```text
configs/
  models/         1B、3B、7B 架构
  variants/       base、siamese_depth
  stages/         Stage 1、2、3、Think SFT、Instruct SFT
  data/           数据来源、tokenizer 和 packing 合同
  topology/       Stage 1–4 与 TP2/SIO 参考拓扑
  performance/    ordinary HSDP、TP2/SIO/MC2、CP single-halo
  pipeline/       生产输入模板
scripts/
  data/           下载、tokenization、index/cache 构建
  train/          通用 1B/3B/7B 训练入口
  checkpoint/     checkpoint 检查与重分片
  inference/      原生 checkpoint 推理
  eval/           PPL 与 OLMES 冻结/评分
src/              配置解析、运行时和模型实现
megatron/         受控 Megatron 修改
third_party/      固定 revision 的上游仓库
```

## 环境

Conda 环境从空前缀创建。该过程不读取、不复制、不 clone 任何已有 Conda
环境。

```bash
cd /path/to/olmo3_mindspeed_pipeline

scripts/bootstrap_conda_env.sh \
  --target-env olmo3-mindspeed-pipeline \
  --ascend-wheelhouse /path/to/cp310-aarch64-wheels \
  --strict-ascend
```

`--ascend-wheelhouse` 包含与
[`ASCEND_RUNTIME_LOCK.json`](ASCEND_RUNTIME_LOCK.json) 匹配的
`torch==2.6.0` 和 `torch-npu==2.6.0` AArch64/CPython 3.10 wheels。
CANN/HCCL、驱动和固件属于宿主机二进制栈，不属于 Conda 环境。

固定的 MindSpeed-LLM revision 上游声明 PyTorch 2.7.1；本部署使用经过项目
验证的 PyTorch/TorchNPU 2.6.0 与 CANN/HCCL 8.3.RC2。
该组合属于项目级集群验证偏差，不代表 MindSpeed 的完整上游支持矩阵。
同样，Dolma 1.1.2 的包元数据限制 `tokenizers<=0.19.1`，而
Transformers 4.46.1 要求 `tokenizers>=0.20,<0.21`；环境固定为实际验证过的
0.20.3，并由 `scripts/verify_python_dependencies.py` 只允许这一项精确的
元数据偏差。其他缺包、版本偏差或依赖冲突均使环境构建失败。

控制节点无 NPU 时省略 `--strict-ascend`；所有训练 Pod 内仍执行：

```bash
conda run -n olmo3-mindspeed-pipeline \
  python scripts/verify_ascend_env.py --strict
```

## 生产输入

[`configs/pipeline/production.env.example`](configs/pipeline/production.env.example)
集中列出路径、不可变 revision、模型、variant、LR、min LR、warmup tokens、
训练 token 数、拓扑、run identity 和端口。模型架构值位于
`configs/models/*.json`，训练超参数不写入模型架构配置。

完整命令顺序位于
[`docs/RUNBOOK.md`](docs/RUNBOOK.md)：

```text
环境
→ 固定上游源码
→ tokenizer
→ 五类数据下载与处理
→ Stage 1 fresh/resume
→ Stage 1→2 transition
→ Stage 2 resume/train
→ Stage 2→3 transition
→ Stage 3 resume/train
→ Stage 3→Think transition
→ Think resume/train
→ Think→Instruct transition
→ Instruct resume/train
→ inference
→ evaluation
```

每次真实提交对应新的 run ID、输出目录和 W&B run identity。多节点路径通过
渲染后的 `launch.sh` 共享，不在脚本中固化集群绝对路径。

非 editable 安装只安装 Python 包，不复制配置、脚本、Megatron overlay 或固定
的第三方仓库。安装后的 `olmo3ctl`、`olmo3ckpt`、`olmo3eval` 以导出的
`OLMO3_ROOT` 为仓库根目录；从仓库内运行时也可自动定位。无效的显式路径会
立即报错，不会退回其他目录。安装包还会逐文件核对该 checkout 中的
`olmo3_pipeline`、`modeling`、`runtime` 和 `llama_config`；源码更新后必须
重新构建环境，旧 wheel 与新 checkout 混用会被拒绝。

## 配置来源

1B、3B、7B 的结构来自固定 OLMo-core revision 中的
`TransformerConfig.olmo3_1B/3B/7B`。其中 7B/32B 存在官方发布的完整训练
recipe；1B/3B 的 LR、min LR、warmup 和部署拓扑属于本项目显式实验输入，
不标记为官方发布 recipe。

两种 variant：

| variant | 模型 | checkpoint keyspace |
|---|---|---|
| `base` | OLMo 3 | `olmo3_base_v1` |
| `siamese_depth` | OLMo 3 + Siamese Norm + Depth Attention | `olmo3_siamese_depth_v1` |

同一 checkpoint 链中的 model size 和 variant 保持不变。

## 验证

控制节点测试：

```bash
python scripts/check_repository_hygiene.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=src:. \
python -m pytest -q
```

公开提交前可通过重复的 `--deny-string` 或逗号分隔的
`OLMO3_HYGIENE_DENY_STRINGS` 增加操作者特有的用户名、挂载点和集群标识。
检查结果只报告文件、行号和规则，不回显疑似凭据内容。

真实 NPU 验证分为严格环境检查、Stage 1–4 smoke、65K cache smoke
和评测 smoke。

## 详细文档

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)：从空环境到评测的命令链
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)：模型与拓扑解析
- [`docs/DATA.md`](docs/DATA.md)：数据合同
- [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)：checkpoint 生命周期
- [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md)：TP/CP/HSDP 优化入口
- [`docs/INFERENCE_EVALUATION.md`](docs/INFERENCE_EVALUATION.md)：推理与评测边界
- [`UPSTREAM_LOCK.json`](UPSTREAM_LOCK.json)：上游源码 revision
- [`ASCEND_RUNTIME_LOCK.json`](ASCEND_RUNTIME_LOCK.json)：Ascend 二进制合同
