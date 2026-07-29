#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_ID="${MODEL_ID:-ArchSpace-Collection/NCP_Olmo3_Stage1_StepLast}"
REVISION="${REVISION:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gsm8k/ncp-stage1-last}"
DATASET_FILE="${DATASET_FILE:-}"

if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "NPROC_PER_NODE must be positive, got $NPROC_PER_NODE" >&2
  exit 2
fi
if [[ -n "$DATASET_FILE" && ! -f "$DATASET_FILE" ]]; then
  echo "DATASET_FILE does not exist: $DATASET_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/cache/hf" "$OUTPUT_DIR/cache/torch"
export HF_HOME="${HF_HOME:-$OUTPUT_DIR/cache/hf}"
export TORCH_HOME="${TORCH_HOME:-$OUTPUT_DIR/cache/torch}"
export TOKENIZERS_PARALLELISM=false

dataset_args=()
if [[ -n "$DATASET_FILE" ]]; then
  dataset_args=(--dataset-file "$DATASET_FILE" --split test)
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  -m reproduce.gsm8k.evaluate \
  --model-id "$MODEL_ID" \
  --revision "$REVISION" \
  "${dataset_args[@]}" \
  --num-samples 1 \
  --seed 1234 \
  --max-sequence-length 4096 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/torchrun.log"

"$PYTHON_BIN" -m reproduce.gsm8k.aggregate \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/aggregate.log"

printf 'model_id=%s\n' "$MODEL_ID"
printf 'revision=%s\n' "$REVISION"
printf 'output_dir=%s\n' "$OUTPUT_DIR"
