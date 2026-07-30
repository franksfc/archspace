#!/usr/bin/env bash
#
# Path-free OLMo 3 training wrapper.
#
# This wrapper only validates, renders, or launches an olmo3ctl composition in
# the current allocation. It does not reserve accelerators or submit a cluster
# job. Render is the default action.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  olmo3_train.sh [--action validate|render|launch] --model 1b|3b|7b [OLMO3CTL_ARGS...]

The default action is "render". "launch" executes torchrun only for a
single-node current allocation; multi-node jobs render once and run the
generated launch.sh on every node with OLMO3_NODE_RANK set. This wrapper never
submits a cluster job.

Required for every stage:
  --model MODEL
  --variant base|siamese_depth
  --stage stage1|stage2|stage3|sft_think|sft_instruct
  --data PROFILE
  --topology PROFILE
  --performance PROFILE
  --run-id ID
  --lifecycle fresh|resume|transition
  --data-root PATH
  --tokenizer PATH
  --save PATH
  --output PATH
  --python PATH
  --peak-lr FLOAT
  --min-lr FLOAT
  --world-size INT
  --nproc-per-node INT
  --nnodes INT
  --node-rank INT
  --master-addr HOST
  --master-port INT

Optional distributed runtime:
  --hccl-if-base-port INT
  Freezes HCCL_IF_BASE_PORT in the rendered environment. Sequential jobs on
  the same allocation must use non-overlapping port ranges.

Stage-specific:
  stage1:
    --data-args-path PATH --train-tokens INT --warmup-tokens INT
    Optional: --data-cache-path PATH reuses Megatron sample-index artifacts
    across compatible runs; otherwise they are stored below --output.
  stage2 or stage3:
    --data-manifest PATH --data-work-dir PATH --train-tokens INT
    --warmup-tokens INT
  sft_think or sft_instruct:
    --data-work-dir PATH --epochs FLOAT
    --expected-instances INT --expected-fingerprint VALUE
    Uses the frozen 3% warmup unless --warmup-tokens is supplied.

Checkpoint rules:
  stage1 accepts lifecycle fresh or resume.
  Every other stage accepts transition or resume.
  resume and transition require --load PATH.

Optional W&B arguments:
  --wandb-project PROJECT [--wandb-entity ENTITY] [--wandb-base-url URL]
  [--wandb-log-style llamafactory|native]
  Omit all W&B arguments to disable W&B.

Save/resume smoke testing:
  [--exit-interval ITERATION]
  This stops cleanly at ITERATION without changing train_iters, LR decay, or
  the checkpoint schedule identity. It is not needed for production runs.

Wrapper-only arguments:
  --action ACTION          Default: render
  --control-python PATH    Python used to run olmo3ctl. Default: python3

All remaining arguments, including topology overrides, validation paths,
intervals, batch sizes, and repeated --set/--performance values, are forwarded
unchanged to olmo3ctl.
USAGE
}

die() {
    printf 'olmo3_train.sh: %s\n' "$*" >&2
    exit 2
}

need_value() {
    local option=$1
    local count=$2
    (( count >= 2 )) || die "${option} requires a value"
}

declare -A seen=()
declare -a forwarded=()
declare -a missing=()

action=render
control_python=${OLMO3_CONTROL_PYTHON:-python3}
model=
stage=
lifecycle=

while (( $# > 0 )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --action)
            need_value "$1" "$#"
            action=$2
            shift 2
            ;;
        --action=*)
            action=${1#*=}
            shift
            ;;
        --control-python)
            need_value "$1" "$#"
            control_python=$2
            shift 2
            ;;
        --control-python=*)
            control_python=${1#*=}
            shift
            ;;
        --model)
            need_value "$1" "$#"
            [[ -z "$model" ]] || die "--model may be supplied only once"
            model=$2
            seen["--model"]=1
            forwarded+=("$1" "$2")
            shift 2
            ;;
        --model=*)
            [[ -z "$model" ]] || die "--model may be supplied only once"
            model=${1#*=}
            seen["--model"]=1
            forwarded+=("$1")
            shift
            ;;
        --stage)
            need_value "$1" "$#"
            stage=$2
            seen["--stage"]=1
            forwarded+=("$1" "$2")
            shift 2
            ;;
        --stage=*)
            stage=${1#*=}
            seen["--stage"]=1
            forwarded+=("$1")
            shift
            ;;
        --lifecycle)
            need_value "$1" "$#"
            lifecycle=$2
            seen["--lifecycle"]=1
            forwarded+=("$1" "$2")
            shift 2
            ;;
        --lifecycle=*)
            lifecycle=${1#*=}
            seen["--lifecycle"]=1
            forwarded+=("$1")
            shift
            ;;
        --*=*)
            seen["${1%%=*}"]=1
            forwarded+=("$1")
            shift
            ;;
        --*)
            seen["$1"]=1
            forwarded+=("$1")
            shift
            ;;
        *)
            # Keep values paired with generic olmo3ctl options. argparse
            # remains the source of truth for option arity and value types.
            forwarded+=("$1")
            shift
            ;;
    esac
done

case "$action" in
    validate|render|launch) ;;
    *) die "--action must be validate, render, or launch; got ${action@Q}" ;;
esac

case "$model" in
    1b|3b|7b) ;;
    "") ;;
    *) die "--model must be 1b, 3b, or 7b; got ${model@Q}" ;;
esac

case "$stage" in
    stage1|stage2|stage3|sft_think|sft_instruct) ;;
    "") ;;
    *) die "unsupported --stage ${stage@Q}" ;;
esac

case "$lifecycle" in
    fresh|resume|transition) ;;
    "") ;;
    *) die "unsupported --lifecycle ${lifecycle@Q}" ;;
esac

require_option() {
    local option=$1
    [[ -n "${seen[$option]+present}" ]] || missing+=("$option")
}

for option in \
    --model \
    --variant \
    --stage \
    --data \
    --topology \
    --performance \
    --run-id \
    --lifecycle \
    --data-root \
    --tokenizer \
    --save \
    --output \
    --python \
    --peak-lr \
    --min-lr \
    --world-size \
    --nproc-per-node \
    --nnodes \
    --node-rank \
    --master-addr \
    --master-port
do
    require_option "$option"
done

case "$stage" in
    stage1)
        require_option --data-args-path
        require_option --train-tokens
        require_option --warmup-tokens
        [[ "$lifecycle" == "fresh" || "$lifecycle" == "resume" ]] \
            || die "stage1 requires --lifecycle fresh or resume"
        ;;
    stage2|stage3)
        require_option --data-manifest
        require_option --data-work-dir
        require_option --train-tokens
        require_option --warmup-tokens
        [[ "$lifecycle" == "transition" || "$lifecycle" == "resume" ]] \
            || die "${stage} requires --lifecycle transition or resume"
        ;;
    sft_think|sft_instruct)
        require_option --data-work-dir
        require_option --epochs
        require_option --expected-instances
        require_option --expected-fingerprint
        [[ "$lifecycle" == "transition" || "$lifecycle" == "resume" ]] \
            || die "${stage} requires --lifecycle transition or resume"
        ;;
esac

if [[ "$lifecycle" == "resume" || "$lifecycle" == "transition" ]]; then
    require_option --load
fi

if [[ -n "${seen[--wandb-entity]+present}" || \
      -n "${seen[--wandb-base-url]+present}" ]]; then
    require_option --wandb-project
fi

if (( ${#missing[@]} > 0 )); then
    printf 'olmo3_train.sh: missing required arguments:' >&2
    printf ' %s' "${missing[@]}" >&2
    printf '\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "${script_dir}/../.." && pwd -P)
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "$control_python" -m olmo3_pipeline.cli "$action" "${forwarded[@]}"
