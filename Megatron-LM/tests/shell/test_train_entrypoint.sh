#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "${test_dir}/../.." && pwd -P)
train_script="${project_root}/scripts/train/olmo3_train.sh"

[[ -x "$train_script" ]] || {
    printf 'training entrypoint is not executable: %s\n' "$train_script" >&2
    exit 1
}
bash -n "$train_script"

if grep -En '(/data/|/afs/|/root/|/home/)' "$train_script"; then
    printf 'training entrypoint contains a fixed filesystem path\n' >&2
    exit 1
fi

if grep -Ein '(kubectl|sbatch|qsub|volcano|submit[_-]?job)' "$train_script"; then
    printf 'training entrypoint contains a cluster submission command\n' >&2
    exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
fake_python="${tmp_dir}/python"
capture="${tmp_dir}/argv"

cat >"$fake_python" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\0' "$@" >"${OLMO3_WRAPPER_CAPTURE:?}"
FAKE
chmod +x "$fake_python"

OLMO3_WRAPPER_CAPTURE="$capture" \
"$train_script" \
    --control-python "$fake_python" \
    --model 3b \
    --variant siamese_depth \
    --stage stage1 \
    --data dolma3_6t \
    --topology tp2_sio_2048npu \
    --performance tp2_sio_mc2 \
    --run-id static-wrapper-test \
    --lifecycle fresh \
    --data-root relative-data \
    --data-args-path relative-data-args.json \
    --tokenizer relative-tokenizer \
    --save relative-checkpoints \
    --output relative-output \
    --python relative-python \
    --train-tokens 6000000000000 \
    --peak-lr 0.00035 \
    --min-lr 0.000035 \
    --warmup-tokens 8388608000 \
    --world-size 2048 \
    --nproc-per-node 16 \
    --nnodes 128 \
    --node-rank 0 \
    --master-addr rendezvous-host \
    --master-port 29501 \
    --hccl-if-base-port 41000

mapfile -d '' -t captured <"$capture"
[[ "${captured[0]}" == "-m" ]]
[[ "${captured[1]}" == "olmo3_pipeline.cli" ]]
[[ "${captured[2]}" == "render" ]]

joined=$(printf '%s\n' "${captured[@]}")
grep -Fxq -- "--model" <<<"$joined"
grep -Fxq -- "3b" <<<"$joined"
grep -Fxq -- "--peak-lr" <<<"$joined"
grep -Fxq -- "--warmup-tokens" <<<"$joined"
grep -Fxq -- "--hccl-if-base-port" <<<"$joined"
grep -Fxq -- "41000" <<<"$joined"

common_transition_args=(
    --control-python "$fake_python"
    --action validate
    --variant base
    --topology stage3_cp16_1024npu
    --performance cp_single_halo
    --run-id transition-wrapper-test
    --lifecycle transition
    --data-root relative-data
    --data-work-dir relative-work
    --tokenizer relative-tokenizer
    --load relative-source-checkpoint
    --save relative-target-checkpoint
    --output relative-output
    --python relative-python
    --peak-lr 0.00025
    --min-lr 0.000025
    --warmup-tokens 0
    --world-size 1024
    --nproc-per-node 16
    --nnodes 64
    --node-rank 0
    --master-addr rendezvous-host
    --master-port 29503
)

for stage in stage2 stage3; do
    OLMO3_WRAPPER_CAPTURE="$capture" \
    "$train_script" \
        "${common_transition_args[@]}" \
        --model 7b \
        --stage "$stage" \
        --data "$([[ "$stage" == "stage2" ]] && printf dolmino_100b || printf longmino_50b)" \
        --data-manifest relative-manifest.json \
        --train-tokens 1000000
    mapfile -d '' -t captured <"$capture"
    [[ "${captured[2]}" == "validate" ]]
done

for stage in sft_think sft_instruct; do
    sft_args=()
    for ((index = 0; index < ${#common_transition_args[@]}; index++)); do
        if [[ "${common_transition_args[index]}" == "--warmup-tokens" ]]; then
            ((index += 1))
            continue
        fi
        sft_args+=("${common_transition_args[index]}")
    done
    OLMO3_WRAPPER_CAPTURE="$capture" \
    "$train_script" \
        "${sft_args[@]}" \
        --model 1b \
        --stage "$stage" \
        --data "$([[ "$stage" == "sft_think" ]] && printf dolci_think || printf dolci_instruct)" \
        --epochs 2 \
        --expected-instances 100 \
        --expected-fingerprint sha256:test
    mapfile -d '' -t captured <"$capture"
    [[ "${captured[2]}" == "validate" ]]
done

if OLMO3_WRAPPER_CAPTURE="$capture" \
    "$train_script" \
    --control-python "$fake_python" \
    --model 1b \
    --variant base \
    --stage stage1 \
    --data dolma3_6t \
    --topology stage1_tp1_4096npu \
    --performance ordinary_hsdp \
    --run-id must-fail \
    --lifecycle fresh \
    --data-root data \
    --data-args-path data-args.json \
    --tokenizer tokenizer \
    --save checkpoints \
    --output output \
    --python python \
    --train-tokens 1 \
    --min-lr 0 \
    --warmup-tokens 0 \
    --world-size 4096 \
    --nproc-per-node 16 \
    --nnodes 256 \
    --node-rank 0 \
    --master-addr rendezvous-host \
    --master-port 29502 \
    >/dev/null 2>&1
then
    printf 'training entrypoint accepted a launch contract without --peak-lr\n' >&2
    exit 1
fi
