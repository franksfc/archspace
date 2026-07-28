#!/usr/bin/env bash
#
# Build the OLMES control environment from an empty Conda prefix.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/eval/bootstrap_eval_env.sh \
    [--target-env NAME] [--conda PATH] [--bootstrap-python PATH]

The target environment must not already exist. The script creates a clean
Python 3.10 environment, installs CPU torch 2.8, installs the exact pinned
third_party/olmes checkout, verifies the selected task import graph, and
verifies that the checkout remains pristine.
USAGE
}

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
target_env=${OLMO3_EVAL_CONDA_ENV:-olmo3-olmes-eval}
conda_exe=${CONDA_EXE:-conda}
bootstrap_python=${OLMO3_BOOTSTRAP_PYTHON:-python3}

while (( $# > 0 )); do
    case "$1" in
        --target-env)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--target-env needs a value" >&2
                exit 2
            }
            target_env=$2
            shift 2
            ;;
        --conda)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--conda needs a value" >&2
                exit 2
            }
            conda_exe=$2
            shift 2
            ;;
        --bootstrap-python)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--bootstrap-python needs a value" >&2
                exit 2
            }
            bootstrap_python=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command -v "$conda_exe" >/dev/null 2>&1 || {
    printf 'conda executable not found: %s\n' "$conda_exe" >&2
    exit 2
}
command -v "$bootstrap_python" >/dev/null 2>&1 || {
    printf 'bootstrap Python executable not found: %s\n' \
        "$bootstrap_python" >&2
    exit 2
}

PYTHONDONTWRITEBYTECODE=1 "$bootstrap_python" \
    "$project_root/scripts/bootstrap_runtime.py" --clone-missing

if "$conda_exe" env list |
    awk 'NF >= 2 && $1 !~ /^#/ {print $1}' |
    grep -Fqx -- "$target_env"; then
    printf 'target conda environment already exists: %s\n' "$target_env" >&2
    exit 2
fi

build_env="${target_env}.__olmes_build_$$"
cleanup_armed=1
cleanup_failed_build() {
    status=$?
    trap - EXIT INT TERM
    if (( cleanup_armed != 0 )); then
        "$conda_exe" env remove --yes --name "$build_env" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_failed_build EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$conda_exe" create \
    --yes \
    --override-channels \
    --channel conda-forge \
    --name "$build_env" \
    python=3.10.20 \
    pip \
    git

"$conda_exe" run --name "$build_env" \
    python -m pip install --upgrade \
        pip==26.0.1 \
        setuptools==80.9.0 \
        wheel==0.46.3

"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0

"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --constraint "$project_root/requirements/eval-constraints.txt" \
        "$project_root/third_party/olmes"

mkdir -p "$project_root/eval_artifacts/nltk_data"
"$conda_exe" run --name "$build_env" \
    python -m nltk.downloader \
        --download-dir "$project_root/eval_artifacts/nltk_data" \
        punkt \
        punkt_tab \
        stopwords \
        words \
        averaged_perceptron_tagger_eng

PYTHONPATH="$project_root/scripts/eval/stubs:$project_root/third_party/olmes" \
NLTK_DATA="$project_root/eval_artifacts/nltk_data" \
    "$conda_exe" run --name "$build_env" \
    python -c \
        'from oe_eval.run_eval import load_task
from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks
import datasets
import transformers
print("OLMO3_OLMES_CONTROL_IMPORTS_OK")'

PYTHONDONTWRITEBYTECODE=1 "$bootstrap_python" \
    "$project_root/scripts/bootstrap_runtime.py"

"$conda_exe" rename --yes --name "$build_env" "$target_env"
cleanup_armed=0
trap - EXIT INT TERM
printf 'OLMO3_EVAL_ENV_READY name=%s\n' "$target_env"
