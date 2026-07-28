#!/usr/bin/env bash
#
# Build the project environment from an empty Conda prefix.
#
# No package or metadata is copied from an existing Conda environment.
# CANN/HCCL, the Ascend driver and firmware remain host prerequisites. The
# matched torch and torch_npu binary wheels must be supplied explicitly in a
# local wheelhouse.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/bootstrap_conda_env.sh \
    --ascend-wheelhouse /path/to/cp310-aarch64-wheels \
    [--target-env NEW_ENV_NAME] [--conda PATH] \
    [--bootstrap-python PATH] [--ascend-home PATH] \
    [--cann-env-script PATH] [--strict-ascend]

Required external inputs:
  * Conda with access to conda-forge.
  * A local wheelhouse containing platform-compatible wheels for every entry
    in requirements/ascend-runtime.txt at the versions sealed by
    ASCEND_RUNTIME_LOCK.json.
  * The CANN/HCCL build locked by ASCEND_RUNTIME_LOCK.json.
  * For --strict-ascend: a matching Ascend driver/firmware, at least one
    visible NPU, and a working one-rank HCCL collective.

Defaults:
  target: OLMO3_TARGET_CONDA_ENV or olmo3-mindspeed-pipeline
  conda: CONDA_EXE or conda
  bootstrap python: OLMO3_BOOTSTRAP_PYTHON or python3
  wheelhouse: OLMO3_ASCEND_WHEELHOUSE (no implicit fallback)
  Ascend home: ASCEND_HOME_PATH, ASCEND_TOOLKIT_HOME, or
               /usr/local/Ascend/ascend-toolkit/latest
  CANN setup: OLMO3_CANN_ENV_SCRIPT or
              /usr/local/Ascend/ascend-toolkit/set_env.sh

The target name must not already exist. A unique staging environment is
created with ``conda create`` using Python 3.10.20 and a bootstrap pip. The
exact pip/setuptools/wheel toolchain, all locked dependencies, and this
project are then installed non-editably. The staging
environment is renamed to the target only after all verification succeeds.

Without --strict-ascend, a control host may finish only when its locked CANN
toolkit and torch/torch_npu installation are valid; absence of a driver or
visible NPU is reported as OLMO3_ASCEND_STACK_DEFERRED. Every training Pod
must run scripts/verify_ascend_env.py --strict before launching training.
USAGE
}

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
target_env=${OLMO3_TARGET_CONDA_ENV:-olmo3-mindspeed-pipeline}
conda_exe=${CONDA_EXE:-conda}
bootstrap_python=${OLMO3_BOOTSTRAP_PYTHON:-python3}
ascend_wheelhouse=${OLMO3_ASCEND_WHEELHOUSE:-}
ascend_home=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}
cann_env_script=${OLMO3_CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
strict_ascend=0

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
        --ascend-wheelhouse)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--ascend-wheelhouse needs a value" >&2
                exit 2
            }
            ascend_wheelhouse=$2
            shift 2
            ;;
        --ascend-home)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--ascend-home needs a value" >&2
                exit 2
            }
            ascend_home=$2
            shift 2
            ;;
        --cann-env-script)
            [[ $# -ge 2 ]] || {
                printf '%s\n' "--cann-env-script needs a value" >&2
                exit 2
            }
            cann_env_script=$2
            shift 2
            ;;
        --strict-ascend)
            strict_ascend=1
            shift
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
if [[ -z "$ascend_wheelhouse" ]]; then
    printf '%s\n' \
        "missing Ascend wheelhouse: pass --ascend-wheelhouse or set OLMO3_ASCEND_WHEELHOUSE" \
        >&2
    exit 2
fi
if [[ ! -d "$ascend_wheelhouse" ]]; then
    printf 'Ascend wheelhouse is not a directory: %s\n' \
        "$ascend_wheelhouse" >&2
    exit 2
fi
if [[ ! -r "$cann_env_script" ]]; then
    printf 'CANN environment script is not readable: %s\n' \
        "$cann_env_script" >&2
    exit 2
fi

read -r locked_torch_version locked_torch_npu_version < <(
    "$bootstrap_python" - "$project_root/ASCEND_RUNTIME_LOCK.json" <<'PY'
import json
import re
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
torch = str(lock["python_packages"]["torch"]["version"])
torch_npu = str(lock["python_packages"]["torch_npu"]["version"])
for name, value in (("torch", torch), ("torch_npu", torch_npu)):
    if re.fullmatch(r"[0-9A-Za-z.+-]+", value) is None:
        raise SystemExit(f"unsafe {name} version in runtime lock: {value!r}")
print(torch, torch_npu)
PY
)

require_wheel() {
    distribution=$1
    pattern=$2
    if ! find "$ascend_wheelhouse" -maxdepth 1 -type f \
        -name "$pattern" -print -quit | grep -q .; then
        printf 'missing locked %s wheel in %s (expected %s)\n' \
            "$distribution" "$ascend_wheelhouse" "$pattern" >&2
        exit 2
    fi
}

require_wheel torch "torch-${locked_torch_version}-*.whl"
require_wheel torch_npu "torch_npu-${locked_torch_npu_version}-*.whl"

"$bootstrap_python" - \
    "$project_root/requirements/ascend-runtime.txt" \
    "$locked_torch_version" \
    "$locked_torch_npu_version" <<'PY'
import sys
from pathlib import Path

requirements = {
    line.split("==", 1)[0]: line.split("==", 1)[1]
    for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if (line := raw.split("#", 1)[0].strip())
}
expected = {"torch": sys.argv[2], "torch-npu": sys.argv[3]}
if requirements != expected:
    raise SystemExit(
        "requirements/ascend-runtime.txt disagrees with "
        f"ASCEND_RUNTIME_LOCK.json: {requirements!r} != {expected!r}"
    )
PY

# This check runs with the bootstrap Python before any Conda mutation. It
# validates the host-owned CANN/HCCL metadata without requiring torch,
# torch_npu, an NPU device, or an existing project environment.
PYTHONDONTWRITEBYTECODE=1 "$bootstrap_python" \
    "$project_root/scripts/verify_ascend_env.py" \
    --host-only \
    --ascend-home "$ascend_home" \
    --cann-env-script "$cann_env_script"

# Export the toolkit's library and Python paths for all subsequent install and
# import probes. This sources host software only; it does not activate or copy
# a Conda environment.
# shellcheck source=/dev/null
source "$cann_env_script"

# A fresh checkout may omit the large pristine upstream repositories. This
# operation clones Git repositories only; it never clones a Conda environment.
PYTHONDONTWRITEBYTECODE=1 "$bootstrap_python" \
    "$project_root/scripts/bootstrap_runtime.py" --clone-missing

env_exists() {
    "$conda_exe" env list |
        awk 'NF >= 2 && $1 !~ /^#/ {print $1}' |
        grep -Fqx -- "$1"
}

if env_exists "$target_env"; then
    printf 'target conda environment already exists: %s\n' "$target_env" >&2
    exit 2
fi

build_env="${target_env}.__olmo3_build_$$"
if env_exists "$build_env"; then
    printf 'staging conda environment already exists: %s\n' "$build_env" >&2
    exit 2
fi

cleanup_armed=1
target_cleanup_armed=0
cleanup_failed_build() {
    status=$?
    trap - EXIT INT TERM
    if (( cleanup_armed != 0 )); then
        candidates=("$build_env")
        if (( target_cleanup_armed != 0 )); then
            candidates+=("$target_env")
        fi
        for candidate in "${candidates[@]}"; do
            if env_exists "$candidate"; then
                printf 'removing incomplete conda environment: %s\n' \
                    "$candidate" >&2
                "$conda_exe" env remove --yes --name "$candidate" >&2 || true
            fi
        done
    fi
    exit "$status"
}
trap cleanup_failed_build EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The environment starts from an empty prefix. No --clone, copied prefix,
# source environment, or inherited site-packages are used.
"$conda_exe" create \
    --yes \
    --override-channels \
    --channel conda-forge \
    --name "$build_env" \
    python=3.10.20 \
    pip

"$conda_exe" run --name "$build_env" \
    python -m pip install --upgrade \
        pip==26.0.1 \
        setuptools==80.9.0 \
        wheel==0.46.3

"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --requirement "$project_root/requirements/dolma-bootstrap.txt"
"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --no-deps \
        --requirement "$project_root/requirements/train.lock.txt"
"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --no-deps \
        --requirement "$project_root/requirements.txt"
"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --no-index \
        --find-links "$ascend_wheelhouse" \
        --no-deps \
        --requirement "$project_root/requirements/ascend-runtime.txt"
"$conda_exe" run --name "$build_env" \
    python -m pip install \
        --no-build-isolation \
        --no-deps \
        "$project_root"

# ``acl`` is a CANN-owned binary binding, not a PyPI package. Add the
# validated host location to this clean environment without copying it.
cann_python_site="$ascend_home/python/site-packages"
if [[ ! -d "$cann_python_site/acl" ]]; then
    printf 'CANN Python binding directory is missing: %s\n' \
        "$cann_python_site/acl" >&2
    exit 2
fi
target_python_site=$(
    "$conda_exe" run --name "$build_env" \
        python -c 'import site; print(site.getsitepackages()[0])'
)
printf '%s\n' "$cann_python_site" \
    >"$target_python_site/olmo3_ascend_cann.pth"
chmod 0644 "$target_python_site/olmo3_ascend_cann.pth"

"$conda_exe" run --name "$build_env" \
    python "$project_root/scripts/bootstrap_runtime.py"
OLMO3_ROOT="$project_root" "$conda_exe" run --name "$build_env" \
    python "$project_root/scripts/verify_python_dependencies.py"
OLMO3_ROOT="$project_root" "$conda_exe" run --name "$build_env" \
    olmo3ctl --help >/dev/null
OLMO3_ROOT="$project_root" "$conda_exe" run --name "$build_env" \
    olmo3ckpt --help >/dev/null
OLMO3_ROOT="$project_root" "$conda_exe" run --name "$build_env" \
    olmo3eval --help >/dev/null
"$conda_exe" run --name "$build_env" \
    python -c \
        'from transformers import AutoTokenizer
import dolma
import einops
import jinja2
import markupsafe
import mpmath
import networkx
import ninja
import psutil
import scipy
import sympy
print("OLMO3_DATA_AND_TRAINING_DEPENDENCIES_OK")'

ascend_verify_args=(
    --ascend-home "$ascend_home"
    --cann-env-script "$cann_env_script"
)
if (( strict_ascend != 0 )); then
    ascend_verify_args+=(--strict)
fi
"$conda_exe" run --name "$build_env" \
    python "$project_root/scripts/verify_ascend_env.py" \
        "${ascend_verify_args[@]}"

OLMO3_LOCKED_TORCH_VERSION="$locked_torch_version" \
OLMO3_LOCKED_TORCH_NPU_VERSION="$locked_torch_npu_version" \
"$conda_exe" run --name "$build_env" \
    python -c \
        'import importlib.metadata as m, json, os, site
from pathlib import Path
for name, expected in (
    ("torch", os.environ["OLMO3_LOCKED_TORCH_VERSION"]),
    ("torch-npu", os.environ["OLMO3_LOCKED_TORCH_NPU_VERSION"]),
):
    actual = m.version(name)
    assert actual == expected, (name, expected, actual)
dist = m.distribution("olmo3-mindspeed-pipeline")
direct = json.loads(dist.read_text("direct_url.json") or "{}")
assert not direct.get("dir_info", {}).get("editable", False), direct
editable_distributions = []
for candidate in m.distributions():
    payload = candidate.read_text("direct_url.json")
    if not payload:
        continue
    value = json.loads(payload)
    if value.get("dir_info", {}).get("editable", False):
        editable_distributions.append(
            (candidate.metadata.get("Name", "<unknown>"), value.get("url"))
        )
editable_files = []
for root in map(Path, site.getsitepackages()):
    editable_files.extend(str(path) for path in root.glob("__editable__*"))
    editable_files.extend(str(path) for path in root.glob("*.egg-link"))
assert not editable_distributions, editable_distributions
assert not editable_files, editable_files
print("OLMO3_CLEAN_NON_EDITABLE_INSTALL_OK")'

if env_exists "$target_env"; then
    printf 'target conda environment appeared during build: %s\n' \
        "$target_env" >&2
    exit 2
fi
target_cleanup_armed=1
"$conda_exe" rename --yes --name "$build_env" "$target_env"
cleanup_armed=0
target_cleanup_armed=0
trap - EXIT INT TERM

printf 'OLMO3_CONDA_ENV_READY name=%s creation=clean\n' "$target_env"
