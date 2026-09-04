#!/usr/bin/env bash
# Install COLA into this Miniconda: /root/autodl-tmp/.sugar_deps/miniconda3
# Usage (from repo root):
#   bash install_cola_env.sh
set -euo pipefail

CONDA_ROOT="/root/autodl-tmp/.sugar_deps/miniconda3"
ENV_NAME="cola"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep pip wheels on the data disk (same pattern as SUGAR setup_env_xhs.sh).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "${CONDA_ROOT}")/.cache/pip}"
# export PIP_CACHE_DIR="/root/autodl-tmp/.sugar_deps/.cache/pip"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

mkdir -p "${PIP_CACHE_DIR}"
echo "[install] PIP_CACHE_DIR=${PIP_CACHE_DIR}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate base

cd "${REPO_ROOT}"

if [[ ! -f IsaacLab/VERSION ]] || [[ ! -f rsl_rl/setup.py ]]; then
  echo "[install] initializing git submodules..."
  git submodule update --init --recursive
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[install] conda env '${ENV_NAME}' already exists, reusing it"
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

ISAACLAB_SETUP="${REPO_ROOT}/IsaacLab/source/isaaclab/setup.py"
if [[ -f "${ISAACLAB_SETUP}" ]]; then
  sed -i 's/flatdict==4\.0\.1/flatdict==4.1.0/g' "${ISAACLAB_SETUP}"
  echo "[install] pinned flatdict==4.1.0 in IsaacLab/source/isaaclab/setup.py"
fi

pip install flatdict==4.1.0
./IsaacLab/isaaclab.sh --install
pip install -r requirements.txt
pip install -e rsl_rl
pip install -e .

echo
echo "[install] done. In a new shell:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  source setup_env.sh"
