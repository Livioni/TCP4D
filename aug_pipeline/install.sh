#!/usr/bin/env bash
# Run inside the existing tcp4d conda environment, from any directory.
set -euo pipefail
AUG_REPO="$(cd "$(dirname "$0")/.." && pwd)"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Activate the Python 3.11 tcp4d environment first.')
try:
    import torch
except ImportError:
    raise SystemExit('Install torch==2.9.1 and torchvision==0.24.1 first; see aug_pipeline/README.md.')
if torch.__version__.split('+')[0] != '2.9.1':
    raise SystemExit('Install the README torch==2.9.1 / torchvision==0.24.1 pair first.')
PY
# CUDA extensions must build against this environment's existing PyTorch.
python -m pip install -c "$AUG_REPO/aug_pipeline/constraints.txt" 'setuptools>=68,<81' 'setuptools-scm>=8,<9' wheel ninja \
  'pybind11>=2.13,<3' 'numpy==1.26.4' 'pandas>=2.2,<3' 'opencv-python>=4.10,<4.12'
if [[ -z "${CUDA_HOME:-}" ]]; then
  AUG_NVCC="$(command -v nvcc || true)"
  [[ -n "$AUG_NVCC" ]] || { echo 'CUDA toolkit nvcc is required for cuRobo.' >&2; exit 1; }
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$AUG_NVCC")")")"
fi
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'CUDA_HOME must contain bin/nvcc.' >&2; exit 1; }
export MAX_JOBS="${MAX_JOBS:-4}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
fi
python -m pip install --no-build-isolation -c "$AUG_REPO/aug_pipeline/constraints.txt" -e "${AUG_REPO}[aug]"
python -m pip check
echo 'Installed augmentation extras in the current tcp4d environment.'
echo 'Next: tcp4d-aug download tcp4d assets sam3 propainter'
