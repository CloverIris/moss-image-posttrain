#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
python_bin="${MOSS_ART_PYTHON:-$CONDA_PREFIX/envs/jax_env/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing project Python: %s\nRun bash scripts/setup.sh first.\n' "$python_bin" >&2
  exit 1
fi
# Set CPU library budgets before Python imports NumPy/JAX. CLI subsequently
# assigns each worker its own physical CUDA device and platform.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1
exec "$python_bin" -u "$project_root/run.py" "$@"
