#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# This suite uses CPU fixtures; real 5B/GPU checks have their own reports.
python_bin="${MOSS_ART_PYTHON:-$CONDA_PREFIX/envs/jax_env/bin/python}"
exec "$python_bin" -m pytest -q "$project_root/tests" "$@"
