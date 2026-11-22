#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
python_bin="${MOSS_ART_PYTHON:-${MOSS_ART_BASE_PYTHON:-$CONDA_PREFIX/envs/jax_env/bin/python}}"
check_only=false
if [[ "${1:-}" == "--check-only" && $# -eq 1 ]]; then
  check_only=true
elif [[ $# -gt 0 ]]; then
  printf 'Usage: bash scripts/setup.sh [--check-only]\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing JAX Python: %s\nSet MOSS_ART_PYTHON to an existing configured interpreter.\n' "$python_bin" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
"$python_bin" - "$project_root" <<'PY'
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
modules = {'jax':'jax', 'optax':'optax', 'numpy':'numpy', 'scipy':'scipy', 'PyYAML':'yaml',
           'safetensors':'safetensors', 'Pillow':'PIL', 'ml_dtypes':'ml_dtypes',
           'tokenizers':'tokenizers', 'setuptools':'setuptools'}
versions, missing = {}, []
for package, module in modules.items():
    try:
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    except ImportError as error:
        missing.append(f'{package}: {error}')
if missing:
    raise SystemExit('Missing project dependencies:\n' + '\n'.join(missing)
                     + '\nDependencies are not installed or upgraded automatically.')
import jax
print(json.dumps({'status':'dependencies_ok', 'python':sys.version.split()[0],
                  'environment':sys.prefix, 'executable':sys.executable, 'versions':versions, 'test_backend':jax.default_backend()}, indent=2))
PY
if ! $check_only; then
  # Install only this project entrypoint into the chosen JAX environment.
  # Dependencies are reused as installed; no dependency resolution or upgrades.
  "$python_bin" -m pip install --no-deps --no-build-isolation -e "$project_root"
fi
printf 'Project environment ready. No GPU job was started.\n'
