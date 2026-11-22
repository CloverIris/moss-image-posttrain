#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -eq 0 ]]; then
  set -- --config "$project_root/configs/moss_lora_4gpu.json"
fi
exec bash "$project_root/scripts/run.sh" train "$@"
