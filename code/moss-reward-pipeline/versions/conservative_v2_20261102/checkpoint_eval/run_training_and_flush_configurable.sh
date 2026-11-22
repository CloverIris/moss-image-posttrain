#!/usr/bin/env bash
# CPU-only supervisor. Qwen cache preparation is an explicit external step.
set -euo pipefail
if [[ $# != 2 || "$1" != --config ]]; then
    printf 'Usage: %s --config /absolute/run/config.json\n' "$0" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_python="${MOSS_LAUNCHER_PYTHON:-${MOSS_PYTHON:-python3}}"
exec "$bootstrap_python" "$script_dir/training_flush_launcher.py" --config "$2"
