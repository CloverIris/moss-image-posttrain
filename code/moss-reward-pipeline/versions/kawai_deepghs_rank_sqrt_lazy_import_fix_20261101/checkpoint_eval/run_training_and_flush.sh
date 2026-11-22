#!/usr/bin/env bash
# Training owns GPUs0..3. GPU4's independent daemon generates validation only.
set -euo pipefail
if [[ $# != 2 || "$1" != --config ]]; then
    printf 'Usage: %s --config /absolute/run/config.json\n' "$0" >&2
    exit 2
fi
config="$(realpath -- "$2")"
run_dir="$(dirname -- "$config")"
repo="$CODE_ROOT/moss-art-jax"
python_bin="${MOSS_PYTHON:-$CONDA_PREFIX/envs/jax_env/bin/python}"
helper="$DATA_DISK/moss-reward-pipeline/checkpoint_eval/flush_validation.py"
[[ -f "$run_dir/flush_validation.config.json" ]]
child_pid=''
stop_stage() {
    trap - INT TERM
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 130
}
trap stop_stage INT TERM
printf 'TRAINING_START %s\n' "$(date --iso-8601=seconds)"
bash "$repo/scripts/run_stages.sh" --config "$config" &
child_pid=$!
wait "$child_pid"
child_pid=''
printf 'TRAINING_COMMITTED_FINAL_VALIDATION_PENDING %s\n' "$(date --iso-8601=seconds)"
[[ -f "$helper" ]]
"$python_bin" "$helper" --config "$run_dir/flush_validation.config.json" &
child_pid=$!
wait "$child_pid"
child_pid=''
printf 'TRAINING_AND_VALIDATION_COMPLETE %s\n' "$(date --iso-8601=seconds)"
