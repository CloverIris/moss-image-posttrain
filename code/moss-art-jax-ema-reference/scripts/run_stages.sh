#!/usr/bin/env bash
# Sequential processes release every stage's CUDA context before the next stage.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ $# != 2 || "$1" != "--config" ]]; then
    printf 'Usage: %s --config experiment.json\n' "$0" >&2
    exit 2
fi
config="$(realpath -- "$2")"
python_bin="${MOSS_PYTHON:-$CONDA_PREFIX/envs/jax_env/bin/python}"
controller="$("$python_bin" -m moss_art.stage_pipeline --config "$config" --action prepare)"
exec 9>"$controller/.run.lock"
flock -n 9 || { printf 'This experiment already has an active controller.\n' >&2; exit 1; }
child_pid=''
stop_stage() {
    trap - INT TERM
    if [[ -n "$child_pid" ]]; then
        kill -TERM -- "-$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 130
}
trap stop_stage INT TERM
while true; do
    phase="$("$python_bin" -m moss_art.stage_pipeline --config "$config" --action next)"
    [[ "$phase" == done ]] && break
    case "$phase" in infer|eval|train) ;; *) printf 'Invalid stage: %s\n' "$phase" >&2; exit 1 ;; esac
    SECONDS=0
    setsid "$python_bin" "$root/scripts/$phase.py" --config "$controller/$phase.json" 9>&- &
    child_pid=$!
    if wait "$child_pid"; then
        child_pid=''
    else
        status=$?
        child_pid=''
        printf '%s failed; committed handoff and durable outputs retained.\n' "$phase" >&2
        exit "$status"
    fi
    "$python_bin" -m moss_art.stage_pipeline --config "$config" --action record --phase "$phase" --seconds "$SECONDS"
done
