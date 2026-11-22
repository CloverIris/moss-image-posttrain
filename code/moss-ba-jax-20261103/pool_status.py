"""Hourly measurement of the independent reference queue and student commits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import shutil
import time
from pathlib import Path
import numpy as np

from common import RUN, read, save
from pool import GPUS, LEARNER, LATENTS, POOL, devices, identity, lock, matches


def optional(path):
    return read(path) if path.is_file() else None


def event_rate(timestamps, since, until, scale=1):
    count = sum(since < timestamp <= until for timestamp in timestamps)
    seconds = until - since
    return dict(events=count, seconds=round(seconds, 2),
                per_hour=round(3600 * count * scale / seconds, 2) if seconds > 0 else None)


def snapshot():
    cfg = read(RUN / "config.json")
    deployment = read(POOL / "deployment.json")
    dp = RUN/"dp4"
    hand = optional(dp/"handoff.json")
    dp_latest = optional(dp/"latest.json") if hand else None
    latest = dp_latest or read(RUN/"latest.json")
    cursor = latest["processed_prompts"]
    initial = deployment["checkpoint"]["cursor"]
    total = len(cfg["order"])
    points = {int(p.stem): p.stat().st_mtime for p in LATENTS.glob("*.npz")}
    now = time.time()
    frontier = next((i for i in range(total) if i not in points), total)
    single_end = hand["processed_prompts"] if hand else cursor
    metrics = [RUN / "metrics" / f"{i:05d}.json" for i in range(single_end)]
    dp_groups = list(range(single_end,cursor,4))
    metrics += [dp/"metrics"/f"{i:05d}.json" for i in dp_groups]
    updates = cfg["updates_per_prompt"]
    if latest["step"] != (single_end+len(dp_groups))*updates or not all(p.is_file() for p in metrics):
        raise ValueError("Learner commit cursor or metric sequence differs")
    last_metrics = read(metrics[-1]) if metrics else None
    if last_metrics and (last_metrics["updates"][-1] != latest["last"] or
                         (last_metrics["prompt_ids"][-1] if dp_groups else last_metrics["prompt_id"]) != cfg["order"][cursor - 1]):
        raise ValueError("Latest prompt, update or diagnostics mismatch")
    commits = [read(p) for p in (POOL / "commits").glob("*.json")]
    per_gpu = Counter(r["gpu"] for r in commits)
    starts = {g: min((r["started"] for r in commits if r["gpu"] == g), default=now) for g in GPUS}
    elapsed_start = max(deployment["activated"], now - 3600)
    window_start = max(max(starts.values()), elapsed_start)
    rates = dict(
        # Ten *intervals* preceding the move, excluding early held-out generation.
        old_single_reference=event_rate([points[i] for i in range(initial)],
                                        points[max(0, initial - 11)], points[initial - 1]),
        pool_reference=event_rate([r["finished"] for r in commits], window_start, now),
        student_optimizer=event_rate([p.stat().st_mtime for p in metrics], elapsed_start, now, updates))
    hardware = devices()
    processes = {}
    for gpu in (*GPUS, LEARNER):
        marker = read(POOL / f"process_{gpu}.json")
        precision = optional(POOL / f"precision_{gpu}.json")
        valid = bool(precision and precision["pid"] == marker["pid"] and
                     precision["uuid"] == marker["uuid"] and marker["uuid"] == hardware[gpu]["uuid"] and
                     precision["contract"]["base_precision"] == "bf16" and
                     precision["contract"]["weight_quantization_mode"] == "none" and
                     precision["contract"]["quantized_modules"] == 0)
        processes[gpu] = dict(pid=marker["pid"], alive=matches(marker), bf16_verified=valid,
                              reference_commits=per_gpu[gpu], **hardware[gpu])
        if hand:
            processes[gpu]["role"]="released"
            if gpu in GPUS:
                rank=optional(dp/f"rank_{gpu}.json")
                live=identity(rank["pid"]) if rank else None
                processes[gpu].update(role="dp_learner",alive=bool(live and live[0]==rank["birth"]),
                        pid=rank["pid"] if rank else None,bf16_verified=bool(rank and rank["uuid"]==hardware[gpu]["uuid"]
                        and rank["contract"]["base_precision"]=="bf16" and rank["contract"]["quantized_modules"]==0))
    evaluated = sorted([*(RUN/"evaluation").glob("*/metrics.json"),*(dp/"evaluation").glob("*/metrics.json")],key=lambda p:int(p.parent.name))
    evaluation = {}
    if evaluated:
        rows = read(evaluated[-1])
        for n in cfg["nfe"]:
            batch = [r for r in rows if r["nfe"] == n]
            evaluation[n] = dict(pairs=len(batch), **{
                field: float(np.mean([r[field] for r in batch]))
                for field in ("ba_mse", "euler_mse", "zero_side_ba_mse")})
    memory_kib = {line.split(":")[0]: int(line.split()[1]) for line in
                  Path("/proc/meminfo").read_text().splitlines() if line.split(":")[0] in
                  ("MemAvailable", "SwapFree", "SwapTotal")}
    gate = optional(POOL / "target_benchmark.json")
    return dict(time_epoch=now, time_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                run=str(RUN), config_sha256=deployment["run_sha"], latest=latest, total_updates=hand["total_updates"] if hand else total*updates,
                dp4_phase=optional(dp/"state.json"), dp4_handoff=hand,
                reference_count=len(points), contiguous_references=frontier, ready_ahead=frontier-cursor,
                completed_beyond_frontier=len(points)-frontier, total_prompts=total, rates=rates,
                qwen=read(RUN/"input_audit.json")["cached_texts_checked"], processes=processes,
                memory_kib=memory_kib, nvme_free_gib=round(shutil.disk_usage(RUN).free/2**30, 2),
                target_benchmark=gate, rpc_stats=optional(POOL/"rpc_stats.json"),
                eval_step=int(evaluated[-1].parent.name) if evaluated else None, evaluation=evaluation,
                complete=(RUN/"complete.json").exists(),
                worker_failures=sorted(p.name for p in POOL.glob("failure_*.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    result = snapshot()
    if args.record:
        with lock(POOL / "status.lock"):
            reports = RUN / "reports"
            save(reports / "hourly" / (str(time.time_ns()) + ".json"), result)
            save(reports / "dispatch_latest.json", result)
    print(json.dumps(result, ensure_ascii=False))
