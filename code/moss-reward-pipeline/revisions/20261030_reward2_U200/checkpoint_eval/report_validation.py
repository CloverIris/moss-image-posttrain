"""Archive fixed-checkpoint reward summaries and publish exactly one TB record.

The scorer's result remains authoritative. This CPU-only reporting stage never
generates images, invokes a reward model, or imports Torch/JAX/TensorFlow.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import time
import types

MODELS = ("kawai", "deepghs", "hps")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def finite(value, description):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{description} must be finite")
    return float(value)


def stats(values):
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values),
            "min": min(values), "max": max(values), "count": len(values)}


def validate_and_summarize(job):
    """Check immutable request/ack alignment and retain every actual component."""
    paths = {}
    for key in ("ack", "evaluation_result", "summary", "logging_log_dir"):
        if not isinstance(job.get(key), str) or not Path(job[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        paths[key] = Path(job[key]).resolve()
    step = job.get("source_step")
    if type(step) is not int or step < 0:
        raise ValueError("source_step must be a nonnegative integer")
    ack, result, summary = (read_json(paths[k]) for k in ("ack", "evaluation_result", "summary"))
    if ack.get("status") not in ("complete", "scored", "done") or result.get("status") != "complete":
        raise ValueError("Scoring must be acknowledged and complete before reporting")
    checkpoint = result.get("checkpoint_id")
    if not isinstance(checkpoint, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", checkpoint):
        raise ValueError("Result needs a safe checkpoint_id")
    if ack.get("checkpoint_id") != checkpoint or summary.get("checkpoint_id") != checkpoint:
        raise ValueError("Checkpoint identity differs between result, ack and summary")
    for owner in (job, ack, result):
        if owner.get("source_step", step) != step:
            raise ValueError("Checkpoint source_step differs")
    if summary.get("update", summary.get("source_step", step)) != step:
        raise ValueError("Daemon summary update differs from source_step")
    if job.get("scores") and file_digest(job["scores"]) != file_digest(paths["evaluation_result"]):
        raise ValueError("scores and evaluation_result must reference the same result bytes")
    request_path = Path(summary["request"]).resolve()
    request = read_json(request_path)
    request_hash = file_digest(request_path)
    for owner in (ack, result, summary):
        if owner.get("request_sha256") != request_hash:
            raise ValueError("Scored request checksum differs from immutable fixed request")
    if summary.get("ack_sha256") != file_digest(paths["ack"]):
        raise ValueError("Daemon summary ack checksum differs")
    for key in ("checkpoint_manifest_sha256", "combined_fingerprint"):
        values = [owner[key] for owner in (ack, result, summary) if key in owner and owner[key] is not None]
        if len(set(values)) > 1:
            raise ValueError(f"Scoring identity differs: {key}")
    items, details, scores = request.get("items"), result.get("details"), result.get("scores")
    if any(not isinstance(rows, list) or len(rows) != 16 for rows in (items, details, scores)):
        raise ValueError("Fixed validation must contain exactly 16 items, scores and details")
    grouped, observed, ids, seeds = defaultdict(list), set(), set(), set()
    for index, (item, detail, score) in enumerate(zip(items, details, scores)):
        key = item.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("Fixed validation IDs must be explicit and unique")
        ids.add(key)
        if detail.get("original_id", detail.get("id")) != key:
            raise ValueError("Score detail order differs from original request IDs")
        if Path(detail["image"]).resolve() != Path(item["image"]).resolve():
            raise ValueError("Score detail image differs from original request")
        prompt_index = item.get("prompt_index")
        repeat = item.get("rollout_id", item.get("repeat"))
        if type(prompt_index) is not int or not 0 <= prompt_index < 4 or type(repeat) is not int or not 0 <= repeat < 4:
            raise ValueError("Validation needs fixed prompt_index 0..3 and rollout_id 0..3")
        if (prompt_index, repeat) in observed:
            raise ValueError("Duplicate fixed validation position")
        observed.add((prompt_index, repeat))
        seed = item.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32 or seed in seeds:
            raise ValueError("Validation needs 16 unique fixed uint32 seeds")
        seeds.add(seed)
        value = finite(score, "reward score")
        if finite(detail.get("reward"), "detail reward") != value:
            raise ValueError("Reward score/detail differs")
        row = {"id": key, "image": item["image"], "prompt_id": item.get("prompt_id"),
               "prompt_index": prompt_index, "rollout_id": repeat, "seed": seed,
               "total": value, "aesthetic": finite(detail.get("aesthetic_reward"), "aesthetic_reward"),
               "raw": {}, "normalized": {}, "contributions": {}}
        for model in MODELS:
            for target, source in (("raw", "raw_scores"), ("normalized", "normalized_scores"),
                                   ("contributions", "contributions")):
                row[target][model] = finite(detail.get(source, {}).get(model), source + "/" + model)
        # SDG is disabled. Its absent values are intentionally not synthesized or
        # propagated from legacy structural zero placeholders into new charts.
        grouped[prompt_index].append(row)
    fixed = summary.get("fixed_seeds")
    if fixed is not None:
        if not isinstance(fixed, list) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in fixed):
            raise ValueError("Daemon frozen seeds must be uint32 integers")
        if len(fixed) == 4:
            # Daemon stores one seed per repeat and derives the next prompt's
            # seed by its fixed four-image offset, matching native jobs_for.
            expected = [(fixed[item.get("rollout_id", item.get("repeat"))]
                         + item["prompt_index"] * 4) % 2**32 for item in items]
        elif len(fixed) == 16:
            expected = fixed
        else:
            raise ValueError("Daemon frozen plan needs four base seeds or 16 expanded seeds")
        if expected != [item["seed"] for item in items]:
            raise ValueError("Image seeds differ from daemon frozen plan")
    if set(grouped) != set(range(4)) or any(len(rows) != 4 for rows in grouped.values()):
        raise ValueError("Fixed validation must have four complete prompt groups")
    per_prompt, all_rows = [], []
    for index in range(4):
        rows = sorted(grouped[index], key=lambda row: row["rollout_id"])
        if len({row["prompt_id"] for row in rows}) != 1:
            raise ValueError("One fixed prompt_index must identify one original prompt")
        all_rows.extend(rows)
        per_prompt.append(component_summary(rows, prompt_index=index, prompt_id=rows[0]["prompt_id"]))
    report = {"version": 1, "status": "complete", "checkpoint_id": checkpoint,
              "source_step": step, "image_count": 16, "prompts": 4, "images_per_prompt": 4,
              "sdg_enabled": False, "overall": component_summary(all_rows),
              "per_prompt": per_prompt, "images": all_rows,
              "input_sha256": {key: file_digest(paths[key]) for key in ("ack", "evaluation_result", "summary")},
              "request": str(request_path), "request_sha256": request_hash,
              "checkpoint_manifest_sha256": result.get("checkpoint_manifest_sha256"),
              "fixed_plan_sha256": result.get("fixed_plan_sha256", summary.get("request_plan_sha256")),
              "combined_archive": result.get("combined_archive"),
              "combined_fingerprint": result.get("combined_fingerprint"),
              "scoring_wall_time": finite(result.get("scored_at", ack.get("acknowledged_at", 0.0)), "scored_at"),
              "logging_log_dir": str(paths["logging_log_dir"]),
              "interpretation": "Fixed prompts and seeds; four images per prompt. Reported std is population spread, not a confidence interval."}
    report["pooled_within_prompt_std"] = math.sqrt(statistics.mean(
        prompt["total"]["std"]**2 for prompt in per_prompt))
    return report, paths


def component_summary(rows, **metadata):
    return {**metadata, "total": stats([row["total"] for row in rows]),
            "aesthetic": stats([row["aesthetic"] for row in rows]),
            "raw": {name: stats([row["raw"][name] for row in rows]) for name in MODELS},
            "normalized": {name: stats([row["normalized"][name] for row in rows]) for name in MODELS},
            "contributions": {name: stats([row["contributions"][name] for row in rows]) for name in MODELS},
            "ranking": [{"id": row["id"], "image": row["image"], "total": row["total"],
                         "rollout_id": row["rollout_id"]} for row in sorted(rows, key=lambda row: (-row["total"], row["rollout_id"]))]}


def scalar_values(report):
    overall = report["overall"]
    values = {"validation/total": overall["total"]["mean"],
              "validation/aesthetic": overall["aesthetic"]["mean"],
              "validation/within_prompt_std": report["pooled_within_prompt_std"]}
    for model in MODELS:
        values["validation/" + model] = overall["raw"][model]["mean"]
        values["validation/normalized/" + model] = overall["normalized"][model]["mean"]
        values["validation/contribution/" + model] = overall["contributions"][model]["mean"]
    for prompt in report["per_prompt"]:
        values[f"validation/per_prompt_{prompt['prompt_index']}"] = prompt["total"]["mean"]
    return values


def event_bytes(report):
    # Use the supported no-TensorFlow sentinel before accessing lazy TB compat.
    from tensorboard import compat
    compat.notf = types.ModuleType("tensorboard.compat.notf")
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter
    import io
    buffer = io.BytesIO()
    writer = RecordWriter(buffer)
    when = report["scoring_wall_time"]
    writer.write(event_pb2.Event(wall_time=when, file_version="brain.Event:2").SerializeToString())
    summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=key, simple_value=value)
                                        for key, value in sorted(scalar_values(report).items())])
    writer.write(event_pb2.Event(wall_time=when, step=report["source_step"], summary=summary).SerializeToString())
    writer.flush()
    return buffer.getvalue()


def publish_event(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("An existing checkpoint TensorBoard event has different contents")
        return
    fd, temporary = tempfile.mkstemp(prefix=".validation-event.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(job):
    report, paths = validate_and_summarize(job)
    root = paths["summary"].parent
    report_path = root / "validation_report.json"
    marker = root / "validation_report.complete.json"
    log_dir = paths["logging_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    event_path = log_dir / (f"events.out.tfevents.{report['source_step']:012d}.checkpoint_validation."
                            f"{report['checkpoint_id']}")
    report_hash = digest(report)
    with locked(log_dir / ".checkpoint_validation_report.lock"):
        if marker.exists():
            done = read_json(marker)
            if done.get("report_digest") != report_hash or file_digest(report_path) != done.get("report_sha256"):
                raise ValueError("Completed validation report identity changed")
            if file_digest(event_path) != done.get("event_sha256"):
                raise ValueError("Completed validation TensorBoard event changed")
            return done
        if report_path.exists() and read_json(report_path) != report:
            raise ValueError("An existing validation report has different immutable input")
        payload = event_bytes(report)
        # One atomic deterministic event file per checkpoint avoids duplicate
        # (tag, step) writes even if interrupted between event and success marker.
        publish_event(event_path, payload)
        atomic_json(report_path, report)
        done = {"version": 1, "status": "complete", "checkpoint_id": report["checkpoint_id"],
                "source_step": report["source_step"], "report": str(report_path),
                "report_digest": report_hash, "report_sha256": file_digest(report_path),
                "event": str(event_path), "event_sha256": file_digest(event_path),
                "completed_at": time.time()}
        atomic_json(marker, done)
    print(json.dumps({"event": "validation_report_complete", "reporting": done}), flush=True)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(read_json(args.config))


if __name__ == "__main__":
    main()
