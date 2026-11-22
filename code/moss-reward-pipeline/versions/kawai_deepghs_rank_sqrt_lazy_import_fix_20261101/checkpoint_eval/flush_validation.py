#!/usr/bin/env python3
"""Drain expected checkpoint eval jobs after training, using the original pipeline.

CPU controller only. The invoked original reward pipeline owns the four scoring
GPUs; no training update is performed. Ready generation may continue on GPU 4.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import signal
import time

import run_reward_with_validation as wrapper


def ready_jobs(settings, expected):
    jobs = []
    for path in wrapper.outbox_path(settings).glob("*.ready.json"):
        ready = wrapper.read_json(path)
        if ready.get("source_step") in expected:
            # This verifies immutable request identity and every image exists.
            jobs.append(wrapper.ready_record(path, settings))
    return sorted(jobs, key=lambda job: (job["source_step"], job["job_id"]))


def committed_ack(job, settings):
    path = wrapper.outbox_path(settings) / (job["job_id"] + ".ack.json")
    if not path.exists():
        return None
    ack = wrapper.read_json(path)
    if ack.get("status") != "complete":
        return None
    for field in ("job_id", "checkpoint_id", "checkpoint_manifest_sha256", "request_sha256"):
        if ack.get(field) != job[field]:
            raise ValueError(f"Checkpoint ACK {field} mismatch: {path}")
    scores = ack.get("scores")
    if scores is None:
        result_path = ack.get("evaluation_result", ack.get("result_path"))
        scores = wrapper.read_json(wrapper.inside(result_path, settings["service_root"], "Eval result"))["scores"]
    expected_images = int(settings.get("expected_prompts", 4)) * int(settings.get("images_per_prompt", 4))
    if len(scores) != expected_images or any(type(value) not in (float, int) or not math.isfinite(value) for value in scores):
        raise ValueError(f"Checkpoint ACK contains invalid scores: {path}")
    return dict(source_step=job["source_step"], job_id=job["job_id"], ack=str(path),
                evaluation_result=ack.get("evaluation_result", ack.get("result_path")),
                images=len(scores))


def run(config, *, pipeline=None, splitter=None, clock=time.monotonic, sleep=time.sleep):
    expected = config.get("expected_updates")
    if (not isinstance(expected, list) or not expected or
            any(type(step) is not int or step < 0 for step in expected) or
            len(expected) != len(set(expected))):
        raise ValueError("expected_updates must be a nonempty list of unique nonnegative integers")
    expected = sorted(expected)
    timeout = float(config.get("timeout_seconds", 21600))
    interval = float(config.get("poll_interval_seconds", 10))
    if timeout <= 0 or interval < 0:
        raise ValueError("Require timeout_seconds > 0 and poll_interval_seconds >= 0")
    cfg = wrapper.read_json(config["pipeline_config"])
    settings = copy.deepcopy(cfg.get("checkpoint_validation", {}))
    settings.update(enabled=True, service_root=str(Path(config["service_root"]).resolve()),
                    expected_updates=expected, max_jobs_per_call=1)
    cfg["checkpoint_validation"] = settings
    archive = Path(config.get("archive_root", Path(settings["service_root"]) / "validation_flush")).resolve()
    wrapper.inside(archive, cfg["allowed_archive_root"], "Validation flush archive")
    archive.mkdir(parents=True, exist_ok=True)
    output = Path(config.get("output", archive / "flush_summary.json")).resolve()
    wrapper.inside(output, cfg["allowed_archive_root"], "Validation flush summary")
    start, failures = clock(), []
    with wrapper.file_lock(archive / ".flush.lock"):
        wrapper.save(archive / "flush.config.json", config)
        while True:
            if clock() - start >= timeout:
                status = dict(status="timeout", expected_updates=expected,
                              elapsed_seconds=clock() - start, failures=failures)
                wrapper.save(output, status)
                raise TimeoutError(f"Fixed checkpoint evaluation did not finish within {timeout:g}s; see {output}")
            try:
                jobs = ready_jobs(settings, expected)
                done, pending = {}, []
                for job in jobs:
                    ack = committed_ack(job, settings)
                    if ack is None:
                        pending.append(job)
                    else:
                        done.setdefault(job["source_step"], ack)
                status = dict(status="complete" if set(done) == set(expected) else "waiting",
                              expected_updates=expected, completed_updates=sorted(done),
                              missing_updates=sorted(set(expected) - set(done)),
                              results=list(done.values()), elapsed_seconds=clock() - start,
                              failures=failures)
                wrapper.save(output, status)
                if status["status"] == "complete":
                    return status
                pending = [job for job in pending if job["source_step"] not in done]
                if pending:
                    job = pending[0]
                    local_cfg = copy.deepcopy(cfg)
                    local_cfg["checkpoint_validation"]["job_id"] = job["job_id"]
                    step_dir = archive / job["job_id"]
                    # Empty train item list is deliberate. The wrapper builds a
                    # nonempty 16-image eval request for the original pipeline.
                    request = dict(version=1, items=[], context=dict(step_dir=str(step_dir),
                        validation_only=True, job_id=job["job_id"],
                        checkpoint_id=job["checkpoint_id"], source_step=job["source_step"]))
                    response = wrapper.run(local_cfg, request, step_dir / "validation_only_response.json",
                                           pipeline=pipeline, splitter=splitter, validation_only=True)
                    if response.get("status") != "no_ready_validation":
                        continue
            except Exception as exc:
                failure = dict(type=type(exc).__name__, message=str(exc),
                               elapsed_seconds=clock() - start, timestamp=time.time())
                failures.append(failure)
                wrapper.save(archive / ("failure." + str(time.time_ns()) + ".json"), failure)
                wrapper.save(output, dict(status="waiting_after_error", expected_updates=expected,
                    elapsed_seconds=clock() - start, failures=failures))
            # No busy wait for GPU 4 generation or for unrelated GPU ownership.
            sleep(min(interval, max(0., timeout - (clock() - start))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    result = run(wrapper.read_json(args.config))
    print(wrapper.json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
