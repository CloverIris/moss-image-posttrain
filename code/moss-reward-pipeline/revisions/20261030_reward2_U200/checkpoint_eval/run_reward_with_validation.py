#!/usr/bin/env python3
"""Append completed fixed-checkpoint images to an existing training reward job.

The original reward pipeline remains the only scoring/fusion implementation.
This wrapper claims ready files without waiting, records an immutable combined
request, verifies HPS batch companions, and returns only training scores to ART.
All checkpoint images and raw four-model results remain in the NVMe archive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time


SCHEMA_VERSION = 1
EVAL_PREFIX = "checkpoint-eval::"
SAFE_JOB = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def file_lock(path):
    """Nonblocking OS lock; a dead process does not leave a stale lock owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
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


def inside(path, root, description):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{description} must stay inside {root}: {path}")
    return path


def load_module(path, name):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def split_groups(items, replicas):
    """Exact existing batch controller policy, also useful for CPU-only tests."""
    grouped = defaultdict(list)
    for item in items:
        key = item.get("prompt_id") or item["prompt"] or item["id"]
        grouped[key].append(item)
    shards = [[] for _ in range(replicas)]
    for group in sorted(grouped.values(), key=lambda rows: -len(rows)):
        rank = min(range(replicas), key=lambda i: len(shards[i]))
        shards[rank].extend(group)
    return shards


def batching(shards, batch_size):
    return [[[row["id"] for row in shard[start:start + batch_size]]
             for start in range(0, len(shard), batch_size)] for shard in shards]


def companion_map(batches):
    return {key: batch for shard in batches for batch in shard for key in batch}


def hps_plan(train_items, eval_items, batch_cfg, splitter=split_groups):
    """Fail closed if appending eval would change any batch-conditioned score."""
    settings = batch_cfg["hps"]
    batch_size = int(settings["batch_size"])
    replicas = len(settings["replica_devices"])
    if batch_size < 1 or replicas < 1:
        raise ValueError("Invalid HPS batch configuration")
    train = batching(splitter(train_items, replicas), batch_size)
    combined = batching(splitter(train_items + eval_items, replicas), batch_size)
    evaluation = batching(splitter(eval_items, replicas), batch_size)
    training_ids = {row["id"] for row in train_items}
    eval_ids = {row["id"] for row in eval_items}
    if training_ids & eval_ids:
        raise ValueError("Training and checkpoint eval IDs overlap")
    for shard in combined:
        for batch in shard:
            if set(batch) & training_ids and set(batch) & eval_ids:
                raise ValueError("HPS would mix train and checkpoint images in one batch")
    actual = companion_map(combined)
    for key, companions in companion_map(train).items():
        if actual[key] != companions:
            raise ValueError("Appending checkpoint images would change HPS training companions")
    for key, companions in companion_map(evaluation).items():
        if actual[key] != companions:
            raise ValueError("HPS checkpoint companions would depend on training data")
    # A full batch per prompt avoids padding or cross-prompt companion drift.
    if eval_items and any(len(batch) != batch_size for shard in evaluation for batch in shard):
        raise ValueError("Checkpoint groups must form full HPS batches")
    return dict(batch_size=batch_size, replica_devices=settings["replica_devices"],
                train_batches=train, eval_batches=evaluation,
                combined_batches=combined,
                batch_companions_preserved=True,
                policy="existing split_groups; stable prompt groups; descending group size")


def group_id(row):
    value = row.get("sample_id", row.get("prompt_index", row.get("prompt_id", "")))
    if str(value) == "":
        raise ValueError("Eval image needs sample_id, prompt_index or prompt_id")
    return str(value)


def checkpoint_items(request, ready, service_root, settings):
    if request.get("version") != 1 or not request.get("items"):
        raise ValueError("Ready checkpoint request must use version=1 and nonempty items")
    groups = defaultdict(list)
    output, ids = [], set()
    prefix = EVAL_PREFIX + ready["job_id"] + "::"
    for index, original in enumerate(request["items"]):
        row = copy.deepcopy(original)
        original_id = str(row.get("id", ""))
        if not original_id or original_id in ids or original_id.startswith(EVAL_PREFIX):
            raise ValueError("Eval input IDs must be explicit, unique and unprefixed")
        ids.add(original_id)
        prompt_id = group_id(row)
        if not isinstance(row.get("prompt"), str):
            raise ValueError("Eval prompt must be an unchanged string")
        image = inside(row["image"], service_root, "Eval image")
        if not image.is_file():
            raise ValueError(f"Eval image is incomplete: {image}")
        groups[prompt_id].append(original_id)
        row.update(id=prefix + original_id, image=str(image),
                   sample_id=prefix + prompt_id, prompt_id=prefix + prompt_id,
                   checkpoint_eval=dict(job_id=ready["job_id"],
                                        checkpoint_id=ready["checkpoint_id"],
                                        source_step=ready["source_step"],
                                        original_id=original_id,
                                        original_prompt_id=prompt_id,
                                        request_index=index))
        output.append(row)
    expected_prompts = int(settings.get("expected_prompts", 4))
    per_prompt = int(settings.get("images_per_prompt", 4))
    if len(groups) != expected_prompts or any(len(g) != per_prompt for g in groups.values()):
        raise ValueError(f"Ready eval requires {expected_prompts} prompts × {per_prompt} images")
    # Preserve original request order, prompt text and each prompt's rollout order.
    return output


def ready_record(path, settings):
    service_root = Path(settings["service_root"]).resolve()
    ready = read_json(path)
    job = str(ready.get("job_id", ""))
    if ready.get("version") != 1 or not SAFE_JOB.fullmatch(job):
        raise ValueError("Ready record has an invalid version or job_id")
    if ready.get("status") != "ready_for_scoring":
        raise ValueError("Ready record status must be ready_for_scoring")
    if Path(path).name != job + ".ready.json":
        raise ValueError("Ready filename and job_id do not match")
    if not isinstance(ready.get("checkpoint_id"), str) or not ready["checkpoint_id"]:
        raise ValueError("Ready record needs a checkpoint_id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(ready.get("checkpoint_manifest_sha256", ""))):
        raise ValueError("Ready record needs the committed checkpoint manifest SHA256")
    if type(ready.get("source_step")) is not int or ready["source_step"] < 0:
        raise ValueError("Ready record needs a nonnegative integer source_step")
    request_path = inside(ready["request_path"], service_root, "Eval request")
    result_dir = inside(ready["result_dir"], service_root, "Eval result directory")
    request_bytes = request_path.read_bytes()
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    if ready.get("request_sha256", request_sha256) != request_sha256:
        raise ValueError("Ready request checksum does not match")
    request = json.loads(request_bytes.decode("utf-8-sig"))
    context = request.get("context", {})
    for field in ("checkpoint_id", "source_step"):
        if field in context and context[field] != ready[field]:
            raise ValueError(f"Ready request context {field} does not match")
    rows = checkpoint_items(request, ready, service_root, settings)
    return dict(ready, ready_path=str(Path(path).resolve()),
                request_path=str(request_path), result_dir=str(result_dir),
                request_sha256=request_sha256, ready_digest=digest(ready), items=rows)


def outbox_path(settings):
    root = Path(settings["service_root"]).resolve()
    return inside(settings.get("outbox", root / "outbox"), root, "Outbox")


def choose_jobs(settings, binding, train_items, batch_cfg, splitter):
    outbox = outbox_path(settings)
    outbox.mkdir(parents=True, exist_ok=True)
    picked, rejected = [], []
    maximum = int(settings.get("max_jobs_per_call", 1))
    if maximum not in (0, 1):
        raise ValueError("max_jobs_per_call must be 0 or 1; never drain an eval backlog into one training job")
    with file_lock(outbox / ".claim.lock"):
        candidates = []
        for path in outbox.glob("*.ready.json"):
            job = path.name.removesuffix(".ready.json")
            if (outbox / (job + ".ack.json")).exists():
                continue
            try:
                ready = ready_record(path, settings)
                if settings.get("job_id") and ready["job_id"] != settings["job_id"]:
                    continue
                if settings.get("expected_updates") is not None and ready["source_step"] not in settings["expected_updates"]:
                    continue
                candidates.append(ready)
            except Exception as exc:
                rejected.append(dict(ready=str(path), reason=str(exc)))
        candidates.sort(key=lambda r: (r["source_step"], r["job_id"]))
        for ready in candidates:
            if len(picked) >= maximum:
                break
            claim_path = outbox / (ready["job_id"] + ".claim.json")
            if claim_path.exists():
                claim = read_json(claim_path)
                if claim.get("binding") != binding:
                    rejected.append(dict(job_id=ready["job_id"], reason="claimed by another training request"))
                    continue
                if claim.get("ready_digest") != ready["ready_digest"]:
                    raise ValueError("A claimed ready record was modified")
            try:
                all_eval = [row for job in picked for row in job["items"]] + ready["items"]
                hps_plan(train_items, all_eval, batch_cfg, splitter)
            except ValueError as exc:
                rejected.append(dict(job_id=ready["job_id"], reason=str(exc)))
                continue
            save(claim_path, dict(version=SCHEMA_VERSION, job_id=ready["job_id"],
                                  binding=binding, ready_digest=ready["ready_digest"],
                                  request_sha256=ready["request_sha256"],
                                  claimed_at=time.time()))
            picked.append(ready)
    return picked, rejected


def validate_response(response, expected_ids):
    scores, details = response["scores"], response["details"]
    if len(scores) != len(expected_ids) or len(details) != len(expected_ids):
        raise ValueError("Combined reward response count mismatch")
    if [row.get("id") for row in details] != expected_ids:
        raise ValueError("Combined reward response ID order changed")
    for score, detail in zip(scores, details):
        if type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("Nonfinite combined reward")
        if detail.get("reward") != score:
            raise ValueError("Reward score/detail mismatch")


def summary_for(details, metadata):
    if not details:
        return dict(version=SCHEMA_VERSION, **metadata, images=0, reward=None,
                    raw_component_mean={}, normalized_component_mean={},
                    contribution_mean={}, groups=[])
    values = [row["reward"] for row in details]
    groups = defaultdict(list)
    for row in details:
        groups[row.get("original_prompt_id", row["prompt_id"])].append(row)
    aesthetic_models = ("kawai", "deepghs", "hps")
    sdg_presence = ["sdg_artifact" in row["raw_scores"] for row in details]
    if any(sdg_presence) and not all(sdg_presence):
        raise ValueError("A score summary cannot mix SDG-enabled and SDG-disabled rows")
    sdg_enabled = metadata.get("sdg_enabled", all(sdg_presence))
    if sdg_enabled != all(sdg_presence):
        raise ValueError("Score summary SDG setting does not match its raw components")
    model_names = aesthetic_models + (("sdg_artifact",) if sdg_enabled else ())
    metadata = dict(metadata, sdg_enabled=sdg_enabled)
    return dict(version=SCHEMA_VERSION, **metadata, images=len(values),
                reward=dict(mean=statistics.mean(values), std=statistics.pstdev(values),
                            min=min(values), max=max(values)),
                raw_component_mean={name: statistics.mean(r["raw_scores"][name] for r in details)
                                    for name in model_names},
                normalized_component_mean={name: statistics.mean(r["normalized_scores"][name] for r in details)
                                           for name in aesthetic_models},
                contribution_mean={name: statistics.mean(r["contributions"][name] for r in details)
                                   for name in model_names},
                groups=[dict(prompt_id=key, mean=statistics.mean(r["reward"] for r in rows),
                             std=statistics.pstdev(r["reward"] for r in rows),
                             ranking=[dict(id=r.get("original_id", r["id"]), image=r["image"], reward=r["reward"])
                                      for r in sorted(rows, key=lambda r: -r["reward"])])
                        for key, rows in groups.items()])


def repair_hps_resume(archive, plan):
    """Retain complete original batches only; preserve every removed raw byte.

    The reused worker writes one JSON line at a time. If interruption occurs
    between companion rows, its pending filter would otherwise change pairs.
    No completed full batch is rerun and no original raw evidence is deleted.
    """
    changes = []
    archive = Path(archive)
    for rank, batches in enumerate(plan["combined_batches"]):
        directory = archive / "deepghs_hps.work" / "hps" / f"dp{rank:02d}"
        path = directory / "hps.jsonl"
        if not path.exists():
            continue
        raw = path.read_bytes()
        lines, invalid_tail = raw.splitlines(keepends=True), False
        rows = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines) - 1:
                    raise ValueError(f"Corrupt non-final HPS row: {path}")
                invalid_tail = True
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate HPS resume rows: {path}")
        expected = {key for batch in batches for key in batch}
        if set(ids) - expected:
            raise ValueError(f"Unexpected HPS resume IDs: {path}")
        present = set(ids)
        keep = {key for batch in batches if set(batch) <= present for key in batch}
        retained = [row for row in rows if row["id"] in keep]
        if len(retained) == len(rows) and not invalid_tail:
            continue
        stamp = str(time.time_ns())
        backup = path.with_name("hps.interrupted_preserved." + stamp + ".jsonl")
        backup.write_bytes(raw)
        temp = path.with_name(path.name + ".repair.tmp")
        with temp.open("w", encoding="utf-8") as stream:
            for row in retained:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        metrics = directory / "hps.metrics.json"
        if metrics.exists():
            metrics.replace(directory / ("hps.metrics.preserved." + stamp + ".json"))
        changes.append(dict(path=str(path), preserved=str(backup),
                            removed_incomplete_batch_ids=[key for key in ids if key not in keep],
                            invalid_tail=invalid_tail, retained_images=len(retained)))
    return changes


def pipeline_identity(pipeline, cfg, combined):
    items = pipeline.canonical_items(combined)
    identity = dict(items=items, config=cfg, calibration=read_json(cfg["calibration"]),
                    batch_config=read_json(cfg["batch_config"]),
                    sdg_config=read_json(cfg["sdg_config"]) if cfg.get("sdg_enabled", True) else None,
                    code=pipeline.provenance(cfg))
    fingerprint = pipeline.digest(identity)
    root = Path(combined.get("context", {}).get("step_dir", cfg["archive_root"])).resolve()
    return fingerprint, root / "reward_models" / fingerprint[:24]


def publish(plan, response, state_dir, settings):
    train_count = plan["train_count"]
    train_response = dict(response, scores=response["scores"][:train_count],
                          details=response["details"][:train_count],
                          checkpoint_validation=dict(version=SCHEMA_VERSION,
                              training_images=train_count,
                              checkpoint_images=len(response["scores"]) - train_count,
                              jobs=[job["job_id"] for job in plan["jobs"]],
                              combined_archive=response["archive"], state_dir=str(state_dir)))
    published = {}
    for job in plan["jobs"]:
        selected_ids = [row["id"] for row in job["items"]]
        table = {row["id"]: row for row in response["details"]}
        metadata = dict(job_id=job["job_id"], checkpoint_id=job["checkpoint_id"],
                        source_step=job["source_step"],
                        checkpoint_manifest_sha256=job.get("checkpoint_manifest_sha256"),
                        fixed_plan_sha256=job.get("fixed_plan_sha256"),
                        scored_with_training_context=plan["training_context"],
                        combined_archive=response["archive"],
                        combined_fingerprint=response["fingerprint"],
                        request_sha256=job["request_sha256"],
                        ready_digest=job["ready_digest"], scored_at=time.time(),
                        sdg_enabled=plan.get("sdg_enabled", True))
        rows = []
        for item in job["items"]:
            row = copy.deepcopy(table[item["id"]])
            source = item["checkpoint_eval"]
            row.update(combined_id=row["id"], original_id=source["original_id"],
                       original_prompt_id=source["original_prompt_id"])
            for key in ("prompt_index", "repeat", "rollout_id", "roll_id", "seed", "index", "source_id"):
                if key in item:
                    row[key] = item[key]
            rows.append(row)
        result = dict(version=SCHEMA_VERSION, status="complete", **metadata,
                      ids=selected_ids, scores=[row["reward"] for row in rows], details=rows,
                      hps_batch_plan=plan["hps_batch_plan"])
        summary = summary_for(rows, metadata)
        published[job["job_id"]] = (result, summary)
        result_dir = Path(job["result_dir"])
        # Idempotent reruns must not overwrite a score from another collection.
        if (result_dir / "result.json").exists():
            previous = read_json(result_dir / "result.json")
            if previous.get("combined_fingerprint") != response["fingerprint"]:
                raise ValueError("Checkpoint result already belongs to another scoring collection")
        save(result_dir / "result.json", result)
        save(result_dir / "summary.json", summary)
        save(state_dir / (job["job_id"] + ".eval_result.json"), result)
    archive = Path(response["archive"])
    domain_meta = dict(binding=plan["binding"], combined_archive=response["archive"],
                       combined_fingerprint=response["fingerprint"],
                       training_context=plan["training_context"],
                       sdg_enabled=plan.get("sdg_enabled", True))
    save(archive / "training_summary.json", summary_for(
        train_response["details"], dict(domain="training", **domain_meta)))
    save(archive / "validation_summary.json", dict(version=SCHEMA_VERSION,
        domain="checkpoint_validation", images=len(response["scores"]) - train_count,
        **domain_meta, jobs=[summary for _, summary in published.values()]))
    save(archive / "evaluation_domains.json", dict(version=SCHEMA_VERSION,
        training_count=train_count, validation_count=len(response["scores"]) - train_count,
        train_ids=[row["id"] for row in train_response["details"]],
        validation_ids=[row["id"] for row in response["details"][train_count:]],
        original_summary_scope="combined_training_and_checkpoint_validation",
        training_summary=str(archive / "training_summary.json"),
        validation_summary=str(archive / "validation_summary.json"),
        validation_jobs=[dict(job_id=job["job_id"], checkpoint_id=job["checkpoint_id"],
            source_step=job["source_step"], checkpoint_manifest_sha256=job["checkpoint_manifest_sha256"],
            result_path=str(Path(job["result_dir"]) / "result.json")) for job in plan["jobs"]],
        **domain_meta))
    # Save the training-only response before acknowledging any checkpoint job.
    save(state_dir / "training_response.json", train_response)
    if plan["jobs"]:
        outbox = outbox_path(settings)
        with file_lock(outbox / ".claim.lock"):
            for job in plan["jobs"]:
                result, summary = published[job["job_id"]]
                ack_path = outbox / (job["job_id"] + ".ack.json")
                ack = dict(version=SCHEMA_VERSION, status="complete", job_id=job["job_id"],
                           checkpoint_id=job["checkpoint_id"], source_step=job["source_step"],
                           checkpoint_manifest_sha256=job.get("checkpoint_manifest_sha256"),
                           binding=plan["binding"], ready_digest=job["ready_digest"],
                           request_sha256=job["request_sha256"],
                           combined_fingerprint=response["fingerprint"],
                           result_path=str(Path(job["result_dir"]) / "result.json"),
                           evaluation_result=str(Path(job["result_dir"]) / "result.json"),
                           summary_path=str(Path(job["result_dir"]) / "summary.json"),
                           scores=result["scores"], details=result["details"], summary=summary,
                           acknowledged_at=time.time())
                if ack_path.exists():
                    previous = read_json(ack_path)
                    if (previous.get("binding"), previous.get("combined_fingerprint")) != (
                            plan["binding"], response["fingerprint"]):
                        raise ValueError("Checkpoint ack identity mismatch")
                else:
                    save(ack_path, ack)
    return train_response


def run(cfg, request, output, *, pipeline=None, splitter=None, validation_only=False):
    cfg = copy.deepcopy(cfg)
    settings = cfg.pop("checkpoint_validation", {})
    pipeline_root = Path(settings.get("pipeline_root", Path(__file__).resolve().parents[1]))
    pipeline = pipeline or load_module(pipeline_root / "run_reward_pipeline.py", "_checkpoint_reward_pipeline")
    if not settings.get("enabled", True) or not settings.get("service_root"):
        if validation_only:
            raise ValueError("Validation-only scoring requires an enabled checkpoint service")
        return pipeline.run(cfg, request, output)
    batch_cfg = read_json(cfg["batch_config"])
    if splitter is None:
        batch_module = load_module(Path(cfg["batch_root"]) / "run.py", "_checkpoint_existing_batch")
        splitter = batch_module.split_groups
    if validation_only:
        if request.get("version") != 1 or request.get("items") != []:
            raise ValueError("Validation-only mode requires version=1 and an empty training item list")
        train_items = []
    else:
        train_items = pipeline.canonical_items(request)
    if any(row["id"].startswith(EVAL_PREFIX) for row in train_items):
        raise ValueError("Training IDs use the reserved checkpoint eval prefix")
    root = Path(request.get("context", {}).get("step_dir", cfg["archive_root"])).resolve()
    inside(root, cfg["allowed_archive_root"], "Training reward directory")
    train_identity = dict(config=cfg, items=train_items, context=request.get("context", {}),
                          mode="validation_only" if validation_only else "training_with_validation")
    binding = digest(train_identity)
    state_dir = root / "reward_validation" / binding[:24]
    state_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(state_dir / ".lock"):
        plan_path = state_dir / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan["binding"] != binding:
                raise ValueError("Existing checkpoint validation binding mismatch")
            combined = read_json(state_dir / "combined_request.json")
            if digest(combined) != plan["combined_request_digest"]:
                raise ValueError("Combined request was modified after claiming")
        else:
            jobs, rejected = choose_jobs(settings, binding, train_items, batch_cfg, splitter)
            if validation_only and not jobs:
                return dict(status="no_ready_validation", scores=[], details=[],
                            rejected_ready=rejected, state_dir=str(state_dir))
            eval_items = [row for job in jobs for row in job["items"]]
            combined = copy.deepcopy(request)
            combined["items"] = train_items + eval_items
            combined.setdefault("context", {})["checkpoint_validation"] = dict(
                binding=binding, jobs=[job["job_id"] for job in jobs],
                training_images=len(train_items), checkpoint_images=len(eval_items))
            batch_plan = hps_plan(train_items, eval_items, batch_cfg, splitter)
            fingerprint, archive = pipeline_identity(pipeline, cfg, combined)
            plan = dict(version=SCHEMA_VERSION, binding=binding,
                        mode="validation_only" if validation_only else "training_with_validation",
                        training_context=request.get("context", {}), train_count=len(train_items),
                        jobs=jobs, rejected_ready=rejected, hps_batch_plan=batch_plan,
                        combined_request_digest=digest(combined),
                        pipeline_fingerprint=fingerprint, pipeline_archive=str(archive),
                        settings=copy.deepcopy(settings), sdg_enabled=cfg.get("sdg_enabled", True),
                        created_at=time.time())
            save(state_dir / "training_request.json", request)
            save(state_dir / "combined_request.json", combined)
            save(state_dir / "pipeline.config.json", cfg)
            save(state_dir / "wrapper.config.json", settings)
            save(state_dir / "plan.json", plan)
        # A retry must use the same claimed jobs/configuration and exact batch plan.
        settings = plan["settings"]
        fingerprint, archive = pipeline_identity(pipeline, cfg, combined)
        if fingerprint != plan["pipeline_fingerprint"]:
            raise ValueError("Reward code/config changed during retry; preserve this collection and inspect it")
        current_plan = hps_plan(train_items, combined["items"][len(train_items):], batch_cfg, splitter)
        if current_plan != plan["hps_batch_plan"]:
            raise ValueError("Existing HPS shard/batch plan changed")
        combined_output = state_dir / "combined_response.json"
        try:
            if combined_output.exists():
                response = read_json(combined_output)
            else:
                # Acquiring this lock also rejects interference from another active scorer.
                with file_lock(archive / ".lock"):
                    repairs = repair_hps_resume(archive, current_plan)
                if repairs:
                    save(state_dir / ("hps_resume_repair." + str(time.time_ns()) + ".json"), repairs)
                response = pipeline.run(cfg, combined, combined_output)
            validate_response(response, [row["id"] for row in combined["items"]])
            if response.get("fingerprint") != fingerprint:
                raise ValueError("Scorer returned a different fingerprint")
            train_response = publish(plan, response, state_dir, settings)
            save(output, train_response)
            save(state_dir / "complete.json", dict(completed_at=time.time(), binding=binding,
                 training_images=len(train_response["scores"]),
                 checkpoint_images=len(response["scores"]) - len(train_response["scores"]),
                 pipeline_archive=str(archive)))
            return train_response
        except BaseException as exc:
            save(state_dir / ("failure." + str(time.time_ns()) + ".json"),
                 dict(type=type(exc).__name__, message=str(exc), timestamp=time.time(),
                      ready_jobs_preserved=[job["ready_path"] for job in plan["jobs"]]))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    response = run(read_json(args.config), read_json(args.request), args.output)
    print(json.dumps(dict(event="composite_reward_complete", images=len(response["scores"]),
                          archive=response["archive"],
                          checkpoint_validation=response.get("checkpoint_validation")),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
