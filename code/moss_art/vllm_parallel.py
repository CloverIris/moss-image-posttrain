"""Host-only orchestration of independent offline vLLM cache replicas.

The parent planner supplies only content-addressed keys and token IDs. Every
child receives its own persistent jobs JSONL, config, cache and log; it never
reloads the source dataset or writes the root manifest. The caller must hold
the root output lock while invoking :func:`run_parallel`.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np

from .data import atomic_json, sha256_file


_KEY = re.compile(r"[0-9a-f]{64}\Z")
_VOLATILE = {"entries", "backend", "dropped_records", "parallel_producers", "parallel_execution", "complete"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _settings(manifest):
    return {key: value for key, value in manifest.items() if key not in _VOLATILE}


def _producer(metadata):
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("Cache producer metadata must be an object")
    # Location is the only permitted replica difference. Versions, source
    # hashes, model architecture and every compute setting remain strict.
    return {key: value for key, value in metadata.items() if key != "model_view"}


def normalize_jobs(jobs, maximum=None):
    """Deduplicate a key/ID plan without retaining or accepting prompt bodies."""
    unique = {}
    for source in jobs:
        if not isinstance(source, dict) or set(source) != {"key", "input_ids"}:
            raise ValueError("A prefill job must contain only key and input_ids")
        key = source["key"]
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise ValueError("Prefill job key must be a lowercase SHA256")
        values = np.asarray(source["input_ids"])
        if (values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu"
                or values.min() < 0 or values.max() > np.iinfo(np.int32).max):
            raise ValueError(f"Invalid token IDs for cache key {key}")
        ids = values.astype(np.int32).tolist()
        if maximum is not None and len(ids) > maximum:
            raise ValueError(f"Job exceeds full context limit: {key}")
        job = {"key": key, "input_ids": ids}
        if key in unique and unique[key] != job:
            raise ValueError(f"Conflicting token IDs for duplicate cache key {key}")
        unique[key] = job
    return [unique[key] for key in sorted(unique)]


def balanced_shards(jobs, replicas):
    """Stable longest-first balancing by token count, not record count."""
    if type(replicas) is not int or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    shards, loads = [[] for _ in range(replicas)], [0] * replicas
    for job in sorted(normalize_jobs(jobs), key=lambda row: (-len(row["input_ids"]), row["key"])):
        rank = min(range(replicas), key=lambda index: (loads[index], index))
        shards[rank].append(job)
        loads[rank] += len(job["input_ids"])
    return shards


def replica_groups(config):
    settings = config["preparation"]["vllm"]
    dp = settings.get("data_parallel_size", 1)
    pp = settings.get("pipeline_parallel_size", 1)
    tp = settings["tensor_parallel_size"]
    if any(type(value) is not int or value < 1 for value in (dp, pp, tp)):
        raise ValueError("DP, PP and TP sizes must be positive integers")
    devices = settings["devices"]
    if (not isinstance(devices, list) or any(type(index) is not int or index < 0 for index in devices)
            or len(set(devices)) != len(devices) or len(devices) != dp * pp * tp):
        raise ValueError("vLLM devices must be unique and match DP times PP times TP")
    width = pp * tp
    return [devices[rank * width:(rank + 1) * width] for rank in range(dp)]


def _checked_entry(directory, key, entry, contract, expected_ids=None):
    if not isinstance(key, str) or _KEY.fullmatch(key) is None or not isinstance(entry, dict):
        raise ValueError("Invalid cache key or entry descriptor")
    filename = entry.get("file")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Missing cache filename: {key}")
    root = Path(directory).resolve()
    source = (root / filename).resolve()
    if root not in source.parents:
        raise ValueError(f"Cache file escapes its directory: {key}")
    if not source.is_file() or sha256_file(source) != entry.get("sha256"):
        raise ValueError(f"Missing or corrupted cache file: {key}")
    with np.load(source, allow_pickle=False) as arrays:
        if not {"embedding", "attention_mask", "input_ids"}.issubset(arrays.files):
            raise ValueError(f"Missing cache arrays: {key}")
        raw, mask, ids = arrays["embedding"], arrays["attention_mask"], arrays["input_ids"]
        from .vllm_contract import decode_embedding_arrays, validate_conditioning_contract
        storage = validate_conditioning_contract(contract)
        if entry.get("embedding_storage_dtype", "float32") != storage:
            raise ValueError(f"Cache descriptor storage dtype mismatch: {key}")
        expected_dtype = np.uint16 if storage == "bfloat16" else np.float32
        if (raw.dtype != expected_dtype or raw.ndim != 3 or raw.shape[0] != 1
                or raw.shape[1] == 0 or raw.shape[2] != contract["hidden_size"]
                or list(raw.shape) != entry.get("shape")
                or ids.dtype != np.int32 or ids.shape != raw.shape[:2]
                or mask.dtype != np.bool_ or mask.shape != ids.shape or not mask.all()
                or entry.get("token_count") != ids.shape[1]):
            raise ValueError(f"Invalid cache shape/dtype/token metadata: {key}")
        if hashlib.sha256(ids.tobytes()).hexdigest() != entry.get("input_ids_sha256"):
            raise ValueError(f"Cache token checksum mismatch: {key}")
        if expected_ids is not None and ids[0].tolist() != expected_ids:
            raise ValueError(f"Cache token IDs differ from planned job: {key}")
        embedding = decode_embedding_arrays(raw, contract)
        if not np.isfinite(embedding).all() or not np.any(embedding):
            raise ValueError(f"Nonfinite or all-zero cache features: {key}")
    return source


def validate_entry(directory, key, entry, contract, job=None):
    """Shared child/merge integrity check; job may be a job dict or ID list."""
    expected_ids = job["input_ids"] if isinstance(job, dict) else job
    return _checked_entry(directory, key, entry, contract, expected_ids)


def _read_shard(directory, parent, jobs, *, require_complete):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        if require_complete and jobs:
            raise ValueError(f"Missing replica cache manifest: {directory}")
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if _digest(_settings(manifest)) != _digest(_settings(parent)):
        raise ValueError(f"Replica cache metadata differs from parent: {directory}")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Replica entries must be an object: {directory}")
    expected = {job["key"]: job["input_ids"] for job in jobs}
    if set(entries) - set(expected):
        raise ValueError(f"Replica contains entries outside its planned jobs: {directory}")
    if require_complete and set(expected) - set(entries):
        raise ValueError(f"Replica did not finish all planned jobs: {directory}")
    for key, entry in entries.items():
        _checked_entry(directory, key, entry, parent["conditioning_contract"], expected[key])
    return manifest


def _atomic_copy(source, target, expected_sha):
    """Prefer same-filesystem hardlinks; never replace a conflicting file."""
    source, target = Path(source), Path(target)
    if target.exists():
        if sha256_file(target) != expected_sha:
            raise ValueError(f"Conflicting destination cache file: {target.name}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_sha:
            raise ValueError(f"Cache file changed during merge: {target.name}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_manifests(manifest, directory, shard_directories, jobs):
    """Validate all producers/files and commit one root manifest atomically."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jobs = normalize_jobs(jobs, manifest["max_sequence_length"])
    expected = {job["key"]: job["input_ids"] for job in jobs}
    merged = copy.deepcopy(manifest)
    entries = merged.setdefault("entries", {})
    origins = {}
    producer = _producer(manifest.get("backend"))
    producers = []
    for key, entry in entries.items():
        origins[key] = _checked_entry(directory, key, entry, manifest["conditioning_contract"], expected.get(key))
    for shard_directory in shard_directories:
        shard_directory = Path(shard_directory).resolve()
        # Each shard may contain any subset of the global missing plan. The
        # runner separately checks stricter per-replica assignment on resume.
        shard = _read_shard(shard_directory, manifest, jobs, require_complete=False)
        if shard is None:
            continue
        if shard["entries"] and not isinstance(shard.get("backend"), dict):
            raise ValueError("Completed replica entries require producer metadata")
        metadata = _producer(shard.get("backend"))
        if metadata is not None:
            if producer is not None and _digest(metadata) != _digest(producer):
                raise ValueError("Replica cache producer versions/source/compute settings differ")
            producer = metadata
            producers.append(shard["backend"])
        for key, entry in shard["entries"].items():
            metadata_without_file = {k: v for k, v in entry.items() if k != "file"}
            if key in entries:
                previous = {k: v for k, v in entries[key].items() if k != "file"}
                if _digest(previous) != _digest(metadata_without_file):
                    raise ValueError(f"Conflicting duplicate cache entry metadata: {key}")
            else:
                entries[key] = copy.deepcopy(entry)
                origins[key] = (shard_directory / entry["file"]).resolve()
    if set(expected) - set(entries):
        raise ValueError("Replica results do not cover all planned cache jobs")
    # Only after validating the complete collection publish any new references.
    # A failed later copy leaves harmless unreferenced files for resume; the
    # previous root manifest remains intact until the single final replace.
    for key, entry in entries.items():
        target = directory / f"{key}.npz"
        _atomic_copy(origins[key], target, entry["sha256"])
        entry["file"] = target.name
    if producer is not None:
        merged["backend"] = producer
    merged["parallel_producers"] = producers
    atomic_json(directory / "manifest.json", merged)
    return merged


def _write_jobs(path, jobs, settings):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"settings": settings}, separators=(",", ":"), allow_nan=False) + "\n")
            for job in jobs:
                stream.write(json.dumps(job, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _sigterm_guard():
    """Turn parent SIGTERM into an unwind; never replace a thread's handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        # A second TERM must not interrupt the cleanup of our owned children.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def _defer_interrupts():
    """Defer Python handlers through spawn/cleanup without child signal masks."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    pending = []

    def defer(number, frame):
        if previous[number] != signal.SIG_IGN and not pending:
            pending.append((number, frame))

    # Do not use pthread_sigmask around Popen: exec'ed vLLM workers would
    # inherit that blocked mask and fail to respond to our later SIGTERM.
    for number in previous:
        signal.signal(number, defer)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if pending:
            number, frame = pending[0]
            handler = previous[number]
            if callable(handler):
                handler(number, frame)
            elif handler == signal.SIG_DFL:
                if number == signal.SIGINT:
                    raise KeyboardInterrupt
                raise SystemExit(128 + number)


def _stop_process(process, *, owned_pgid=None, timeout=10):
    """Stop only this Popen child and its explicitly recorded new session.

    The leader can already have exited while its PP children still hold GPU
    memory. Never discover a group from an arbitrary PID or enumerate/kill GPU
    users: ``owned_pgid`` is recorded exclusively at our start_new_session spawn.
    """
    if os.name == "posix" and owned_pgid is not None:
        if type(owned_pgid) is not int or owned_pgid <= 1 or owned_pgid == os.getpgrp():
            raise ValueError("Refusing cleanup of an unowned/parent process group")
        try:
            os.killpg(owned_pgid, signal.SIGTERM)
        except ProcessLookupError:
            process.poll()  # Reap our direct child if it has already exited.
            return
        deadline = time.monotonic() + timeout
        while True:
            process.poll()
            try:
                os.killpg(owned_pgid, 0)
            except ProcessLookupError:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        try:
            os.killpg(owned_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)
        return
    # Non-POSIX, or a test double with no owned process group: do not infer one.
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)


@_sigterm_guard()
def run_parallel(config, jobs, manifest, directory, *, popen_factory=None):
    """Run DP replicas over disjoint PP×TP device groups and merge their caches.

    ``config`` is already validated and topology-ordered by the parent. This
    module does not discover GPUs, import accelerator libraries, read dataset
    text, or alter source prompt files. The root output lock belongs to caller.
    ``popen_factory`` is injectable solely for host-only orchestration tests.
    """
    started = time.perf_counter()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    groups = replica_groups(config)
    jobs = normalize_jobs(jobs, manifest["max_sequence_length"])
    existing = manifest.get("entries", {})
    planned = {job["key"]: job["input_ids"] for job in jobs}
    for key, entry in existing.items():
        _checked_entry(directory, key, entry, manifest["conditioning_contract"], planned.get(key))
    missing = [job for job in jobs if job["key"] not in existing]
    if not missing:
        result = {"event": "vllm_parallel_ready", "cache_dir": str(directory), "unique_texts": len(existing),
                  "encoded_now": 0, "replicas": len(groups), "launched_workers": 0,
                  "seconds": time.perf_counter() - started}
        atomic_json(directory / "parallel_report.json", result)
        return copy.deepcopy(manifest)
    plan = {"settings": _settings(manifest), "jobs": missing, "groups": groups,
            "vllm": config["preparation"]["vllm"]}
    workspace = directory / ".vllm-dp" / _digest(plan)
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_json(workspace / "plan.json", plan)
    shards = balanced_shards(missing, len(groups))
    workers, handles, shard_directories = [], [], []
    owned_groups = {}
    encoded_now = 0
    factory = popen_factory or subprocess.Popen
    try:
        for rank, (group, assigned) in enumerate(zip(groups, shards)):
            rank_directory = workspace / f"replica-{rank:03d}"
            rank_directory.mkdir(exist_ok=True)
            shard_directories.append(rank_directory)
            partial = _read_shard(rank_directory, manifest, assigned, require_complete=False)
            completed = len(partial["entries"]) if partial else 0
            if completed == len(assigned):
                continue
            jobs_path = workspace / f"jobs-{rank:03d}.jsonl"
            _write_jobs(jobs_path, assigned, _settings(manifest))
            child = copy.deepcopy(config)
            child["data"]["cache"] = str(rank_directory)
            child["data"]["prompts"] = str(jobs_path)
            child["data"]["context_length"] = manifest["max_sequence_length"]
            child["data"]["short_context_length"] = manifest["prefill_buckets"]["short"]
            child["preparation"]["limit"] = 0
            child["preparation"]["vllm"]["devices"] = group
            child["preparation"]["vllm"]["data_parallel_size"] = 1
            model_view = child["preparation"]["vllm"].get("model_path", "")
            if model_view:
                child["preparation"]["vllm"]["model_path"] = str(Path(model_view) / f"dp-{rank}")
            child["runtime"]["devices"] = group
            config_path = workspace / f"config-{rank:03d}.json"
            atomic_json(config_path, child)
            command = [child["preparation"]["vllm"]["python"], "-m", "moss_art.vllm_prepare",
                       "--config", str(config_path), "--worker-jobs", str(jobs_path),
                       "--worker-cache", str(rank_directory)]
            log = (workspace / f"worker-{rank:03d}.log").open("ab")
            handles.append(log)
            with _defer_interrupts():
                process = factory(command, cwd=str(Path(__file__).resolve().parents[1]), env=dict(os.environ),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                workers.append((rank, process))
                # A successful new POSIX session has PGID == its Popen PID.
                # Fake host-test processes need not provide a PID.
                owned_groups[id(process)] = getattr(process, "pid", None) if os.name == "posix" else None
            encoded_now += len(assigned) - completed
            print(json.dumps({"event": "vllm_replica_started", "replica": rank, "devices": group,
                              "jobs": len(assigned), "missing": len(assigned) - completed,
                              "tokens": sum(len(row["input_ids"]) for row in assigned)}), flush=True)
        while any(process.poll() is None for _, process in workers):
            failure = next(((rank, process.poll()) for rank, process in workers
                            if process.poll() not in (None, 0)), None)
            if failure is not None:
                raise RuntimeError(f"vLLM replica {failure[0]} exited with status {failure[1]}")
            time.sleep(0.1)
        for rank, process in workers:
            if process.wait() != 0:
                raise RuntimeError(f"vLLM replica {rank} failed; inspect its worker log")
        for shard_directory, assigned in zip(shard_directories, shards):
            _read_shard(shard_directory, manifest, assigned, require_complete=True)
        merged = merge_manifests(manifest, directory, shard_directories, missing)
    finally:
        with _defer_interrupts():
            cleanup_errors = []
            try:
                for _, process in workers:
                    try:
                        _stop_process(process, owned_pgid=owned_groups[id(process)])
                    except Exception as error:
                        cleanup_errors.append(error)
            finally:
                for handle in handles:
                    handle.close()
            if cleanup_errors:
                raise RuntimeError("Could not clean up an owned vLLM replica process group") from cleanup_errors[0]
    result = {"event": "vllm_parallel_ready", "cache_dir": str(directory), "unique_texts": len(merged["entries"]),
              "encoded_now": encoded_now, "replicas": len(groups), "launched_workers": len(workers),
              "seconds": time.perf_counter() - started}
    atomic_json(directory / "parallel_report.json", result)
    print(json.dumps(result), flush=True)
    return merged
