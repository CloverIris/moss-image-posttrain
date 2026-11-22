"""Config-driven, resumable offline vLLM preparation for ART prompt caches."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .data import CACHE_VERSION, PromptCache, atomic_json, load_prompts, model_identity, text_key
from .vllm_contract import ENCODER, storage_contract, encode_embedding_arrays


def select_devices(devices):
    """Resolve physical nvidia-smi indices to UUIDs before importing Torch."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {int(a.strip()): b.strip() for a, b in (line.split(",", 1) for line in rows.splitlines())}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapping[index] for index in devices)
    except KeyError as exc:
        raise ValueError(f"Unknown physical GPU index {exc.args[0]}") from exc


def build_plan(config):
    from .prompt_buckets import PromptBuckets, log_drop
    from .vllm_qwen import tokenizer_for, text_tokens
    c = config
    directory = Path(c["data"]["cache"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    maximum = c["data"]["context_length"]
    records = load_prompts(c["data"]["prompts"], default_negative_prompt=c["data"]["negative_prompt"])
    if c["preparation"]["limit"]:
        records = records[:c["preparation"]["limit"]]
    tokenizer = tokenizer_for(c["model"]["path"])
    texts = list(dict.fromkeys(text.strip() for r in records for text in (r.prompt, r.negative_prompt)))
    ids_by_text = {text: text_tokens(tokenizer, text) for text in texts}
    kept, dropped, lengths = [], [], []
    for record in records:
        required = max(len(ids_by_text[text.strip()]) for text in (record.prompt, record.negative_prompt))
        lengths.append(required)
        if required > maximum:
            event = log_drop(record.id, required, maximum, "vllm_prepare_text")
            event["text_keys"] = [text_key(record.prompt), text_key(record.negative_prompt)]
            dropped.append(event)
        else:
            kept.append(record)
    buckets = PromptBuckets.resolve(lengths, c["data"]["short_context_length"], maximum)
    identity = model_identity(c["model"]["path"], full_weight_hash=c["preparation"]["full_weight_hash"])
    manifest_path = directory / "manifest.json"
    settings = dict(schema_version=CACHE_VERSION, encoder=ENCODER, model_identity=identity,
                    max_sequence_length=maximum, prefill_buckets={"short": buckets.short, "full": buckets.full},
                    conditioning_contract=storage_contract(c["preparation"]["vllm"]["storage_dtype"]))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, expected in settings.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Existing cache {key} differs; choose a new cache directory")
    else:
        manifest = {**settings, "embedding_storage_dtype": c["preparation"]["vllm"]["storage_dtype"],
                    "padding": "vLLM packed prefill; stored valid prompt tokens only", "entries": {}}
        atomic_json(manifest_path, manifest)
    # A prior dataset may have been complete. This invocation has not yet
    # validated every selected prompt, including when no encoding is needed.
    manifest["complete"] = False
    manifest["dropped_records"] = dropped
    atomic_json(manifest_path, manifest)
    (directory / "dropped.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dropped))
    texts = list(dict.fromkeys(text.strip() for r in kept for text in (r.prompt, r.negative_prompt)))
    cache = PromptCache(directory, model_dir=c["model"]["path"])
    missing = []
    for text in texts:
        if text_key(text) in manifest["entries"]:
            cache.load_text(text)
        else:
            missing.append(text)
    # Resume and fully prepared datasets must never allocate GPU model weights.
    missing.sort(key=lambda text: len(ids_by_text[text]))
    itemsize = 2 if c["preparation"]["vllm"]["storage_dtype"] == "bfloat16" else 4
    estimated_bytes = sum(len(ids_by_text[t]) * 4096 * itemsize for t in missing)
    free_bytes = shutil.disk_usage(directory).free
    print(json.dumps({"event": "vllm_prefill_plan", "records": len(kept), "dropped": len(dropped),
                      "unique_texts": len(texts), "missing": len(missing),
                      "feature_bytes_uncompressed": estimated_bytes, "free_disk_bytes": free_bytes}), flush=True)
    if estimated_bytes > free_bytes:
        raise ValueError("Not enough free space for uncompressed conditioning cache")
    jobs = [{"key": text_key(text), "input_ids": ids_by_text[text].tolist()} for text in missing]
    return directory, manifest, jobs, kept, dropped, len(texts)


def encode_jobs(config, jobs, directory, manifest, *, mark_complete=True):
    """One PP replica; consumes token jobs, never rereads the source dataset."""
    from .sft_data import write_npz
    from .vllm_qwen import VllmPrefill
    c = config
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    maximum = c["data"]["context_length"]
    storage_dtype = c["preparation"]["vllm"]["storage_dtype"]
    manifest_path = directory / "manifest.json"
    remaining = []
    for job in jobs:
        key = job["key"]
        if not isinstance(key, str) or not key or key in (".", "..") or Path(key).name != key or "\\" in key:
            raise ValueError("Invalid hidden-state job key")
        ids = np.asarray(job["input_ids"])
        if ids.ndim != 1 or not 0 < len(ids) <= maximum or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > 2**31-1):
            raise ValueError(f"Invalid token IDs for job {key}")
        if key in manifest["entries"]:
            # The shard can be resumed without constructing another model.
            from .vllm_parallel import validate_entry
            validate_entry(directory, key, manifest["entries"][key], manifest["conditioning_contract"], job)
        else:
            remaining.append(job)
    if remaining:
        manifest["complete"] = False
        atomic_json(manifest_path, manifest)
        encoder = VllmPrefill(c["model"]["path"], c["preparation"]["vllm"],
                              directory / ".vllm-work", max_sequence_length=maximum)
        if "backend" in manifest and manifest["backend"] != encoder.metadata:
            raise ValueError("Partial-cache producer version/source changed; use a new cache directory")
        manifest["backend"] = encoder.metadata
        atomic_json(manifest_path, manifest)
        batch_size = c["preparation"]["vllm"]["batch_size"]
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            arrays = encoder.encode_token_lists([job["input_ids"] for job in batch])
            if len(arrays) != len(batch):
                raise RuntimeError("Incorrect prefill batch result count")
            for job, embedding in zip(batch, arrays):
                ids = np.asarray(job["input_ids"], dtype=np.int32)[None]
                key = job["key"]
                embedding = np.asarray(embedding, dtype=np.float32)
                if embedding.shape != (1, len(job["input_ids"]), 4096) or not np.isfinite(embedding).all() or not np.any(embedding):
                    raise ValueError(f"Invalid extracted hidden-state features: {key}")
                stored = encode_embedding_arrays(embedding, storage_dtype)
                descriptor = write_npz(directory, f"{key}.npz", embedding=stored,
                                       attention_mask=np.ones(ids.shape, dtype=bool), input_ids=ids)
                manifest["entries"][key] = {**descriptor, "shape": list(embedding.shape),
                                            "token_count": int(ids.shape[1]),
                                            "embedding_storage_dtype": storage_dtype,
                                            "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
            atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "vllm_cache_progress", "encoded": offset + len(batch),
                              "total": len(remaining)}), flush=True)
    manifest["complete"] = bool(mark_complete)
    atomic_json(manifest_path, manifest)
    return len(remaining)


def prepare_cache(config):
    """Standalone producer. Training does not call this function."""
    started = time.perf_counter()
    c = config
    directory, manifest, jobs, kept, dropped, unique_count = build_plan(c)
    manifest["complete"] = False
    atomic_json(directory / "manifest.json", manifest)
    if jobs:
        if c["preparation"]["vllm"]["data_parallel_size"] > 1:
            from .vllm_parallel import run_parallel
            result_manifest = run_parallel(c, jobs, manifest, directory)
            if result_manifest is not None:
                manifest = result_manifest
        else:
            select_devices(c["preparation"]["vllm"]["devices"])
            encode_jobs(c, jobs, directory, manifest, mark_complete=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    # Root completion is a commit marker, only published after the exact
    # selected dataset has passed the same checked reader used by training.
    PromptCache(directory, model_dir=c["model"]["path"]).validate_prompts(kept)
    manifest["complete"] = True
    atomic_json(directory / "manifest.json", manifest)
    result = {"event": "text_cache_ready", "encoder": ENCODER, "cache_dir": str(directory),
              "records": len(kept), "dropped": len(dropped), "unique_texts": unique_count,
              "encoded_now": len(jobs), "seconds": time.perf_counter() - started}
    atomic_json(directory / "preparation_report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="store_true", help="Show parallel/device plan without reading data or loading a model")
    parser.add_argument("--worker-jobs", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from .config import load_config
    c = load_config(args.config)
    if c["preparation"]["backend"] != "vllm":
        raise ValueError("vllm_prepare requires preparation.backend=vllm")
    v = c["preparation"]["vllm"]
    if args.worker_jobs or args.worker_cache:
        if not args.worker_jobs or not args.worker_cache:
            parser.error("worker jobs and cache must be specified together")
        select_devices(v["devices"])
        from .locking import output_lock
        with output_lock(args.worker_cache):
            with open(args.worker_jobs, encoding="utf-8") as source:
                header = json.loads(next(source))
                settings = header["settings"]
                jobs = [json.loads(line) for line in source if line.strip()]
            target = Path(args.worker_cache) / "manifest.json"
            manifest = json.loads(target.read_text()) if target.exists() else {**settings, "entries": {}}
            for key, expected in settings.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"Worker cache {key} mismatch")
            encode_jobs(c, jobs, args.worker_cache, manifest)
        return
    if v["topology_aware"]:
        from .gpu_topology import discover_gpu_topology
        report = discover_gpu_topology(v["devices"])
        v["devices"] = report["flattened_indices"]
    else:
        width = v["pipeline_parallel_size"] * v["tensor_parallel_size"]
        report = {"groups": [v["devices"][i:i+width] for i in range(0, len(v["devices"]), width)]}
    report.update(data_parallel_size=v["data_parallel_size"], pipeline_parallel_size=v["pipeline_parallel_size"],
                  tensor_parallel_size=v["tensor_parallel_size"], storage_dtype=v["storage_dtype"])
    print(json.dumps({"event": "vllm_parallel_plan", **report}), flush=True)
    if args.plan:
        return report
    atomic_json(Path(c["data"]["cache"]) / "topology.json", report)
    from .locking import output_lock
    with output_lock(c["data"]["cache"]):
        prepare_cache(c)


if __name__ == "__main__":
    main()
