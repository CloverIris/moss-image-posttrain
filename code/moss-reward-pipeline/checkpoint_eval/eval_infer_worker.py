"""Generate the frozen checkpoint validation set using MOSS's public rollout CLI.

This worker owns no checkpoint watcher, GPU scheduler, reward model, or sampler.
The caller validates/exports an adapter with the original training config first.
Only --config is accepted. All native rollout products remain on disk on failure.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time

VERSION = 1
PROMPTS = 4
REPEATS = 4
RESOLUTION = 768
STEPS = 48


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_job(raw):
    """Validate the intentionally small job contract; all paths are absolute."""
    job = copy.deepcopy(raw)
    for key in ("repo", "python", "source_config", "fixed_prompts", "adapter", "output_dir"):
        value = job.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        job[key] = str(Path(value).resolve())
    for key in ("repo", "python", "source_config", "fixed_prompts", "adapter"):
        if not Path(job[key]).exists():
            raise FileNotFoundError(job[key])
    for key in ("job_id", "checkpoint_id"):
        if not isinstance(job.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", job[key]):
            raise ValueError(f"{key} must contain only letters, digits, dot, underscore or dash")
    if type(job.get("source_step")) is not int or job["source_step"] < 0:
        raise ValueError("source_step must be a nonnegative integer")
    job.setdefault("seed", 20261029)
    if type(job["seed"]) is not int or not 0 <= job["seed"] < 2**32:
        raise ValueError("seed must be uint32")
    for key, expected in (("device", 4), ("prompt_count", PROMPTS), ("images_per_prompt", REPEATS)):
        if key in job and (type(job[key]) is not int or job[key] != expected):
            raise ValueError(f"{key} is fixed at {expected}")
        job[key] = expected
    job.setdefault("timeout_seconds", 14400)
    if type(job["timeout_seconds"]) is not int or job["timeout_seconds"] < 1:
        raise ValueError("timeout_seconds must be positive")
    if not Path(job["repo"], "run.py").is_file():
        raise FileNotFoundError(Path(job["repo"], "run.py"))
    if Path(job["fixed_prompts"]).suffix.lower() not in (".jsonl", ".ndjson"):
        raise ValueError("fixed_prompts must be immutable JSONL with explicit original IDs")
    if Path(job["adapter"]).stat().st_size == 0:
        raise ValueError("adapter must not be empty")
    return job


def _merge_known(default, source):
    value = copy.deepcopy(default)
    for key in value.keys() & source.keys():
        value[key] = (_merge_known(value[key], source[key])
                      if isinstance(value[key], dict) and isinstance(source[key], dict)
                      else copy.deepcopy(source[key]))
    return value


def build_rollout_config(source, defaults, job):
    """Adapt only the public rollout schema, retaining the native sampler setup."""
    result = copy.deepcopy(defaults)
    for key in ("model", "lora", "data", "runtime", "sampling", "preparation"):
        result[key] = _merge_known(defaults[key], source.get(key, {}))
    if result["model"]["kind"] != "moss":
        raise ValueError("Checkpoint validation requires the real MOSS model")
    result["model"].update(base_precision="mxfp8", dtype="bfloat16",
                           attention_backend="cudnn", vae_precision="float16",
                           activation_storage="gpu")
    # configure_process resolves PHYSICAL indices using nvidia-smi UUIDs. [0]
    # here would select the training card even under CUDA_VISIBLE_DEVICES=4.
    result["runtime"].update(execution_backend="workers", devices=[4], platform="cuda",
                             allow_busy=False, output_dir=job["output_dir"])
    if "tensor_parallel_size" in result["runtime"]:
        result["runtime"]["tensor_parallel_size"] = 1
    result["data"]["prompts"] = job["fixed_prompts"]
    result["sampling"].update(height=RESOLUTION, width=RESOLUTION, steps=STEPS,
                              rollout_mode="bf16", vae_resident_rollout=False,
                              split_group_across_workers=False, pair_cached_points=False,
                              guidance_scale=3.0, rollout_batch_size=1)
    # Legacy single-card workers select actual quantization through
    # model.base_precision; rollout_mode is the shared ART validator's neutral
    # mode. worker_info.json and every sample must confirm actual MXFP8.
    result["sampling"]["spectrum"]["enabled"] = False
    result["sampling"]["rollout_mixing"]["enabled"] = False
    result["preparation"].update(backend="vllm", auto_prepare=False, async_prefetch=False)
    result["run"] = {"adapter": job["adapter"]}
    result["rollout"].update(count=REPEATS, seed=job["seed"], prompts_per_batch=1,
                             decode=True, save_latents=False, cache_intermediates=False,
                             host_cache_limit_gib=1.0)
    return result


def check_cache(config, repo):
    """CPU-only use of the SAME prompt escaping and cache reader as inference."""
    sys.path.insert(0, str(repo))
    from moss_art.data import PromptCache, text_key
    from moss_art.rollout_only import read_prompt_file
    records = read_prompt_file(config["data"]["prompts"], config["data"]["negative_prompt"])
    if len(records) != PROMPTS:
        raise ValueError(f"Expected exactly {PROMPTS} frozen prompts; got {len(records)}")
    # IDs are explicit in the frozen source: no filename/line-number reassignment.
    raw_records = [json.loads(line) for line in Path(config["data"]["prompts"]).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    if any("id" not in row and "sample_id" not in row for row in raw_records):
        raise ValueError("Every frozen prompt must retain its original id/sample_id")
    cache = PromptCache(config["data"]["cache"], model_dir=config["model"]["path"])
    if cache.manifest.get("complete") is False:
        raise ValueError("Cached Qwen features are incomplete; worker will not run Qwen")
    entries = cache.manifest["entries"]
    plan = []
    for index, record in enumerate(records):
        keys = [text_key(record.prompt), text_key(record.negative_prompt)]
        if any(key not in entries for key in keys):
            raise ValueError(f"Frozen prompt {record.id} is not fully cached; no Qwen fallback")
        lengths = [entries[key]["token_count"] for key in keys]
        if max(lengths) > config["data"]["context_length"]:
            raise ValueError(f"Frozen prompt {record.id} exceeds the existing context bucket")
        features = cache.load(record)  # checks SHA256, dtype/shape, finite values
        del features
        plan.append({"prompt_id": record.id, "prompt_index": index, "text_keys": keys,
                     "token_counts": lengths,
                     "seeds": [(config["rollout"]["seed"] + index * REPEATS + i) % 2**32
                               for i in range(REPEATS)]})
    report = {"prompt_count": len(records), "plan": plan,
              "manifest_sha256": digest(Path(config["data"]["cache"], "manifest.json")),
              "model_identity": cache.manifest["model_identity"],
              "conditioning_contract": cache.manifest.get("conditioning_contract")}
    del cache, records
    gc.collect()
    return report


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def validate_products(job, contract, cache_report):
    """Require all 16 native outputs before publishing a scorer request."""
    output = Path(job["output_dir"])
    complete = read_json(output / "complete.json")
    if complete.get("images") != PROMPTS * REPEATS or complete.get("prompts") != PROMPTS:
        raise ValueError("Native rollout did not finish the complete frozen validation set")
    info = read_json(output / "run_info.json")
    if (info.get("retained_prompts") != PROMPTS or info.get("dropped_ids")
            or info.get("expected_images") != PROMPTS * REPEATS):
        raise ValueError("Native rollout dropped or changed frozen validation prompts")
    workers = read_json(output / "worker_info.json")
    if len(workers) != 1 or any(w.get("base_precision") != "mxfp8"
                                or w.get("attention_backend") != "cudnn" for w in workers):
        raise ValueError("Native worker did not report MXFP8 and cuDNN attention")
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != PROMPTS * REPEATS:
        raise ValueError("Incomplete or duplicate sample manifest")
    samples.sort(key=lambda row: (row["prompt_index"], row["repeat"]))
    items, images, used = [], set(), set()
    for row in samples:
        pair = row["prompt_index"], row["repeat"]
        if pair in used or not (0 <= pair[0] < PROMPTS and 0 <= pair[1] < REPEATS):
            raise ValueError("Invalid or repeated fixed seed position")
        used.add(pair)
        fixed = cache_report["plan"][pair[0]]
        if row.get("prompt_id") != fixed["prompt_id"] or row.get("seed") != fixed["seeds"][pair[1]]:
            raise ValueError("Sample identity/seed differs from the frozen plan")
        keys = [hashlib.sha256(row[field].strip().encode()).hexdigest()
                for field in ("prompt", "negative_prompt")]
        if keys != fixed["text_keys"]:
            raise ValueError("Sample text differs from the frozen cached prompt")
        if row.get("base_precision") != "mxfp8" or row.get("actual_steps") != STEPS:
            raise ValueError("Sample precision or actual DiT steps differ from the frozen plan")
        sampling = row.get("sampling", {})
        if (sampling.get("height") != RESOLUTION or sampling.get("width") != RESOLUTION
                or sampling.get("steps") != STEPS or sampling.get("guidance_scale") != 3.0
                or sampling.get("spectrum", {}).get("enabled")
                or sampling.get("rollout_mixing", {}).get("enabled")):
            raise ValueError("Sample resolution/sampling differs from full-DiT validation")
        if row.get("adapter", {}).get("sha256") != contract["adapter_sha256"]:
            raise ValueError("Sample used a different LoRA adapter")
        image = (output / row["image"]).resolve()
        if output.resolve() not in image.parents or image in images:
            raise ValueError("Sample image escapes result directory or is duplicated")
        images.add(image)
        if png_dimensions(image) != (RESOLUTION, RESOLUTION):
            raise ValueError("Sample image has the wrong dimensions")
        items.append({"id": f"eval:{job['job_id']}:p{pair[0]:04d}:r{pair[1]:02d}",
                      "prompt_id": row["prompt_id"], "image": str(image),
                      "prompt": row["prompt"], "negative_prompt": row["negative_prompt"],
                      "prompt_index": pair[0], "repeat": pair[1], "rollout_id": pair[1], "seed": row["seed"],
                      "checkpoint_id": job["checkpoint_id"], "source_step": job["source_step"]})
    request = {"version": VERSION, "context": {"job_id": job["job_id"],
               "checkpoint_id": job["checkpoint_id"], "source_step": job["source_step"],
               "kind": "fixed_checkpoint_validation", "fixed_plan_sha256": contract["fixed_plan_sha256"],
               "adapter_sha256": contract["adapter_sha256"]}, "items": items}
    return request


def run(raw):
    job = resolve_job(raw)
    source = read_json(job["source_config"])
    prompt_hash = digest(job["fixed_prompts"])
    expected = job.get("expected_prompt_sha256")
    if expected is not None and expected != prompt_hash:
        raise ValueError("Frozen prompt file hash does not match the daemon's plan")
    sys.path.insert(0, job["repo"])
    from moss_art.rollout_only import DEFAULTS, load_config
    rollout = build_rollout_config(source, DEFAULTS, job)
    work = Path(job["output_dir"] + ".worker")
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "rollout.config.json"
    if config_path.exists() and read_json(config_path) != rollout:
        raise ValueError("Existing native config belongs to a different immutable job")
    atomic_json(config_path, rollout)
    load_config(config_path)  # same strict schema and ART compatibility validation
    cache_report = check_cache(rollout, job["repo"])
    contract = {"version": VERSION, "job_id": job["job_id"],
                "checkpoint_id": job["checkpoint_id"], "source_step": job["source_step"],
                "source_config_sha256": digest(job["source_config"]),
                "prompt_file_sha256": prompt_hash, "adapter_sha256": digest(job["adapter"]),
                "rollout_config_sha256": digest(config_path),
                "fixed_plan_sha256": canonical_digest({"prompts": prompt_hash,
                    "plan": cache_report["plan"], "resolution": RESOLUTION, "steps": STEPS,
                    "guidance_scale": 3.0, "sampler": rollout["sampling"]["sampler"],
                    "scheduler": rollout["sampling"]["scheduler"], "shift": rollout["sampling"]["shift"],
                    "base_precision": "mxfp8", "vae_precision": "float16"}),
                "physical_gpu": 4, "logical_gpu": 0,
                "cache_manifest_sha256": cache_report["manifest_sha256"],
                "model_identity": cache_report["model_identity"]}
    # Existing offline cache manifest includes each quantized payload's SHA256
    # and the explicit source-weight fingerprint method. Do not label the
    # source's header/mtime fingerprint as a full BF16 payload checksum.
    if rollout["model"].get("mxfp8_weight_mode") == "offline":
        manifest_path = Path(rollout["model"]["mxfp8_cache_dir"], "manifest.json")
        contract["mxfp8_manifest_sha256"] = digest(manifest_path)
        contract["mxfp8_manifest_path"] = str(manifest_path.resolve())
    previous = work / "contract.json"
    if previous.exists() and read_json(previous) != contract:
        raise ValueError("Output job already belongs to a different immutable contract")
    atomic_json(previous, contract)
    atomic_json(work / "cache_precheck.json", cache_report)
    if "mxfp8_manifest_path" in contract:
        atomic_json(work / "mxfp8_manifest.json", read_json(contract["mxfp8_manifest_path"]))
    output = Path(job["output_dir"])
    if not (output / "complete.json").is_file():
        if output.exists() and any(output.iterdir()):
            raise ValueError("Incomplete native output retained; use a new result directory to retry")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "4"
        # No XLA/global CUDA tuning flags are added or changed here.
        started = time.time()
        with (work / "inference.log").open("a", encoding="utf-8") as log:
            # Keep CLI + its spawned MOSS/VAE workers in the daemon's one job
            # process group. The daemon owns TERM/KILL + stage timeout, including
            # cleanup when a group leader exits. A new session here could orphan
            # GPU workers when the service is stopped.
            subprocess.run([job["python"], str(Path(job["repo"], "run.py")),
                            "rollout-only", "--config", str(config_path)], cwd=job["repo"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(work / "timing.json", {"started_unix": started, "finished_unix": time.time()})
    if digest(job["fixed_prompts"]) != prompt_hash or digest(job["adapter"]) != contract["adapter_sha256"]:
        raise ValueError("Immutable prompt or adapter changed during inference")
    request = validate_products(job, contract, cache_report)
    request_path = output / "eval_request.json"
    atomic_json(request_path, request)
    atomic_json(output / "inference_contract.json", contract)
    ready = {"version": VERSION, "job_id": job["job_id"], "checkpoint_id": job["checkpoint_id"],
             "source_step": job["source_step"], "request_path": str(request_path),
             "result_dir": str(output), "request_sha256": digest(request_path),
             "image_count": len(request["items"]), "fixed_plan_sha256": contract["fixed_plan_sha256"]}
    atomic_json(output / "inference_ready.json", ready)
    print(json.dumps({"event": "checkpoint_eval_inference_ready", **ready}), flush=True)
    return ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(read_json(args.config))


if __name__ == "__main__":
    main()
