#!/usr/bin/env python3
"""Measure real, resident DiT image microbatches with only --config.

Each model.predict receives [batch_size,C,H,W] latents, repeated cached prompt
embeddings, and batch-length vectors. This does not call NativeWorker.rollout:
its group_size is a serial group, not an image microbatch. Independent
(1,C,H,W) seed draws preserve the serial path's initial noise per image.
The cold run and at least two complete batch warmups are excluded. No VAE,
Qwen, reward, ART capture, or optimizer is instantiated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_dit_precision import GPUMonitor, latent_summary, write_json


def read_settings(path):
    settings = json.loads(path.read_text())
    expected = {"engine_config", "device", "batch_size", "precision", "prompt_id",
                "warmup_seed_batches", "seed_batches", "monitor_interval_seconds", "output_dir"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError(f"Benchmark config requires exactly {sorted(expected)}")
    for key in ("engine_config", "prompt_id", "output_dir"):
        if not isinstance(settings[key], str) or not settings[key] or "\x00" in settings[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if type(settings["device"]) is not int or settings["device"] < 0:
        raise ValueError("device must be a nonnegative physical nvidia-smi index")
    if type(settings["batch_size"]) is not int or settings["batch_size"] not in (1, 2):
        raise ValueError("batch_size must be 1 or 2 for this comparison")
    if settings["precision"] not in ("bf16", "mxfp8"):
        raise ValueError("precision must be bf16 or mxfp8")
    for key, minimum in (("warmup_seed_batches", 2), ("seed_batches", 1)):
        batches = settings[key]
        if not isinstance(batches, list) or len(batches) < minimum:
            raise ValueError(f"{key} must contain at least {minimum} seed batches")
        for batch in batches:
            if (not isinstance(batch, list) or len(batch) != settings["batch_size"] or
                    any(type(seed) is not int or not 0 <= seed < 2**32 for seed in batch)):
                raise ValueError(f"Every {key} row must contain batch_size uint32 seeds")
    interval = settings["monitor_interval_seconds"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or not .05 <= interval <= 2:
        raise ValueError("monitor_interval_seconds must be between .05 and 2")
    return settings


def batch_rollout(worker, record, seeds):
    """Actual simultaneous image batch, native model and sampler arithmetic."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from moss_art.conditioning import pad_context
    from moss_art.prompt_buckets import PromptBuckets
    from moss_art.sampling import schedule, step_coefficients, sample_step

    size, s = len(seeds), worker.c["sampling"]
    positive, negative = worker._conditioning(record)
    length = max(positive.shape[1], negative.shape[1])
    buckets = PromptBuckets.resolve([length], worker.c["data"]["short_context_length"],
                                    worker.c["data"]["context_length"], emit=False)
    capacity = buckets.capacity(length)
    if worker.c["model"]["kind"] == "moss" and capacity != 512:
        raise ValueError("This comparison requires the selected prompt to fit the 512 bucket")
    positive, pl = pad_context(positive, capacity)
    negative, nl = pad_context(negative, capacity)
    positive = jax.device_put(np.repeat(positive, size, axis=0), worker.device)
    negative = jax.device_put(np.repeat(negative, size, axis=0), worker.device)
    # Host int32 vectors match NativeWorker; MossModel validates then uploads.
    lengths = (np.repeat(pl, size, axis=0), np.repeat(nl, size, axis=0))
    _, sigmas = schedule(s["steps"], scheduler=s["scheduler"], shift=s["shift"],
                         training_shift=s["training_shift"], beta_alpha=s["beta_alpha"],
                         beta_beta=s["beta_beta"])
    coefficients = step_coefficients(sigmas, sampler=s["sampler"])
    single_shape = (1, worker.model.config["latent_channels"], s["height"] // 16, s["width"] // 16)
    # Drawing one [B,...] tensor from one seed changes image noise relative to
    # NativeWorker. Separate draws give an identical seed-to-row mapping.
    z = jnp.concatenate([jax.random.normal(jax.random.PRNGKey(seed), single_shape,
                                          dtype=jnp.float32) for seed in seeds], axis=0)
    assert z.shape[0] == size and positive.shape[0] == size and lengths[0].shape == (size,)
    previous_denoised = None
    for index, coefficient in enumerate(coefficients):
        velocity = worker.model.predict(worker.lora, z, sigmas[index], positive,
                                        **worker._kwargs(negative, lengths)).astype(jnp.float32)
        z, previous_denoised = sample_step(z, velocity, coefficient, previous_denoised)
    # D2H and finiteness are inside the duration, exactly as in NativeWorker.
    endpoints = np.asarray(z).copy()
    if not np.isfinite(endpoints).all():
        raise FloatingPointError("Nonfinite batched rollout endpoint")
    return {"endpoints": endpoints, "actual_steps": len(coefficients), "capacity": capacity,
            "context_lengths": [lengths[0].tolist(), lengths[1].tolist()],
            "sampler": s["sampler"], "scheduler": s["scheduler"]}


def memory_stats(worker):
    import numpy as np
    return {key: int(value) for key, value in (worker.device.memory_stats() or {}).items()
            if isinstance(value, (int, np.integer))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    settings = read_settings(config_path)
    os.chdir(ROOT)
    from moss_art.config import load_config
    from moss_art.runtime import configure_process, preflight
    from moss_art.locking import output_lock
    engine_path = Path(settings["engine_config"]).resolve()
    c = load_config(engine_path)
    if c["sampling"]["steps"] != 16:
        raise ValueError("This comparison requires engine sampling.steps=16")
    if c["model"]["kind"] == "moss":
        required = {"height": 1024, "width": 1024, "sampler": "euler", "scheduler": "moss",
                    "guidance_scale": 3.0}
        for key, expected in required.items():
            if c["sampling"][key] != expected:
                raise ValueError(f"This MOSS comparison requires sampling.{key}={expected!r}")
        if c["lora"]["rank"] != 32 or c["lora"]["alpha"] != 64:
            raise ValueError("This MOSS comparison requires LoRA rank32/alpha64")
        if c["data"]["short_context_length"] != 512:
            raise ValueError("This MOSS comparison requires a 512 short prompt bucket")
    c["runtime"]["devices"] = [settings["device"]]
    c["runtime"]["tensor_parallel_size"] = 1
    c["preparation"]["async_prefetch"] = False
    c["preparation"]["auto_prepare"] = False
    c["sampling"]["group_size"] = 1
    c["sampling"]["prompts_per_rollout"] = 1
    c["sampling"]["train_timesteps"] = [0]
    output = Path(settings["output_dir"]).resolve()
    c["runtime"]["output_dir"] = str(output)
    inventory = preflight(c)
    configure_process(c, settings["device"])
    import numpy as np
    import jax
    import jaxlib
    from moss_art.worker import NativeWorker
    from moss_art.trainer import read_records
    with output_lock(output):
        report_path = output / "benchmark.json"
        if report_path.exists():
            raise FileExistsError(f"Choose an unused output_dir: {output}")
        report = {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "settings": settings, "benchmark_config_path": str(config_path),
                  "engine_config_path": str(engine_path), "effective_engine_config": c,
                  "inventory": inventory, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "versions": {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__},
                  "environment": {key: os.environ.get(key) for key in
                    ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_COMPILATION_CACHE_DIR")},
                  "timing_scope": "Synchronized real-B image batch, model.predict plus native Python sampler and host dispatch. Includes cached-condition padding/repeat/upload, per-image independent RNG, final latent D2H and finiteness check. Excludes model loading, condition cache priming, the cold compilation run, at least two complete batch warmups, VAE, Qwen, reward, ART capture, optimizer, and report/latent serialization.",
                  "batch_definition": "Simultaneous [B,C,H,W] latents in every model.predict; same cached prompt repeated B times; CFG positive and negative branches remain sequential as in production.",
                  "memory_note": "JAX peak_bytes_in_use is the process lifetime active allocation peak, including loading and warmup/compilation; NVML memory_used includes allocator reservations. Per-batch JAX current and lifetime-peak snapshots are recorded.",
                  "vae_loaded": False, "qwen_loaded": False, "art_cache_retained": False,
                  "cold_compile": None, "warmups": [], "measurements": []}
        write_json(report_path, report)
        monitor = worker = None
        phase, overall_start = "load_records", time.monotonic()
        try:
            records = read_records(c)
            matching = [record for record in records if record["sample_id"] == settings["prompt_id"]]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one cached prompt {settings['prompt_id']!r}")
            record = matching[0]
            report["record"] = record
            phase, started = "load_model", time.monotonic()
            worker = NativeWorker(c, 0, base_precision=settings["precision"])
            report["load_seconds_excluded"] = time.monotonic() - started
            report["model"] = {"actual_base_precision": worker.model.base_precision,
                               "mxfp8_modules": worker.model.mxfp8_modules,
                               "base_storage_bytes": worker.model.base_storage_bytes,
                               "base_storage_gib": worker.model.base_storage_bytes / 2**30,
                               "attention_backend": worker.model.attention_backend,
                               "dit_autotune_level": worker.model.dit_autotune_level,
                               "config": dict(worker.model.config), "lora_rank": c["lora"]["rank"],
                               "lora_alpha": c["lora"]["alpha"], "lora_preset": c["lora"]["preset"]}
            if worker.model.base_precision != settings["precision"]:
                raise RuntimeError("Actual precision differs from requested precision")
            if settings["precision"] == "mxfp8" and c["model"]["kind"] == "moss" and not worker.model.mxfp8_modules:
                raise RuntimeError("MXFP8 requested but no projections were quantized")
            phase = "prime_cached_condition"
            condition = worker._conditioning(record)
            jax.block_until_ready(condition)
            report["conditioning"] = {"positive_unpadded_shape": list(condition[0].shape),
                                      "negative_unpadded_shape": list(condition[1].shape)}
            del condition
            uuid = inventory[settings["device"]]["uuid"] if c["runtime"]["platform"] == "cuda" else None
            monitor = GPUMonitor(uuid, settings["monitor_interval_seconds"], enabled=uuid is not None)
            monitor.start()
            print(json.dumps({"event": "loaded", "model": report["model"]}), flush=True)
            write_json(report_path, report)
            kinds = (("cold_compile", [settings["warmup_seed_batches"][0]]),
                     ("warmup", settings["warmup_seed_batches"]), ("measured", settings["seed_batches"]))
            for kind, seed_batches in kinds:
                for index, seeds in enumerate(seed_batches):
                    phase = f"{kind}_{index:03d}"
                    monitor.phase = phase
                    jax.block_until_ready((worker.model.base, worker.lora))
                    started = time.monotonic()
                    value = batch_rollout(worker, record, seeds)
                    jax.block_until_ready(value["endpoints"])
                    ended = time.monotonic()
                    monitor.phase = "between_batches"
                    seconds, size = ended - started, settings["batch_size"]
                    if value["actual_steps"] != 16 or value["endpoints"].shape[0] != size:
                        raise RuntimeError("Incorrect actual image batch or step count")
                    entry = {"index": index, "seeds": seeds, "batch_size": size,
                             "seconds": seconds, "seconds_per_batch": seconds,
                             "seconds_per_image": seconds / size,
                             "seconds_per_step_per_batch": seconds / 16,
                             "images_per_second": size / seconds,
                             "started_monotonic_seconds": started, "ended_monotonic_seconds": ended,
                             "actual_denoising_steps": value["actual_steps"],
                             "sampler": value["sampler"], "scheduler": value["scheduler"],
                             "prompt_capacity": value["capacity"], "context_lengths": value["context_lengths"],
                             "latent": latent_summary(value["endpoints"]),
                             "jax_memory_stats_after_batch": memory_stats(worker)}
                    if kind == "measured":
                        filename = f"latent_batch{index:03d}.npy"
                        np.save(output / filename, value["endpoints"], allow_pickle=False)
                        entry["latent_file"] = filename
                        report["measurements"].append(entry)
                    elif kind == "warmup":
                        report["warmups"].append(entry)
                    else:
                        report["cold_compile"] = entry
                    if worker.cache is not None:
                        raise RuntimeError("Unexpected ART cache allocation")
                    write_json(report_path, report)
                    print(json.dumps({"event": kind + "_complete", **entry}), flush=True)
                    del value
            phase = "summarize"
            seconds = [entry["seconds"] for entry in report["measurements"]]
            size, total = settings["batch_size"], sum(seconds)
            windows = [(entry["started_monotonic_seconds"], entry["ended_monotonic_seconds"])
                       for entry in report["measurements"]]
            for entry, window in zip(report["measurements"], windows):
                entry["gpu_metrics_full_window"] = monitor.summarize([window])
                entry["gpu_metrics_interior"] = monitor.summarize([window], trim_seconds=1)
            report["summary"] = {
                "batches": len(seconds), "batch_size": size, "images": size * len(seconds),
                "steps_per_image": 16, "total_measured_seconds": total,
                "seconds_per_batch_mean": statistics.fmean(seconds),
                "seconds_per_batch_median": statistics.median(seconds),
                "seconds_per_batch_min": min(seconds), "seconds_per_batch_max": max(seconds),
                "seconds_per_image_mean": statistics.fmean(seconds) / size,
                "seconds_per_image_median": statistics.median(seconds) / size,
                "seconds_per_step_per_batch_median": statistics.median(seconds) / 16,
                "images_per_second_aggregate": size * len(seconds) / total,
                "gpu_metrics_full_windows": monitor.summarize(windows),
                "gpu_metrics_interior": monitor.summarize(windows, trim_seconds=1)}
            report["jax_memory_stats_after_measurement"] = memory_stats(worker)
            if monitor.enabled and (not report["summary"]["gpu_metrics_full_windows"]["samples"] or monitor.errors):
                raise RuntimeError("NVML did not capture complete utilization evidence")
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failure_phase=phase, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            if worker is not None:
                try:
                    report["jax_memory_stats_at_failure"] = memory_stats(worker)
                except Exception as memory_error:
                    report["memory_stats_error"] = str(memory_error)
            raise
        finally:
            if monitor is not None:
                report["monitor"] = monitor.finish(output / "gpu_samples.csv")
            report["total_process_seconds"] = time.monotonic() - overall_start
            write_json(report_path, report)
            print(json.dumps({"event": "benchmark_finished", "status": report["status"],
                              "report": str(report_path), "summary": report.get("summary")}), flush=True)


if __name__ == "__main__":
    main()
