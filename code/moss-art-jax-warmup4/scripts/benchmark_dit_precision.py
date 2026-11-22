#!/usr/bin/env python3
"""Compare resident BF16/MXFP8 denoising through the native rollout path.

Only --config is accepted. Model loading, cached-condition priming, two or more
full warmups, and compilation are outside measured durations. No VAE, Qwen,
reward model, ART cache, or optimizer is instantiated. A measured duration is
one synchronized NativeWorker.rollout(..., keep_cache=False): it includes the
native Python sampler/dispatch, cached-condition padding/upload and the final
latent transfer/finiteness check, and excludes monitor/report serialization.
Thus this is production denoising wall time, not isolated CUDA kernel time.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_settings(path):
    settings = json.loads(path.read_text())
    expected = {"engine_config", "device", "precision", "prompt_id", "warmup_seeds",
                "measured_seeds", "monitor_interval_seconds", "output_dir"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError(f"Benchmark config requires exactly {sorted(expected)}")
    for key in ("engine_config", "prompt_id", "output_dir"):
        if not isinstance(settings[key], str) or not settings[key] or "\x00" in settings[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if type(settings["device"]) is not int or settings["device"] < 0:
        raise ValueError("device must be a nonnegative physical nvidia-smi index")
    if settings["precision"] not in ("bf16", "mxfp8"):
        raise ValueError("precision must be bf16 or mxfp8")
    for key, minimum in (("warmup_seeds", 2), ("measured_seeds", 1)):
        values = settings[key]
        if (not isinstance(values, list) or len(values) < minimum or
                any(type(value) is not int or not 0 <= value < 2**32 for value in values)):
            raise ValueError(f"{key} must contain at least {minimum} uint32 seeds")
    interval = settings["monitor_interval_seconds"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or not .05 <= interval <= 2:
        raise ValueError("monitor_interval_seconds must be between .05 and 2")
    return settings


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class MemoryInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class GPUMonitor:
    """Read NVML without importing an accelerator framework or spawning a process.

    utilization.gpu is sampled GPU-busy time, not SM occupancy, Tensor Core
    utilization or achieved FLOPs. NVML may return duplicate readings because
    hardware sample windows can be longer than our polling period.
    """
    FIELDS = ("monotonic_seconds", "phase", "gpu_busy_percent",
              "memory_busy_percent", "memory_used_mib", "power_watts",
              "sm_clock_mhz", "memory_clock_mhz", "temperature_c", "pstate")

    def __init__(self, gpu_uuid, interval, enabled=True):
        self.enabled, self.interval = enabled, float(interval)
        self.rows, self.errors = [], []
        self.phase = "startup"
        self.stop = threading.Event()
        self.thread = None
        self.lib = None
        if not enabled:
            return
        self.lib = ctypes.CDLL(ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1")
        self._require("nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        fn = self.lib.nvmlDeviceGetHandleByUUID
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self._require("nvmlDeviceGetHandleByUUID", gpu_uuid.encode(), ctypes.byref(self.handle))
        # Fail before warmup if this GPU does not expose the principal metric.
        self.sample()

    def _require(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"NVML {name} failed with code {code}")

    def _optional_uint(self, name, *args, divisor=1):
        result = ctypes.c_uint()
        code = getattr(self.lib, name)(self.handle, *args, ctypes.byref(result))
        return result.value / divisor if not code else None

    def sample(self):
        utilization, memory = Utilization(), MemoryInfo()
        self._require("nvmlDeviceGetUtilizationRates", self.handle, ctypes.byref(utilization))
        self._require("nvmlDeviceGetMemoryInfo", self.handle, ctypes.byref(memory))
        return {
            "monotonic_seconds": time.monotonic(), "phase": self.phase,
            "gpu_busy_percent": int(utilization.gpu),
            "memory_busy_percent": int(utilization.memory),
            "memory_used_mib": memory.used / 2**20,
            "power_watts": self._optional_uint("nvmlDeviceGetPowerUsage", divisor=1000),
            "sm_clock_mhz": self._optional_uint("nvmlDeviceGetClockInfo", ctypes.c_uint(1)),
            "memory_clock_mhz": self._optional_uint("nvmlDeviceGetClockInfo", ctypes.c_uint(2)),
            "temperature_c": self._optional_uint("nvmlDeviceGetTemperature", ctypes.c_uint(0)),
            "pstate": self._optional_uint("nvmlDeviceGetPowerState"),
        }

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="precision-nvml", daemon=True)
            self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                self.rows.append(self.sample())
            except Exception as error:
                if len(self.errors) < 20:
                    self.errors.append(f"{type(error).__name__}: {error}")
            self.stop.wait(self.interval)

    def summarize(self, windows, trim_seconds=0):
        selected = [row for row in self.rows if any(
            start + trim_seconds <= row["monotonic_seconds"] <= end - trim_seconds
            for start, end in windows)]
        result = {"samples": len(selected), "boundary_trim_seconds": trim_seconds}
        for key in self.FIELDS[2:]:
            values = [row[key] for row in selected if row[key] is not None]
            result[key] = ({"mean": statistics.fmean(values), "median": statistics.median(values),
                            "min": min(values), "max": max(values)} if values else None)
        return result

    def finish(self, path):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("NVML monitor did not stop")
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        if self.lib is not None:
            self._require("nvmlShutdown")
        return {"backend": "NVML ctypes" if self.enabled else "disabled for CPU",
                "requested_poll_interval_seconds": self.interval,
                "samples": len(self.rows), "errors": self.errors,
                "metric_definition": "NVML utilization.gpu: percentage of its hardware sample window with one or more GPU kernels executing; not MFU, occupancy or Tensor Core utilization",
                "sampling_note": "Hardware sample windows may exceed the poll interval, producing duplicate values; phase boundaries may overlap the hardware window. Interior summaries trim one second at both ends of each measured image."}


def latent_summary(value):
    import numpy as np
    value = np.asarray(value)
    finite = bool(np.isfinite(value).all())
    if not finite:
        raise FloatingPointError("Nonfinite denoising endpoint")
    as64 = value.astype(np.float64)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
            "mean": float(as64.mean()), "std": float(as64.std()),
            "rms": float(np.sqrt(np.mean(as64 * as64))),
            "min": float(as64.min()), "max": float(as64.max()),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


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
    # A repeat is one image. Four measured seeds represent K=4 serial rollout.
    # Never prepare Qwen or activate ART capture while benchmarking DiT.
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
                  "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_COMPILATION_CACHE_DIR")},
                  "timing_scope": "Synchronized single-image NativeWorker.rollout(keep_cache=False) wall time. Includes native sampler/host dispatch, cached-condition padding/upload, final latent D2H and finiteness check. Excludes loading, condition cache priming, full warmups/compilation, VAE, Qwen, reward, ART cache capture, optimizer, and report serialization.",
                  "vae_loaded": False, "qwen_loaded": False, "art_cache_retained": False,
                  "warmups": [], "measurements": []}
        write_json(report_path, report)
        monitor = None
        worker = None
        phase = "load_records"
        overall_start = time.monotonic()
        try:
            records = read_records(c)
            matching = [record for record in records if record["sample_id"] == settings["prompt_id"]]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one cached prompt {settings['prompt_id']!r}")
            record = matching[0]
            report["record"] = record
            phase = "load_model"
            started = time.monotonic()
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
                raise RuntimeError("Actual precision differs from the requested precision")
            if settings["precision"] == "mxfp8" and c["model"]["kind"] == "moss" and worker.model.mxfp8_modules == 0:
                raise RuntimeError("MXFP8 requested but no model projections were quantized")
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
            for kind, seeds in (("warmup", settings["warmup_seeds"]), ("measured", settings["measured_seeds"])):
                for index, seed in enumerate(seeds):
                    phase = f"{kind}_{index:03d}_seed{seed}"
                    monitor.phase = phase
                    # Loading and previous work must be complete before starting.
                    jax.block_until_ready((worker.model.base, worker.lora))
                    started = time.monotonic()
                    value = worker.rollout(records=[record], seeds=[seed], keep_cache=False)
                    # NativeWorker materializes endpoints as NumPy before return;
                    # this explicit synchronization documents the timing boundary.
                    jax.block_until_ready(value["endpoints"])
                    ended = time.monotonic()
                    monitor.phase = "between_images"
                    seconds = ended - started
                    if value["actual_steps"] != 16 or len(value["endpoints"]) != 1:
                        raise RuntimeError("Measured rollout did not produce one image at 16 steps")
                    entry = {"index": index, "seed": seed, "seconds": seconds,
                             "seconds_per_step": seconds / 16, "images_per_second": 1 / seconds,
                             "started_monotonic_seconds": started, "ended_monotonic_seconds": ended,
                             "actual_denoising_steps": value["actual_steps"],
                             "sampler": value["sampler"], "scheduler": value["scheduler"],
                             "latent": latent_summary(value["endpoints"])}
                    if kind == "measured":
                        np.save(output / f"latent_seed{seed}_repeat{index:03d}.npy", value["endpoints"], allow_pickle=False)
                        report["measurements"].append(entry)
                    else:
                        report["warmups"].append(entry)
                    if worker.cache is not None:
                        raise RuntimeError("Unexpected ART cache allocation")
                    write_json(report_path, report)
                    print(json.dumps({"event": kind + "_complete", **entry}), flush=True)
                    del value
            phase = "summarize"
            seconds = [entry["seconds"] for entry in report["measurements"]]
            windows = [(entry["started_monotonic_seconds"], entry["ended_monotonic_seconds"])
                       for entry in report["measurements"]]
            for entry in report["measurements"]:
                window = [(entry["started_monotonic_seconds"], entry["ended_monotonic_seconds"])]
                entry["gpu_metrics_full_window"] = monitor.summarize(window)
                entry["gpu_metrics_interior"] = monitor.summarize(window, trim_seconds=1)
            report["summary"] = {
                "images": len(seconds), "steps_per_image": 16,
                "total_measured_seconds": sum(seconds), "seconds_per_image_median": statistics.median(seconds),
                "seconds_per_image_mean": statistics.fmean(seconds),
                "seconds_per_image_min": min(seconds), "seconds_per_image_max": max(seconds),
                "seconds_per_step_median": statistics.median(seconds) / 16,
                "images_per_second_aggregate": len(seconds) / sum(seconds),
                "gpu_metrics_full_windows": monitor.summarize(windows),
                "gpu_metrics_interior": monitor.summarize(windows, trim_seconds=1)}
            stats = worker.device.memory_stats() or {}
            report["jax_memory_stats_after_measurement"] = {key: int(value) for key, value in stats.items() if isinstance(value, (int, np.integer))}
            if monitor.enabled and (not report["summary"]["gpu_metrics_full_windows"]["samples"] or monitor.errors):
                raise RuntimeError("NVML did not capture complete utilization evidence")
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failure_phase=phase, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
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