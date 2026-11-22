#!/usr/bin/env python3
"""Measure an actual native worker rollout, retained ART cache, decode and reward.

One device per invocation. No optimizer step and no training checkpoint changes.
Use --no-decode for a CPU tiny fixture or for isolating transformer capacity.
--gradient-probe checks one real ART replay item after the final reward pass;
it never applies an optimizer update or represents a complete training rollout.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moss_art.config import load_config
from moss_art.runtime import preflight, configure_process
from moss_art.locking import output_lock


class GPUMonitor:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.phase, self.rows, self.errors = "startup", [], []
        self.stop = threading.Event()
        self.thread = None
        self.started = time.monotonic()

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-i", str(self.device),
                     "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, check=True, timeout=5)
                used, util = [int(x.strip()) for x in result.stdout.strip().split(",")]
                self.rows.append([time.monotonic()-self.started, self.phase, used, util])
            except Exception as error:
                self.errors.append(type(error).__name__)
            self.stop.wait(.5)

    def finish(self, path):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
        with Path(path).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["seconds", "phase", "nvidia_smi_used_mib", "gpu_utilization_percent"])
            writer.writerows(self.rows)
        return {"poll_interval_seconds": .5, "samples": len(self.rows),
                "max_used_mib": max((r[2] for r in self.rows), default=0),
                "max_used_mib_by_phase": {phase: max(r[2] for r in self.rows if r[1] == phase)
                                        for phase in sorted({r[1] for r in self.rows})},
                "errors": self.errors}



def run_gradient_probe(worker, scores, seed, memory, probe):
    """Measure one active replay row/slot, using ALL endpoint-group rewards.

    Mutate the supplied report so a failed/OOM probe retains its selected item
    and pre-backward memory. This deliberately performs no optimizer update.
    """
    import numpy as np

    probe.update(status="running", optimizer_updates=0, items=1, updates=0,
                 scope="one active ART row and one selected timestep; no optimizer update",
                 reward_source=worker.c["reward"]["kind"])
    begun = time.monotonic()
    try:
        prepared = worker.prepare_update(rewards=np.asarray(scores, np.float32), seed=seed)
        probe["prepared_complete_groups"] = prepared
        active_groups = np.flatnonzero(worker.stats["group_active"])
        active_rows = np.flatnonzero(worker.stats["row_active"])
        if not len(active_groups) or not len(active_rows):
            probe.update(status="skipped", reason="No active reward group/row; backward was not executed.",
                         seconds=time.monotonic()-begun, after_probe=memory())
            return probe

        row = int(active_rows[0])
        group = int(worker.cache["groups"][row])
        slot = len(worker.cache["selected"]) // 2
        step = int(worker.cache["selected"][slot])
        probe.update(row=row, group=group, sample_id=worker.cache["records"][row]["sample_id"],
                     selected_slot=slot, denoising_step=step, actual_timestep=step,
                     sigma=float(worker.cache["sigmas"][step]),
                     before_gradient=memory())
        # Keep guidance derived from every full K-member group. The custom
        # window only selects the single replay item whose VJP is measured.
        worker.prepare_update(stats=worker.stats, windows=[
            np.asarray([[row, slot]], dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
        ])
        gradient_start = time.monotonic()
        result = worker.gradient(window=0)
        gradient_seconds = time.monotonic()-gradient_start
        finite = all(bool(np.isfinite(value).all()) for value in result["gradients"].values())
        # Accumulate in host FP64, avoiding FP32 norm overflow in the report.
        norm_squared = sum(float(np.sum(np.square(np.asarray(value, dtype=np.float64))))
                           for value in result["gradients"].values())
        norm = float(np.sqrt(norm_squared))
        nonzero = sum(int(np.count_nonzero(value)) for value in result["gradients"].values())
        passed = (result["denominator"] == 1 and result["active"] > 0
                  and result["masked"] == 0 and finite and np.isfinite(norm) and norm > 0)
        probe.update(status="passed" if passed else "failed",
                     seconds=time.monotonic()-begun, gradient_seconds=gradient_seconds,
                     denominator=int(result["denominator"]), active=int(result["active"]),
                     masked=int(result["masked"]), loss_sum=float(result["loss_sum"]),
                     gradients_finite=finite, gradient_l2_norm=norm if np.isfinite(norm) else None,
                     nonzero_gradient_elements=nonzero,
                     gradient_parameter_elements=sum(int(value.size) for value in result["gradients"].values()),
                     after_gradient=memory())
        if not passed:
            raise RuntimeError("Gradient probe did not produce an active, unmasked, finite, nonzero LoRA gradient")
        return probe
    except BaseException as error:
        probe.update(status="failed", seconds=time.monotonic()-begun,
                     error_type=type(error).__name__, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt-ids", help="Comma-separated IDs; count must match prompts_per_rollout")
    parser.add_argument("--repeats", type=int, default=2, help="Identical seeds, first/warm in this process; persistent compilation cache may be reused")
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--gradient-probe", action="store_true",
                        help="After final scoring, measure one active ART LoRA gradient without updating weights")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    override = {"runtime": {"output_dir": args.output}} if args.output else None
    c = load_config(args.config, override)
    if len(c["runtime"]["devices"]) != 1:
        parser.error("This benchmark measures one worker; configure exactly one device")
    if c["reward"]["cpu_decode"] and not args.no_decode:
        parser.error("This single-worker benchmark measures GPU-local VAE decoding; set reward.cpu_decode=false")
    if args.gradient_probe and args.no_decode and c["reward"]["kind"] != "toy_latent":
        parser.error("--gradient-probe requires image decoding/scoring, except for toy_latent smoke rewards")
    inventory = preflight(c)
    configure_process(c, c["runtime"]["devices"][0])
    import numpy as np
    import jax
    import jaxlib
    from moss_art.worker import NativeWorker
    from moss_art.trainer import read_records, make_payloads, write_json
    from moss_art.evaluate import save_images, write_gallery
    from moss_art.rewards import score_images
    from moss_art.assets import experiment_identity
    output = Path(c["runtime"]["output_dir"]).resolve()
    with output_lock(output):
        if (output/"benchmark.json").exists():
            raise FileExistsError("Benchmark report already exists; choose a new --output")
        records = read_records(c)
        if args.prompt_ids:
            wanted = args.prompt_ids.split(",")
            mapping = {r["sample_id"]: r for r in records}
            if len(set(wanted)) != len(wanted) or any(key not in mapping for key in wanted):
                raise ValueError("--prompt-ids must contain existing unique prompt IDs")
            records = [mapping[key] for key in wanted]
        else:
            records = records[:c["sampling"]["prompts_per_rollout"]]
        if len(records) != c["sampling"]["prompts_per_rollout"]:
            raise ValueError("Selected prompt count must equal prompts_per_rollout")
        payload = make_payloads(c, records, rollout=0, evaluation=False)[0]
        report = {"status":"running", "recorded_utc":datetime.now(timezone.utc).isoformat(),
                  "config":c, "inventory":inventory, "records":records, "repeats":[],
                  "native_art_cache_retained":True, "optimizer_updates":0,
                  "seeds":payload["seeds"], "gradient_probe_requested":args.gradient_probe,
                  "jax_peak_memory_scope":"process_lifetime_high_watermark",
                  "memory_note":"Later repeats inherit earlier decode/backward peaks; nvidia-smi samples include reserved memory.",
                  "unique_samples_count":len(payload["seeds"]),
                  "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "argv":list(sys.argv),
                  "software_versions":{"python":sys.version, "jax":jax.__version__, "jaxlib":jaxlib.__version__},
                  "compilation_cache":{
                      "path":os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                      "policy":"persistent disk cache may be reused; no cold-cache guarantee; worker retains up to 16 historical text lengths",
                      "enabled":getattr(jax.config,"jax_enable_compilation_cache",None),
                      "min_compile_time_seconds":getattr(jax.config,"jax_persistent_cache_min_compile_time_secs",None),
                      "min_entry_size_bytes":getattr(jax.config,"jax_persistent_cache_min_entry_size_bytes",None)},
                  "runtime_environment":{key:os.environ.get(key) for key in
                      ("CUDA_VISIBLE_DEVICES","JAX_PLATFORMS","XLA_FLAGS","XLA_PYTHON_CLIENT_PREALLOCATE","XLA_PYTHON_CLIENT_MEM_FRACTION")},
                  "adapter":str(Path(args.adapter).resolve()) if args.adapter else "zero-initialized LoRA (base MOSS)",
                  "source_asset_identity":experiment_identity(c)}
        if args.adapter:
            report["adapter_sha256"] = hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest()
        write_json(output/"benchmark.json", report)
        monitor = GPUMonitor(c["runtime"]["devices"][0], c["runtime"]["platform"]=="cuda")
        monitor.start()
        worker = None
        started = time.monotonic()
        def memory():
            stats = worker.device.memory_stats() or {}
            return {"jax":{k:int(v) for k,v in stats.items() if isinstance(v, (int,np.integer))},
                    "host_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        try:
            monitor.phase = "load_model"
            begin = time.monotonic()
            worker = NativeWorker(c, 0)
            if args.adapter:
                from moss_art.lora import import_safetensors
                worker.set_lora(import_safetensors(args.adapter, worker.lora, alpha=c["lora"]["alpha"]))
            report["load_seconds"] = time.monotonic()-begin
            report["attention_backend"] = worker.model.attention_backend
            report["dit_autotune_level"] = worker.model.dit_autotune_level
            report["after_load"] = memory()
            print(json.dumps({"event":"loaded", "seconds":report["load_seconds"], **report["after_load"]}), flush=True)
            baseline = None
            baseline_images = None
            for repeat in range(args.repeats):
                phase = f"rollout_{repeat}"
                monitor.phase = phase
                print(json.dumps({"event":"phase_start", "phase":phase}), flush=True)
                begin = time.monotonic()
                values = worker.rollout(**payload)
                elapsed = time.monotonic()-begin
                count = len(values["endpoints"])
                state_bytes = sum(worker.cache[key].nbytes for key in ("states","old","endpoints","groups","sigmas","ode_velocity_coefficients"))
                entry = {"repeat":repeat, "repeat_label":"first_in_process" if repeat == 0 else "warm_in_process",
                         "rollout_seconds":elapsed, "images":count, "unique_samples_count":count,
                         "sigmas":[float(value) for value in values["sigmas"]],
                         "images_per_second":count/elapsed, "seconds_per_image":elapsed/count,
                         "actual_denoising_steps":len(values["sigmas"])-1,
                         "sampler":values["sampler"], "scheduler":values["scheduler"],
                         "model_predictions_per_image":values["actual_steps"],
                         "ode_velocity_coefficients":[float(v) for v in values["ode_velocity_coefficients"]],
                         "endpoints_finite":bool(np.isfinite(values["endpoints"]).all()),
                         "retained_art_numpy_bytes":state_bytes, "after_rollout":memory()}
                if baseline is None:
                    baseline = values["endpoints"].copy()
                else:
                    entry["same_seed_max_abs_latent_difference"] = float(np.max(np.abs(baseline-values["endpoints"])))
                folder = output/f"repeat_{repeat:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder/"endpoints.npy", values["endpoints"], allow_pickle=False)
                write_json(folder/"prompts.json", values["prompts"])
                report["repeats"].append(entry)
                write_json(output/"benchmark.json",report)
                print(json.dumps({"event":"rollout_complete", **entry}), flush=True)
                images, paths, scores = None, None, None
                if not args.no_decode:
                    monitor.phase = f"decode_{repeat}"
                    begin = time.monotonic()
                    images = worker.decode(values["endpoints"])["images"]
                    entry["decode_seconds"] = time.monotonic()-begin
                    entry["after_decode"] = memory()
                    assert images.shape == (count,c["sampling"]["height"],c["sampling"]["width"],3)
                    if baseline_images is None:
                        baseline_images = images.copy()
                    else:
                        difference = np.abs(images.astype(np.int16)-baseline_images.astype(np.int16))
                        entry["same_seed_images_equal"] = bool(np.array_equal(images, baseline_images))
                        entry["same_seed_max_abs_pixel_difference"] = int(difference.max())
                        entry["same_seed_pixel_mean_abs_difference"] = float(difference.mean())
                    paths = save_images(images, folder)
                    write_json(output/"benchmark.json",report)
                    print(json.dumps({"event":"decode_complete", "repeat":repeat,
                                      "seconds":entry["decode_seconds"],"memory":entry["after_decode"]}), flush=True)
                if c["reward"]["kind"] == "toy_latent" or images is not None:
                    monitor.phase = f"reward_{repeat}"
                    begin = time.monotonic()
                    if c["reward"]["kind"] == "toy_latent":
                        endpoints = values["endpoints"].astype(np.float32)
                        scores = -np.mean(np.square(endpoints - np.float32(.25)),
                                          axis=tuple(range(1, endpoints.ndim)))
                    else:
                        scores = np.asarray(score_images(images, values["prompts"], c["reward"]), np.float32)
                    if scores.shape != (count,) or not np.isfinite(scores).all():
                        raise ValueError("Reward must contain one finite scalar per endpoint")
                    entry["reward_seconds"] = time.monotonic()-begin
                    entry["reward_kind"] = c["reward"]["kind"]
                    entry["rewards"] = scores.tolist()
                    entry["reward_mean"] = float(scores.mean())
                    entry["reward_std"] = float(scores.std())
                    if paths is not None:
                        write_gallery([{"image":p,"prompt":prompt,"score":float(score)}
                                       for p,prompt,score in zip(paths,values["prompts"],scores)], folder,
                                      title=f"MOSS rollout - repeat {repeat}")
                    print(json.dumps({"event":"reward_complete", "repeat":repeat,"rewards":scores.tolist()}),flush=True)
                write_json(output/"benchmark.json",report)
                if args.gradient_probe and repeat == args.repeats - 1:
                    monitor.phase = "gradient_probe"
                    report["gradient_probe"] = {"status":"running"}
                    write_json(output/"benchmark.json",report)
                    print(json.dumps({"event":"phase_start", "phase":"gradient_probe"}),flush=True)
                    run_gradient_probe(worker, scores, c["training"]["seed"], memory, report["gradient_probe"])
                    write_json(output/"benchmark.json",report)
                    print(json.dumps({"event":"gradient_probe_complete", **report["gradient_probe"]}),flush=True)
            report["status"] = ("passed_with_gradient_probe_skipped"
                                if report.get("gradient_probe", {}).get("status") == "skipped" else "passed")
        except BaseException as error:
            report.update(status="failed",failure_phase=monitor.phase,
                          error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            raise
        finally:
            report["total_seconds"] = time.monotonic()-started
            report["nvidia_smi"] = monitor.finish(output/"gpu_memory.csv")
            if worker is not None:
                try:
                    report["final_memory"] = memory()
                except BaseException as memory_error:
                    report["final_memory_error"] = {"error_type":type(memory_error).__name__,
                                                    "error":str(memory_error)}
            write_json(output/"benchmark.json",report)
            print(json.dumps({"event":"benchmark_finished","status":report["status"],
                              "seconds":report["total_seconds"],"nvidia_smi":report["nvidia_smi"]}),flush=True)


if __name__ == "__main__":
    main()
