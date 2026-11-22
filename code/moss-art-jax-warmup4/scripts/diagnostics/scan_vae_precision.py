#!/usr/bin/env python3
"""Run the authorized GPU2 BF16 / GPU3 FP16 one-sample VAE stage screen.

Every case is a fresh benchmark subprocess with its own normal busy-GPU check.
No overwrite or retry of prior measurements; --summarize-only is CPU-only and
can add reference metrics after FP32 has completed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_vae_precision import atomic_json, comparison_metrics, comparison_panel, sha256
import numpy as np

PROFILES = [(name, name) for name in ("input", "mid", "up0", "up1", "up2", "up3", "output")] + [
    ("early", "input,mid,up0"), ("late", "up1,up2,up3"), ("ups", "up0,up1,up2,up3")]
PRECISIONS = (("bf16", "bfloat16", 2), ("fp16", "float16", 3))
LOCK = threading.Lock()


def summarize(base, latents):
    reference_path = base / "fp32_reference" / "summary.json"
    reference_report = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    reference_ready = reference_report.get("status") == "passed"
    reference = None
    if reference_ready:
        if reference_report["input_sha256"] != sha256(latents):
            raise ValueError("FP32 reference latent input differs")
        mapping = {row["sample_index"]: row["raw_path"] for row in reference_report["repeats"][0]["samples"]}
        reference = np.load(base / "fp32_reference" / mapping[3], allow_pickle=False)
    rows = []
    for prefix, precision, device in PRECISIONS:
        for profile, stages in [("all", "input,mid,up0,up1,up2,up3,output"), *PROFILES]:
            name = prefix + "_" + profile
            path = base / name / "summary.json"
            if not path.exists():
                rows.append({"profile": name, "status": "not_started"})
                continue
            report = json.loads(path.read_text())
            row = {"profile": name, "precision": precision, "stages": stages.split(","),
                   "device": device, "status": report["status"], "measurement_report": str(path),
                   "measurement_report_sha256": sha256(path),
                   "compile_seconds": report.get("compile_seconds"),
                   "total_seconds": report.get("total_seconds"),
                   "peak_active_bytes": report.get("final_memory", {}).get("jax", {}).get("peak_bytes_in_use"),
                   "reserved_dit_buffer_bytes": report.get("reservation", {}).get("buffer_bytes"),
                   "error": report.get("error")}
            if report["repeats"] and report["repeats"][0]["samples"]:
                sample = report["repeats"][0]["samples"][0]
                row["first_decode_seconds"] = sample.get("decode_seconds")
                row["raw_finite"] = sample.get("raw_finite")
                if row["first_decode_seconds"] is not None and row["compile_seconds"] is not None:
                    row["first_decode_minus_explicit_compile_seconds"] = row["first_decode_seconds"] - row["compile_seconds"]
                row["warm_decode_seconds"] = [repeat["samples"][0].get("decode_seconds") for repeat in report["repeats"][1:]]
                if reference is not None and report["status"] == "passed":
                    if report["input_sha256"] != reference_report["input_sha256"]:
                        raise ValueError("Candidate/reference latent input differs: " + name)
                    if sample["sample_index"] != 3:
                        raise ValueError("Stage screen must use original sample 3")
                    raw = np.load(base / name / sample["raw_path"], allow_pickle=False)
                    row["comparison"] = comparison_metrics(raw, reference)
                    panel = ROOT / "reports" / "vae_precision_panels" / (name + ".png")
                    panel.parent.mkdir(exist_ok=True)
                    comparison_panel(reference, raw, panel, name)
                    row["comparison_panel"] = str(panel)
            evidence = report.get("hlo_evidence", {})
            conv_counts, conv_entries = {}, []
            for entry in evidence.get("convolution_and_linear_instructions", []):
                target = entry.get("custom_call_target") or ""
                prefix_text = entry["instruction_prefix"]
                if "conv" in target.lower() or "convolution(" in prefix_text:
                    match = re.search(r"=\s*\(?\s*(bf16|f16|f32)\[", prefix_text)
                    dtype = match.group(1) if match else "unknown"
                    conv_counts[dtype] = conv_counts.get(dtype, 0) + 1
                    conv_entries.append({"line": entry["line"], "target": target, "output_dtype": dtype})
            row["compiled_convolution_output_dtype_counts"] = conv_counts
            row["convolution_evidence"] = conv_entries
            row["optimized_hlo"] = str(base / name / evidence.get("path", "optimized.hlo.txt"))
            rows.append(row)
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if reference_ready and all(row["status"] == "passed" for row in rows) else "in_progress",
              "reference_status": reference_report.get("status", "not_started"),
              "reference_report": str(reference_path),
              "input_latents": str(latents), "input_sha256": sha256(latents),
              "sample_index": 3, "unique_samples": 1,
              "scope": "One 1024 sample stage screening; all profiles have two repeats, isolated stage/combination scans have one first-in-process repeat; no optimizer update.",
              "timing_note": "Single-repeat first decode includes compilation. Subtracting explicit compile time is only a screening estimate, not a warm measurement.",
              "memory_note": "9.6GiB buffers approximate resident DiT capacity, not its computation or full trainer coexistence.",
              "driver_sha256": sha256(__file__), "profiles": rows}
    atomic_json(ROOT / "reports" / "vae_precision_screening.json", report)
    return report


def run_queue(prefix, precision, device, base, latents):
    initial = json.loads((base / (prefix + "_all") / "summary.json").read_text())
    if initial.get("status") != "passed":
        raise RuntimeError(prefix + "_all must complete before this device's queue")
    results = []
    for profile, stages in PROFILES:
        name = prefix + "_" + profile
        directory = base / name
        if directory.exists():
            # Never overwrite any measured or partially measured profile.
            existing = directory / "summary.json"
            status = json.loads(existing.read_text()).get("status") if existing.exists() else "exists_without_report"
            results.append({"profile": name, "status": status, "action": "preserved_existing"})
            continue
        argv = ['$CONDA_PREFIX/envs/jax_env/bin/python', str(ROOT / "scripts/benchmark_vae_precision.py"),
                "--device", str(device), "--precision", precision, "--stages", stages,
                "--latents", str(latents), "--samples", "3", "--repeats", "1",
                "--reserve-dit-gib", "9.6", "--output", str(directory)]
        log = ROOT / "reports" / ("vae_" + name + ".log")
        environment = os.environ.copy()
        environment.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        with log.open("x") as file:
            completed = subprocess.run(argv, cwd=ROOT, env=environment, stdout=file, stderr=subprocess.STDOUT, check=False)
        result = {"profile": name, "returncode": completed.returncode, "log": str(log)}
        results.append(result)
        with LOCK:
            summary = summarize(base, latents)
            row = next(row for row in summary["profiles"] if row["profile"] == name)
            print(json.dumps({"event": "profile_complete", "profile": name, "returncode": completed.returncode,
                              "first_seconds": row.get("first_decode_seconds"), "comparison": row.get("comparison")}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", default="outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy")
    parser.add_argument("--base-output", default="outputs/vae_precision")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    base, latents = (ROOT / args.base_output).resolve(), (ROOT / args.latents).resolve()
    if args.summarize_only:
        print(json.dumps({"status": summarize(base, latents)["status"]}), flush=True)
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(run_queue, prefix, precision, device, base, latents)
                   for prefix, precision, device in PRECISIONS]
        for future in as_completed(pending):
            print(json.dumps({"event": "queue_finished", "results": future.result()}), flush=True)
    final = summarize(base, latents)
    print(json.dumps({"event": "screen_finished", "status": final["status"],
                      "report": str(ROOT / "reports/vae_precision_screening.json")}), flush=True)


if __name__ == "__main__":
    main()
