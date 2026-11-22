#!/usr/bin/env python3
"""Summarize completed FP32, late-FP16 and all-stage mixed-FP16 VAE probes."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_vae_precision import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-full", action="store_true", help="Also compare the continuous-FP16 decoder experiment")
    args = parser.parse_args()
    base = ROOT / "outputs/vae_precision"
    names = ("fp32_reference", "fp16_late_four", "fp16_all_four")
    if args.include_full:
        names += ("full_fp16_all",)
    reports = []
    input_sha = None
    for name in names:
        path = base / name / "summary.json"
        report = json.loads(path.read_text())
        if report["status"] != "passed" or len(report["repeats"]) != 2 or report["sample_indices"] != [0, 1, 2, 3]:
            raise ValueError(name + " requires four samples and two passed repeats")
        input_sha = input_sha or report["input_sha256"]
        if report["input_sha256"] != input_sha:
            raise ValueError("Latent input mismatch")
        first = sorted(report["repeats"][0]["samples"], key=lambda row: row["sample_index"])
        warm = sorted(report["repeats"][1]["samples"], key=lambda row: row["sample_index"])
        timing = [sample["decode_seconds"] for sample in warm]
        metrics = [sample.get("comparison") for sample in first]
        quality = None
        if all(metric is not None for metric in metrics):
            finite_psnr = [metric["pixel_psnr_db"] for metric in metrics if metric["pixel_psnr_db"] is not None]
            quality = {
                "pixel_mae_mean": float(np.mean([metric["pixel_mae_0_255"] for metric in metrics])),
                "pixel_mae_worst": max(metric["pixel_mae_0_255"] for metric in metrics),
                "pixel_max_abs_worst": max(metric["pixel_max_abs_0_255"] for metric in metrics),
                "psnr_db_mean": float(np.mean(finite_psnr)) if len(finite_psnr) == 4 else None,
                "psnr_db_worst": min(finite_psnr) if finite_psnr else None,
                "ssim_mean": float(np.mean([metric["ssim"]["rgb_mean"] for metric in metrics])),
                "ssim_worst": min(metric["ssim"]["rgb_mean"] for metric in metrics),
                "raw_relative_l2_mean": float(np.mean([metric["raw_relative_l2"] for metric in metrics])),
                "raw_relative_l2_worst": max(metric["raw_relative_l2"] for metric in metrics),
                "raw_global_relative_l2": math.sqrt(sum(metric["raw_error_l2"] ** 2 for metric in metrics) /
                                                    sum(metric["raw_reference_l2"] ** 2 for metric in metrics)),
                "raw_max_abs_worst": max(metric["raw_max_abs"] for metric in metrics),
            }
        dtypes = {}
        for evidence in report["hlo_evidence"]["convolution_and_linear_instructions"]:
            if "conv" in (evidence.get("custom_call_target") or "").lower() or "convolution(" in evidence["instruction_prefix"]:
                match = re.search(r"=\s*\(?\s*(bf16|f16|f32)\[", evidence["instruction_prefix"])
                dtype = match.group(1) if match else "unknown"
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
        reports.append({
            "run": name, "precision": report["precision"], "low_precision_stages": report["low_precision_stages"],
            "benchmark_variant": report.get("benchmark_variant", "stage_selective_operand_precision"),
            "precision_policy": report["policy"],
            "physical_gpu": report["physical_device"],
            "report": str(path), "report_sha256": sha256(path), "source_hashes": report["source_hashes"],
            "warm_seconds": timing, "warm_mean_seconds": float(np.mean(timing)),
            "warm_median_seconds": float(np.median(timing)), "first_compile_seconds": report["compile_seconds"],
            "quality_against_fp32": quality,
            "per_sample": [{"sample_index": sample["sample_index"], "first_seconds": sample["decode_seconds"],
                            "warm_seconds": repeated["decode_seconds"], "comparison": sample.get("comparison"),
                            "raw_finite_both": sample["raw_finite"] and repeated["raw_finite"],
                            "bitwise_repeated": repeated["same_seed_raw_bitwise_equal"]}
                           for sample, repeated in zip(first, warm)],
            "all_outputs_finite": all(sample["raw_finite"] for repeat in report["repeats"] for sample in repeat["samples"]),
            "all_raw_bitwise_repeated": all(sample["same_seed_raw_bitwise_equal"] for sample in warm),
            "peak_active_bytes": report["final_memory"]["jax"]["peak_bytes_in_use"],
            "max_reserved_nvidia_mib": report["nvidia_smi"]["max_used_mib"],
            "dit_reservation_bytes": report["reservation"]["buffer_bytes"],
            "conv_output_dtype_counts": dtypes,
            "optimized_hlo": str(base / name / "optimized.hlo.txt"),
            "optimized_hlo_sha256": report["hlo_evidence"]["sha256"],
            "runtime_environment": report["runtime_environment"],
        })
    for report in reports:
        report["observed_speedup_vs_fp32"] = reports[0]["warm_mean_seconds"] / report["warm_mean_seconds"]
        report["observed_speedup_vs_late_fp16"] = reports[1]["warm_mean_seconds"] / report["warm_mean_seconds"]
    output = ROOT / ("reports/vae_full_fp16_comparison.json" if args.include_full
                     else "reports/vae_all_fp16_four_samples.json")
    if output.exists():
        raise FileExistsError(output)
    result = {
        "status": "passed", "recorded_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": input_sha,
        "scope": ("FP32, late/all-stage operand precision and continuous-FP16 decoder comparison; see each run's precision_policy."
                  if args.include_full else "Existing mixed-precision helper with all selected conv/linear stages. This is NOT a continuously FP16 VAE."),
        "policy": "Per-run precision_policy is authoritative; full FP16 decoder keeps FP32 BN, GN statistics/affine, QK logits/softmax and accumulation.",
        "timing": "Four second-repeat synchronized decode times per profile; excludes parameters, compilation, PNG and metrics.",
        "memory": "Process lifetime active peak includes 9.6GiB buffer reservation, not a real resident DiT.",
        "limitations": ["FP32 and continuous-FP16 measured on GPU1; late FP16 on GPU3; all-stage FP16 on GPU2. Background load and run times differ.",
                       "Four fixed latents; cannot infer general quality, reward ordering or universal numerical stability.",
                       "Any CPU reward/ART comparison is separate and performs no model gradient or optimizer update."],
        "runs": reports,
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "report": str(output),
                      "results": [{key: report[key] for key in ("run", "warm_mean_seconds", "observed_speedup_vs_fp32",
                          "observed_speedup_vs_late_fp16", "quality_against_fp32", "peak_active_bytes",
                          "all_outputs_finite", "all_raw_bitwise_repeated", "conv_output_dtype_counts")}
                                  for report in reports]}), flush=True)


if __name__ == "__main__":
    main()
