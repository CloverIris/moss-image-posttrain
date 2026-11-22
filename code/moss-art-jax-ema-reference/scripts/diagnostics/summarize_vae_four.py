#!/usr/bin/env python3
"""CPU-only aggregation of completed FP32/BF16/FP16 four-sample VAE runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_vae_precision import atomic_json, comparison_metrics, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combo", choices=("late", "ups"), default="late")
    parser.add_argument("--base-output", default="outputs/vae_precision")
    args = parser.parse_args()
    base = (ROOT / args.base_output).resolve()
    names = ("fp32_reference", "bf16_" + args.combo + "_four", "fp16_" + args.combo + "_four")
    rows = []
    input_sha = None
    for name in names:
        file = base / name / "summary.json"
        report = json.loads(file.read_text())
        if report.get("status") != "passed" or len(report["repeats"]) != 2:
            raise ValueError(name + " requires a passed two-repeat benchmark")
        if report["sample_indices"] != [0, 1, 2, 3]:
            raise ValueError(name + " did not validate all four original samples")
        input_sha = input_sha or report["input_sha256"]
        if report["input_sha256"] != input_sha:
            raise ValueError(name + " latent input differs")
        first = {sample["sample_index"]: sample for sample in report["repeats"][0]["samples"]}
        warm = {sample["sample_index"]: sample for sample in report["repeats"][1]["samples"]}
        samples = []
        for index in range(4):
            sample, repeated = first[index], warm[index]
            if name == "fp32_reference":
                raw = np.load(base / name / sample["raw_path"], allow_pickle=False)
                metrics = comparison_metrics(raw, raw)
            else:
                metrics = sample["comparison"]
            samples.append({"sample_index": index, "first_decode_seconds": sample["decode_seconds"],
                            "warm_decode_seconds": repeated["decode_seconds"],
                            "raw_finite_both_repeats": sample["raw_finite"] and repeated["raw_finite"],
                            "raw_bitwise_repeat": repeated["same_seed_raw_bitwise_equal"],
                            "raw_repeat_max_abs": repeated["same_seed_raw_max_abs"],
                            "raw_path": str(base / name / sample["raw_path"]),
                            "png_path": str(base / name / sample["png_path"]),
                            "comparison_against_fp32": metrics})
        timing = [sample["warm_decode_seconds"] for sample in samples]
        metrics = [sample["comparison_against_fp32"] for sample in samples]
        finite_psnr = [value["pixel_psnr_db"] for value in metrics if value["pixel_psnr_db"] is not None]
        aggregate = {
            "pixel_mae_mean": float(np.mean([value["pixel_mae_0_255"] for value in metrics])),
            "pixel_mae_worst": max(value["pixel_mae_0_255"] for value in metrics),
            "pixel_max_abs_worst": max(value["pixel_max_abs_0_255"] for value in metrics),
            "pixel_psnr_db_mean": float(np.mean(finite_psnr)) if len(finite_psnr) == 4 else None,
            "pixel_psnr_db_worst": min(finite_psnr) if finite_psnr else None,
            "pixel_psnr_infinite_sample_count": sum(value["pixel_psnr_infinite"] for value in metrics),
            "ssim_mean": float(np.mean([value["ssim"]["rgb_mean"] for value in metrics])),
            "ssim_worst": min(value["ssim"]["rgb_mean"] for value in metrics),
            "raw_relative_l2_mean": float(np.mean([value["raw_relative_l2"] for value in metrics])),
            "raw_relative_l2_worst": max(value["raw_relative_l2"] for value in metrics),
            "raw_global_relative_l2": math.sqrt(sum(value["raw_error_l2"] ** 2 for value in metrics) /
                                                sum(value["raw_reference_l2"] ** 2 for value in metrics)),
            "raw_max_abs_worst": max(value["raw_max_abs"] for value in metrics),
        }
        conv_counts = {}
        for evidence in report["hlo_evidence"]["convolution_and_linear_instructions"]:
            if "conv" in (evidence.get("custom_call_target") or "").lower() or "convolution(" in evidence["instruction_prefix"]:
                match = re.search(r"=\s*\(?\s*(bf16|f16|f32)\[", evidence["instruction_prefix"])
                dtype = match.group(1) if match else "unknown"
                conv_counts[dtype] = conv_counts.get(dtype, 0) + 1
        rows.append({"run": name, "precision": report["precision"],
                     "stages": report["low_precision_stages"], "physical_gpu": report["physical_device"],
                     "measurement_report": str(file), "measurement_report_sha256": sha256(file),
                     "source_hashes": report["source_hashes"], "input_sha256": report["input_sha256"],
                     "warm_seconds_per_sample": timing, "warm_seconds_mean": float(np.mean(timing)),
                     "warm_seconds_median": float(np.median(timing)), "warm_seconds_total": float(np.sum(timing)),
                     "compile_seconds": report["compile_seconds"], "total_benchmark_seconds": report["total_seconds"],
                     "all_outputs_finite": all(sample["raw_finite_both_repeats"] for sample in samples),
                     "all_raw_bitwise_repeated": all(sample["raw_bitwise_repeat"] for sample in samples),
                     "peak_active_bytes": report["final_memory"]["jax"]["peak_bytes_in_use"],
                     "max_nvidia_smi_used_mib": report["nvidia_smi"]["max_used_mib"],
                     "reserved_dit_buffer_bytes": report["reservation"]["buffer_bytes"],
                     "reservation_requested_gib": report["reservation"]["requested_gib"],
                     "compiled_convolution_output_dtype_counts": conv_counts,
                     "convolution_accumulation_environment": report["runtime_environment"].get("TF_FP16_CONV_USE_FP32_COMPUTE"),
                     "optimized_hlo": str(base / name / report["hlo_evidence"]["path"]),
                     "optimized_hlo_sha256": report["hlo_evidence"]["sha256"],
                     "quality_aggregate": aggregate, "samples": samples})
    for row in rows:
        row["warm_observed_speedup_vs_fp32"] = rows[0]["warm_seconds_mean"] / row["warm_seconds_mean"]
        row["warm_observed_seconds_saved_mean"] = rows[0]["warm_seconds_mean"] - row["warm_seconds_mean"]
    final = {"status": "passed", "recorded_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Four fixed 1024 outputs from identical packed latents; no new diffusion rollout or optimizer update.",
             "timing_scope": "Second in-process repeat only; each sample includes H2D, synchronized decode and D2H; excludes PNG, quality metrics, loading and initial compilation.",
             "quality_scope": "Quantized clipped RGB MAE/PSNR/SSIM and unclipped FP32 NCHW raw relative L2; SSIM Gaussian11 sigma1.5, RGB channel mean, population covariance, valid5px crop.",
             "memory_scope": "Process-lifetime JAX active high watermark includes 9.6GiB reserved device buffers; buffers are not a real DiT model or concurrent training.",
             "limitations": ["Reference GPU1 and candidate GPU2/GPU3 are separate cards of the same model.",
                            "Profiles overlap with server background workloads; observed gains are not strictly isolated paired-card measurements.",
                            "Four examples do not establish safety for every latent or prompt.",
                            "Convolution output dtype evidence does not independently prove internal accumulation; environment explicitly requests FP32 compute."],
             "selection_report": str(ROOT / "reports/vae_precision_selection.json"),
             "runs": rows}
    destination = ROOT / "reports/vae_precision_four_samples.json"
    atomic_json(destination, final)
    print(json.dumps({"status": final["status"], "report": str(destination),
                      "results": [{key: row[key] for key in ("run", "warm_seconds_mean", "warm_seconds_median",
                        "warm_observed_speedup_vs_fp32", "all_outputs_finite", "all_raw_bitwise_repeated",
                        "quality_aggregate", "peak_active_bytes", "compiled_convolution_output_dtype_counts")} for row in rows]}), flush=True)


if __name__ == "__main__":
    main()
