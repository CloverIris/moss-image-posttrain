#!/usr/bin/env python3
"""Reproduce compact VAE convolution results and figure from saved reports, CPU only.

Run from any directory:
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/summarize_vae_conv_optimization.py
Use --overwrite to replace the two generated report files.
No JAX/model imports, inference, recompilation, or source-artifact edits.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GIB = 1024 ** 3
PROFILES = ("baseline_fp32", "cudnn_fp32_four", "cudnn_late_fp16_four",
            "tiled_fp32_four")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_report(root, relative, sources):
    path = root / relative
    raw = path.read_bytes()
    result = json.loads(raw)
    if result.get("status") not in ("passed", "completed"):
        raise ValueError(f"Incomplete input {relative}: {result.get('status')}")
    sources[relative] = {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "status": result["status"],
        "recorded_utc": result.get("recorded_utc"),
    }
    return result


def compact_quality(samples):
    rows = []
    for sample in samples:
        c = sample["comparison"]
        rows.append({
            "sample_index": sample["sample_index"],
            "raw_max_abs": c["raw_max_abs"],
            "raw_relative_l2": c["raw_relative_l2"],
            "pixel_mae_0_255": c["pixel_mae_0_255"],
            "pixel_max_abs_0_255": c["pixel_max_abs_0_255"],
            "pixel_psnr_db": c["pixel_psnr_db"],
            "pixel_psnr_infinite": c["pixel_psnr_infinite"],
            "ssim_rgb": c["ssim"]["rgb_mean"],
        })
    psnr = [r["pixel_psnr_db"] for r in rows]
    return {
        "samples": rows,
        "raw_max_abs_worst": max(r["raw_max_abs"] for r in rows),
        "raw_relative_l2_worst": max(r["raw_relative_l2"] for r in rows),
        "pixel_mae_mean_0_255": statistics.mean(r["pixel_mae_0_255"] for r in rows),
        "pixel_max_abs_worst_0_255": max(r["pixel_max_abs_0_255"] for r in rows),
        "pixel_psnr_mean_db": statistics.mean(psnr) if all(x is not None for x in psnr) else None,
        "pixel_psnr_all_infinite": all(r["pixel_psnr_infinite"] for r in rows),
        "ssim_rgb_mean": statistics.mean(r["ssim_rgb"] for r in rows),
        "ssim_rgb_worst": min(r["ssim_rgb"] for r in rows),
        "ssim_definition": samples[0]["comparison"]["ssim"]["definition"],
        "scope": "One unique set of samples, warm repeat; not independent repeated draws.",
    }


def summarize_profile(d):
    warm = [r for r in d["repeats"] if r["repeat"] == 1]
    if len(warm) != 1:
        raise ValueError("Require exactly one repeat 1")
    samples = warm[0]["samples"]
    if sorted(s["sample_index"] for s in samples) != sorted(d["sample_indices"]):
        raise ValueError("Warm sample indices differ from declared indices")
    seconds = [float(s["decode_seconds"]) for s in samples]
    if not all(math.isfinite(t) and t > 0 for t in seconds):
        raise ValueError("Invalid timing")
    first = d["repeats"][0]["samples"]
    memory = d["final_memory"]["jax"]
    plans = [p["selected_plan"] for p in d.get("convolution_plans", [])
             if p.get("selected_plan")]
    if not all(s["raw_finite"] for r in d["repeats"] for s in r["samples"]):
        raise ValueError("Nonfinite raw decoder output")
    return {
        "status": d["status"], "physical_gpu": d["physical_device"],
        "precision": d["precision"], "low_precision_stages": d["low_precision_stages"],
        "conv_backend": d["conv_backend"], "workspace_cap_mib": d.get("conv_workspace_mib"),
        "sample_indices": d["sample_indices"], "input_sha256": d["input_sha256"],
        "input_shape": d["input_shape"],
        "warm_repeat": 1, "warm_samples_seconds": seconds,
        "warm_mean_seconds": statistics.mean(seconds),
        "warm_median_seconds": statistics.median(seconds),
        "first_repeat_samples_seconds": [s["decode_seconds"] for s in first],
        "compile_seconds": d["compile_seconds"],
        "all_outputs_finite": True,
        "warm_outputs_bitwise_equal_first": all(s["same_seed_raw_bitwise_equal"] for s in samples),
        "quality_vs_original_fp32": compact_quality(samples),
        "compiled_memory_bytes": d["compiled_memory"],
        "jax_process_peak_bytes": memory["peak_bytes_in_use"],
        "jax_allocator_limit_bytes": memory["bytes_limit"],
        "jax_peak_headroom_bytes": memory["bytes_limit"] - memory["peak_bytes_in_use"],
        "reserved_dit_buffer_bytes": d["reservation"]["buffer_bytes"],
        "reserved_dit_buffer_gib": d["reservation"]["requested_gib"],
        "hlo_sha256": d["hlo_evidence"]["sha256"],
        "hlo_custom_call_counts": d["hlo_evidence"]["custom_call_counts"],
        "selected_plan_count": len(plans),
        "selected_plan_max_workspace_bytes": max((p["workspace_bytes"] for p in plans), default=None),
        "runtime_environment": d["runtime_environment"],
        "compiler_options": d.get("vae_compiler_options"),
        "software_versions": d["versions"],
        "timing_policy": d["policy"]["timing"],
        "compilation_cache_policy": d["policy"]["compilation_cache"],
        "source_hashes": d["source_hashes"],
    }


def summarize_rollout(d):
    warm = next(r for r in d["repeats"] if r["repeat"] == 1)
    n = warm["images"]
    stage = {k: warm[k] for k in ("rollout_seconds", "decode_seconds", "reward_seconds")}
    return {
        "physical_gpu": d["config"]["runtime"]["devices"],
        "image_count": n, "warm_repeat": 1,
        "stages_seconds": stage,
        "stages_seconds_per_image": {k: v / n for k, v in stage.items()},
        "three_stage_sum_seconds": sum(stage.values()),
        "three_stage_sum_seconds_per_image": sum(stage.values()) / n,
        "sampling": d["config"]["sampling"], "seeds": d["seeds"],
        "adapter_sha256": d["adapter_sha256"],
        "native_art_cache_retained": d["native_art_cache_retained"],
        "optimizer_updates": d["optimizer_updates"],
        "endpoints_finite": warm["endpoints_finite"],
        "same_seed_latent_max_abs": warm["same_seed_max_abs_latent_difference"],
        "same_seed_images_equal": warm["same_seed_images_equal"],
        "jax_process_peak_bytes": d["final_memory"]["jax"]["peak_bytes_in_use"],
        "timing_scope": "Sum of separately timed rollout, decode and CPU reward stages; excludes load/compile setup outside these timers and is not full process wall time.",
    }


def draw(report, path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.axisbelow": True,
        "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 9.5))
    fig.subplots_adjust(left=.10, right=.975, top=.87, bottom=.20, hspace=.62, wspace=.30)
    fig.suptitle("1024 x 1024 MOSS VAE: bounded cuDNN convolution", fontsize=18,
                 fontweight="bold", y=.972)
    fig.text(.5, .932, "Saved measurements | GB203(SM120) 16 GB | synchronized warm repeats",
             ha="center", color="#555555", fontsize=11)
    blue, green, orange, gray = "#0072B2", "#009E73", "#D55E00", "#757575"
    profiles = report["vae_profiles"]
    same = report["same_gpu_sample3"]
    ax = axs[0, 0]
    vals = [same["baseline_seconds"], same["cudnn_seconds"]]
    bars = ax.bar(["XLA default\nFP32", "Bounded cuDNN\nFP32"], vals,
                  color=[gray, blue], width=.55)
    ax.set_yscale("log")
    ax.set_ylim(.1, 100)
    ax.set_ylabel("Seconds / image (log scale)")
    ax.set_title("A  Same GPU 3, same latent sample 3", loc="left", pad=12)
    ax.grid(axis="y", alpha=.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.17, f"{val:.6f} s",
                ha="center", fontsize=11)
    ax.text(.5, .57, f"{same['speedup']:.2f}x faster", transform=ax.transAxes,
            ha="center", fontsize=18, color=blue, fontweight="bold")

    ax = axs[0, 1]
    names = ["cudnn_fp32_four", "cudnn_late_fp16_four", "tiled_fp32_four"]
    vals = [profiles[n]["warm_mean_seconds"] for n in names]
    bars = ax.bar(["cuDNN FP32\nGPU 3", "cuDNN late FP16\nGPU 1",
                   "Tiled FP32\nGPU 2"], vals, color=[blue, green, orange], width=.65)
    ax.set_ylim(0, max(vals)*1.27)
    ax.set_ylabel("Mean seconds / image")
    ax.set_title("B  Four-image decoder comparison", loc="left", pad=12)
    ax.grid(axis="y", alpha=.2)
    for bar, val, name in zip(bars, vals, names):
        x = bar.get_x()+bar.get_width()/2
        ax.text(x, val+.025, f"{val:.3f} s", ha="center", fontweight="bold")
        ax.scatter(np.linspace(x-.06, x+.06, 4), profiles[name]["warm_samples_seconds"], s=13,
                   color="#222222", zorder=3)
    ax.text(.02, .94, "Dots: 4 fixed samples; different physical GPUs", transform=ax.transAxes,
            va="top", fontsize=9, color="#555555")

    ax = axs[1, 0]
    old, new = report["real_rollout"]["old"], report["real_rollout"]["new"]
    left = [0., 0.]
    for key, color, label in [("rollout_seconds", blue, "DiT"),
                               ("decode_seconds", orange, "VAE"),
                               ("reward_seconds", green, "CPU SigLIP")]:
        widths = [x["stages_seconds_per_image"][key] for x in (old, new)]
        ax.barh([1, 0], widths, left=left, color=color, height=.55, label=label)
        left = [l+w for l, w in zip(left, widths)]
    ax.set_yticks([1, 0], ["Old / GPU 1", "New / GPU 2"])
    ax.set_xlim(0, max(left)*1.23)
    ax.set_ylim(-.6, 1.75)
    for y, value in zip([1, 0], left):
        ax.text(value+.8, y, f"{value:.3f}", va="center", fontweight="bold")
    ax.set_xlabel("Seconds / image: DiT + VAE + CPU reward")
    ax.set_title("C  Real rollout with DiT resident", loc="left", pad=12)
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False,
              columnspacing=.8, handlelength=1.4)
    ax.grid(axis="x", alpha=.2)
    ax.text(.02, .06, f"{report['real_rollout']['three_stage_speedup']:.3f}x overall; "
            "10 steps, CFG 3, 2 prompts x K=2", transform=ax.transAxes,
            fontsize=9, color="#555555")

    ax = axs[1, 1]
    names = ["baseline_fp32", "cudnn_fp32_four", "cudnn_late_fp16_four", "tiled_fp32_four"]
    vals = [profiles[n]["compiled_memory_bytes"]["temp_size_in_bytes"]/GIB for n in names]
    bars = ax.bar(["XLA\nFP32", "cuDNN\nFP32", "cuDNN\nlate FP16", "Tiled\nFP32"],
                  vals, color=[gray, blue, green, orange], width=.65)
    ax.set_ylim(0, max(vals)*1.30)
    ax.set_ylabel("Compiled temporary buffers (GiB)")
    ax.set_title("D  VAE executable memory", loc="left", pad=12)
    ax.grid(axis="y", alpha=.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+.10, f"{val:.3f}",
                ha="center", fontweight="bold")
    ax.text(.02, .94, "Tiled: experimental; allocator limit reached", transform=ax.transAxes,
            color=orange, fontsize=9, va="top")

    q = profiles["cudnn_fp32_four"]["quality_vs_original_fp32"]
    reward = report["reward_audit"]
    fig.text(.075, .136,
             f"FP32 fidelity: mean RGB SSIM {q['ssim_rgb_mean']:.9f}; "
             f"max raw error {q['raw_max_abs_worst']:.2e}; "
             f"max reward delta {reward['max_abs_score_delta']:.2e}.",
             fontsize=10, color="#333333")
    fig.text(.075, .108,
             "Timing: A/B include latent H2D + synchronized decode + raw FP32 D2H; "
             "warm repeat excludes compilation and file writes.",
             fontsize=9, color="#555555")
    trace_ms = report["separate_gpu_trace"]["time_accounting"]["kernel_sum_seconds"] * 1000
    benchmark_ms = profiles["cudnn_fp32_four"]["warm_mean_seconds"] * 1000
    fig.text(.075, .083,
             f"Separate CUPTI trace: {trace_ms:.0f} ms GPU kernel sum; this is not the "
             f"{benchmark_ms:.0f} ms four-image mean benchmark. No confidence intervals.",
             fontsize=9, color="#555555")
    fig.text(.075, .058,
             "A/B/D reserve 9.6 GiB dummy buffers; C runs the actual DiT. "
             "Late FP16 = up1/up2/up3 conv/linear operands; sensitive ops remain FP32.",
             fontsize=9, color="#555555")
    fig.text(.075, .033, "Reproduction and source SHA256: reports/vae_conv_optimization.json",
             fontsize=9, color="#555555")
    fig.savefig(path, dpi=180, metadata={"Software": "summarize_vae_conv_optimization.py"})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output-json", default="reports/vae_conv_optimization.json")
    p.add_argument("--output-figure", default="reports/vae_conv_optimization.png")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    json_path, figure_path = root / args.output_json, root / args.output_figure
    for path in (json_path, figure_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing overwrite: {path}; use --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
    sources = {}
    raw = {n: read_report(root, f"outputs/vae_conv_optimization/{n}/summary.json", sources)
           for n in PROFILES}
    profiles = {n: summarize_profile(d) for n, d in raw.items()}
    if len({d["input_sha256"] for d in raw.values()}) != 1:
        raise ValueError("Standalone profiles use different latent inputs")
    baseline, candidate = raw["baseline_fp32"], raw["cudnn_fp32_four"]
    if baseline["physical_device"] != candidate["physical_device"]:
        raise ValueError("Primary comparison must use the same physical GPU")
    b = next(s["decode_seconds"] for r in baseline["repeats"] if r["repeat"] == 1
             for s in r["samples"] if s["sample_index"] == 3)
    c = next(s["decode_seconds"] for r in candidate["repeats"] if r["repeat"] == 1
             for s in r["samples"] if s["sample_index"] == 3)
    rewards = read_report(root, "reports/vae_conv_rewards.json", sources)
    reward_candidate = next(d for d in rewards["profiles"]
                            if Path(d["directory"]).name == "cudnn_fp32_four")
    art = next(d for d in rewards["art_comparison"]["profiles"]
               if d["label"] == reward_candidate["label"])
    train = read_report(root, "reports/train_1024_fast_vae_validation.json", sources)
    old = read_report(root, "outputs/rollout_1024_cudnn_tuned/benchmark.json", sources)
    new = read_report(root, "outputs/rollout_1024_fast_vae_matched/benchmark.json", sources)
    validation = read_report(root, "reports/vae_conv_matched_rollout_validation.json", sources)
    trace = read_report(root, "reports/vae_conv_optimized_trace.json", sources)
    if not all(validation["matching_checks"].values()):
        raise ValueError("Matched rollout verification failed")
    for name, value in (("new", new), ("old", old)):
        source = f"outputs/{'rollout_1024_fast_vae_matched' if name == 'new' else 'rollout_1024_cudnn_tuned'}/benchmark.json"
        if validation["inputs"][f"{name}_benchmark_sha256"] != sources[source]["sha256"]:
            raise ValueError("Matched rollout audit references a different benchmark")
    old_summary, new_summary = summarize_rollout(old), summarize_rollout(new)
    report = {
        "schema_version": 1, "status": "passed",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True, "script_sha256": sha256(Path(__file__)),
        "argv": sys.argv, "root": str(root),
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                    "numpy": np.__version__, "thread_limit": 2,
                    "gpu_visibility": os.environ["CUDA_VISIBLE_DEVICES"]},
        "sources": sources, "vae_profiles": profiles,
        "same_gpu_sample3": {
            "physical_gpu": baseline["physical_device"], "sample_index": 3, "repeat": 1,
            "baseline_seconds": b, "cudnn_seconds": c, "speedup": b/c,
            "scope": "Same GPU, same latent sample; one warm observation per backend, not a distribution.",
        },
        "real_rollout": {
            "old": old_summary, "new": new_summary,
            "three_stage_speedup": old_summary["three_stage_sum_seconds"]/new_summary["three_stage_sum_seconds"],
            "decode_speedup": old_summary["stages_seconds"]["decode_seconds"]/new_summary["stages_seconds"]["decode_seconds"],
            "independent_validation_status": validation["status"],
            "matching_checks": validation["matching_checks"],
            "endpoints_bitwise_equal_between_backends": all(r["endpoints_bitwise_equal_to_old"] for r in validation["repeats"]),
            "scope": "Same model, adapter, prompts, seeds and sampling; old GPU1 versus new GPU2, same GPU model. Summary reads the independent array audit without rerunning it.",
        },
        "reward_audit": {
            "status": rewards["status"], "cpu_only": rewards["cpu_only"],
            "reference_scores": rewards["profiles"][0]["scores"],
            "candidate_scores": reward_candidate["scores"],
            "max_abs_score_delta": reward_candidate["max_abs_score_delta_reference"],
            "score_deltas": reward_candidate["score_delta_reference"],
            "groups": reward_candidate["groups"],
            "art": {k: art[k] for k in (
                "group_direction_cosine_reference", "group_direction_relative_l2_reference",
                "direction_finite", "group_active_equal_reference", "row_active_equal_reference")},
            "limitations": rewards["limitations"],
        },
        "training_validation": {
            k: train[k] for k in (
                "status", "complete", "adam_counts", "adapter_tensor_count", "adapter_parameters",
                "adapter_l2_delta_from_initial", "changed_adapter_tensors",
                "export_equals_checkpoint_bitwise", "checkpoint_repeat_load_bitwise",
                "all_adapter_and_optimizer_leaves_finite",
                "checkpoint_integrity_and_config_fingerprint_verified",
                "run_current_and_checkpoint_experiment_identity_match",
                "vae_backend_binding", "metrics", "peak_active_gib", "scope", "weight_identity_limit")},
        "separate_gpu_trace": {
            "sample_index": trace["sample_index"], "time_accounting": trace["time_accounting"],
            "traced_synchronized_host_seconds": trace["traced_synchronized_host_seconds"],
            "scope": "CUPTI device event diagnostic, not the transfer-inclusive unprofiled VAE benchmark; kernel sum counts overlap, union does not. Do not substitute 432 ms for 491 ms.",
        },
        "limitations": [
            "Only four fixed samples and one warm repeat; timings are observations, not confidence intervals or general quality claims.",
            "Same-card 64.85x claim uses sample 3 only; four-image alternatives and real-rollout comparison use different physical cards of the same model.",
            "VAE-only tests reserve 9.6 GiB device buffers, not an actual DiT; real rollout and training validations do retain the actual model.",
            "Compiled temporary memory is not whole-process peak; JAX peak is process-lifetime high watermark and includes reservation. NVIDIA reserved memory has a different meaning.",
            "Tiled GEMM is an explicit experimental fallback: whole-VAE HLO has 36 static cuBLASLt sites inside loops, not 36 dynamic launches; standalone microkernels used Triton and their timing/memory must not replace whole-VAE measurements.",
            "Late FP16 only changes selected up1/up2/up3 conv/linear operand precision; FP32 sensitive operations and stored weights remain. Reward audit here covers cuDNN FP32 only.",
            "Persistent compilation cache may be reused; first-in-process timing is not guaranteed cold. Warm tables use repeat 1, not compile-inclusive repeat 0.",
        ],
    }
    draw(report, figure_path)
    report["figure"] = {"path": str(figure_path), "sha256": sha256(figure_path)}
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "json": str(json_path), "figure": str(figure_path),
        "same_gpu_sample3_speedup": b/c,
        "four_image_warm_means": {n: profiles[n]["warm_mean_seconds"] for n in PROFILES[1:]},
        "rollout_old_seconds_per_image": old_summary["three_stage_sum_seconds_per_image"],
        "rollout_new_seconds_per_image": new_summary["three_stage_sum_seconds_per_image"],
        "rollout_speedup": report["real_rollout"]["three_stage_speedup"],
    }, indent=2))


if __name__ == "__main__":
    main()
