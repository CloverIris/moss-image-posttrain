#!/usr/bin/env python3
"""Render VAE measurements from saved reports only; performs no GPU work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def short(name):
    return (name.replace("decoder.", "").replace("up_blocks.", "up")
            .replace("mid_block.", "mid.").replace("resnets.", "res")
            .replace("upsamplers.0.conv", "upsample.conv"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/vae_gpu_trace.json")
    parser.add_argument("--output", default="reports/vae_layer_profile.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report = json.loads(Path(args.report).read_text())
    assert report["status"] == "completed"
    runs = {r["variant"]: r for r in report["runs"]}
    first, second = runs["fp32"], runs["full_fp16"]
    convs = {name: {c["prefix"]: c for c in run["convolutions"]} for name, run in runs.items()}
    ordered = [c["prefix"] for c in first["convolutions"]]
    assert len(ordered) == 36 and set(ordered) == set(convs["full_fp16"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(16, 16))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.36, 1], hspace=.38, wspace=.30)
    ax = fig.add_subplot(grid[:, 0])
    y = np.arange(len(ordered))
    for offset, key, color, label in [(-.18, "fp32", "#2864A0", "FP32"),
                                    (.18, "full_fp16", "#D47B26", "Continuous FP16")]:
        vals = [convs[key][name]["gpu_seconds_per_decode"] for name in ordered]
        ax.barh(y+offset, vals, height=.34, color=color, label=label)
    ax.set_yticks(y, [short(name) for name in ordered], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GPU kernel time per decode (seconds)")
    ax.set_title("All 36 convolutions, in model order", loc="left", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(c["gpu_seconds_per_decode"] for c in first["convolutions"])*1.14)

    stages = ["input", "mid_pre", "mid_post", "up0", "up1", "up2", "up3", "output"]
    stageax = fig.add_subplot(grid[0, 1])
    sy = np.arange(len(stages))
    for offset, key, color in [(-.18, "fp32", "#2864A0"), (.18, "full_fp16", "#D47B26")]:
        run = runs[key]
        vals = [run["convolution_stage_subtotals"][stage]["convolution_gpu_seconds_per_decode"] for stage in stages]
        bars = stageax.barh(sy+offset, vals, height=.34, color=color)
        stageax.bar_label(bars, labels=[f"{v:.3f}s" for v in vals], padding=3, fontsize=9)
    stageax.set_yticks(sy, stages)
    stageax.invert_yaxis()
    stageax.set_xlim(0, 15.8)
    stageax.set_xlabel("GPU time in convolutions only (seconds)")
    stageax.set_title("Convolution subtotals by stage", loc="left", fontsize=14, weight="bold")
    stageax.grid(axis="x", alpha=.16)
    stageax.set_axisbelow(True)

    catkeys = [
        ("convolution_including_fused_bias_and_optional_residual", "Convolution + fused bias/residual"),
        ("matmul_including_attention_and_projections", "Attention/projection matmuls"),
        ("other_fusion_including_elementwise_norm_silu_residual", "Other fused operations"),
        ("reduction_kernel_including_norm_and_softmax", "Norm / softmax reductions"),
        ("layout_transpose_or_copy", "Layout transpose / copy"),
        ("device_copy_helper_kernel", "Device-copy helper kernels"),
        ("softmax_kernel", "Standalone softmax kernel"),
        ("unclassified_kernel", "Unclassified kernels")]
    catkeys = [(k, label) for k, label in catkeys if any(run["kernel_categories"][k]["events"] for run in runs.values())]
    catax = fig.add_subplot(grid[1, 1])
    cy = np.arange(len(catkeys))
    for offset, key, color in [(-.18, "fp32", "#2864A0"), (.18, "full_fp16", "#D47B26")]:
        cats = runs[key]["kernel_categories"]
        vals = [cats.get(k, {}).get("gpu_seconds_per_decode", 0)*1000 for k,_ in catkeys]
        catax.barh(cy+offset, vals, height=.34, color=color)
        for yy,(k,_),v in zip(cy+offset,catkeys,vals):
            if not v:
                continue
            label = f'{v:.3f} ms  ({cats[k]["percent_all_gpu_time"]:.3f}%)'
            if v > 10000:
                catax.text(v/1.15, yy, label, va="center", ha="right", color="white", fontsize=8)
            else:
                catax.text(v*1.15, yy, label, va="center", fontsize=8)
    catax.set_yticks(cy, [label for _,label in catkeys], fontsize=9)
    catax.invert_yaxis()
    catax.set_xscale("log")
    catax.set_xlim(.04, 120000)
    catax.set_xlabel("GPU kernel time (milliseconds; logarithmic scale)")
    catax.set_title("Operator categories in the complete graph", loc="left", fontsize=14, weight="bold")
    catax.grid(axis="x", which="major", alpha=.16)
    catax.set_axisbelow(True)
    fig.suptitle("MOSS VAE decode: 1024 x 1024, batch 1, GB203(SM120)", y=.98, fontsize=19, weight="bold")
    fig.text(.5,.948,"Actual whole-graph CUPTI trace | global XLA autotune=0 | same latent | one warmed decode per dtype",
             ha="center", fontsize=11)
    fig.text(.5,.924,
             f'All GPU kernels: FP32 {first["time_accounting"]["sum_kernel_seconds_per_decode"]:.3f}s; '
             f'FP16 {second["time_accounting"]["sum_kernel_seconds_per_decode"]:.3f}s. '
             "Convolutions consume 99.67% / 99.72%.",
             ha="center", fontsize=12, weight="bold")
    fig.subplots_adjust(left=.19, right=.985, top=.89, bottom=.10)
    fig.text(.19,.025,
             "Convolution timings include fused bias and, where present, residual addition. Other kernels are not attributed to individual layers.\n"
             "FP16 keeps BN, GN statistics and attention score/softmax accumulation in FP32. FP16 capture enabled cuDNN debug logging.\n"
             "Single traces locate bottlenecks; small timing differences are not a stable speedup claim. Raw reports: reports/vae_gpu_trace.json",
             fontsize=9, linespacing=1.5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    print(output)

if __name__ == "__main__":
    main()
