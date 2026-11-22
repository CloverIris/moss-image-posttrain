"""Generate the report figures from analysis CSVs."""

import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotlib import Canvas, PALETTE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(ROOT, "analysis")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SHORT = {
    "rank_sqrt_rerank_top1991_P24_K8_T4_1024_U200_mxfp8_calibrated_nol2_20261031": "rank-sqrt 1024 / 校准 FP8",
    "rank_sqrt_rerank_top1991_P24_K8_T4_1024_U200_nol2_20261031": "rank-sqrt 1024 / 未校准 FP8",
    "rank_sqrt_kawai_deepghs_w1_1_top1975v2_P24_K8_T4_1024_U200_calibrated_nol2_20261101_015727": "kawai+deepGHS 1024 / 校准",
    "reward3_top1991_P24_K8_T4_768_warmup4_20261029": "reward3 融合 768（完整 20 步）",
    "rank_sqrt_rerank_top1991_P24_K8_T4_768_U200_20261030_l2b1_from2": "rank-sqrt 768（46 步）",
    "reward2_top1991_P24_K8_T4_768_warmup4_U200_20261030": "reward2 768（2 步）",
    "reward3_P24_K8_T4_768_eval4x4_20261029": "reward3 768 eval4x4",
}

eval_rows = list(csv.DictReader(open(os.path.join(AN, "eval_scores.csv"), encoding="utf-8")))
by_run = collections.defaultdict(list)
for r in eval_rows:
    if r["fused_mean"]:
        by_run[r["run"]].append((int(r["step"]), float(r["fused_mean"])))

# ---------------------------------------------------------------- fig 1
order = [
    "rank_sqrt_rerank_top1991_P24_K8_T4_1024_U200_mxfp8_calibrated_nol2_20261031",
    "rank_sqrt_rerank_top1991_P24_K8_T4_1024_U200_nol2_20261031",
    "rank_sqrt_kawai_deepghs_w1_1_top1975v2_P24_K8_T4_1024_U200_calibrated_nol2_20261101_015727",
    "reward3_top1991_P24_K8_T4_768_warmup4_20261029",
    "rank_sqrt_rerank_top1991_P24_K8_T4_768_U200_20261030_l2b1_from2",
]
c = Canvas(1500, 850, title="固定 4 prompt × 4 seed 评估：融合奖励随优化器更新变化",
           xlabel="优化器更新步", ylabel="融合奖励（0–1）", margins=(135, 45, 78, 150))
c.set_scale(xlim=(0, 44), ylim=(0.15, 0.65))
c.frame()
legend = []
for i, run in enumerate(order):
    pts = sorted(by_run.get(run, []))
    if not pts:
        continue
    xs, ys = zip(*pts)
    col = c.line(xs, ys, color_idx=i, marker=5)
    legend.append((SHORT.get(run, run[:16]), col, "line"))
c.legend(legend, pos="tr")
c.save(os.path.join(FIG, "fig1-fused-reward-curves.png"))

# ---------------------------------------------------------------- fig 2
deltas = []
for run, pts in by_run.items():
    pts.sort()
    if len(pts) < 3:
        continue
    deltas.append((SHORT.get(run, run[:18]), pts[-1][1] - pts[0][1]))
deltas.sort(key=lambda x: x[1])
c = Canvas(1500, 620, title="末次固定评估相对首次的变化（Δ融合奖励）",
           xlabel="Δ 融合奖励", margins=(430, 60, 78, 110))
lo = min(d for _, d in deltas) - 0.03
hi = max(d for _, d in deltas) + 0.03
c.set_scale(xlim=(lo, hi), ylim=(0, len(deltas)), yticks=[])
c.frame()
colors = [(44, 160, 44) if d > 0 else (214, 39, 40) for _, d in deltas]
c.hbar([lab for lab, _ in deltas], [d for _, d in deltas], colors=colors, fmt="{:+.3f}")
c.save(os.path.join(FIG, "fig2-delta-fused.png"))

# ---------------------------------------------------------------- fig 3
runs = list(csv.DictReader(open(os.path.join(AN, "runs_table.csv"), encoding="utf-8")))
formal = [r for r in runs if r["max_updates"]]
rows3 = []
for r in sorted(formal, key=lambda r: (int(r["max_updates"]), int(r["n_updates_logged"] or 0))):
    rows3.append((SHORT.get(r["run"], r["run"][:26]),
                  int(r["n_updates_logged"] or 0), int(r["max_updates"])))
n = len(rows3)
c = Canvas(1500, 820, title="各正式 run 的更新预算与实跑步数（灰 = 预算上限，蓝 = 实际完成）",
           xlabel="优化器更新步数", margins=(470, 90, 78, 110))
c.set_scale(xlim=(0, 210), ylim=(0, n), yticks=[])
c.frame()
from plotlib import _font  # noqa: E402
for i, (lab, act, bud) in enumerate(rows3):
    y = n - (i + 0.5)
    py = c._py(y)
    hh = abs(c._py(n - (i + 0.5) + 0.33) - py)
    x0 = c._px(0)
    c.d.rectangle([x0, py - hh, c._px(bud), py + hh], fill=(214, 219, 227))
    c.d.rectangle([x0, py - hh, c._px(act), py + hh], fill=PALETTE[0])
    c.d.text((x0 - 12 * 2, py), lab, font=_font(13 * 2), fill=(50, 54, 60), anchor="rm")
    c.d.text((c._px(bud) + 8 * 2, py), f"{act}/{bud}", font=_font(12 * 2),
             fill=(60, 64, 70), anchor="lm")

c.save(os.path.join(FIG, "fig3-budget-vs-actual.png"))

# ---------------------------------------------------------------- fig 4
dist_rows = list(csv.DictReader(open(os.path.join(AN, "distill_metrics.csv"), encoding="utf-8")))
by_dist = collections.defaultdict(list)
for r in dist_rows:
    if r["mse"]:
        by_dist[r["run"]].append((int(r["step"]), float(r["mse"]), float(r["zero_offset_mse"] or 0)))


def rolling(seq, w=25):
    out, acc = [], []
    for i, v in enumerate(seq):
        acc.append(v)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


smoothed = {}
for run, pts in by_dist.items():
    pts.sort()
    smoothed[run] = ([p[0] for p in pts], rolling([p[1] for p in pts]), rolling([p[2] for p in pts]))

allv = [v for _, a, b in smoothed.values() for v in a + b]
c = Canvas(1500, 830, title="少步蒸馏：latent MSE 滑动平均（窗口 25 次更新）",
           xlabel="优化器更新步", ylabel="latent MSE", margins=(140, 45, 130, 150))
c.set_scale(xlim=(0, 1320), ylim=(min(allv) * 0.97, max(allv) * 1.03))
c.frame()
legend = []
for i, run in enumerate(sorted(smoothed)):
    xs, a, b = smoothed[run]
    name = "MOSS 形状 101.8M" if "smallmoss" in run else "SideNet v2"
    base = PALETTE[0] if i == 0 else PALETTE[2]
    light = tuple(int(255 - (255 - ch) * 0.55) for ch in base)
    c.line(xs, a, marker=0, width=3, color=base)
    c.line(xs, b, marker=0, width=2, dash=True, color=light)
    legend.append((f"{name}：学生修正后", base, "line"))
    legend.append((f"{name}：零修正基线", light, "line"))
c.legend(legend, pos="tr")
c.save(os.path.join(FIG, "fig4-distill-mse.png"))

# ---------------------------------------------------------------- fig 5
cost = []
for r in runs:
    n = int(r["n_updates_logged"] or 0)
    s = float(r["sum_grad_seconds"] or 0)
    if n >= 8 and s > 0:
        cost.append((SHORT.get(r["run"], r["run"][:22]), s / n / 60))
cost.sort(key=lambda x: x[1])
c = Canvas(1500, 560, title="单次优化器更新的平均耗时（含采样、奖励与反传）",
           xlabel="分钟 / 更新", margins=(470, 60, 78, 110))
c.set_scale(xlim=(0, max(v for _, v in cost) * 1.18), ylim=(0, len(cost)), yticks=[])
c.frame()
c.hbar([lab for lab, _ in cost], [v for _, v in cost], colors=[PALETTE[3]] * len(cost), fmt="{:.1f}")
c.save(os.path.join(FIG, "fig5-cost-per-update.png"))

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f, os.path.getsize(os.path.join(FIG, f)), "bytes")
