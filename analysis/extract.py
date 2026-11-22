import json, os, glob, csv

B = os.path.expanduser("~/nvme2T/moss-art-jax")
OUT = os.path.expanduser("~/nvme2T/cloveriris_export/analysis")
os.makedirs(OUT, exist_ok=True)

def jload(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return default

def run_cfg(d):
    for fn in ("config.json","effective_config.json"):
        c = jload(os.path.join(d,fn))
        if c: return c
    return {}

summary = {"runs": [], "evals": [], "distillation": [], "source": B}

for d in sorted(x for x in glob.glob(B+"/runs/*") if os.path.isdir(x)):
    name = os.path.basename(d); cfg = run_cfg(d)
    tr = cfg.get("training",{}) or {}; rew = cfg.get("reward",{}) or {}
    rec = {"run": name, "max_updates": tr.get("max_updates"),
           "P": tr.get("prompts_per_update"), "K": tr.get("rollouts_per_prompt"),
           "T": tr.get("training_points_per_trajectory"),
           "base": (cfg.get("model") or {}).get("base") or (cfg.get("model") or {}).get("path"),
           "lora": cfg.get("lora"), "reward": rew.get("models") or rew.get("weights")}
    ms = []
    mp = os.path.join(d,"training","metrics.jsonl")
    if os.path.exists(mp):
        for line in open(mp):
            line = line.strip()
            if not line: continue
            try: ms.append(json.loads(line))
            except Exception: pass
    rec["n_updates_logged"] = len(ms)
    if ms:
        rec.update({"reward_mean_first": ms[0].get("reward_mean"), "reward_mean_last": ms[-1].get("reward_mean"),
                    "loss_first": ms[0].get("loss"), "loss_last": ms[-1].get("loss"),
                    "grad_norm_last": ms[-1].get("grad_norm"), "lr_last": ms[-1].get("learning_rate"),
                    "optimizer_updates": ms[-1].get("optimizer_updates"),
                    "sum_grad_seconds": sum(m.get("gradient_update_seconds") or 0 for m in ms),
                    "reward_curve": [[m.get("update"), m.get("reward_mean")] for m in ms]})
    rec["checkpoints"] = [os.path.basename(c) for c in sorted(glob.glob(d+"/training/checkpoints/step_*"))]
    rec["complete"] = os.path.exists(d+"/training/complete.json")
    rec["n_png"] = len(glob.glob(d+"/**/*.png", recursive=True))
    rec["run_metadata"] = jload(d+"/run_metadata.json")
    summary["runs"].append(rec)

for f in sorted(glob.glob(B+"/fixed_eval_5060/evaluations/*/attempt_*/scoring/result.json")):
    r = jload(f) or {}
    ctx = r.get("scored_with_training_context") or {}
    sd = str(ctx.get("step_dir") or r.get("combined_archive") or "").rstrip("/")
    det = r.get("details") or []
    def mean(key):
        v = [ (x.get("raw_scores") or {}).get(key) for x in det ]
        v = [x for x in v if isinstance(x,(int,float))]
        return sum(v)/len(v) if v else None
    sc = r.get("scores") or []
    summary["evals"].append({"run": os.path.basename(sd), "step": r.get("source_step"),
        "n_images": len(det), "fused_mean": (sum(sc)/len(sc)) if sc else None,
        "kawai_mean": mean("kawai"), "deepghs_mean": mean("deepghs"), "hps_mean": mean("hps"),
        "status": r.get("status")})

for dd in sorted(x for x in glob.glob(B+"/distillation/*") if os.path.isdir(x)):
    rec = {"run": os.path.basename(dd)}
    pts = []
    for f in sorted(glob.glob(dd+"/metrics/*.json")):
        m = jload(f) or {}
        pts.append([m.get("step"), m.get("mse"), m.get("zero_offset_mse")])
    ok = [p for p in pts if isinstance(p[1],(int,float))]
    rec["n_metrics"] = len(pts)
    rec["first"] = pts[0] if pts else None
    rec["last"] = pts[-1] if pts else None
    if ok:
        rec["best"] = min(ok, key=lambda p: p[1])
        tail = ok[-50:]
        rec["mse_mean_tail50"] = sum(p[1] for p in tail)/len(tail)
        zs = [p[2] for p in tail if isinstance(p[2],(int,float))]
        rec["zero_mse_mean_tail50"] = sum(zs)/len(zs) if zs else None
    rec["cancelled"] = jload(dd+"/cancelled.json")
    rec["latest"] = jload(dd+"/latest.json")
    summary["distillation"].append(rec)

json.dump(summary, open(OUT+"/summary.json","w"), indent=1, ensure_ascii=False)

with open(OUT+"/runs_table.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","max_updates","P","K","T","n_updates_logged","complete","n_png","reward_first","reward_last","loss_first","loss_last","sum_grad_seconds"])
    for r in summary["runs"]:
        w.writerow([r["run"],r["max_updates"],r["P"],r["K"],r["T"],r["n_updates_logged"],r["complete"],r["n_png"],
                    r.get("reward_mean_first"),r.get("reward_mean_last"),r.get("loss_first"),r.get("loss_last"),r.get("sum_grad_seconds")])

with open(OUT+"/eval_scores.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","step","n_images","fused_mean","kawai_mean","deepghs_mean","hps_mean","status"])
    for e in sorted(summary["evals"], key=lambda x:(x["run"], x["step"] if x["step"] is not None else -1)):
        w.writerow([e["run"],e["step"],e["n_images"],e["fused_mean"],e["kawai_mean"],e["deepghs_mean"],e["hps_mean"],e["status"]])

with open(OUT+"/distill_metrics.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["run","step","mse","zero_offset_mse"])
    for dd in sorted(x for x in glob.glob(B+"/distillation/*") if os.path.isdir(x)):
        for p in sorted(glob.glob(dd+"/metrics/*.json")):
            m = jload(p) or {}
            w.writerow([os.path.basename(dd), m.get("step"), m.get("mse"), m.get("zero_offset_mse")])
print("summary written:", OUT)
