import json, os, glob, csv, collections

B = os.path.expanduser("~/nvme2T/moss-art-jax")
OUT = os.path.expanduser("~/nvme2T/cloveriris_export/analysis")

def jload(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return default

rows = []
seen = set()
for f in sorted(glob.glob(B+"/**/scoring/result.json", recursive=True)):
    r = jload(f) or {}
    parts = f.split(os.sep)
    run = "ROOT_fixed_eval_5060"
    if "runs" in parts:
        i = parts.index("runs")
        if i+1 < len(parts): run = parts[i+1]
    det = r.get("details") or []
    sc = r.get("scores") or []
    if not det: continue
    key = (run, r.get("source_step"))
    if key in seen: continue
    seen.add(key)
    def m(k):
        v = [(x.get("raw_scores") or {}).get(k) for x in det]
        v = [x for x in v if isinstance(x,(int,float))]
        return sum(v)/len(v) if v else None
    rows.append({"run": run, "step": r.get("source_step"), "n": len(det),
                 "fused": (sum(sc)/len(sc)) if sc else None,
                 "kawai": m("kawai"), "deepghs": m("deepghs"), "hps": m("hps"),
                 "status": r.get("status")})
rows.sort(key=lambda x: (x["run"], x["step"] if x["step"] is not None else -1))

with open(OUT+"/eval_scores.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["run","step","n_images","fused_mean","kawai_mean","deepghs_mean","hps_mean","status"])
    for e in rows:
        w.writerow([e["run"],e["step"],e["n"],e["fused"],e["kawai"],e["deepghs"],e["hps"],e["status"]])

json.dump(rows, open(OUT+"/eval_scores.json","w"), indent=1)

print(f"{'run':58s} {'n_eval':>6} {'steps':>18} {'fused0':>8} {'fusedN':>8} {'d_fused':>8} {'d_kawai':>8} {'d_deepghs':>9} {'d_hps':>8}")
by = collections.defaultdict(list)
for e in rows: by[e["run"]].append(e)
for run, es in sorted(by.items()):
    es.sort(key=lambda x: x["step"])
    a, b = es[0], es[-1]
    def d(k): 
        return (b[k]-a[k]) if isinstance(a[k],(int,float)) and isinstance(b[k],(int,float)) else None
    fmt = lambda v: f"{v:8.4f}" if isinstance(v,(int,float)) else "     n/a"
    dfmt = lambda v: f"{v:+8.4f}" if isinstance(v,(int,float)) else "     n/a"
    print(f"{run[:58]:58s} {len(es):>6} {str(a['step'])+'->'+str(b['step']):>18} {fmt(a['fused'])} {fmt(b['fused'])} {dfmt(d('fused'))} {dfmt(d('kawai'))} {dfmt(d('deepghs'))} {dfmt(d('hps'))}")
print()
print("total eval records:", len(rows))
