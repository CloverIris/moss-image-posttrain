import json, os, glob, csv
B = os.path.expanduser("~/nvme2T/moss-art-jax/distillation")
OUT = os.path.expanduser("~/nvme2T/cloveriris_export/analysis")

allrows = []
for dd in sorted(os.path.isdir(x) and x or "" for x in glob.glob(B+"/*")):
    if not dd: continue
    run = os.path.basename(dd)
    recs = []
    for f in sorted(glob.glob(dd+"/metrics/*.json")):
        d = json.load(open(f))
        for u in d.get("updates", []):
            recs.append({"run": run, "step": u.get("step"), "mse": u.get("mse"),
                         "zero": u.get("zero_offset_mse"), "grad": u.get("gradient_norm")})
    recs.sort(key=lambda r: r["step"] or 0)
    allrows.extend(recs)
    good = [r for r in recs if isinstance(r["mse"],(int,float))]
    print(f'== {run}: {len(good)} update records, step {good[0]["step"]}..{good[-1]["step"]}' if good else f'== {run}: 0')
    for label, sel in (("first50", good[:50]), ("last50", good[-50:])):
        if not sel: continue
        m = sum(r["mse"] for r in sel)/len(sel)
        z = sum(r["zero"] for r in sel)/len(sel)
        print(f'   {label}: mse={m:.5f} zero_offset_mse={z:.5f} improvement={(z-m)/z*100:+.2f}%  n={len(sel)}')
    # per-100 buckets
    for i in range(0, len(good), max(len(good)//6,1)):
        blk = good[i:i+max(len(good)//6,1)]
        print(f'   step {blk[0]["step"]:>4}-{blk[-1]["step"]:>4}: mse={sum(r["mse"] for r in blk)/len(blk):.5f} zero={sum(r["zero"] for r in blk)/len(blk):.5f}')

with open(OUT+"/distill_metrics.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["run","step","mse","zero_offset_mse","grad_norm"])
    for r in allrows: w.writerow([r["run"],r["step"],r["mse"],r["zero"],r["grad"]])
print("rows:", len(allrows))
