import collections
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "analysis", "eval_scores.csv")
rows = list(csv.DictReader(open(p, encoding="utf-8")))
print("records:", len(rows))
by = collections.defaultdict(list)
for r in rows:
    by[r["run"]].append(r)

for run, es in sorted(by.items()):
    es.sort(key=lambda x: int(x["step"]))
    a, b = es[0], es[-1]

    def f(v):
        return float(v) if v else None

    def d(k):
        x, y = f(a[k]), f(b[k])
        return None if x is None or y is None else y - x

    dv = d("fused_mean")
    line = "{:<50s} n={:2d} {:>3}->{:<3} fused {:>7}->{:>7} d={}".format(
        run[:50], len(es), a["step"], b["step"],
        a["fused_mean"][:7], b["fused_mean"][:7],
        ("{:+.4f}".format(dv) if dv is not None else "n/a"))
    print(line)
