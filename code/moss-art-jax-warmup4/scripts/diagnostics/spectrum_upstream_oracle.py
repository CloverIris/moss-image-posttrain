"""Generate a tiny CPU Torch oracle from the pinned upstream forecaster class."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch

root = Path(__file__).resolve().parents[2]
source_dir = root / "inputs/spectrum_reference_20261025"
revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
assert revision == "1baeab6f9ee94a75011353c6ef41963888cc8979"
source = (source_dir / "moss.py").read_text()
tree = ast.parse(source)
forecaster = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Forecaster")
namespace = {"torch": torch}
exec("from __future__ import annotations\n" + ast.unparse(forecaster), namespace)
cases = []
for times in ([0, 1, 2, 3, 4, 5], [6, 9, 13, 18, 23], [1, 5, 11, 22, 30, 39]):
    for degree in (0, 1, 3):
        for blend in (0., .3, 1.):
            predictor = namespace["_Forecaster"](degree, .1, blend, 48, 100)
            predictor.t = list(times)
            target = times[-1] + 2
            weights = predictor.weights(target)
            cases.append(dict(times=times, target=target, total_steps=48, degree=degree,
                              ridge_lambda=.1, blend_weight=blend, weights=weights.tolist()))
output = root / "tests/spectrum_upstream_oracle.json"
output.write_text(json.dumps(dict(revision=revision, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                 cases=cases), indent=2) + "\n")
print(json.dumps(dict(output=str(output), cases=len(cases), revision=revision)))
