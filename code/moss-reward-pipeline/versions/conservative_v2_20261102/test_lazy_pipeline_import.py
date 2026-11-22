"""Exercise the actual delayed Kawai dispatch after wrapper sys.path cleanup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROBE = r'''
import importlib.util, json, pathlib, shutil, sys, types
source, root = map(pathlib.Path, sys.argv[1:3])
deployment = root / "isolated-deployment"
deployment.mkdir()
for name in ("run_reward_pipeline.py", "reward_fusion.py", "sdg_parallel.py"):
    shutil.copyfile(source / name, deployment / name)
(deployment / "score_kawai.py").write_text("""import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args()
cfg=json.loads(pathlib.Path(a.config).read_text())
rows=json.loads(pathlib.Path(cfg['input']).read_text())['items']
pathlib.Path(cfg['output']).write_text(json.dumps(dict(
 items=[dict(id=r['id'],kawai=float(r['fixture_score'])) for r in rows],
 metadata=dict(fixture=True,device=cfg['device']))))
""")
spec = importlib.util.spec_from_file_location(
    "isolated_wrapper", source / "checkpoint_eval" / "run_reward_with_validation.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
original_path = list(sys.path)
pipeline = wrapper.load_module(deployment / "run_reward_pipeline.py", "_isolated_pipeline")
assert sys.path == original_path, "wrapper must restore import path"
assert str(deployment) not in sys.path
assert not any(k.startswith("_moss_reward_parallel_") for k in sys.modules)
# A foreign cached module must not be selected over this version's helpers.
if len(sys.argv) > 3:
    sys.modules["sdg_parallel"] = types.SimpleNamespace(
        balanced_shards=lambda *_: (_ for _ in ()).throw(AssertionError("foreign helper used")))
items = []
for domain, groups, repeats in (("train",24,8),("eval",4,4)):
    for prompt in range(groups):
        for sample in range(repeats):
            items.append(dict(id=f"{domain}:{prompt}:{sample}",
                prompt_id=f"{domain}:{prompt}",fixture_score=len(items)/208))
archive = root / "archive"
archive.mkdir()
cfg = dict(kawai_python=sys.executable,kawai={},stage_timeout=15,kawai_devices=[0,1,2,3])
result = pipeline.run_kawai_parallel(items,cfg,archive)
assert [r["id"] for r in result["items"]] == [r["id"] for r in items]
assert [r["kawai"] for r in result["items"]] == [r["fixture_score"] for r in items]
assert result["metadata"]["shard_images"] == [52,52,52,52]
assert len(result["metadata"]["launched_jobs"]) == 4
# A completed shard retry must read durable outputs and launch no new worker.
again = pipeline.run_kawai_parallel(items,cfg,archive)
assert again["items"] == result["items"]
assert again["metadata"]["launched_jobs"] == []
assert sys.path == original_path, "lazy dispatch must not leak paths"
helpers = pipeline._parallel_helpers()
assert pathlib.Path(helpers.__file__).parent == deployment
helpers.run_sdg_parallel = lambda *a, **kw: (a,kw)
assert pipeline.run_sdg_parallel("probe", enabled=True) == (("probe",),dict(enabled=True))
assert not any(name.split(".")[0] in ("torch","jax","vllm") for name in sys.modules)
print(json.dumps(dict(images=len(items),shard_images=result["metadata"]["shard_images"],
    workers=len(result["metadata"]["launched_jobs"]),retry_workers=0,gpu_imports=False)))
'''


class LazyImportTests(unittest.TestCase):
    def invoke(self, poison=False):
        with tempfile.TemporaryDirectory() as directory:
            args = [sys.executable, "-I", "-c", PROBE,
                    str(Path(__file__).resolve().parent), directory]
            if poison:
                args.append("poison-global-module")
            result = subprocess.run(args, cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(record["images"], 208)
            self.assertEqual(record["workers"], 4)
            self.assertEqual(record["retry_workers"], 0)

    def test_delayed_real_kawai_dispatch_after_wrapper_path_restored(self):
        self.invoke()

    def test_version_local_helper_ignores_foreign_sys_modules(self):
        self.invoke(poison=True)


if __name__ == "__main__":
    unittest.main()
