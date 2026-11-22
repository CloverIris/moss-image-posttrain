"""Frozen full-DiT BF16 oracle. Both branches consume the existing cached Qwen tokens."""
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from common import RUN, NATIVE, arrays, read, save, sha

sys.path.insert(0, str(NATIVE))
import jax
import jax.numpy as jnp
from moss_art.conditioning import pad_context
from moss_art.data import PromptCache, load_prompts
from moss_art.model import MossModel
from moss_art.sampling import schedule

class Teacher:
    def __init__(self, cfg):
        self.cfg = cfg
        assert cfg["sampling"]["rollout_mode"] == "bf16"
        assert cfg["train_sha"] == sha(cfg["train"]) and cfg["eval_sha"] == sha(cfg["eval"])
        assert cfg["cache_manifest_sha"] == sha(Path(cfg["cache"])/"manifest.json")
        self.records = {r.id:r for path in (cfg["train"],cfg["eval"]) for r in load_prompts(path)}
        self.cache = PromptCache(cfg["cache"], model_dir=cfg["model"])
        self.model = MossModel.from_pretrained(cfg["model"], base_precision="bf16", attention_backend="cudnn",
                                              dit_autotune_level=4, device=jax.devices()[0])
        assert self.model.config["text_dim"] == cfg["qwen_width"]
        print("teacher_ready", self.model.config, jax.devices(), flush=True)

    def condition(self, prompt_id):
        cached = self.cache.load(self.records[prompt_id])
        size = max(cached[k].shape[1] for k in ("positive","negative"))
        capacity = 288 if size <= 288 else 1024
        pos, lp = pad_context(cached["positive"], capacity)
        neg, ln = pad_context(cached["negative"], capacity)
        return tuple(jnp.asarray(v) for v in (pos,neg,lp,ln))

    def velocity(self, cond, x, t):
        pos,neg,lp,ln = cond
        return self.model.predict_base(x,jnp.float32(t),pos,cfg_scale=self.cfg["sampling"]["guidance_scale"],
                                       negative_context=neg,negative_skip_layers=tuple(self.cfg["sampling"]["skip_layers"]),
                                       context_lengths=lp,negative_context_lengths=ln)

    def grid(self,n):
        return schedule(n,scheduler="moss",shift=self.cfg["sampling"]["shift"])[1]

    def noise(self,seed):
        side = self.cfg["sampling"]["resolution"]//16
        return jax.random.normal(jax.random.PRNGKey(seed),(1,128,side,side),dtype=jnp.float32)

    def euler(self,cond,seed,n):
        grid = self.grid(n)
        x = self.noise(seed)
        for t,end in zip(grid,grid[1:]):
            x = x + np.float32(end-t)*self.velocity(cond,x,t)
        return x

def produce():
    cfg = read(RUN/"config.json")
    oracle = Teacher(cfg)
    jobs = [("train",i,p,20261103+i) for i,p in enumerate(cfg["order"])]
    heldout = [("eval",i,p["prompt_id"],p["seed"]) for i,p in enumerate(cfg["eval_plan"])]
    jobs[1:1] = heldout
    for split,index,pid,seed in jobs:
        path = RUN/"latents"/split/f"{index:05d}.npz"
        if path.exists():
            with np.load(path) as saved:
                assert str(saved["prompt_id"]) == pid and int(saved["seed"]) == seed
            continue
        start = time.monotonic()
        cond = oracle.condition(pid)
        values = {}
        for n in ([48,8,12] if split=="eval" else [48]):
            x = np.asarray(oracle.euler(cond,seed,n))
            if not np.isfinite(x).all():
                raise FloatingPointError("Nonfinite reference")
            values[f"euler{n}"] = x.astype(np.float32 if split=="eval" else np.float16)
        arrays(path,prompt_id=np.array(pid),seed=np.array(seed),**values)
        save(RUN/"producer.json",dict(split=split,index=index,prompt_id=pid,seed=seed,seconds=time.monotonic()-start))
        print(split,index,pid,round(time.monotonic()-start,2),flush=True)
    save(RUN/"producer_complete.json",dict(training=len(cfg["order"]),eval=len(heldout)))

