import copy
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from moss_art.config import DEFAULTS
from moss_art.worker import NativeWorker
from moss_art.trainer import paired_replay_payloads, aggregate_gradients
from moss_art.art import covariance, prompt_windows

class Policy:
    config={"latent_channels":2}
    def predict(self,a,x,sigma,context,**kw):
        return .15*x + a["theta"]*jnp.tanh(x+context.mean()) + .03*jnp.float32(sigma)**2
    def predict_with_pullback(self,a,x,sigma,context,**kw):
        v,pb=jax.vjp(lambda aa:self.predict(aa,x,sigma,context,**kw),a)
        return v,lambda cot:pb(cot)[0]

def worker(c):
    from collections import OrderedDict
    w=NativeWorker.__new__(NativeWorker);w.c=copy.deepcopy(c);w.model=Policy();w.device=jax.devices()[0]
    w.lora={"theta":jnp.asarray(.02,jnp.float32)};w.cache=None;w.text_cache=OrderedDict();w._compiled_text_lengths=set();w.buckets=None
    w._conditioning=lambda rec:(jnp.ones((1,3,1),jnp.float32)*.3,jnp.zeros((1,2,1),jnp.float32))
    return w

class Pool:
    def __init__(self,ws):self.workers=ws
    def call(self,op,ps=None):return [getattr(w,op)(**p) for w,p in zip(self.workers,ps or [{} for _ in self.workers])]

def test_paired_cache_distribution_preserves_both_optimizer_updates():
    c=copy.deepcopy(DEFAULTS);c["sampling"].update(height=32,width=32,steps=4,train_timesteps=[0,1,2,3],group_size=4,prompts_per_rollout=1,split_group_across_workers=True,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[])
    record={"sample_id":"synthetic","prompt":"synthetic fixture","negative_prompt":""}
    ws=[worker(c) for _ in range(4)];pool=Pool(ws)
    outs=[w.rollout([record],[100+i],samples_per_prompt=1) for i,w in enumerate(ws)]
    endpoints=np.concatenate([o["endpoints"] for o in outs]);groups=np.zeros(4,np.int32)
    stats={k:np.asarray(v) for k,v in covariance(endpoints,np.array([.1,.4,.7,.95],np.float32),groups,group_size=4).items()}
    windows=prompt_windows(groups,4,seed=42)
    ref=worker(c);ref.rollout([record],[100,101,102,103]);ref.prepare_update(stats=stats,windows=windows)
    payloads=paired_replay_payloads(pool,groups,stats,windows);prep=pool.call("prepare_update",payloads)
    assert [p["items_per_window"] for p in prep]==[[2,2]]*4
    tx=optax.adamw(.004);p={"theta":jnp.asarray(.02,jnp.float32)};st=tx.init(p)
    for window in range(2):
        actual,metrics=aggregate_gradients(pool.call("gradient",[{"window":window}]*4))
        expected,reference_metrics=aggregate_gradients([ref.gradient(window)])
        np.testing.assert_allclose(actual["theta"],expected["theta"],rtol=3e-6,atol=3e-7)
        assert metrics["items"]==reference_metrics["items"]==8
        np.testing.assert_allclose(metrics["loss"],reference_metrics["loss"],rtol=3e-6)
        delta,st=tx.update(actual,st,p);p=optax.apply_updates(p,delta)
        for w in ws+[ref]:w.set_lora(p)
