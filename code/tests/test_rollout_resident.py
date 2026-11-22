"""Inference residency contract without initializing an accelerator."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from moss_art.rollout_only import DEFAULTS,engine_config,_run_resident


def config():
    c=copy.deepcopy(DEFAULTS)
    c["model"]["base_precision"]="mxfp8"
    c["sampling"].update(vae_resident_rollout=True,height=32,width=32,steps=2)
    c["rollout"].update(count=5,prompts_per_batch=2,save_latents=True)
    return c


def test_resident_engine_uses_compatible_precision_source():
    c=config()
    assert c["sampling"]["rollout_mode"]=="bf16"  # Legacy default is not authoritative.
    e=engine_config(c)
    assert e["sampling"]["rollout_mode"]=="mxfp8"
    assert e["sampling"]["group_size"]==4
    assert e["sampling"]["prompts_per_rollout"]==1
    assert e["sampling"]["split_group_across_workers"]
    assert e["runtime"]["execution_backend"]=="mesh"
    assert c["runtime"]["execution_backend"]=="workers"  # Input is unchanged.


@pytest.mark.parametrize("change",["bf16","cpu","one_device","odd_devices","no_decode"])
def test_reject_incompatible_residency(change):
    c=config()
    if change=="bf16":c["model"]["base_precision"]="bf16"
    if change=="cpu":c["runtime"]["platform"]="cpu"
    if change=="one_device":c["runtime"]["devices"]=[0]
    if change=="odd_devices":c["runtime"]["devices"]=[0,1,2]
    if change=="no_decode":c["rollout"]["decode"]=False
    with pytest.raises(ValueError):engine_config(c)


@pytest.mark.parametrize("keep",[True,False])
@pytest.mark.parametrize("asynchronous",[True,False])
def test_resident_tail_streams_unique_jobs_and_matching_trajectories(tmp_path,monkeypatch,keep,asynchronous):
    from moss_art import runtime,prefetch
    c=config();c["rollout"]["cache_intermediates"]=keep
    c["preparation"]["async_prefetch"]=asynchronous
    prompt_file=tmp_path/"input.txt";prompt_file.write_text("cat\ndog\n")
    c["data"]["prompts"]=str(prompt_file)
    e=engine_config(c);out=tmp_path/"out";out.mkdir()
    rows=[dict(sample_id=str(i),prompt=t,negative_prompt="blurry") for i,t in enumerate(("cat","dog"))]
    sources=[SimpleNamespace(id=r["sample_id"]) for r in rows]
    events=[]
    class Buffer:
        def __init__(self,c,records):
            assert c["sampling"]["prompts_per_rollout"]==2
            self.buckets=SimpleNamespace(short=8,full=16)
        def __enter__(self):return self
        def __exit__(self,*args):events.append("qwen_closed")
        def submit(self,records):return (0,7)
        def acquire(self,ticket):return ({}, {})
        def release(self,ticket):events.append("qwen_released")
    class Pool:
        def __init__(self,c):
            assert c["sampling"]["prompts_per_rollout"]==1
            self.workers=list(range(4));self.calls=0
            self.worker=SimpleNamespace(cache=None,lora={},model=SimpleNamespace(
                base_precision="mxfp8",mxfp8_modules=252,base_storage_bytes=123,attention_backend="cudnn"))
        def __enter__(self):return self
        def __exit__(self,*args):events.append("mesh_closed")
        def _ensure_precision(self,precision):assert precision=="mxfp8"
        def ensure_rollout_vae(self):events.append("vae_loaded")
        def call(self,op,payloads=None):
            if op=="bind_conditions":return [dict(generation=7,copied=2)]*4
            if op=="clear_replay":self.worker.cache=None;events.append("clear");return
            assert op=="rollout"
            assert self.worker.cache is None
            expected=[0,4,5,9][self.calls]
            assert len(list(out.glob("p*.png")))==expected  # Saved before next sampling.
            self.calls+=1
            assert len({p["records"][0]["sample_id"] for p in payloads})==1
            seeds=[p["seeds"][0] for p in payloads]
            states=np.stack([np.full((2,128,2,2),seed,np.float32) for seed in seeds])
            self.worker.cache=dict(states=states,old=states+1,selected=[0,1],sigmas=np.array([1,.5,0])) if keep else None
            events.append(("rollout",seeds))
            return [dict(endpoints=np.full((1,128,2,2),seed,np.float32),actual_steps=2) for seed in seeds]
        def decode(self,endpoints):
            events.append("decode")
            return [dict(images=np.zeros((1,16,16,3),np.uint8),memory_stats={}) for _ in endpoints]
    monkeypatch.setattr(runtime,"configure_process",lambda c,devices:events.append(("devices",devices)))
    monkeypatch.setattr(prefetch,"DoubleBuffer",Buffer)
    monkeypatch.setitem(sys.modules,"moss_art.mesh_runtime",SimpleNamespace(MeshPool=Pool))
    _run_resident(c,e,sources,rows,{r["sample_id"]:i for i,r in enumerate(rows)},out)
    samples=[json.loads(s) for s in (out/"samples.jsonl").read_text().splitlines()]
    assert len(samples)==10 and len({s["seed"] for s in samples})==10
    assert len(list(out.glob("p*.png")))==10
    assert len(list(out.glob("*-trajectory.npz")))==(10 if keep else 0)
    for sample in samples:
        np.testing.assert_array_equal(np.load(out/sample["latent"]),np.full((128,2,2),sample["seed"],np.float32))
        if keep:
            with np.load(out/sample["trajectory"]) as trajectory:
                assert trajectory["states"].shape==(1,2,128,2,2)
                assert np.all(trajectory["states"]==sample["seed"])
                assert np.all(trajectory["old"]==sample["seed"]+1)
    assert events.count("vae_loaded")==1
    assert events.count("decode")==4 and events.count("clear")==4
    assert "mesh_closed" in events
    tails=[v[1] for v in events if isinstance(v,tuple) and v[0]=="rollout"][1::2]
    assert all(len(set(seeds))==1 for seeds in tails)
    info=json.loads((out/"run_info.json").read_text())
    assert info["execution_backend"]=="mesh"
    assert info["peak_wave_cache_bytes"]==4*128*2*2*4*(5 if keep else 1)
