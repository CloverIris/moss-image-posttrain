import copy
import json
import os
from pathlib import Path
from multiprocessing import shared_memory
import subprocess
import sys
import time
import numpy as np
import pytest
from moss_art.config import DEFAULTS,validate
from moss_art.prefetch import DoubleBuffer,TinyEncoder,plan_records,read_buffer,READY,READING

class SlowEncoder(TinyEncoder):
    def encode(self,record,branch):
        value,new=super().encode(record,branch)
        if new:time.sleep(.12)
        return value,new

class BrokenEncoder:
    def __init__(self,*args):raise ValueError("intentional encoder failure")

class CopyPool:
    workers=[None,None]
    def __init__(self):self.copies=[]
    def call(self,op,payloads):
        assert op=="bind_conditions"
        self.copies=[read_buffer(p["descriptor"]) for p in payloads]
        return [dict(copied=len(x),generation=p["descriptor"]["generation"]) for x,p in zip(self.copies,payloads)]


def config(tmp_path):
    path=tmp_path/"prompts.jsonl"
    path.write_text("".join(json.dumps(dict(id=str(i),prompt=f"image {i}"))+"\n" for i in range(8)))
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0)
    c["data"].update(prompts=str(path),cache=str(tmp_path/"cache"),short_context_length=4,context_length=8)
    c["preparation"]["async_prefetch"]=True
    c["runtime"].update(platform="cpu",devices=[0],cpu_threads=1,worker_timeout=60,output_dir=str(tmp_path/"run"))
    c["sampling"].update(height=32,width=32,steps=2,train_timesteps=[0,1],prompts_per_rollout=2,skip_layers=[],guidance_scale=1.)
    c["training"].update(max_updates=4,checkpoint_every=2,evaluate_every=0)
    c["reward"].update(kind="toy_latent",decode=False)
    return validate(c)


def test_double_buffer_overlap_ack_reuse_and_cache(tmp_path):
    c=config(tmp_path);rows=plan_records(c);pool=CopyPool()
    with DoubleBuffer(c,rows,factory=SlowEncoder) as b:
        names=[h.name for h in b.handles]
        first=b.submit(rows[:2]);b.bind(pool,rows[:2],first)
        original=[[tuple(x.copy() for x in pair) for pair in worker] for worker in pool.copies]
        second=b.submit(rows[2:4]);assert second[0]!=first[0]
        compute_start=time.monotonic();time.sleep(.6);compute_end=time.monotonic()
        assert b.states[second[0]]==READY
        descriptor,event=b.acquire(second)
        assert max(compute_start,event["fill_started"])<min(compute_end,event["fill_finished"])
        assert b.states[second[0]]==READING
        third=b.submit(rows[4:6])
        with pytest.raises(RuntimeError,match="Both"):b.submit(rows[6:8])
        # First consumer copies survive multiple producer writes.
        for workers,want in zip(pool.copies,original):
            for pair,expected in zip(workers,want):
                for x,y in zip(pair,expected):np.testing.assert_array_equal(x,y)
        b.release(second)
        b.bind(pool,rows[4:6],third)
        again=b.submit(rows[:2]);_,event=b.acquire(again)
        assert event["encoded"]==0 and event["cache_hits"]==4
        b.release(again)
    for name in names:
        with pytest.raises(FileNotFoundError):shared_memory.SharedMemory(name=name)


def test_stale_generation_and_double_release_rejected(tmp_path):
    c=config(tmp_path);rows=plan_records(c)
    with DoubleBuffer(c,rows) as b:
        ticket=b.submit(rows[:2]);descriptor,_=b.acquire(ticket)
        expected=read_buffer(descriptor)
        b.release(ticket)
        with pytest.raises(RuntimeError,match="release"):b.release(ticket)
        other=b.submit(rows[2:4]);b.acquire(other);b.release(other)
        reuse=b.submit(rows[4:6]);b.acquire(reuse)
        with pytest.raises(RuntimeError,match="Stale"):b.acquire(ticket)
        with pytest.raises(RuntimeError,match="generation"):read_buffer(descriptor)
        assert expected[0][0].shape==(1,3,24)
        b.release(reuse)


def test_producer_failure_and_timeout_propagate(tmp_path):
    c=config(tmp_path);rows=plan_records(c)
    with pytest.raises(RuntimeError,match="intentional encoder failure"):
        DoubleBuffer(c,rows,factory=BrokenEncoder)
    with DoubleBuffer(c,rows,factory=SlowEncoder) as b:
        ticket=b.submit(rows[:2]);b.timeout=.03
        with pytest.raises(TimeoutError):b.acquire(ticket)
        b.timeout=60;b.acquire(ticket);b.release(ticket)
        ticket=b.submit(rows[2:4]);b.process.kill();b.process.join()
        with pytest.raises(RuntimeError,match="exited"):b.acquire(ticket)


def test_config_rejects_shared_encoder_training_gpu(tmp_path):
    c=config(tmp_path);c["runtime"]["platform"]="cuda";c["runtime"]["devices"]=[4]
    with pytest.raises(ValueError,match="separate"):validate(c)


def run_cli(c,path,op="train"):
    path.write_text(json.dumps(c))
    result=subprocess.run([sys.executable,"-m","moss_art.cli",op,"--config",str(path)],
        env={**os.environ,"JAX_PLATFORMS":"cpu","CUDA_VISIBLE_DEVICES":""},capture_output=True,text=True,timeout=180)
    path.with_suffix(".log").write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr


def test_async_training_matches_sync_and_resumes_with_evaluation(tmp_path):
    c=config(tmp_path);c["runtime"]["devices"]=[0,1];c["training"]["evaluate_every"]=2
    c["runtime"]["output_dir"]=str(tmp_path/"async")
    run_cli(c,tmp_path/"async.json")
    # Compare same seeds and branches against the original worker path.
    synchronous=copy.deepcopy(c);synchronous["preparation"]["async_prefetch"]=False
    synchronous["runtime"]["output_dir"]=str(tmp_path/"sync")
    run_cli(synchronous,tmp_path/"sync.json")
    from moss_art.checkpoint import load_checkpoint
    a=load_checkpoint(tmp_path/"async/checkpoints/step_00000004")
    b=load_checkpoint(tmp_path/"sync/checkpoints/step_00000004")
    import jax
    for x,y in zip(jax.tree.leaves(a[:2]),jax.tree.leaves(b[:2])):np.testing.assert_array_equal(x,y)
    split=copy.deepcopy(c);split["training"]["max_updates"]=2;split["runtime"]["output_dir"]=str(tmp_path/"resumed")
    run_cli(split,tmp_path/"split.json")
    split["training"]["max_updates"]=4;split["run"]["resume"]=str(tmp_path/"resumed")
    run_cli(split,tmp_path/"resume.json")
    r=load_checkpoint(tmp_path/"resumed/checkpoints/step_00000004")
    for x,y in zip(jax.tree.leaves(a[:2]),jax.tree.leaves(r[:2])):np.testing.assert_array_equal(x,y)
    for key in ("update","rollout","prompt_cursor"):assert a[2][key]==r[2][key]
    ev=copy.deepcopy(c);ev["runtime"]["output_dir"]=str(tmp_path/"eval")
    ev["run"]["checkpoint"]=str(tmp_path/"async/checkpoints/step_00000004")
    run_cli(ev,tmp_path/"eval.json","evaluate")
    expected=json.loads((tmp_path/"async/evaluation/step_00000004/metrics.json").read_text())
    actual=json.loads((tmp_path/"eval/evaluation/step_00000004/metrics.json").read_text())
    assert expected==actual


def test_no_signal_prefetch_advances_without_deadlock(tmp_path,monkeypatch):
    c=config(tmp_path);c["training"]["max_no_signal_rollouts"]=2
    import moss_art.trainer as trainer
    monkeypatch.setattr(trainer,"reward_round",lambda c,pool,outputs:(np.zeros(sum(len(x["endpoints"]) for x in outputs),np.float32),None))
    with pytest.raises(RuntimeError,match="no-signal"):
        trainer.train(c)
    events=[json.loads(s) for s in (tmp_path/"run/metrics.jsonl").read_text().splitlines()]
    assert [e["rollout"] for e in events]==[1,2]


def test_incremental_cache_does_not_change_async_resume_identity(tmp_path,monkeypatch):
    import moss_art.assets as assets
    c=config(tmp_path);c["model"]["kind"]="moss"
    monkeypatch.setattr(assets,"component_identity",lambda *a:{"fixture":"weights"})
    monkeypatch.setattr("moss_art.data.model_identity",lambda *a,**kw:{"fingerprint":"encoder-a"})
    before=assets.experiment_identity(c)
    directory=Path(c["data"]["cache"]);directory.mkdir()
    (directory/"manifest.json").write_text(json.dumps({"entries":{"new":"features"}}))
    assert assets.experiment_identity(c)==before
    monkeypatch.setattr("moss_art.data.model_identity",lambda *a,**kw:{"fingerprint":"encoder-b"})
    assert assets.experiment_identity(c)["fingerprint"]!=before["fingerprint"]


def test_missing_consumer_ack_does_not_release_slot(tmp_path):
    class WrongAck(CopyPool):
        def call(self,op,payloads):
            result=super().call(op,payloads)
            result[0]["generation"]=-1
            return result
    c=config(tmp_path);rows=plan_records(c)
    with DoubleBuffer(c,rows) as b:
        ticket=b.submit(rows[:2])
        with pytest.raises(RuntimeError,match="acknowledgement"):
            b.bind(WrongAck(),rows[:2],ticket)
        assert b.states[ticket[0]]==READING
