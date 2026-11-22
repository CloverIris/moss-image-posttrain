import copy
import numpy as np
from moss_art.config import load_config
from moss_art.runtime import WorkerPool


def test_pause_resume_preserves_art_replay_and_gradient():
    c = load_config("configs/smoke_cpu.yaml")
    c["runtime"].update(devices=[0],cpu_threads=1)
    c["sampling"].update(steps=2,train_timesteps=[0,1],group_size=2,height=32,width=32,skip_layers=[])
    records = [dict(sample_id="a",prompt="a red bird",negative_prompt="",metadata={})]
    with WorkerPool(c) as pool:
        outputs = pool.call("rollout",[dict(records=records,seeds=[1,2])])
        pool.call("prepare_update",[dict(rewards=np.array([-1.,1.]),seed=7)])
        before = pool.call("gradient",[dict(window=0)])[0]
        processes = [process for process,_ in pool.workers]
        pool.pause()
        assert not pool.workers and all(not process.is_alive() for process in processes)
        state = pool.saved_states[0]
        np.testing.assert_array_equal(state["cache"]["endpoints"],outputs[0]["endpoints"])
        assert all(isinstance(x,np.ndarray) for pair in state["cache"]["contexts"] for x in pair)
        pool.resume()
        pool.call("prepare_update",[dict(rewards=np.array([-1.,1.]),seed=7)])
        after = pool.call("gradient",[dict(window=0)])[0]
        assert before["loss_sum"] == after["loss_sum"]
        for key in before["gradients"]:
            np.testing.assert_array_equal(before["gradients"][key],after["gradients"][key])


def test_reward_runs_between_release_and_resume(monkeypatch):
    from moss_art.trainer import reward_round
    import moss_art.phases as phases
    import moss_art.rewards as rewards
    events = []
    class Pool:
        def pause(self): events.append("release_dit")
        def resume(self): events.append("restore_dit")
    def decode(c,values):
        events.append("vae")
        return [{"images":np.zeros((2,2,2,3),np.uint8)}]
    def score(*args):
        events.append("reward")
        return np.array([.1,.2],np.float32)
    monkeypatch.setattr(phases,"decode_phase",decode)
    monkeypatch.setattr(rewards,"score_images",score)
    c=load_config("configs/smoke_cpu.yaml")
    c["reward"].update(kind="toy",cpu_decode=False)
    reward_round(c,Pool(),[dict(endpoints=np.zeros((2,128,1,1),np.float32),prompts=["a","b"])])
    assert events == ["release_dit","vae","reward","restore_dit"]


def test_shared_cpu_base_has_one_readonly_backing(monkeypatch):
    import ml_dtypes
    from moss_art.host_weights import HostWeights,attach
    import vendor.moss_jax.weights as weights
    source={"layers.0.weight":np.arange(12,dtype=np.float32).reshape(3,4),"rope.bands":np.array([1.,2.],np.float32)}
    monkeypatch.setattr(weights,"load_host_params",lambda *_:source)
    owner=HostWeights("unused")
    try:
        left,a=attach(owner.descriptor)
        right,b=attach(owner.descriptor)
        try:
            assert a["layers.0.weight"].dtype==np.dtype(ml_dtypes.bfloat16)
            assert a["rope.bands"].dtype==np.float32
            assert not a["layers.0.weight"].flags.writeable
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            at,shape,dtype=owner.descriptor["entries"]["rope.bands"]
            writable=np.ndarray(shape,dtype=dtype,buffer=owner.memory.buf,offset=at)
            writable[0]=9
            assert a["rope.bands"][0]==b["rope.bands"][0]==9
            del writable
        finally:
            del a,b
            left.close();right.close()
    finally:owner.close()


def test_rollout_only_streams_all_steps_and_bounds_cache(tmp_path):
    import os,sys,json,subprocess
    prompts=tmp_path/"p.txt";prompts.write_text("a red bird")
    c={"data":{"prompts":str(prompts),"cache":str(tmp_path/"cache"),"short_context_length":4,"context_length":8},
       "model":{"kind":"tiny","dtype":"float32","attention_backend":"native","dit_autotune_level":0},
       "sampling":{"height":32,"width":32,"steps":3,"skip_layers":[]},
       "rollout":{"count":3,"decode":False,"save_latents":True,"cache_intermediates":True},
       "runtime":{"platform":"cpu","devices":[0,1],"cpu_threads":1,"output_dir":str(tmp_path/"out")}}
    path=tmp_path/"c.json";path.write_text(json.dumps(c))
    env=dict(os.environ,JAX_PLATFORMS="cpu",CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1")
    run=subprocess.run([sys.executable,"run.py","rollout","--config",str(path)],env=env,capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    events=[json.loads(x) for x in run.stdout.splitlines() if x.startswith("{")]
    names=[e.get("event") for e in events]
    assert names.count("dit_host_cache_progress")==2
    assert names.index("dit_released")>max(i for i,n in enumerate(names) if n=="dit_host_cache_progress")
    assert names.index("rollout_only_progress")>names.index("dit_released")
    files=list((tmp_path/"out").glob("*-trajectory.npz"))
    assert len(files)==3
    for f in files:
        with np.load(f) as trajectory:
            assert trajectory["states"].shape==trajectory["old"].shape==(1,3,8,2,2)
            np.testing.assert_array_equal(trajectory["selected"],[0,1,2])
            assert np.isfinite(trajectory["states"]).all()
    c["runtime"]["output_dir"]=str(tmp_path/"limited")
    c["rollout"]["host_cache_limit_gib"]=1e-12
    path.write_text(json.dumps(c))
    run=subprocess.run([sys.executable,"run.py","rollout","--config",str(path)],env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode!=0 and "Host trajectory cache needs" in run.stderr
    assert "rollout_workers_ready" not in run.stdout
