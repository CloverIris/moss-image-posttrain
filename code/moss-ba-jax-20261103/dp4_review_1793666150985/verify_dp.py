"""CPU-only protocol regression using real SideNet, AdamW and the four-rank loop."""
import os
from pathlib import Path
import tempfile
import numpy as np


class TinyTeacher:
    def grid(self,n):return np.linspace(1.,0.,n+1,dtype=np.float32)
    def noise(self,seed):
        import jax
        import jax.numpy as jnp
        return jax.random.normal(jax.random.key(seed),(1,2,4,4),dtype=jnp.float32)
    def condition(self,pid):
        import jax.numpy as jnp
        return (jnp.ones((1,4,8),jnp.float32)*(int(pid)+1),jnp.ones((1,4,8),jnp.float32),
                jnp.int32([2]),jnp.int32([3]))
    def velocity(self,c,x,t):return .2*x+t*t+.03*c[0].mean()


def fixture_rank(rank,uuids,parent):
    import dp_replica
    from common import read
    root=Path(os.environ["BA_TEST_ROOT"])
    dp_replica.RUN=root
    dp_replica.replica=dp_replica.Replica(read(root/"config.json"),TinyTeacher(),16)
    return rank


def test(root):
    import jax
    import optax
    import dp_train, dp_replica, dp_control, pool, train
    from flax import serialization
    from common import save,read,sha
    from side import SideNet
    root=Path(root); dp=root/"dp4"
    os.environ["BA_TEST_ROOT"]=str(root)
    dp_train.RUN=dp_control.RUN=train.RUN=dp_replica.RUN=root
    dp_train.DP=dp_control.DP=dp_replica.DP=dp
    (root/"pool").mkdir(); dp.mkdir()
    (root/"latents/train").mkdir(parents=True); (root/"latents/eval").mkdir()
    cfg=dict(order=list(map(str,range(7))),seed=2701,nfe=[8,12],updates_per_prompt=4,
             eval_plan=[dict(prompt_id="7",seed=87)],lr=.0003,width=16,chain=4)
    save(root/"config.json",cfg)
    assert not dp_control.coverage(cfg)
    assert dp_control.advance(cfg,{})=="sampling"
    save(root/"producer_complete.json",dict(training=7,eval=1,verified=True))
    assert dp_control.coverage(cfg)
    teacher=TinyTeacher()
    for i,pid in enumerate(cfg["order"]):
        np.savez(root/f"latents/train/{i:05d}.npz",prompt_id=pid,seed=cfg["seed"]+i,euler48=np.asarray(teacher.noise(i)).astype(np.float16))
    ev=np.asarray(teacher.noise(87))
    np.savez(root/"latents/eval/00000.npz",prompt_id="7",seed=87,euler48=ev,euler8=ev,euler12=ev)
    plan=dict(run_sha=sha(root/"config.json"),uuids=list(range(4)))
    save(dp/"plan.json",plan)
    # Exercise the watchdog's cutover without a CUDA allocation or real signals.
    from unittest.mock import patch
    dp_control.POOL=root/"pool"
    for i in range(5):save(dp_control.POOL/f"process_{i}.json",dict(pid=100+i))
    with patch.object(dp_control,"matches",side_effect=lambda m:m["pid"] in (104,200)),patch.object(dp_control,"devices",return_value={i:dict(uuid=i,used_mib=32) for i in range(4)}):
        save(dp/"coordinator.json",dict(pid=200))
        assert dp_control.advance(cfg,plan)=="preflight"
        save(dp/"ready.json",dict(pid=200))
        save(root/"latest.json",dict(step=4,processed_prompts=1))
        assert dp_control.advance(cfg,plan)=="waiting_for_commit"
        assert not (dp/"handoff.json").exists()
        (dp/"cancelled.json").touch()
        assert dp_control.advance(cfg,plan)=="done"
        (dp/"cancelled.json").unlink()
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.linear_schedule(cfg["lr"]/32,cfg["lr"],31),weight_decay=.01))
    x=teacher.noise(4)
    p=SideNet(16).init(jax.random.key(13),x,x,.5,.1,teacher.condition("0"))["params"]
    st=optimizer.init(p)
    for _ in range(4):
        delta,st=optimizer.update(jax.tree.map(lambda a:a*.01,p),st,p)
        p=optax.apply_updates(p,delta)
    source=root/"latest.msgpack"
    source.write_bytes(serialization.to_bytes(dict(params=p,opt=st,step=4,next_index=1,config_sha=plan["run_sha"])))
    save(dp/"handoff.json",dict(step=4,processed_prompts=1,checkpoint=str(source),checkpoint_sha256=sha(source),total_updates=12))
    for step in (4,12):
        folder=dp/"evaluation"/f"{step:06d}"; folder.mkdir(parents=True)
        (folder/"gallery.jpg").touch()  # The synthetic two-channel tensor has no MOSS VAE.
    dp_train.run(fixture_rank)
    latest=read(dp/"latest.json")
    p_dp,st_dp,step,cursor=dp_train.restore(Path(latest["checkpoint"]),optimizer)
    assert (step,cursor)==(12,7) and read(root/"complete.json")["step"]==12
    replica=dp_replica.replica
    seen=[]
    for group in ([1,2,3,4],[5,6]):
        rows=read(dp/"metrics"/f"{group[0]:05d}.json")
        assert rows["indices"]==group
        for repeat,logged in enumerate(rows["updates"]):
            assert [(v["index"],v["repeat"]) for v in logged["trajectories"]]==[(i,repeat) for i in group]
            seen.extend((i,repeat) for i in group)
            batch=[replica.trajectory(p,i,repeat)[0] for i in group]
            # Independent serial reduction, including the two-sample tail.
            g=jax.tree.map(lambda *v:np.mean(np.stack(v),axis=0),*batch)
            delta,st=optimizer.update(g,st,p); p=optax.apply_updates(p,delta)
    assert len(seen)==len(set(seen))==24
    for a,b in zip(jax.tree.leaves((p,st)),jax.tree.leaves((p_dp,st_dp)),strict=True):
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
    final_sha=latest["checkpoint_sha256"]
    # Resume from the previous published group and reproduce the final checksum.
    prior=dp/"checkpoint_1.msgpack"
    save(dp/"latest.json",dict(checkpoint=str(prior),checkpoint_sha256=sha(prior),step=8,processed_prompts=5))
    dp_train.run(fixture_rank)
    assert read(dp/"latest.json")["checkpoint_sha256"]==final_sha
    print("PASS: four CPU replicas vs serial batch mean, 24 distinct trajectories, short tail, Adam counts, checkpoint replay, evaluation/export",flush=True)


if __name__=="__main__":
    if os.environ.get("JAX_PLATFORMS")!="cpu":raise RuntimeError("Run this regression on CPU")
    with tempfile.TemporaryDirectory(prefix="ba-dp4-test-") as directory:test(directory)
