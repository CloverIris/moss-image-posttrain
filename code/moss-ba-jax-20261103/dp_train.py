"""Synchronous prompt data parallelism. Only rank 0 publishes optimizer state."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from common import RUN, read, save, sha, arrays, PYTHON
from dp_replica import DP, initialize, host_gradient, mean_gradient
from pool import lock


def budget(step,cursor,total):
    return step + 4*((total-cursor+3)//4)


def restore(path,opt):
    from flax import serialization
    import jax
    raw=path.read_bytes()
    untyped=serialization.msgpack_restore(raw)
    p=untyped["params"]
    template=dict(params=p,opt=opt.init(p),next_index=0,step=0,config_sha="")
    state=serialization.from_bytes(template,raw)
    if state["config_sha"]!=sha(RUN/"config.json"):raise ValueError("Checkpoint configuration differs")
    counts=[int(v) for v in jax.tree.leaves(state["opt"]) if np.shape(v)==() and np.issubdtype(np.asarray(v).dtype,np.integer)]
    if not counts or any(c!=state["step"] for c in counts):raise ValueError("Adam or schedule counter mismatch")
    return jax.device_put(state["params"]),jax.device_put(state["opt"]),int(state["step"]),int(state["next_index"])


def evaluate(step,rank,params,cfg):
    import jax.numpy as jnp
    from flax import serialization
    from train import atomic_bytes
    from solver import sample
    folder=DP/"evaluation"/f"{step:06d}"
    folder.mkdir(parents=True,exist_ok=True)
    weights=DP/f"side_{step:06d}.msgpack"
    if not weights.exists():atomic_bytes(weights,serialization.to_bytes(params))
    if not (folder/"metrics.json").exists():
        rows=[]
        for i,plan in enumerate(cfg["eval_plan"]):
            with np.load(RUN/"latents/eval"/f"{i:05d}.npz",allow_pickle=False) as refs:
                pid,seed=str(refs["prompt_id"]),int(refs["seed"])
                if (pid,seed)!=(plan["prompt_id"],plan["seed"]):raise ValueError("Fixed evaluation changed")
                ref=refs["euler48"]
                cond=rank.teacher.condition(pid)
                output={"euler48":ref}
                v=lambda x,t:rank.teacher.velocity(cond,x,t)
                probe=lambda x,v,t,h:rank.probes(params,x,v,t,h,cond)
                for n in cfg["nfe"]:
                    initial,grid=rank.teacher.noise(seed),rank.teacher.grid(n)
                    started=time.monotonic()
                    pred=np.asarray(sample(v,probe,initial,grid))
                    sec=time.monotonic()-started
                    zero=DP/"zero_side_control"/f"{i:02d}_{n}.npz"
                    if not zero.exists():
                        value=np.asarray(sample(v,lambda x,v,t,h:jnp.zeros((len(h),*x.shape),jnp.float32),initial,grid))
                        if not np.isfinite(value).all():raise FloatingPointError("Nonfinite BA control")
                        arrays(zero,latent=value)
                    with np.load(zero) as z:
                        zero_mse=float(np.mean((z["latent"]-ref)**2))
                    if not np.isfinite(pred).all():raise FloatingPointError("Nonfinite held-out prediction")
                    baseline=refs[f"euler{n}"]
                    rows.append(dict(prompt_id=pid,seed=seed,nfe=n,dit_branch_forwards=2*n,side_queries=4*n-1,
                                ba_mse=float(np.mean((pred-ref)**2)),euler_mse=float(np.mean((baseline-ref)**2)),
                                zero_side_ba_mse=zero_mse,ba_seconds_including_possible_compile=sec,device="GB203(SM120) GPU0"))
                    output[f"ba{n}"]=pred
                    output[f"euler{n}"]=baseline
            arrays(folder/f"{i:02d}.npz",**output)
        save(folder/"metrics.json",rows)
    if not (folder/"gallery.jpg").exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu")
        with (folder/"decode.log").open("a") as log:
            subprocess.Popen([PYTHON,str(Path(__file__).with_name("gallery.py")),str(folder)],env=env,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def run(replica_initializer=initialize):
    from pool import GPUS
    plan=read(DP/"plan.json")
    cfg=read(RUN/"config.json")
    done=read(RUN/"producer_complete.json")
    total=len(cfg["order"])
    if (done.get("verified") is not True or done["training"]!=total or done["eval"]!=len(cfg["eval_plan"])
        or plan["run_sha"]!=sha(RUN/"config.json") or cfg["updates_per_prompt"]!=4):
        raise ValueError("DP requires verified full reference coverage")
    uuids=plan["uuids"]
    with ExitStack() as stack:
        for gpu in GPUS:stack.enter_context(lock(RUN/"pool"/f"worker_{gpu}.lock",True))
        workers={}
        for gpu in (1,2,3):
            proc=stack.enter_context(ProcessPoolExecutor(1,mp_context=mp.get_context("spawn")))
            # Stagger model loading and mapping to bound host memory.
            if proc.submit(replica_initializer,gpu,uuids,os.getpid()).result(240)!=gpu:raise ValueError("Wrong rank")
            workers[gpu]=proc
        replica_initializer(0,uuids,os.getpid())
        import dp_replica
        import jax
        import optax
        from flax import serialization
        from train import atomic_bytes
        root=dp_replica.replica
        opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(
            optax.linear_schedule(cfg["lr"]/32,cfg["lr"],31),weight_decay=.01))
        @jax.jit
        def update(params,opt_state,g):
            delta,new_state=opt.update(g,opt_state,params)
            return optax.apply_updates(params,delta),new_state,optax.global_norm(g)
        previous=DP/"latest.json"
        source=Path(read(previous)["checkpoint"]) if previous.exists() else RUN/"latest.msgpack"
        p,st,_,_=restore(source,opt)
        host=jax.device_get(p)
        futures=[w.submit(host_gradient,host,0,0) for w in workers.values()]
        sample=[root.trajectory(p,0,0),*(f.result(240) for f in futures)]
        if not np.isfinite([v[k] for _,v in sample for k in ("mse","zero_offset_mse")]).all():
            raise FloatingPointError("Invalid four-card test loss")
        gs=[item[0] for item in sample]
        probe_p,probe_st,norm=update(p,st,mean_gradient(gs))
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((probe_p,probe_st,norm))):
            raise FloatingPointError("Invalid four-card test update")
        save(DP/"ready.json",dict(pid=os.getpid(),gpu_probe_mse=[v["mse"] for _,v in sample],
                                 gradient_norm=float(norm),source=str(source),test_update_discarded=True,time=time.time()))
        del host,p,st,sample,gs,probe_p,probe_st,futures
        while not (DP/"handoff.json").is_file():
            if (DP/"cancelled.json").exists() or (RUN/"complete.json").exists():return
            time.sleep(1)
        stack.enter_context(lock(RUN/"learner.lock",True))
        hand=read(DP/"handoff.json")
        current=read(previous) if previous.exists() else hand
        path=Path(current["checkpoint"])
        if sha(path)!=current["checkpoint_sha256"]:raise ValueError("Published checkpoint checksum mismatch")
        p,st,step,cursor=restore(path,opt)
        if (step,cursor)!=(current["step"],current["processed_prompts"]):raise ValueError("Published cursor mismatch")
        target=budget(hand["step"],hand["processed_prompts"],total)
        if step+4*((total-cursor+3)//4)!=target:raise ValueError("Prompt coverage or DP step budget mismatch")
        due={hand["step"],target}
        if hand["step"]<2048<=target:due.add(2048)
        if step in due:evaluate(step,root,p,cfg)
        while cursor<total:
            if (DP/"cancelled.json").is_file():return
            group=list(range(cursor,min(cursor+4,total)))
            logs=[]
            started=time.monotonic()
            for repeat in range(4):
                host=jax.device_get(p)
                pending=[workers[i].submit(host_gradient,host,index,repeat) for i,index in enumerate(group) if i]
                result=[root.trajectory(p,cursor,repeat),*(f.result(240) for f in pending)]
                grads=mean_gradient([g for g,_ in result])
                p,st,norm=update(p,st,grads)
                step+=1
                rows=[row for _,row in result]
                diagnostics=[float(norm)]+[v[k] for v in rows for k in ("mse","zero_offset_mse")]
                if not np.isfinite(diagnostics).all():raise FloatingPointError("Nonfinite DP update")
                logs.append(dict(step=step,mse=float(np.mean([r["mse"] for r in rows])),
                       zero_offset_mse=float(np.mean([r["zero_offset_mse"] for r in rows])),
                       gradient_norm=float(norm),trajectories=rows))
            slot=((step-hand["step"])//4)%2
            dest=DP/f"checkpoint_{slot}.msgpack"
            group_path=DP/"metrics"/f"{cursor:05d}.json"
            cursor+=len(group)
            state=dict(params=p,opt=st,next_index=cursor,step=step,config_sha=plan["run_sha"])
            atomic_bytes(dest,serialization.to_bytes(state))
            save(group_path,dict(indices=group,prompt_ids=[cfg["order"][i] for i in group],
                                 updates=logs,seconds=time.monotonic()-started))
            save(previous,dict(step=step,processed_prompts=cursor,total_prompts=total,total_updates=target,
                               last=logs[-1],checkpoint=str(dest),checkpoint_sha256=sha(dest),time=time.time()))
            if step in due:evaluate(step,root,p,cfg)
        save(DP/"complete.json",dict(step=step,prompts=cursor,final_eval=str(DP/"evaluation"/f"{step:06d}"/"metrics.json")))
        save(RUN/"complete.json",dict(read(DP/"complete.json"),backend="dp4",checkpoint=read(previous)["checkpoint"]))


if __name__=="__main__":
    try:run()
    except BaseException:
        save(DP/f"failure_{time.time_ns()}.json",dict(pid=os.getpid(),traceback=traceback.format_exc()))
        raise
