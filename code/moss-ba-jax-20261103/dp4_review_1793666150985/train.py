"""Online bidirectional residual distillation with a frozen MOSS teacher."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from common import RUN, arrays, read, save, sha
from side import SideNet
from solver import gl_step, sample
from teacher import Teacher

def atomic_bytes(path, data):
    if shutil.disk_usage(RUN).free < 10*1024**3:
        raise OSError("NVMe reserve reached")
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,path)

def sample_interval(rng,t,deploy):
    sign = 1 if rng.random()<.5 else -1
    limit = t if sign>0 else 1-t
    if limit < 1e-5:
        sign = -sign
        limit = 1-limit
    if rng.random()<.5:
        length = min(limit,float(rng.choice(deploy))*float(rng.uniform(.25,1.)))
    else:
        length = -np.log1p(rng.random()*np.expm1(-12.*limit))/12.
    return np.float32(sign*length)

def train():
    cfg = read(RUN/"config.json")
    oracle = Teacher(cfg)
    net = SideNet(cfg["width"])
    condition = oracle.condition(cfg["order"][0])
    initial_x = oracle.noise(0)
    params = net.init(jax.random.key(cfg["seed"]),initial_x,initial_x,np.float32(.5),np.float32(.1),condition)["params"]
    opt = optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.linear_schedule(cfg["lr"]/32,cfg["lr"],31),weight_decay=.01))
    opt_state = opt.init(params)
    fingerprint = sha(RUN/"config.json")
    checkpoint = RUN/"latest.msgpack"
    state = dict(params=params,opt=opt_state,next_index=0,step=0,config_sha=fingerprint)
    if checkpoint.exists():
        state=serialization.from_bytes(state,checkpoint.read_bytes())
        if state["config_sha"] != fingerprint or state["step"]!=state["next_index"]*cfg["updates_per_prompt"]:
            raise ValueError("Incompatible checkpoint or cursor")
    params,opt_state,step = state["params"],state["opt"],int(state["step"])
    probes = jax.jit(jax.vmap(lambda p,x,v,t,h,c:net.apply({"params":p},x,v,t,h,c),
                              in_axes=(None,None,None,None,0,None)))
    def objective(p,x,v,t,h,c,target):
        delta = net.apply({"params":p},x,v,t,h,c)
        truth = jax.lax.stop_gradient(target-v)
        return jnp.mean(jnp.square(delta-truth)),jnp.mean(jnp.square(truth))
    grad = jax.jit(jax.value_and_grad(objective,has_aux=True))
    @jax.jit
    def update(p,st,g):
        updates,st = opt.update(g,st,p)
        return optax.apply_updates(p,updates),st,optax.global_norm(g)
    deploy = np.concatenate([-np.diff(oracle.grid(n)) for n in cfg["nfe"]])
    count = sum(int(v.size) for v in jax.tree.leaves(params))
    save(RUN/"student.json",dict(trainable_parameters=count,qwen_projection=[4096,cfg["width"]],
                                dtype="bf16_compute_fp32_master",base_precision=oracle.model.base_precision,
                                guidance_scale=cfg["sampling"]["guidance_scale"],chain=cfg["chain"]))
    evaluation_due = {4,32,128,512,2048,len(cfg["order"])*cfg["updates_per_prompt"]}
    def evaluate(current_step):
        folder = RUN/"evaluation"/f"{current_step:06d}"
        metrics=[]
        for i,plan in enumerate(cfg["eval_plan"]):
            ref_path=RUN/"latents/eval"/f"{i:05d}.npz"
            if not ref_path.exists(): return False
            with np.load(ref_path,allow_pickle=False) as f:
                pid,seed = str(f["prompt_id"]),int(f["seed"])
                assert pid==plan["prompt_id"] and seed==plan["seed"]
                ref=f["euler48"]
                c=oracle.condition(pid)
                output={"euler48":ref}
                for n in cfg["nfe"]:
                    baseline=f[f"euler{n}"]
                    start=time.monotonic()
                    x=sample(lambda x,t:oracle.velocity(c,x,t),lambda x,v,t,h:probes(params,x,v,t,h,c),
                             oracle.noise(seed),oracle.grid(n))
                    generated=np.asarray(x)
                    seconds=time.monotonic()-start
                    if not np.isfinite(generated).all():raise FloatingPointError("Nonfinite BA sample")
                    zero_file=RUN/"zero_side_control"/f"{i:02d}_{n}.npz"
                    if not zero_file.exists():
                        zero=sample(lambda x,t:oracle.velocity(c,x,t),
                                    lambda x,v,t,h:jnp.zeros((len(h),*x.shape),jnp.float32),
                                    oracle.noise(seed),oracle.grid(n))
                        arrays(zero_file,latent=np.asarray(zero))
                    with np.load(zero_file) as z:
                        zero_error=float(np.mean((z["latent"]-ref)**2))
                    metrics.append(dict(prompt_id=pid,seed=seed,nfe=n,dit_branch_forwards=2*n,side_queries=4*n-1,
                                        ba_mse=float(np.mean((generated-ref)**2)),
                                        zero_side_ba_mse=zero_error,euler_mse=float(np.mean((baseline-ref)**2)),
                                        ba_seconds_including_possible_compile=seconds))
                    output[f"ba{n}"]=generated
                    output[f"euler{n}"]=baseline
            arrays(folder/f"{i:02d}.npz",**output)
        save(folder/"metrics.json",metrics)
        return True

    def finish_evaluation(current_step):
        if current_step not in evaluation_due:
            return
        folder=RUN/"evaluation"/f"{current_step:06d}"
        weights=RUN/f"side_{current_step:06d}.msgpack"
        if not weights.exists():
            atomic_bytes(weights,serialization.to_bytes(params))
        if not (folder/"metrics.json").exists():
            while not (RUN/"latents/eval/00003.npz").exists():
                if (RUN/"producer_failure.json").exists():raise RuntimeError("Reference producer failed")
                time.sleep(2)
            evaluate(current_step)
        if not (folder/"gallery.jpg").exists():
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
            with (folder/"decode.log").open("a") as log:
                subprocess.Popen([sys.executable,str(Path(__file__).with_name("gallery.py")),str(folder)],
                                 env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

    finish_evaluation(step)
    for index in range(int(state["next_index"]),len(cfg["order"])):
        path=RUN/"latents/train"/f"{index:05d}.npz"
        while not path.exists():
            if (RUN/"producer_failure.json").exists():raise RuntimeError("Reference producer failed")
            time.sleep(2)
        with np.load(path,allow_pickle=False) as f:
            assert str(f["prompt_id"])==cfg["order"][index]
            clean=jnp.asarray(f["euler48"],jnp.float32)
        cond=oracle.condition(cfg["order"][index])
        logs=[]
        for repeat in range(cfg["updates_per_prompt"]):
            rng=np.random.default_rng(np.random.SeedSequence([cfg["seed"],index,repeat]))
            t=np.float32(rng.uniform(.001,.999))
            x=(1-t)*clean+t*oracle.noise(int(rng.integers(0,2**31)))
            v=oracle.velocity(cond,x,t)
            grads=None
            losses=[]; baselines=[]; intervals=[]
            for _ in range(cfg["chain"]):
                h=sample_interval(rng,t,deploy)
                xn=gl_step(lambda x,v,t,h:probes(params,x,v,t,h,cond),x,v,t,h)
                tn=np.float32(np.clip(t-h,0.,1.))
                vn=oracle.velocity(cond,xn,tn)
                (loss,base),g=grad(params,x,v,t,h,cond,vn)
                grads=g if grads is None else jax.tree.map(jnp.add,grads,g)
                losses.append(float(loss));baselines.append(float(base));intervals.append(float(h))
                x,v,t=xn,vn,tn
            grads=jax.tree.map(lambda g:g/len(losses),grads)
            params,opt_state,norm=update(params,opt_state,grads)
            if not np.isfinite(losses+baselines+[float(norm)]).all():
                raise FloatingPointError("Nonfinite SideNet loss or gradient")
            step+=1
            logs.append(dict(step=step,mse=float(np.mean(losses)),zero_offset_mse=float(np.mean(baselines)),
                             gradient_norm=float(norm),signed_intervals=intervals))
        save(RUN/"metrics"/f"{index:05d}.json",dict(prompt_id=cfg["order"][index],updates=logs))
        state=dict(params=params,opt=opt_state,next_index=index+1,step=step,config_sha=fingerprint)
        atomic_bytes(checkpoint,serialization.to_bytes(state))
        save(RUN/"latest.json",dict(step=step,processed_prompts=index+1,total_prompts=len(cfg["order"]),last=logs[-1],checkpoint=str(checkpoint)))
        print("committed",step,logs[-1],flush=True)
        finish_evaluation(step)
    save(RUN/"complete.json",dict(step=step,prompts=len(cfg["order"]),final_eval=str(RUN/"evaluation"/f"{step:06d}"/"metrics.json")))

