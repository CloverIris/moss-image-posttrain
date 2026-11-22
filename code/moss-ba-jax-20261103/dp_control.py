"""Arm a single, reference-complete transition for the existing BA run."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import PYTHON, RUN, read, save, sha
from pool import GPUS, POOL, devices, identity, lock, matches, retire
from dp_replica import DP

HERE=Path(__file__).resolve().parent


def start(script, args, name, env):
    command=[PYTHON,str(HERE/script),*args]
    with (DP/f"{name}.log").open("a") as log:
        child=subprocess.Popen(command,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    born=identity(child.pid)
    if born is None:raise RuntimeError(f"{name} exited at launch")
    receipt=dict(pid=child.pid,birth=born[0],command=command,time=time.time())
    save(DP/f"{name}.json",receipt)
    return receipt


def coverage(cfg):
    path=RUN/"producer_complete.json"
    if not path.exists():return False
    r=read(path)
    if r.get("verified") is not True or (r["training"],r["eval"])!=(len(cfg["order"]),len(cfg["eval_plan"])):
        raise ValueError("Reference completion record is incomplete")
    return True


def advance(cfg,plan):
    if (DP/"cancelled.json").exists() or (RUN/"complete.json").exists():return "done"
    if not coverage(cfg):return "sampling"
    with lock(RUN/"launch.lock"):
        if (DP/"cancelled.json").exists() or (RUN/"complete.json").exists():return "done"
        source=DP/"coordinator.json"
        process=read(source) if source.is_file() else None
        handoff=DP/"handoff.json"
        legacy=read(POOL/"process_4.json")
        retired=DP/"learner_retired.json"
        if not handoff.exists() and not retired.exists() and not matches(legacy):
            raise RuntimeError("The original GB206(SM120) learner is no longer live; check stop intent")
        if not process or not matches(process):
            if process:raise RuntimeError("DP coordinator exited; inspect logs and use recover after resolving the cause")
            for gpu in GPUS:
                if matches(read(POOL/f"process_{gpu}.json")):return "draining"
            hw=devices()
            if any(hw[i]["uuid"]!=plan["uuids"][i] for i in GPUS):raise RuntimeError("GPU UUID mapping changed")
            if any(hw[i]["used_mib"]>=300 for i in GPUS):return "waiting_for_free_cards"
            pinned=dict(read(RUN/"training_sources.json"),**plan["sources"])
            if any(sha(k)!=v for k,v in pinned.items()):raise ValueError("Pinned DP or training code differs")
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan["uuids"][0],CUDA_DEVICE_ORDER="PCI_BUS_ID",
                     JAX_PLATFORMS="cuda",JAX_ENABLE_X64="true",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     JAX_COMPILATION_CACHE_DIR=str(RUN/"jax_cache"),OPENBLAS_NUM_THREADS="2",
                     OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1")
            start("dp_train.py",[],"coordinator",env)
            return "preflight"
        if not (DP/"ready.json").is_file() or read(DP/"ready.json")["pid"]!=process["pid"]:return "preflight"
        if handoff.exists():
            latest=DP/"latest.json"
            if latest.exists() and read(latest)["step"]>read(handoff)["step"]:
                if not (DP/"first_commit.json").exists():
                    point=read(latest)
                    if sha(point["checkpoint"])!=point["checkpoint_sha256"]:raise IOError("DP commit checksum")
                    save(DP/"first_commit.json",dict(point,coordinator=process["pid"],checked=time.time()))
                return "dp4_running"
            return "resuming"
        if not retired.exists():
            current=read(RUN/"latest.json")
            gate=DP/"boundary.json"
            if not gate.exists() or read(gate)["pid"]!=legacy["pid"]:
                save(gate,dict(pid=legacy["pid"],step=current["step"]))
            if current["step"]<=read(gate)["step"]:return "waiting_for_commit"
            proc=Path(f"/proc/{legacy['pid']}")
            if (proc/"cwd").resolve()!=RUN.resolve() or (
                  ("CUDA_VISIBLE_DEVICES="+legacy["uuid"]).encode() not in (proc/"environ").read_bytes().split(b"\0")):
                raise RuntimeError("Unexpected GB206(SM120) process identity")
            retire("learner",DP,4)
        if matches(legacy):raise RuntimeError("Original learner still active after retirement")
        proof=read(DP/"checkpoint.json")
        backup=DP/"before_migration.msgpack"
        if sha(backup)!=proof["checkpoint_sha256"]:raise IOError("Handoff checksum mismatch")
        save(handoff,dict(step=proof["step"],processed_prompts=proof["cursor"],checkpoint=str(backup),
             checkpoint_sha256=proof["checkpoint_sha256"],total_updates=proof["step"]+4*((len(cfg["order"])-proof["cursor"]+3)//4),
             run_sha=plan["run_sha"],time=time.time()))
        return "resuming"


def watch():
    with lock(DP/"watch.lock",True):
        cfg=read(RUN/"config.json"); plan=read(DP/"plan.json")
        if sha(RUN/"config.json")!=plan["run_sha"]:raise ValueError("Run identity changed")
        last=None
        while True:
            phase=advance(cfg,plan)
            if phase!=last:
                save(DP/"state.json",dict(phase=phase,time=time.time(),pid=os.getpid()))
                last=phase
            if phase in ("done","dp4_running"):return
            time.sleep(15 if phase in ("sampling","draining","waiting_for_free_cards") else 1)


def arm(action):
    DP.mkdir(exist_ok=True)
    with lock(RUN/"launch.lock"):
        if (DP/"cancelled.json").exists() or (RUN/"complete.json").exists():return
        path=DP/"plan.json"
        if not path.exists():
            if action!="arm":raise ValueError("No DP plan armed")
            cfg=read(RUN/"config.json"); hw=devices()
            if len(cfg["order"])!=1971 or cfg["sampling"]["rollout_mode"]!="bf16":raise ValueError("Wrong run")
            if not all(matches(read(POOL/f"process_{i}.json")) for i in (*GPUS,4)):raise RuntimeError("Initial role identity")
            files=["dp_control.py","dp_replica.py","dp_train.py"]
            save(path,dict(time=time.time(),run_sha=sha(RUN/"config.json"),uuids=[hw[g]["uuid"] for g in GPUS],
                training_gpus=list(GPUS),prompts=1971,repeats=4,global_batch=4,
                budget="four trajectories per remaining prompt; actual tail size; carry Adam state",
                sources={str(HERE/n):sha(HERE/n) for n in files}))
        if (DP/"watcher.json").exists() and matches(read(DP/"watcher.json")):return
        process=DP/"coordinator.json"
        if process.exists() and not matches(read(process)):
            if action!="recover":raise RuntimeError("Failed DP process; explicit recovery required")
            os.replace(process,DP/f"coordinator_failed_{time.time_ns()}.json")
        if process.exists() and (DP/"first_commit.json").exists():return
        start("dp_control.py",["watch"],"watcher",dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu"))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="watch":
        try:watch()
        except BaseException:
            save(DP/"state.json",dict(phase="attention",time=time.time(),traceback=traceback.format_exc()));raise
    elif action in ("arm","ensure","recover"):arm(action)
    else:raise ValueError(action)
