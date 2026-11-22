"""Asynchronous BF16 reference dispatch and single-writer learner placement."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from common import PYTHON, RUN, arrays, read, save, sha

POOL = RUN / "pool"
GPUS = (0, 1, 2, 3)
LEARNER = 4
CODE = Path(__file__).resolve()
LATENTS = RUN / "latents/train"

@contextmanager
def lock(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield

def identity(pid):
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc/"stat").read_text().rsplit(") ",1)[1].split()
        if stat[0]=="Z":
            return None
        return (stat[19], (proc/"cmdline").read_bytes().rstrip(b"\0").split(b"\0"))
    except (FileNotFoundError,ProcessLookupError):
        return None

def matches(marker):
    live = identity(marker["pid"])
    return live is not None and live == (str(marker["birth"]), [os.fsencode(s) for s in marker["command"]])

def devices():
    columns="index,uuid,memory.used,utilization.gpu,temperature.gpu"
    rows=subprocess.check_output(["nvidia-smi","--query-gpu="+columns,"--format=csv,noheader,nounits"],text=True)
    result={}
    for row in rows.splitlines():
        i,uid,mem,util,temp=map(str.strip,row.split(","))
        result[int(i)] = dict(uuid=uid,used_mib=int(mem),util_pct=int(util),temp_c=int(temp))
    return result

def log_precision(teacher,gpu):
    contract=teacher.model.precision_contract()
    dtypes={}
    for key,value in teacher.model.base.items():
        dt=str(value.dtype)
        dtypes[dt]=dtypes.get(dt,0)+1
        if key!="rope.bands" and dt!="bfloat16":
            raise RuntimeError(f"Unexpected DiT array {key} {dt}")
    if contract["base_precision"]!="bf16" or contract["quantized_modules"] or contract["weight_quantization_mode"]!="none":
        raise RuntimeError("DiT precision contract differs from BF16")
    save(POOL/f"precision_{gpu}.json",dict(contract=contract,weight_array_dtypes=dtypes,pid=os.getpid(),
                                        gpu=gpu,uuid=os.environ["CUDA_VISIBLE_DEVICES"],time=time.time()))

def reference_worker(gpu):
    from teacher import Teacher
    cfg=read(RUN/"config.json")
    if cfg["reference_steps"]!=48 or not all((RUN/"latents/eval"/f"{i:05d}.npz").is_file() for i in range(len(cfg["eval_plan"]))):
        raise RuntimeError("Expected frozen 48-step setup and already committed validation references")
    teacher=Teacher(cfg)
    log_precision(teacher,gpu)
    service=None
    if gpu==0 and (POOL/"try_rpc.json").exists():
        from fast_targets import Server
        service=Server(teacher)
    n=len(cfg["order"])
    frontier=0
    while True:
        while frontier<n and (LATENTS/f"{frontier:05d}.npz").is_file():
            frontier+=1
        if frontier==n:break
        for index in range(frontier,n):
            path=LATENTS/f"{index:05d}.npz"
            if path.exists():continue
            claim=POOL/"claims"/f"{index:05d}.lock"
            with claim.open("a") as stream:
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if path.exists():continue
                pid=cfg["order"][index]
                seed=cfg["seed"]+index
                started=time.time()
                state=dict(gpu=gpu,index=index,prompt_id=pid,seed=seed,started=started,phase="generating")
                save(POOL/f"gpu_{gpu}.json",state)
                cond=teacher.condition(pid)
                conditioned=time.time()
                integrate=service.euler if service else teacher.euler
                output=np.asarray(integrate(cond,seed,cfg["reference_steps"]),dtype=np.float32)
                stored=output.astype(np.float16)
                if stored.shape!=(1,128,64,64) or not np.isfinite(stored).all():
                    raise FloatingPointError("Invalid 1024 reference endpoint")
                arrays(path,prompt_id=np.array(pid),seed=np.array(seed),euler48=stored)
                state.update(phase="committed",finished=time.time(),cache_read_seconds=conditioned-started,
                             artifact_sha256=sha(path))
                save(POOL/"commits"/f"{index:05d}.json",state)
                save(POOL/f"gpu_{gpu}.json",state)
                print("committed",gpu,index,state["finished"]-started,flush=True)
                break
        else:
            if service:service.poll(1.)
            else:time.sleep(1)
    with lock(POOL/"finish.lock"):
        if not (RUN/"producer_complete.json").exists():
            for i,pid in enumerate(cfg["order"]):
                with np.load(LATENTS/f"{i:05d}.npz",allow_pickle=False) as f:
                    data=f["euler48"]
                    if str(f["prompt_id"])!=pid or int(f["seed"])!=cfg["seed"]+i or data.shape!=(1,128,64,64) or not np.isfinite(data).all():
                        raise ValueError(f"Bad committed reference {i}")
            save(RUN/"producer_complete.json",dict(training=n,eval=len(cfg["eval_plan"]),verified=True))
    if service:
        while not (RUN/"complete.json").exists():
            outcome=POOL/"target_benchmark.json"
            if outcome.exists() and not read(outcome)["using_rpc"]:break
            service.poll(1.)

def worker(gpu):
    with lock(POOL/f"worker_{gpu}.lock",True):
        if gpu==LEARNER:
            with lock(RUN/"learner.lock",True):
                import train
                original=train.Teacher
                # Instrument the unmodified trainer's process-local teacher factory.
                def audited_teacher(cfg):
                    model=original(cfg)
                    log_precision(model,gpu)
                    if (POOL/"try_rpc.json").exists():
                        gate=POOL/"target_benchmark.json"
                        if gate.is_file() and read(gate)["using_rpc"] is False:
                            return model
                        from fast_targets import validated_proxy
                        return validated_proxy(model)
                    return model
                train.Teacher=audited_teacher
                train.train()
        elif gpu in GPUS:
            if gpu==1:
                with lock(RUN/"producer.lock",True):reference_worker(gpu)
            else:reference_worker(gpu)
        else:raise ValueError(gpu)

def launch(gpu):
    if gpu not in (*GPUS,LEARNER):raise ValueError(gpu)
    student_plan=RUN/"control/student_restart_100m.json"
    if gpu==LEARNER and student_plan.is_file():
        return
    if any((RUN/"dp4"/name).exists() for name in ("learner_retired.json","handoff.json")):
        return
    marker=POOL/f"process_{gpu}.json"
    if marker.exists() and matches(read(marker)):return
    if marker.exists():
        old=read(marker)
        found=identity(old["pid"])
        if found is not None and found[0]==old["birth"]:raise RuntimeError("Unknown live worker identity")
    if (RUN/"complete.json").exists():return
    gate=POOL/"target_benchmark.json"
    rpc_needed=(POOL/"try_rpc.json").is_file() and (not gate.is_file() or read(gate)["using_rpc"])
    if gpu!=LEARNER and (RUN/"producer_complete.json").exists() and not (gpu==0 and rpc_needed):return
    info=devices()[gpu]
    if info["used_mib"]>=300:raise RuntimeError(f"GPU {gpu} occupied: {info}")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=info["uuid"],CUDA_DEVICE_ORDER="PCI_BUS_ID",JAX_PLATFORMS="cuda",
             JAX_ENABLE_X64="true",XLA_PYTHON_CLIENT_PREALLOCATE="false",JAX_COMPILATION_CACHE_DIR=str(RUN/"jax_cache"),
             OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1")
    cmd=[PYTHON,str(CODE),"worker",str(gpu)]
    with (POOL/f"worker_{gpu}.log").open("a") as log:
        p=subprocess.Popen(cmd,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    born=identity(p.pid)
    if born is None:raise RuntimeError(f"GPU {gpu} worker exited at launch")
    record=dict(pid=p.pid,birth=born[0],command=cmd,gpu=gpu,uuid=info["uuid"],started=time.time())
    save(marker,record)
    if gpu in (1,LEARNER):save(RUN/("producer_process.json" if gpu==1 else "learner_process.json"),record)
    print("started",record,flush=True)

def checkpoint_audit():
    from flax.serialization import msgpack_restore
    raw=(RUN/"latest.msgpack").read_bytes()
    state=msgpack_restore(raw)
    latest=read(RUN/"latest.json")
    step=state["step"]
    if state["config_sha"]!=sha(RUN/"config.json") or step!=latest["step"] or step!=4*state["next_index"] or state["next_index"]!=latest["processed_prompts"]:
        raise RuntimeError("Checkpoint not at an externally published boundary")
    def leaves(tree):
        if isinstance(tree,dict):
            for v in tree.values():yield from leaves(v)
        elif isinstance(tree,(list,tuple)):
            for v in tree:yield from leaves(v)
        elif isinstance(tree,np.ndarray):yield tree
    if not all(np.isfinite(a).all() for a in leaves(state)):raise ValueError("Nonfinite checkpoint")
    if not state["opt"] or not state["params"]:raise ValueError("Incomplete training state")
    return dict(step=step,cursor=state["next_index"],checkpoint_sha256=__import__("hashlib").sha256(raw).hexdigest())

def retire(role,transition,gpu=None):
    marker=read(RUN/(role+"_process.json") if gpu is None else POOL/f"process_{gpu}.json")
    expected=[PYTHON,str(CODE.with_name("start.py")),role,"--worker"] if gpu is None else [PYTHON,str(CODE),"worker",str(gpu)]
    if marker["command"]!=expected or not matches(marker):
        raise RuntimeError(f"Unverified legacy {role}; migration halted")
    pid=marker["pid"]
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            stat=Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()
            if stat[0]=="T":break
            time.sleep(.05)
        else:raise RuntimeError("Worker did not reach stopped state")
        if not matches(marker):raise RuntimeError("Worker birth changed")
        if role=="learner":
            point=checkpoint_audit()
            shutil.copy2(RUN/"latest.msgpack",transition/"before_migration.msgpack")
            if sha(transition/"before_migration.msgpack")!=point["checkpoint_sha256"]:raise IOError("Checkpoint backup checksum")
            save(transition/"checkpoint.json",point)
        save(transition/(role+"_retired.json"),dict(marker,stopped_at=time.time()))
        os.kill(pid,signal.SIGTERM)
    finally:
        if matches(marker):os.kill(pid,signal.SIGCONT)
    for _ in range(300):
        # During CUDA teardown Linux can already clear cmdline while the
        # original process still owns its GPU context.
        live=identity(pid)
        if live is None or live[0]!=marker["birth"]:
            physical_gpu=marker["gpu"]
            deadline=time.monotonic()+10
            while devices()[physical_gpu]["used_mib"]>=300 and time.monotonic()<deadline:
                time.sleep(.25)
            return
        time.sleep(.1)
    raise RuntimeError("Owned legacy worker did not exit")

def controller(action):
    if any((RUN/"dp4"/name).exists() for name in ("learner_retired.json","handoff.json")):
        if action!="ensure":raise RuntimeError("The learner has been handed to four-card DP")
        from dp_control import arm
        return arm("ensure")
    POOL.mkdir(exist_ok=True)
    (POOL/"claims").mkdir(exist_ok=True)
    with lock(RUN/"launch.lock"):
        for p,expected in read(RUN/"training_sources.json").items():
            if sha(p)!=expected:raise ValueError(f"Frozen source changed: {p}")
        cfg=read(RUN/"config.json")
        if cfg["sampling"]["rollout_mode"]!="bf16" or cfg["sampling"]["guidance_scale"]!=3.:raise ValueError("Unexpected run")
        layout=POOL/"deployment.json"
        if action=="migrate" and not layout.exists():
            inventory=devices()
            for gpu in (0,3,4):
                if inventory[gpu]["used_mib"]>=300:raise RuntimeError(f"GPU {gpu} is occupied")
            prior=POOL/"migration"
            prior.mkdir(exist_ok=True)
            for role in ("producer","learner"):
                item=read(RUN/(role+"_process.json"))
                recorded=prior/(role+"_retired.json")
                if not matches(item) and not (recorded.exists() and read(recorded)["birth"]==item["birth"]):
                    raise RuntimeError("Legacy worker identity check failed")
            if not (prior/"before.json").exists():
                save(prior/"before.json",dict(time=time.time(),latest=read(RUN/"latest.json"),gpu=inventory,
                                             ref_count=len(list(LATENTS.glob("*.npz"))),config_sha=sha(RUN/"config.json")))
                shutil.copy2(RUN/"producer.log",prior/"producer_before.log")
            if matches(read(RUN/"producer_process.json")):
                previous=read(RUN/"producer.json")["index"]
                deadline=time.monotonic()+100
                while read(RUN/"producer.json")["index"]==previous and time.monotonic()<deadline:
                    time.sleep(.2)
                retire("producer",prior)
            if matches(read(RUN/"learner_process.json")):
                deadline=time.monotonic()+300
                ready=len(list(LATENTS.glob("*.npz")))
                while read(RUN/"latest.json")["processed_prompts"]<ready and time.monotonic()<deadline:
                    time.sleep(.25)
                retire("learner",prior)
            save(layout,dict(activated=time.time(),reference_gpus=list(GPUS),learner_gpu=LEARNER,
                             checkpoint=read(prior/"checkpoint.json"),run_sha=sha(RUN/"config.json")))
        if not layout.exists():raise RuntimeError("Pool not activated")
        if action=="accelerate":
            transition=POOL/"target_migration"
            transition.mkdir(exist_ok=True)
            flag=POOL/"try_rpc.json"
            if not flag.exists():
                old=read(POOL/"process_0.json")
                if not matches(old):raise RuntimeError("Reference worker unavailable for rolling change")
                last=read(POOL/"gpu_0.json")
                deadline=time.monotonic()+110
                while read(POOL/"gpu_0.json")["index"]==last["index"] and time.monotonic()<deadline:
                    time.sleep(.1)
                retire("reference",transition,0)
                save(flag,dict(created=time.time(),policy="require 4 warm finite BF16 comparisons, relative RMS <=0.001 and median latency gain >5%"))
            launch(0)
            server=POOL/"rpc_server.json"
            deadline=time.monotonic()+120
            while not (server.exists() and read(server)["pid"]==read(POOL/"process_0.json")["pid"]):
                if time.monotonic()>deadline:raise TimeoutError("Target server did not initialize")
                time.sleep(.5)
            if not (transition/"learner_retired.json").exists():
                step=read(RUN/"latest.json")["step"]
                deadline=time.monotonic()+400
                while read(RUN/"latest.json")["step"]==step and time.monotonic()<deadline:
                    time.sleep(.1)
                retire("learner",transition,LEARNER)
        # Each role can recover independently. Retain diagnostics for any busy card.
        errors={}
        for gpu in (0,LEARNER,1,2,3):
            try:
                marker=POOL/f"process_{gpu}.json"
                already=marker.exists() and matches(read(marker))
                launch(gpu)
                if not already and marker.exists():
                    entry=read(marker)
                    deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        ready=POOL/f"precision_{gpu}.json"
                        if ready.exists() and read(ready)["pid"]==entry["pid"] and matches(entry):break
                        found=identity(entry["pid"])
                        if found is None or found[0]!=entry["birth"]:break
                        time.sleep(1)
                    if not matches(entry) and not (RUN/("complete.json" if gpu==LEARNER else "producer_complete.json")).exists():
                        raise RuntimeError("Worker exited; inspect its log")
            except Exception as error:errors[str(gpu)]=str(error)
        if errors:raise RuntimeError(errors)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("migrate","ensure","accelerate","worker"))
    parser.add_argument("gpu",nargs="?",type=int)
    args=parser.parse_args()
    if args.action=="worker":
        try:worker(args.gpu)
        except BaseException:
            save(POOL/f"failure_{args.gpu}_{time.time_ns()}.json",dict(traceback=traceback.format_exc()))
            raise
    else:controller(args.action)

