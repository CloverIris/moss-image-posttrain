"""Start only this isolated experiment. Each role holds its own exclusive lock."""
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import RUN, PYTHON, initialize, save, read, sha

def run(role):
    lock = (RUN/(role+".lock")).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if role=="producer":
            from teacher import produce
            produce()
        else:
            from train import train
            train()
    except BaseException:
        save(RUN/(role+"_failure.json"),dict(time=time.time(),traceback=traceback.format_exc()))
        raise

def launch(role):
    if role not in ("producer","learner"):
        raise ValueError(role)
    if (RUN/"pool/deployment.json").is_file():
        from pool import controller
        return controller("ensure")
    RUN.mkdir(parents=True,exist_ok=True)
    cfg = initialize()
    source_lock=RUN/"training_sources.json"
    if source_lock.exists():
        for source,expected in read(source_lock).items():
            if sha(source)!=expected:raise ValueError(f"Training source differs from committed run: {source}")
    with (RUN/"launch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (RUN/"pool/deployment.json").is_file():
            raise RuntimeError("GPU pool activated during launch. Use pool.py ensure.")
        completed=RUN/("producer_complete.json" if role=="producer" else "complete.json")
        if completed.exists():
            print("already complete",read(completed))
            return
        marker = RUN/(role+"_process.json")
        if marker.exists():
            prev = read(marker)
            stat = Path(f"/proc/{prev['pid']}/stat")
            if stat.exists() and stat.read_text().split(") ")[1].split()[19]==prev["birth"] and stat.read_text().split(") ")[1][0]!="Z":
                print("already active",prev); return
        gpu = cfg["teacher_gpu" if role=="producer" else "learner_gpu"]
        uuid,used = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],text=True).strip().split(",")
        if int(used)>300:
            raise RuntimeError(f"GPU {gpu} already in use")
        failure=RUN/(role+"_failure.json")
        if failure.exists():
            os.replace(failure,RUN/(role+"_failure_"+str(time.time_ns())+".json"))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid.strip(),JAX_PLATFORMS="cuda",JAX_ENABLE_X64="true",
                 XLA_PYTHON_CLIENT_PREALLOCATE="false",JAX_COMPILATION_CACHE_DIR=str(RUN/"jax_cache"),
                 OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1")
        cmd=[PYTHON,str(Path(__file__).resolve()),role,"--worker"]
        with (RUN/(role+".log")).open("a") as log:
            p=subprocess.Popen(cmd,env=env,cwd=RUN,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=p.pid,birth=Path(f"/proc/{p.pid}/stat").read_text().split(") ")[1].split()[19],gpu=gpu,uuid=uuid.strip(),started=time.time(),command=cmd)
        save(marker,info);print(info)

if __name__=="__main__":
    role=sys.argv[1]
    run(role) if "--worker" in sys.argv else launch(role)

