"""Serve the unchanged, frozen guided velocity over a private Unix socket."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import os
import select
import socket
import stat
import struct
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from common import RUN, read, save, sha
from teacher import Teacher

POOL = RUN/"pool"
DIGEST = sha(RUN/"config.json")
SOCKET_DIR = Path("/tmp")/f"moss-ba-{os.getuid()}-{DIGEST[:12]}"
ADDRESS = str(SOCKET_DIR/"anchor.sock")
SHAPE = (1,128,64,64)
SIZE = int(np.prod(SHAPE))*4

def send_frame(sock, data):
    sock.sendall(struct.pack("!I",len(data)))
    sock.sendall(data)

def read_frame(stream, limit):
    header=stream.read(4)
    if len(header)!=4:raise ConnectionError("Missing frame header")
    size,=struct.unpack("!I",header)
    if size>limit:raise ValueError("Oversized frame")
    data=stream.read(size)
    if len(data)!=size:raise ConnectionError("Incomplete frame")
    return data

class Server:
    def __init__(self, teacher):
        self.teacher=teacher
        self.conditions=OrderedDict()
        self.count=0
        self.compute=0.
        self.last_response=0.
        SOCKET_DIR.mkdir(mode=0o700,exist_ok=True)
        info=SOCKET_DIR.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
            raise PermissionError("Socket directory must be private")
        address=Path(ADDRESS)
        if address.exists():
            if not address.is_socket():raise RuntimeError("Unexpected file at socket path")
            address.unlink()  # caller holds the exclusive GPU0 worker lock
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.bind(ADDRESS)
        self.socket.listen(4)
        save(POOL/"rpc_server.json",dict(pid=os.getpid(),gpu=0,config_sha=DIGEST,address=ADDRESS,started=time.time()))

    def poll(self,timeout):
        if not select.select([self.socket],[],[],timeout)[0]:return False
        conn,_=self.socket.accept()
        with conn:
            conn.settimeout(90)
            try:
                with conn.makefile("rb") as stream:
                    meta=json.loads(read_frame(stream,4096))
                    raw=read_frame(stream,SIZE)
                if meta["config_sha"]!=DIGEST or len(raw)!=SIZE:raise ValueError("Wrong run or latent shape")
                pid=meta["prompt_id"]
                if pid not in self.conditions:
                    self.conditions[pid]=self.teacher.condition(pid)
                    if len(self.conditions)>3:self.conditions.popitem(last=False)
                self.conditions.move_to_end(pid)
                x=jnp.asarray(np.frombuffer(raw,dtype="<f4").reshape(SHAPE))
                started=time.monotonic()
                v=np.asarray(self.teacher.velocity(self.conditions[pid],x,np.float32(meta["t"])),dtype="<f4")
                if v.shape!=SHAPE or not np.isfinite(v).all():raise ValueError("Nonfinite teacher response")
                elapsed=time.monotonic()-started
                send_frame(conn,json.dumps(dict(id=meta["id"],config_sha=DIGEST,gpu=0,compute_seconds=elapsed)).encode())
                send_frame(conn,memoryview(v).cast("B"))
                self.count+=1
                self.compute+=elapsed
                self.last_response=time.monotonic()
                if self.count==1 or self.count%20==0:
                    save(POOL/"rpc_stats.json",dict(pid=os.getpid(),requests=self.count,mean_compute_seconds=self.compute/self.count,time=time.time()))
            except (ConnectionError,TimeoutError):
                pass  # A disconnected client owns no training state here.
        return True

    def yield_to_student(self):
        # During a chain the student usually asks for its next target within
        # a few milliseconds. Outside a chain resume background Euler work.
        grace=.3 if time.monotonic()-self.last_response<.3 else 0.
        while self.poll(grace):
            grace=.3

    def euler(self,cond,seed,n):
        grid=self.teacher.grid(n)
        x=self.teacher.noise(seed)
        for t,end in zip(grid,grid[1:]):
            self.yield_to_student()
            x=x+np.float32(end-t)*self.teacher.velocity(cond,x,t)
        return x

class RemoteTeacher:
    grid=Teacher.grid
    noise=Teacher.noise

    def __init__(self,local):
        self.cfg=local.cfg
        self.cache=local.cache
        self.records=local.records
        self.model=SimpleNamespace(base_precision="bf16")

    def condition(self,pid):
        self.bound=Teacher.condition(self,pid)
        self.pid=pid
        return self.bound

    def velocity(self,cond,x,t):
        if cond is not self.bound:raise ValueError("Gradient batch and Qwen cache binding differ")
        qid=uuid.uuid4().hex
        meta=dict(id=qid,prompt_id=self.pid,t=float(t),config_sha=DIGEST)
        data=np.ascontiguousarray(x,dtype="<f4")
        if data.shape!=SHAPE:raise ValueError("Unexpected latent shape")
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.settimeout(120)
            sock.connect(ADDRESS)
            send_frame(sock,json.dumps(meta).encode())
            send_frame(sock,memoryview(data).cast("B"))
            with sock.makefile("rb") as stream:
                reply=json.loads(read_frame(stream,4096))
                velocity=read_frame(stream,SIZE)
        if reply["id"]!=qid or reply["config_sha"]!=DIGEST or len(velocity)!=SIZE:raise ValueError("Bad teacher response")
        return jax.lax.stop_gradient(jnp.asarray(np.frombuffer(velocity,dtype="<f4").reshape(SHAPE)))

def validated_proxy(local):
    proxy=RemoteTeacher(local)
    measurements=[]
    # Time the steady local and RPC bursts separately. Interleaving a slow
    # local forward between requests would benchmark an idle RPC lane.
    expected=[]
    cases=[(0,901,.5),(0,902,.75),(0,903,.25),(1,904,.13),(1,905,.94),(0,901,.5)]
    for prompt_index,seed,t in cases:
        pid=local.cfg["order"][prompt_index]
        lc=local.condition(pid)
        x=local.noise(seed)
        start=time.monotonic()
        direct=np.asarray(local.velocity(lc,x,t),dtype=np.float32)
        middle=time.monotonic()
        expected.append((pid,seed,t,x,direct,middle-start))
    for pid,seed,t,x,direct,local_seconds in expected:
        rc=proxy.condition(pid)
        middle=time.monotonic()
        remote=np.asarray(proxy.velocity(rc,x,t),dtype=np.float32)
        end=time.monotonic()
        rmse=float(np.sqrt(np.mean((remote-direct)**2)))
        measurements.append(dict(prompt_id=pid,seed=seed,t=t,direct=local_seconds,remote=end-middle,
                                 direct_rms=float(np.sqrt(np.mean(direct**2))),rmse=rmse,
                                 relative_rmse=rmse/max(float(np.sqrt(np.mean(direct**2))),1e-8),
                                 max_abs=float(np.max(np.abs(remote-direct))),bitwise=bool(np.array_equal(remote,direct))))
    warm=measurements[1:]
    # Device-specific BF16 kernels may have small rounding differences. A
    # conservative acceptance bound prevents unverified labels reaching Adam.
    passed=all(r["relative_rmse"]<=.001 for r in measurements)
    direct_median=float(np.median([r["direct"] for r in warm]))
    rpc_median=float(np.median([r["remote"] for r in warm]))
    use=passed and rpc_median < .95*direct_median
    report=dict(learner_pid=os.getpid(),teacher_gpu=0,student_gpu=4,passed_numerical=passed,
                direct_median_seconds=direct_median,rpc_median_seconds=rpc_median,using_rpc=bool(use),
                measured=measurements,at=time.time())
    save(POOL/"benchmarks"/f"{time.time_ns()}.json",report)
    save(POOL/"target_benchmark.json",report)
    if not use:return local
    local.model.release_base()
    gc.collect()
    return proxy

