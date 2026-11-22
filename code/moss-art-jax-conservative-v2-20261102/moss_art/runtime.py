"""Synchronous host-reduced data parallelism with one JAX worker per GPU group.

Only LoRA arrays/gradients cross the host boundary. Every worker owns a logical
frozen BF16 model (optionally tensor-sharded within its pair) and complete prompt groups. Reducing unnormalised window sums
with one global ORIGINAL denominator preserves ART under microbatching/masking.
"""
from __future__ import annotations
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,uuid,pci.bus_id",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        i, name, used, total, uuid, pci_bus_id = [x.strip() for x in line.split(",")]
        rows[int(i)] = {"name": name, "used_mib": int(used), "total_mib": int(total),
                        "uuid": uuid, "pci_bus_id": pci_bus_id}
    return rows

def preflight(config):
    r = config["runtime"]
    if r["platform"] == "cpu":
        return {"platform": "cpu", "workers": len(device_groups(config))}
    available = gpu_inventory()
    for i in r["devices"]:
        if i not in available:
            raise RuntimeError(f"Requested GPU {i} does not exist")
        if available[i]["used_mib"] > r["busy_memory_mib"] and not r["allow_busy"]:
            raise RuntimeError(f"GPU {i} is busy ({available[i]['used_mib']} MiB); no existing processes were touched. Select free GPUs or wait.")
    return {i: available[i] for i in r["devices"]}

def device_groups(config):
    devices=config["runtime"]["devices"]
    size=config["runtime"].get("tensor_parallel_size",1)
    if type(size) is not int or size not in (1,2) or len(devices)%size:
        raise ValueError("device count must divide tensor_parallel_size (1 or 2)")
    return [devices[i:i+size] for i in range(0,len(devices),size)]

def configure_process(config, device=None):
    r = config["runtime"]
    visible = device if isinstance(device,(list,tuple)) else [device]
    if device is None or r["platform"] == "cpu":
        cuda_visible = ""
    else:
        # Config indices refer to nvidia-smi, not CUDA's potentially different
        # ordinal order. UUIDs preserve the requested physical TP grouping.
        available = gpu_inventory()
        missing = [i for i in visible if i not in available]
        if missing:
            raise RuntimeError(f"Requested GPUs do not exist: {missing}")
        cuda_visible = ",".join(available[i]["uuid"] for i in visible)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    os.environ["JAX_PLATFORMS"] = "cpu" if device is None else r["platform"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(r["memory_fraction"])
    os.environ["JAX_ENABLE_X64"] = "true"
    # Keep process-wide compiler and convolution precision policy untouched.
    # XLA uses its defaults or the caller's exact environment; bounded cuDNN
    # plans specify their own FP32 compute and workspace limits locally.
    os.environ["OMP_NUM_THREADS"] = str(r["cpu_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(r["cpu_threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path(".jax_cache").resolve())
    if hasattr(os, "sched_getaffinity"):
        original = os.environ.setdefault("MOSS_ART_CPUSET", ",".join(map(str, sorted(os.sched_getaffinity(0)))))
        cores = [int(x) for x in original.split(",")]
        width = min(len(cores), r["cpu_threads"])
        offset = 0 if device is None else (visible[0] * width) % len(cores)
        chosen = (cores + cores)[offset:offset + width]
        os.sched_setaffinity(0, chosen)

class WorkerPool:
    def __init__(self, config, worker_main=None, *, cache_base=True):
        if worker_main is None:
            from .worker import worker_main
        preflight(config)
        self.host_weights = None
        if cache_base and config["model"]["kind"] == "moss":
            from .host_weights import HostWeights
            import copy
            self.host_weights = HostWeights(config["model"]["path"])
            config = copy.deepcopy(config)
            config["_host_base"] = self.host_weights.descriptor
        self.worker_main = worker_main
        self.timeout = config["runtime"]["worker_timeout"]
        self.workers = []
        self.config = config
        self._start()

    def _start(self):
        context = mp.get_context("spawn")
        config = self.config
        worker_main = self.worker_main
        try:
            for rank, group in enumerate(device_groups(config)):
                device=group[0] if len(group)==1 else group
                parent, child = context.Pipe()
                process = context.Process(target=worker_main, args=(child, config, rank, device),
                                          name=f"moss-art-{device}", daemon=False)
                process.start()
                child.close()
                self.workers.append((process, parent))
            self.info = self._receive_all()
        except BaseException:
            self.close()
            raise

    def _receive_all(self):
        results = []
        for process, conn in self.workers:
            deadline = time.monotonic() + self.timeout
            while not conn.poll(min(1, max(0, deadline - time.monotonic()))):
                if not process.is_alive():
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{process.name} did not respond within {self.timeout}s")
            response = conn.recv()
            if response.get("error"):
                raise RuntimeError(f"{process.name}: {response['error']}\n{response.get('traceback', '')}")
            results.append(response["result"])
        return results

    def call(self, op, payloads=None):
        payloads = [{} for _ in self.workers] if payloads is None else payloads
        if len(payloads) != len(self.workers):
            raise ValueError("one payload per worker required")
        for (_, conn), payload in zip(self.workers, payloads):
            conn.send({"op": op, **payload})
        return self._receive_all()

    def pause(self):
        """Export only replay/adapter state, then physically release GPU contexts."""
        self.saved_states = self.call("export_state")
        self._stop_workers()

    def resume(self):
        if self.workers: raise RuntimeError("Cannot resume a running pool")
        self._start()
        self.call("import_state", [{"state": state} for state in self.saved_states])
        del self.saved_states

    def close(self):
        self._stop_workers()
        if self.host_weights is not None:
            self.host_weights.close()
            self.host_weights = None

    def _stop_workers(self):
        for process, conn in self.workers:
            try:
                if process.is_alive():
                    conn.send({"op": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process, conn in self.workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            conn.close()
        self.workers = []

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
