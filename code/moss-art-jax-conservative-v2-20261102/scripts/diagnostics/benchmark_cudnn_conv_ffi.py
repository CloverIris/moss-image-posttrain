"""Single realistic-shape convolution: bounded cuDNN backend plans vs native JAX."""
from __future__ import annotations
import argparse, hashlib, json, time, gc, sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.profile_vae_stages import configure,hlo_evidence
from scripts.diagnostics.profile_vae_layers import difference
from scripts.benchmark_vae_precision import atomic_json,GPUMonitor,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--device",type=int,default=1);p.add_argument("--size",type=int,default=1024)
 p.add_argument("--channels",type=int,default=192);p.add_argument("--workspace-mib",type=int,default=512)
 p.add_argument("--reserve-dit-gib",type=float,default=9.6);p.add_argument("--warm-repeats",type=int,default=3)
 p.add_argument("--cpu-threads",type=int,default=4)
 p.add_argument("--output",default="outputs/cudnn_conv_ffi_1024")
 a=p.parse_args()
 inventory=configure(a)
 import numpy as np,jax
 from scripts.diagnostics.cudnn_conv_prototype import CudnnPrototype
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/"hlo").mkdir()
 report={"status":"running","arguments":vars(a),"inventory":inventory,
  "runtime_environment":{k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","JAX_PLATFORMS","XLA_FLAGS","TF_FP16_CONV_USE_FP32_COMPUTE","XLA_PYTHON_CLIENT_PREALLOCATE","XLA_PYTHON_CLIENT_MEM_FRACTION")},
  "policy":{"compute":"FFI explicitly requests FP32 for both precisions; native baseline follows XLA/backend defaults and caller environment","precision_filter":"reject DOWN_CONVERT_INPUTS/REDUCED_PRECISION_REDUCTION; strict FP32 also rejects TENSOR_CORE because local cuDNN omits down-conversion note",
  "timing":"synchronized host call latency includes dispatch/sync, excludes compilation and plan building",
  "workspace":"JAX-owned explicit U8 custom-call output, bounded by plan workspace query and runtime check; no cudaMalloc in prototype",
  "input":"fixed synthetic standard_normal(seed=20261020), x scale .5, weight scale 1/sqrt(9*C), NHWC/OHWI, no bias/residual"},
  "modes":{},"source_hashes":{str(p):sha256(p) for p in [Path(__file__),Path("scripts/diagnostics/cudnn_conv_prototype.py"),Path("scripts/diagnostics/cudnn_conv_ffi.cc")]}}
 monitor=GPUMonitor(a.device,True);monitor.start()
 try:
  device=jax.devices()[0];reserve=[]
  remaining=round(a.reserve_dit_gib*1024**3)
  while remaining:
   count=min(remaining,256*1024**2);reserve.append(jax.device_put(np.zeros(count,np.uint8)));reserve[-1].block_until_ready();remaining-=count
  lib=CudnnPrototype("outputs/cudnn_conv_ffi.so")
  for dtype in (np.float32,np.float16):
   name=str(np.dtype(dtype));monitor.phase=name+"_prepare";begin=time.monotonic()
   manifest=lib.prepare(a.size,a.size,a.channels,a.channels,3,dtype,a.workspace_mib)
   build_seconds=time.monotonic()-begin
   result={"manifest":manifest,"plan_build_seconds":build_seconds,"benchmarks":[]}
   report["modes"][name]=result
   print("PREPARED",name,len(manifest["ready"]),build_seconds,flush=True)
   atomic_json(out/"summary.json",report)
   rng=np.random.default_rng(20261020)
   host_x=(rng.standard_normal((1,a.size,a.size,a.channels),dtype=np.float32)*.5).astype(dtype)
   host_w=(rng.standard_normal((a.channels,3,3,a.channels),dtype=np.float32)/np.sqrt(9*a.channels)).astype(dtype)
   result["input_sha256"]=hashlib.sha256(host_x.tobytes()).hexdigest()
   result["weight_sha256"]=hashlib.sha256(host_w.tobytes()).hexdigest()
   x,w=jax.device_put(host_x),jax.device_put(host_w);del host_x,host_w;jax.block_until_ready((x,w))
   def measure(label,fn):
    monitor.phase=name+"_"+label
    start=time.monotonic();compiled=jax.jit(fn).lower(x,w).compile();compile_s=time.monotonic()-start
    times=[];y=None
    for i in range(a.warm_repeats+1):
     if y is not None:y.delete()
     start=time.monotonic();y=compiled(x,w);y.block_until_ready();times.append(time.monotonic()-start)
    raw=np.asarray(jax.device_get(y));y.delete()
    entry={"name":label,"compile_seconds":compile_s,"first_seconds":times[0],"warm_seconds":times[1:],"warm_median_seconds":float(np.median(times[1:])),"finite":bool(np.all(np.isfinite(raw))),
     "hlo":hlo_evidence(compiled,out/"hlo"/(name+"_"+label+".txt")),"jax_memory":device.memory_stats()}
    return raw,entry
   ref,baseline=measure("native",lambda u,v:jax.lax.conv_general_dilated(u,v,(1,1),"SAME",dimension_numbers=("NHWC","OHWI","NHWC"),precision=jax.lax.Precision.HIGHEST,preferred_element_type=dtype))
   result["baseline"]=baseline;atomic_json(out/"summary.json",report)
   for index,plan in enumerate(manifest["ready"]):
    try:
     raw,entry=measure("plan"+str(index)+"_engine"+str(plan["engine"]),lib.operation(manifest,plan))
     entry.update(plan=plan,error=difference(raw,ref),speedup=baseline["warm_median_seconds"]/entry["warm_median_seconds"])
     entry["accepted_accuracy"]=entry["finite"] and entry["error"]["max_abs"] < (1e-4 if dtype==np.float32 else .01)
     print("PLAN",name,index,plan["engine"],plan["workspace_bytes"],entry["warm_median_seconds"],entry["speedup"],entry["error"],flush=True)
     del raw
    except Exception as error:
     entry={"plan":plan,"error_type":type(error).__name__,"error":str(error),"accepted_accuracy":False}
     print("PLAN_FAILED",name,index,repr(error),flush=True)
    result["benchmarks"].append(entry);atomic_json(out/"summary.json",report);gc.collect()
   for cap in (256,512):
    valid=[e for e in result["benchmarks"] if e["accepted_accuracy"] and e["plan"]["workspace_bytes"]<=cap*1024**2]
    result["best_"+str(cap)+"mib"]=min(valid,key=lambda e:e["warm_median_seconds"]) if valid else None
   del ref;x.delete();w.delete();gc.collect()
  report["status"]="passed"
 except BaseException as error:
  report.update(status="failed",error=repr(error));raise
 finally:
  report["nvidia_smi"]=monitor.finish(out/"gpu_memory.csv")
  atomic_json(out/"summary.json",report)
  print("FINISHED",report["status"],flush=True)
if __name__=="__main__":main()
