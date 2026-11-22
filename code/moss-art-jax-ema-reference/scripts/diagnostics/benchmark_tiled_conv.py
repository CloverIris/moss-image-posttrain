#!/usr/bin/env python3
"""Experimental bounded-tile im2col+GEMM convolution. No production/vendor edits."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.benchmark_vae_precision import sha256,atomic_json
from moss_art.config import DEFAULTS
from moss_art.runtime import preflight,configure_process
from moss_art.locking import output_lock


def tiled_conv(x,weight,bias,*,tile=4096):
    """N=1, OIHW 3x3 same convolution, NCHW result, low operands/F32 GEMM sum."""
    import jax.numpy as jnp
    from jax import lax
    if x.ndim!=4 or x.shape[0]!=1 or weight.shape[2:]!=(3,3) or x.shape[1]!=weight.shape[1]:
        raise ValueError("Expected N=1 NCHW and 3x3 OIHW")
    _,ci,h,w=x.shape
    co=weight.shape[0]
    if h*w%tile:
        raise ValueError("This prototype requires tile to divide output pixels")
    flat_input=x.reshape(ci,h*w)
    matrix=weight.transpose(2,3,1,0).reshape(9*ci,co)
    offsets=jnp.arange(tile,dtype=jnp.int32)
    output=jnp.zeros((co,h*w),x.dtype)
    def body(block,out):
        flat=block*tile+offsets
        row,col=flat//w,flat%w
        def patch(dy,dx):
            rr,cc=row+dy-1,col+dx-1
            valid=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
            index=jnp.clip(rr,0,h-1)*w+jnp.clip(cc,0,w-1)
            gathered=flat_input[:,index].T
            return jnp.where(valid[:,None],gathered,jnp.zeros((),x.dtype))
        patches=jnp.concatenate([patch(dy,dx) for dy in range(3) for dx in range(3)],axis=1)
        values=jnp.matmul(patches,matrix,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
        # Match native low-output conv followed by low-dtype bias interface.
        values=values.astype(x.dtype)+bias[None,:]
        return lax.dynamic_update_slice(out,values.T,(0,block*tile))
    output=lax.fori_loop(0,h*w//tile,body,output)
    return output.reshape(1,co,h,w)


def direct_conv(x,weight,bias):
    from jax import lax
    y=lax.conv_general_dilated(x,weight,(1,1),((1,1),(1,1)),
        dimension_numbers=("NCHW","OIHW","NCHW"),precision=lax.Precision.HIGHEST,
        preferred_element_type=x.dtype)
    return y if bias is None else y+bias[None,:,None,None]


def cpu_test():
    import jax
    import jax.numpy as jnp
    rng=np.random.default_rng(17)
    results=[]
    for dtype in (np.float32,np.float16):
        x=jnp.asarray(rng.normal(size=(1,3,8,8)).astype(dtype))
        w=jnp.asarray((rng.normal(size=(4,3,3,3))*.1).astype(dtype))
        b=jnp.asarray((rng.normal(size=(4,))*.1).astype(dtype))
        a=np.asarray(jax.jit(direct_conv)(x,w,b),np.float32)
        v=np.asarray(jax.jit(lambda a,b,c:tiled_conv(a,b,c,tile=8))(x,w,b),np.float32)
        np.testing.assert_allclose(v,a,rtol=2e-5 if dtype==np.float32 else .01,atol=2e-6 if dtype==np.float32 else .001)
        results.append({"dtype":np.dtype(dtype).name,"max_abs":float(np.max(np.abs(v-a)))})
    print(json.dumps({"cpu_equivalence":"passed","cases":results}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--device",type=int,default=2)
    parser.add_argument("--output")
    parser.add_argument("--shape",choices=("1024x192","512x384"),required=False)
    parser.add_argument("--dtype",choices=("float32","float16"),required=False)
    parser.add_argument("--tile",type=int,default=4096)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--implementation",choices=("prototype","module"),default="prototype")
    parser.add_argument("--no-bias",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        os.environ.update(CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        cpu_test()
        return
    if not args.output or not args.shape or not args.dtype:
        parser.error("--output, --shape, --dtype required")
    h,ci=(1024,192) if args.shape=="1024x192" else (512,384)
    itemsize=4 if args.dtype=="float32" else 2
    if args.tile<=0 or h*h%args.tile or args.repeats<1:
        parser.error("Tile must divide H*W; repeats positive")
    if args.tile*9*ci*itemsize>256*1024**2:
        parser.error("Materialized patch tile exceeds 256MiB")
    config=copy.deepcopy(DEFAULTS)
    config["runtime"].update(devices=[args.device],cpu_threads=4,memory_fraction=.9)
    inventory=preflight(config)
    # Compiler/precision environment remains caller-controlled.
    configure_process(config,args.device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    from moss_art.tiled_conv import conv2d_nchw
    output=Path(args.output).resolve()
    with output_lock(output):
        if any(p.name!=".run.lock" for p in output.iterdir()):
            raise FileExistsError(output)
        report={"status":"running","recorded_utc":datetime.now(timezone.utc).isoformat(),"argv":sys.argv,
            "source_sha256":sha256(Path(__file__)),"module_sha256":sha256(ROOT/"moss_art/tiled_conv.py"),"implementation":args.implementation,"bias_enabled":not args.no_bias,"physical_gpu":args.device,"inventory":inventory,
            "versions":{"jax":jax.__version__,"jaxlib":jaxlib.__version__},
            "case":{"height":h,"width":h,"input_channels":ci,"output_channels":ci,"kernel":[3,3],"dtype":args.dtype,"tile":args.tile},
            "data":"Fixed synthetic seed=20261020; N(0,.25^2) activations, N(0,1/(9*Cin)) weights, N(0,.01^2) bias. No real activation/model-quality claim.",
            "policy":{"input_output":"NCHW","weights":"OIHW","precision":"HIGHEST; no requested TF32 low precision",
                      "gemm_accumulator_output":"FP32; convolution result cast to input dtype before bias",
                      "autotune":"Explicit per-JIT A/B: direct cuDNN level0, tiled GEMM level4; global XLA_FLAGS preserved",
                      "native_convolution_accumulator":"XLA/backend default unless caller supplied a precision environment override; no environment is forced here",
                      "memory":"No DiT reservation; direct NCHW gather and update avoid full padded/layout copies. Compiler temp_size is verified separately from required input/output buffers",
                      "timing":"explicit already-device x,w,b; compile excluded; requested synchronized warm executions"},
            "logical_patch_tile_bytes":args.tile*9*ci*itemsize,
            "full_im2col_bytes_avoided":h*h*9*ci*itemsize,
            "runtime_environment":{k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","XLA_FLAGS","XLA_PYTHON_CLIENT_PREALLOCATE","XLA_PYTHON_CLIENT_MEM_FRACTION","TF_FP16_CONV_USE_FP32_COMPUTE")},
            "methods":{}}
        atomic_json(output/"summary.json",report)
        try:
            device=jax.devices()[0]
            rng=np.random.default_rng(20261020)
            x=rng.standard_normal((1,ci,h,h),dtype=np.float32)*.25
            w=rng.standard_normal((ci,ci,3,3),dtype=np.float32)/np.sqrt(9*ci)
            b=rng.standard_normal((ci,),dtype=np.float32)*.01
            x,w,b=[jax.device_put(np.asarray(a,args.dtype),device) for a in (x,w,b)]
            jax.block_until_ready((x,w,b))
            reference_fn=lambda x,w,b:direct_conv(x,w,None if args.no_bias else b)
            if args.implementation=="module":
                def tiled_fn(x,w,b):
                    out=conv2d_nchw(x,w,tile_pixels=args.tile)
                    return out if args.no_bias else out+b[None,:,None,None]
            elif args.no_bias:
                tiled_fn=lambda x,w,b:tiled_conv(x,w,jnp.zeros_like(b),tile=args.tile)
            else:
                tiled_fn=lambda x,w,b:tiled_conv(x,w,b,tile=args.tile)
            for name,fn,level in (("direct_cudnn",reference_fn,0),("tiled_im2col_gemm",tiled_fn,4)):
                begun=time.monotonic()
                compiled=jax.jit(fn,compiler_options={"xla_gpu_autotune_level":level}).lower(x,w,b).compile()
                compile_seconds=time.monotonic()-begun
                hlo=compiled.as_text()
                (output/(name+".hlo.txt")).write_text(hlo)
                stats=compiled.memory_analysis()
                method={"compiler_options":{"xla_gpu_autotune_level":level},"compile_seconds":compile_seconds,"compiled_memory":{k:int(getattr(stats,k)) for k in
                    ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")},
                    "hlo_sha256":sha256(output/(name+".hlo.txt")),"custom_call_targets":{},
                    "dot_evidence":[],"times":[]}
                for line in hlo.splitlines():
                    m=re.search(r'custom_call_target="([^"]+)"',line)
                    if m:
                        method["custom_call_targets"][m.group(1)]=method["custom_call_targets"].get(m.group(1),0)+1
                    if " dot(" in line or m and any(t in m.group(1).lower() for t in ("gemm","matmul")):
                        method["dot_evidence"].append(line[:2200])
                if name=="tiled_im2col_gemm" and method["compiled_memory"]["temp_size_in_bytes"]>256*1024**2:
                    raise MemoryError("Tiled executable temporary workspace exceeds 256MiB")
                if name=="tiled_im2col_gemm" and any("conv" in target.lower() for target in method["custom_call_targets"]):
                    raise RuntimeError("Unexpected convolution inside supposed GEMM path")
                print(json.dumps({"event":"compiled","method":name,"seconds":compile_seconds,"memory":method["compiled_memory"]}),flush=True)
                result=compiled(x,w,b)
                result.block_until_ready()
                result.delete()
                for repeat in range(args.repeats):
                    begun=time.monotonic()
                    result=compiled(x,w,b)
                    result.block_until_ready()
                    method["times"].append(time.monotonic()-begun)
                    if repeat+1<args.repeats:
                        result.delete()
                raw=np.asarray(result,np.float32).copy()
                result.delete()
                method["mean_seconds"]=float(np.mean(method["times"]))
                method["finite"]=bool(np.isfinite(raw).all())
                method["raw_max_abs"]=float(np.max(np.abs(raw)))
                if not method["finite"]:
                    raise FloatingPointError(name+" produced nonfinite output")
                if name=="direct_cudnn":
                    reference=raw
                else:
                    # Streaming CPU error accumulation avoids another full float64 batch.
                    maxabs=0.;sumerr=0.;sumref=0.;mae=0.
                    a,bref=raw.reshape(-1),reference.reshape(-1)
                    for start in range(0,a.size,1048576):
                        aa=a[start:start+1048576].astype(np.float64);bb=bref[start:start+1048576].astype(np.float64)
                        delta=aa-bb
                        maxabs=max(maxabs,float(np.max(np.abs(delta))))
                        sumerr+=float(np.sum(delta**2));sumref+=float(np.sum(bb**2));mae+=float(np.sum(np.abs(delta)))
                    method["comparison_to_direct"]={"max_abs":maxabs,"relative_l2":(sumerr/sumref)**.5,
                                                   "mae":mae/a.size,"bitwise_equal":bool(np.array_equal(raw,reference))}
                report["methods"][name]=method
                report["jax_memory_after_"+name]={k:int(v) for k,v in device.memory_stats().items() if isinstance(v,(int,np.integer))}
                atomic_json(output/"summary.json",report)
                print(json.dumps({"event":"measured","method":name,"mean_seconds":method["mean_seconds"],"comparison":method.get("comparison_to_direct")}),flush=True)
                del raw,compiled
                gc.collect()
            report["observed_speedup"]=report["methods"]["direct_cudnn"]["mean_seconds"]/report["methods"]["tiled_im2col_gemm"]["mean_seconds"]
            report["status"]="passed"
        except BaseException as error:
            report.update(status="failed",error_type=type(error).__name__,error=str(error))
            raise
        finally:
            atomic_json(output/"summary.json",report)
            print(json.dumps({"status":report["status"],"report":str(output/"summary.json")}),flush=True)


if __name__=="__main__":
    main()
