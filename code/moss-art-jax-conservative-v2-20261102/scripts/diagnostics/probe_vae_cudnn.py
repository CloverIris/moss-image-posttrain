#!/usr/bin/env python3
"""Audit current attention backends and test the actual VAE head width on CUDA.

This diagnostic never changes model heads or head_dim. Small smoke uses
[B,T,H,D]=[1,16,1,384]; --actual-tokens additionally tests T=16384 only when the
small test supports the same dtype/head shape. No model weights are loaded.
Import/--help does not import JAX or initialize an accelerator.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def hlo_evidence(path):
    source=path.read_text()
    targets=re.findall(r'custom_call_target="([^"]+)"',source)
    lines=source.splitlines()
    fmha=[{"line":i+1,"text":line} for i,line in enumerate(lines) if "__cudnn$fmha" in line]
    return dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
                custom_call_counts={target:targets.count(target) for target in sorted(set(targets))},
                fmha_instructions=fmha,
                flash_true_count=len(re.findall(r'"is_flash_attention"\s*:\s*true',source)),
                attention_loop_evidence=[{"line":i+1,"text":line[:1800]} for i,line in enumerate(lines)
                    if ("while(" in line or "while (" in line) and ("attention" in line or "16384" in line)][:5])


def static_audit():
    sys.path.insert(0,str(ROOT))
    from moss_art.config import load_config
    configs=[]
    for path in sorted((ROOT/"configs").glob("moss*.yaml")):
        model=load_config(path)["model"]
        configs.append(dict(path=str(path),sha256=sha256(path),
                            attention_backend=model["attention_backend"],
                            dit_autotune_level=model["dit_autotune_level"],
                            vae_precision=model["vae_precision"],
                            vae_low_precision_stages=model["vae_low_precision_stages"]))
    sources=["moss_art/attention.py","moss_art/model.py","moss_art/worker.py",
             "moss_art/config.py","moss_art/vae_precision.py","moss_art/evaluate.py",
             "vendor/moss_jax/vae.py"]
    dit=[]
    for path in sorted((ROOT/"outputs/gradient_1024_cudnn_tuned/hlo").glob("*after_optimizations.txt")):
        if "jit__dit_block" in path.name or "jit__block_vjp_impl" in path.name:
            dit.append(hlo_evidence(path))
    vae=[]
    for name in ("fp32_reference","bf16_late_four","fp16_late_four","bf16_all","fp16_all"):
        directory=ROOT/"outputs/vae_precision"/name
        evidence=hlo_evidence(directory/"optimized.hlo.txt")
        summary=json.loads((directory/"summary.json").read_text())
        evidence.update(profile=name,summary_path=str(directory/"summary.json"),
                        summary_sha256=sha256(directory/"summary.json"),
                        benchmark_status=summary["status"],precision=summary["precision"],
                        low_precision_stages=summary["low_precision_stages"],
                        source_hashes_at_benchmark=summary.get("source_hashes"))
        vae.append(evidence)
    weights=Path("$MODEL_ROOT/MOSS-Image-SFT/vae/diffusion_pytorch_model.safetensors")
    with weights.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        header=json.loads(stream.read(size))
    attention_weights={key:dict(dtype=value["dtype"],shape=value["shape"]) for key,value in header.items()
                       if key.startswith("decoder.mid_block.attentions.0.") and key.endswith(".weight")}
    return dict(configs=configs,
                current_source_sha256={name:sha256(ROOT/name) for name in sources},
                dit_hlo=dit,vae_hlo=vae,
                vae_shape=dict(output_resolution=[1024,1024],packed_latents=[1,128,64,64],
                               unpacked_latents=[1,32,128,128],attention_tokens=16384,
                               attention_heads=1,attention_head_dim=384,
                               projection_weights=attention_weights),
                conclusions=["Current production DiT configurations explicitly request cuDNN; historical native baselines remain explicit.",
                             "Measured DiT forward and backward HLO contain cuDNN FMHA custom calls with is_flash_attention=true.",
                             "Measured FP32, late-mixed and all-stage-mixed VAE HLO contain no cuDNN FMHA.",
                             "VAE spatial attention has one head of width 384 over 16384 tokens at 1024 output.",
                             "HLO files are saved historical compiled evidence; current source hashes and benchmark source hashes are recorded separately."])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",required=True,type=int,help="Physical CUDA GPU; reject cards already using >1500MiB")
    parser.add_argument("--output",default=str(ROOT/"reports/vae_attention_backend_audit.json"))
    parser.add_argument("--actual-tokens",action="store_true")
    args=parser.parse_args(argv)
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    inventory=subprocess.check_output(["nvidia-smi","-i",str(args.device),
        "--query-gpu=index,name,uuid,memory.used","--format=csv,noheader,nounits"],text=True).strip()
    if int(inventory.split(",")[-1].strip())>1500: raise RuntimeError("Requested GPU is busy: "+inventory)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device),JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",XLA_PYTHON_CLIENT_MEM_FRACTION=".5",
        JAX_ENABLE_COMPILATION_CACHE="false",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    report=dict(status="running",recorded_utc=datetime.now(timezone.utc).isoformat(),
        argv=list(sys.argv if argv is None else [sys.argv[0],*argv]),script_sha256=sha256(__file__),
        inventory=inventory,physical_device=args.device,static_audit=static_audit(),probes=[])
    write_json(output,report)
    started=time.monotonic()
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import jaxlib
        from jax._src.cudnn import fused_attention_stablehlo as fused_source
        device=jax.devices()[0]
        source_path=Path(fused_source.__file__).resolve()
        source_lines=source_path.read_text().splitlines()
        report.update(versions=dict(python=sys.version,jax=jax.__version__,jaxlib=jaxlib.__version__),
            device=str(device),runtime_environment={key:os.environ.get(key) for key in
            ("CUDA_VISIBLE_DEVICES","JAX_PLATFORMS","XLA_FLAGS","JAX_ENABLE_COMPILATION_CACHE")},
            installed_jax_constraint=dict(path=str(source_path),sha256=sha256(source_path),
                excerpt=[dict(line=i+1,text=source_lines[i]) for i in range(353,399)]))
        def forward(q,k,v):
            return jax.nn.dot_product_attention(q,k,v,scale=384**-.5,
                                               is_causal=False,implementation="cudnn")
        def forward_vjp(q,k,v,cotangent):
            value,pullback=jax.vjp(forward,q,k,v)
            return value,pullback(cotangent)
        for dtype_name in ("float16","bfloat16"):
            dtype=getattr(jnp,dtype_name)
            small_supported=False
            for tokens in ([16,16384] if args.actual_tokens else [16]):
                if tokens!=16 and not small_supported:
                    report["probes"].append(dict(dtype=dtype_name,tokens=tokens,heads=1,head_dim=384,
                        status="skipped",reason="Identical head_dim=384 rejected by small-token probe; do not change model heads"))
                    continue
                shape=(1,tokens,1,384)
                rng=np.random.default_rng(20261020)
                arrays=[jax.device_put(rng.normal(size=shape).astype(np.float32),device).astype(dtype) for _ in range(4)]
                passed=True
                for phase,fn,inputs in (("forward",forward,arrays[:3]),
                                       ("forward_vjp",forward_vjp,arrays)):
                    item=dict(dtype=dtype_name,shape=list(shape),heads=1,head_dim=384,
                              tokens=tokens,phase=phase,implementation="cudnn")
                    report["probes"].append(item)
                    begun=time.monotonic()
                    try:
                        compiled=jax.jit(fn).lower(*inputs).compile()
                        item["compile_seconds"]=time.monotonic()-begun
                        result=compiled(*inputs)
                        jax.block_until_ready(result)
                        item["finite"]=all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(result))
                        item["status"]="supported" if item["finite"] else "nonfinite"
                        hlo=compiled.as_text()
                        folder=ROOT/"outputs/vae_cudnn_compatibility"
                        folder.mkdir(parents=True,exist_ok=True)
                        hlo_path=folder/f"{dtype_name}_{tokens}_{phase}.hlo.txt"
                        hlo_path.write_text(hlo)
                        item["hlo"]=hlo_evidence(hlo_path)
                        if not item["hlo"]["fmha_instructions"]:
                            raise AssertionError("Explicit cuDNN test produced no FMHA evidence")
                        for leaf in jax.tree.leaves(result): leaf.delete()
                        del result,compiled
                    except Exception as error:
                        passed=False
                        item.update(status="rejected",error_type=type(error).__name__,
                                    error=str(error),traceback=traceback.format_exc())
                    item["seconds"]=time.monotonic()-begun
                    write_json(output,report)
                small_supported=passed if tokens==16 else small_supported
                for array in arrays: array.delete()
                del arrays
                jax.clear_caches();gc.collect()
        report["status"]="audit_completed"
        report["memory_stats"]=device.memory_stats()
        report["compatibility_conclusion"]=("VAE head_dim384 is rejected by installed explicit cuDNN API."
            if all(p["status"]=="rejected" for p in report["probes"] if p["tokens"]==16)
            else "Inspect individual probe records; no blanket backend claim.")
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"]=time.monotonic()-started
        write_json(output,report)
        print(json.dumps(dict(status=report["status"],output=str(output))),flush=True)


if __name__=="__main__":
    main()
