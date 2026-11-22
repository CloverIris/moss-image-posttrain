#!/usr/bin/env python3
"""Capture optimized strict-FP32 VAE cuDNN FFI GPU events; original trace tool unchanged."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from moss_art.config import DEFAULTS
from moss_art.runtime import preflight, configure_process
from moss_art.locking import output_lock
from scripts.benchmark_vae_precision import atomic_json, sha256, load_latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--variant", default="fp32", choices=("fp32",))
    parser.add_argument("--conv-workspace-mib", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latents", default=str(ROOT/"outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy"))
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=1)
    parser.add_argument("--reserve-dit-gib", type=float, default=9.6)
    parser.add_argument("--model-dir", default="$MODEL_ROOT/MOSS-Image-SFT")
    args = parser.parse_args()
    if args.trace_repeats < 1 or args.device < 0 or args.reserve_dit_gib < 0:
        parser.error("Invalid repetitions, device or reservation")
    latents = load_latents(args.latents)
    if not 0 <= args.sample < len(latents):
        parser.error("Sample index outside latent batch")
    config = copy.deepcopy(DEFAULTS)
    config["runtime"].update(devices=[args.device], cpu_threads=4, memory_fraction=.9)
    inventory = preflight(config)
    # Preserve caller compiler/precision environment; do not inject global XLA policy.
    cupti_dir = "/usr/local/cuda/extras/CUPTI/lib64"
    os.environ["LD_LIBRARY_PATH"] = cupti_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    configure_process(config, args.device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    from vendor.moss_jax import vae
    from vendor.moss_jax.weights import load_config, load_host_params
    from moss_art.assets import component_identity

    output = Path(args.output).resolve()
    with output_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Refusing to overwrite trace output")
        report = {
            "status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv, "variant": args.variant, "physical_device": args.device, "inventory": inventory,
            "input_sha256": sha256(args.latents), "sample_index": args.sample,
            "model_identity": component_identity(args.model_dir, "vae"),
            "versions": {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__},
            "runtime_environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "TF_FP16_CONV_USE_FP32_COMPUTE", "LD_LIBRARY_PATH", "CUDNN_LOGLEVEL_DBG", "CUDNN_LOGDEST_DBG")},
            "sources": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/"vendor/moss_jax/vae.py",
                       ROOT/"moss_art/vae_precision.py", ROOT/"moss_art/cudnn_conv.py", ROOT/"moss_art/native/cudnn_conv_ffi.cc"]},
            "policy": {"timing": "preloaded p,z explicit JIT arguments; compile then one synchronized warmup; trace executions synchronize; no host output transfer inside trace",
                       "whole_graph": True, "tiling": True, "attention_chunk_size": 256,
                       "compiler_environment": "Caller environment preserved; bounded cuDNN FFI requests FP32 compute explicitly without a global environment override",
                       "conv_backend": "cudnn_bounded", "workspace_mib": args.conv_workspace_mib, "precision": "strict FP32 including FP32 compute and rejection of Tensor Core/implicit precision reduction",
                       "named_scopes_added": False, "reservation": "device uint8 buffers, not actual DiT model"},
        }
        atomic_json(output/"capture.json", report)
        reservations = []
        try:
            device = jax.devices()[0]
            report["device"] = str(device)
            remaining = round(args.reserve_dit_gib * 1024**3)
            while remaining:
                count = min(remaining, 256*1024**2)
                reservations.append(jnp.zeros((count,), jnp.uint8))
                reservations[-1].block_until_ready()
                remaining -= count
            report["reserved_bytes"] = sum(x.nbytes for x in reservations)
            vae_config = load_config(args.model_dir, "vae")
            host = load_host_params(args.model_dir, "vae")
            params = {
                k: jax.device_put(np.asarray(v, np.float16 if args.variant == "full_fp16" and not k.startswith("bn.") else np.float32), device)
                for k,v in host.items() if k.startswith(("decoder.", "post_quant_conv.", "bn."))
            }
            del host
            z = jax.device_put(latents[args.sample:args.sample+1], device)
            jax.block_until_ready((params, z))
            report["parameter_bytes"] = sum(x.nbytes for x in params.values())
            from moss_art.vae_precision import decode_latents
            from moss_art.cudnn_conv import backend_info, plan_inventory
            def decode(p,z):
                return decode_latents(p,z,vae_config,precision="fp32",conv_backend="cudnn_bounded",
                    conv_workspace_mib=args.conv_workspace_mib,tiling=True,attention_chunk_size=256,clamp=False)
            began = time.monotonic()
            compiled = jax.jit(decode).lower(params,z).compile()
            report["compile_seconds"] = time.monotonic()-began
            report["backend_info"] = backend_info()
            report["plan_inventory"] = plan_inventory()
            (output/"optimized.hlo.txt").write_text(compiled.as_text())
            report["hlo_sha256"] = sha256(output/"optimized.hlo.txt")
            stats = compiled.memory_analysis()
            report["compiled_memory"] = {k:int(getattr(stats,k)) for k in
                ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}
            print(json.dumps({"event":"compiled","variant":args.variant,"seconds":report["compile_seconds"]}),flush=True)
            began = time.monotonic()
            warm = compiled(params,z)
            warm.block_until_ready()
            report["warmup_seconds"] = time.monotonic()-began
            raw = np.asarray(warm,dtype=np.float32).copy()
            report["warmup_finite"] = bool(np.isfinite(raw).all())
            warm.delete()
            if not report["warmup_finite"]:
                raise FloatingPointError("Warmup output nonfinite")
            np.save(output/"warmup.raw.npy",raw,allow_pickle=False)
            print(json.dumps({"event":"warmup_done","variant":args.variant,"seconds":report["warmup_seconds"]}),flush=True)
            report["traced_seconds"] = []
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = 2
            options.python_tracer_level = 0
            options.enable_hlo_proto = True
            options.raise_error_on_start_failure = True
            with jax.profiler.trace(output/"trace", create_perfetto_trace=True, profiler_options=options):
                for repeat in range(args.trace_repeats):
                    with jax.profiler.TraceAnnotation("vae_decode_traced", variant=args.variant, repeat=repeat):
                        began = time.monotonic()
                        result = compiled(params,z)
                        result.block_until_ready()
                        report["traced_seconds"].append(time.monotonic()-began)
            candidate = np.asarray(result,dtype=np.float32)
            report["traced_output_finite"] = bool(np.isfinite(candidate).all())
            report["traced_matches_warmup_bitwise"] = bool(np.array_equal(raw,candidate))
            report["final_memory"] = {k:int(v) for k,v in device.memory_stats().items() if isinstance(v,(int,np.integer))}
            report["trace_files"] = [{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":sha256(p)}
                                      for p in sorted((output/"trace").rglob("*")) if p.is_file()]
            report["status"] = "captured"
        except BaseException as error:
            report.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            raise
        finally:
            atomic_json(output/"capture.json",report)
            print(json.dumps({"status":report["status"],"report":str(output/"capture.json")}),flush=True)


if __name__ == "__main__":
    main()
