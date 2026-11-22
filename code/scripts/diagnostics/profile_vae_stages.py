#!/usr/bin/env python3
"""Locate VAE cost by separately compiling FP32 decoder stages.

Stage times are diagnostic and must not be presented as complete-decode time:
stage boundaries alter fusion, dispatch, and memory lifetime. Only the isolated
convolution-free attention stage compares local GPU autotune levels 0 and 4.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from moss_art.config import DEFAULTS
from moss_art.locking import output_lock
from moss_art.runtime import configure_process, preflight
from scripts.benchmark_rollout import GPUMonitor
from scripts.benchmark_vae_precision import atomic_json, load_latents, sha256

STAGES = ("input", "mid_pre", "mid_attention", "mid_post", "up0", "up1", "up2", "up3", "output")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--sample", type=int, default=3, help="Zero-based sample index; default fourth image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reserve-dit-gib", type=float, default=9.6)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--attention-chunk-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-dir", default="$MODEL_ROOT/MOSS-Image-SFT")
    parser.add_argument("--reference-raw", help="Optional full FP32 decoder raw NCHW RGB NPY for comparison")
    args = parser.parse_args(argv)
    if args.device < 0 or args.sample < 0 or min(args.warm_repeats, args.attention_chunk_size, args.cpu_threads) < 1:
        parser.error("device/sample must be nonnegative; repeat/chunk/thread counts must be positive")
    if not math.isfinite(args.reserve_dit_gib) or args.reserve_dit_gib < 0:
        parser.error("reserve-dit-gib must be finite and nonnegative")
    return args


def configure(args):
    config = copy.deepcopy(DEFAULTS)
    config["runtime"].update(devices=[args.device], cpu_threads=args.cpu_threads, memory_fraction=.9)
    inventory = preflight(config)
    # Preserve caller compiler/precision environment; do not inject global XLA policy.
    configure_process(config, args.device)
    return inventory


def stage_function(name, config, chunk_size):
    """Static configuration closes over no weights or device arrays."""
    from vendor.moss_jax import vae
    import jax.numpy as jnp
    groups = int(vae._get(config, "norm_num_groups"))

    def stage(params, x):
        if name == "input":
            x = vae.unpack_latents(params, x, config)
            if bool(vae._get(config, "use_post_quant_conv")):
                x = vae._conv2d(params, x, "post_quant_conv")
            return vae._conv2d(params, x, "decoder.conv_in")
        if name == "mid_pre":
            return vae._resnet(params, x, "decoder.mid_block.resnets.0", groups)
        if name == "mid_attention":
            if bool(vae._get(config, "mid_block_add_attention")):
                return vae._attention(params, x, "decoder.mid_block.attentions.0", groups, chunk_size)
            return x
        if name == "mid_post":
            return vae._resnet(params, x, "decoder.mid_block.resnets.1", groups)
        if name.startswith("up"):
            index = int(name[2:])
            prefix = f"decoder.up_blocks.{index}"
            for resnet in range(int(vae._get(config, "layers_per_block")) + 1):
                x = vae._resnet(params, x, f"{prefix}.resnets.{resnet}", groups)
            if index < len(vae._get(config, "up_block_types")) - 1:
                x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
                x = vae._conv2d(params, x, prefix + ".upsamplers.0.conv")
            return x
        if name == "output":
            return vae._conv2d(params, vae._silu(vae._group_norm(
                params, x, "decoder.conv_norm_out", groups)), "decoder.conv_out")
        raise ValueError(f"Unknown stage {name}")
    return stage


def stage_params(params, name):
    prefixes = {
        "input": ("bn.", "post_quant_conv.", "decoder.conv_in."),
        "mid_pre": ("decoder.mid_block.resnets.0.",),
        "mid_attention": ("decoder.mid_block.attentions.0.",),
        "mid_post": ("decoder.mid_block.resnets.1.",),
        "output": ("decoder.conv_norm_out.", "decoder.conv_out."),
    }
    selected = prefixes.get(name, (f"decoder.up_blocks.{name[2:]}.",))
    return {key: value for key, value in params.items() if key.startswith(selected)}


def array_stats(value):
    array = np.asarray(value)
    finite = bool(np.isfinite(array).all())
    return {
        "shape": list(array.shape), "dtype": str(array.dtype), "bytes": int(array.nbytes),
        "finite": finite, "min": float(array.min()) if finite else None,
        "max": float(array.max()) if finite else None,
    }


def hlo_evidence(compiled, path):
    hlo = compiled.as_text()
    path.write_text(hlo)
    targets, instructions, parameters = {}, [], []
    for number, line in enumerate(hlo.splitlines(), 1):
        match = re.search(r'custom_call_target="([^"]+)"', line)
        if match:
            targets[match.group(1)] = targets.get(match.group(1), 0) + 1
        if " parameter(" in line:
            parameters.append({"line": number, "instruction": line.strip()[:1000]})
        if "convolution(" in line or " dot(" in line or match:
            instructions.append({"line": number, "target": match.group(1) if match else None,
                                 "instruction": line.strip()[:1800]})
    memory = compiled.memory_analysis()
    return {
        "path": path.name, "sha256": sha256(path), "custom_call_counts": targets,
        "parameter_instructions": parameters, "operation_instructions": instructions,
        "compiled_memory": {key: int(getattr(memory, key)) for key in (
            "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")},
    }


def comparison(actual, reference):
    a, b = np.asarray(actual, np.float32), np.asarray(reference, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Comparison shape differs: {a.shape} vs {b.shape}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite": False}
    diff = a.astype(np.float64) - b.astype(np.float64)
    denominator = float(np.linalg.norm(b.astype(np.float64).ravel()))
    return {"finite": True, "max_abs": float(np.max(np.abs(diff))),
            "relative_l2": float(np.linalg.norm(diff.ravel())) / denominator if denominator else None,
            "bitwise_equal": bool(np.array_equal(a, b))}


def main(argv=None):
    args = arguments(argv)
    latents = load_latents(args.latents)
    if args.sample >= len(latents):
        raise ValueError("Selected sample does not exist")
    inventory = configure(args)
    import jax
    import jax.numpy as jnp
    from vendor.moss_jax.weights import load_config, load_host_params
    from vendor.moss_jax import vae

    output = Path(args.output).resolve()
    with output_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Use a new empty output directory")
        report = {
            "status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "argv": vars(args), "physical_device": args.device, "inventory": inventory,
            "input_sha256": sha256(args.latents), "input_shape": list(latents.shape),
            "sample_index": args.sample, "dtype": "float32", "stages": [],
            "source_hashes": {name: sha256(ROOT / name) for name in (
                "scripts/diagnostics/profile_vae_stages.py", "vendor/moss_jax/vae.py",
                "moss_art/runtime.py", "scripts/benchmark_vae_precision.py")},
            "runtime_environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "TF_FP16_CONV_USE_FP32_COMPUTE",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
            "timing_scope": "Independent stage compile, synchronized device-only first/warm calls; excludes D2H statistics/HLO; do not equate summed stage times with complete-decode latency",
            "reservation_scope": "Device buffers only; no DiT computation",
            "pipeline_selection": "Every downstream stage receives explicit per-JIT autotune0 FP32 output; per-JIT attention autotune4 is a side comparison only; global XLA_FLAGS preserved",
            "convolution_accumulator_policy": "XLA/backend default unless caller supplied a precision environment override",
        }
        atomic_json(output / "summary.json", report)
        monitor = GPUMonitor(args.device, True)
        monitor.start()
        reservation = []
        started = time.monotonic()
        try:
            device = jax.devices()[0]
            monitor.phase = "reserve_dit_memory"
            remaining = round(args.reserve_dit_gib * 1024**3)
            while remaining:
                count = min(remaining, 256 * 1024**2)
                reservation.append(jax.device_put(np.zeros(count, np.uint8), device))
                reservation[-1].block_until_ready()
                remaining -= count
            report["reservation_bytes"] = sum(value.nbytes for value in reservation)
            monitor.phase = "load_vae"
            config = load_config(args.model_dir, "vae")
            if tuple(vae._get(config, "up_block_types")) != ("UpDecoderBlock2D",) * 4:
                raise ValueError("This diagnostic requires the actual four-stage MOSS VAE")
            unpacked_height = latents.shape[2] * vae._get(config, "patch_size")[0]
            unpacked_width = latents.shape[3] * vae._get(config, "patch_size")[1]
            sample_size = vae._get(config, "sample_size")
            sample_size = sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
            threshold = int(sample_size) // (2 ** (len(vae._get(config, "block_out_channels")) - 1))
            if max(unpacked_height, unpacked_width) > threshold:
                raise ValueError("Stage diagnostics target direct-path latents at/below the tiling threshold")
            host = load_host_params(args.model_dir, "vae")
            params = {key: jax.device_put(np.asarray(value, np.float32), device)
                      for key, value in host.items()
                      if key.startswith(("bn.", "post_quant_conv.", "decoder."))}
            del host
            jax.block_until_ready(params)
            report["vae_config"] = config
            report["parameter_bytes"] = sum(value.nbytes for value in params.values())
            x = jax.device_put(latents[args.sample:args.sample+1], device)
            x.block_until_ready()

            def run_stage(name, function, local_params, value, level):
                label = f"{name}_autotune{level}"
                monitor.phase = label
                entry = {"name": name, "autotune_level": level, "input": array_stats(value),
                         "parameter_dtypes": sorted({str(p.dtype) for p in local_params.values()}),
                         "parameter_bytes": sum(p.nbytes for p in local_params.values())}
                print(json.dumps({"event": "stage_start", "name": name, "autotune_level": level}), flush=True)
                options = {"xla_gpu_autotune_level": level}
                entry["compiler_options"] = options
                jit = jax.jit(function, compiler_options=options)
                begin = time.monotonic()
                compiled = jit.lower(local_params, value).compile()
                entry["compile_seconds"] = time.monotonic() - begin
                result = None
                times = []
                for _ in range(args.warm_repeats + 1):
                    if result is not None:
                        result.delete()
                    begin = time.monotonic()
                    result = compiled(local_params, value)
                    result.block_until_ready()
                    times.append(time.monotonic() - begin)
                entry["first_run_seconds"], entry["warm_seconds"] = times[0], times[1:]
                entry["warm_median_seconds"] = float(np.median(times[1:]))
                entry["output"] = array_stats(result)
                if not entry["output"]["finite"]:
                    raise FloatingPointError(f"{label} output is nonfinite")
                entry["hlo"] = hlo_evidence(compiled, output / f"{label}.hlo.txt")
                if level == 4 and any("conv" in target.lower() for target in entry["hlo"]["custom_call_counts"]):
                    raise AssertionError("Attention-only tuned program unexpectedly contains convolution")
                entry["jax_memory"] = device.memory_stats()
                report["stages"].append(entry)
                atomic_json(output / "summary.json", report)
                print(json.dumps({"event": "stage_done", "name": name, "autotune_level": level,
                                  "warm_median_seconds": entry["warm_median_seconds"],
                                  "output": entry["output"]}), flush=True)
                return result, entry

            for name in STAGES:
                local = stage_params(params, name)
                fn = stage_function(name, config, args.attention_chunk_size)
                result, baseline = run_stage(name, fn, local, x, 0)
                if name == "mid_attention":
                    alternative, tuned = run_stage(name, fn, local, x, 4)
                    report["attention_comparison"] = {
                        "numeric": comparison(alternative, result),
                        "warm_speedup": baseline["warm_median_seconds"] / tuned["warm_median_seconds"],
                        "baseline_seconds": baseline["warm_median_seconds"],
                        "tuned_seconds": tuned["warm_median_seconds"],
                    }
                    alternative.delete()
                    atomic_json(output / "summary.json", report)
                x.delete()
                x = result
                gc.collect()
            np.save(output / "stagewise_fp32.raw.npy", np.asarray(x, np.float32), allow_pickle=False)
            report["final_output"] = array_stats(x)
            if args.reference_raw:
                report["whole_decoder_reference"] = {
                    "path": str(Path(args.reference_raw).resolve()),
                    "sha256": sha256(args.reference_raw),
                    "comparison": comparison(x, np.load(args.reference_raw, allow_pickle=False)),
                }
            report["baseline_stage_sum_seconds_not_end_to_end"] = sum(
                stage["warm_median_seconds"] for stage in report["stages"] if stage["autotune_level"] == 0)
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failure_phase=monitor.phase, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report["total_seconds"] = time.monotonic() - started
            report["nvidia_smi"] = monitor.finish(output / "gpu_memory.csv")
            atomic_json(output / "summary.json", report)
            print(json.dumps({"event": "profile_finished", "status": report["status"],
                              "total_seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
