#!/usr/bin/env python3
"""Reproduce isolated per-jit autotune 0/4 MOSS linear benchmarks.

Example:
  $CONDA_PREFIX/envs/jax_env/bin/python scripts/diagnostics/gemm_autotune.py --device 3 \
    --output-prefix reports/gemm_repro

Every shape/level runs in a fresh child process with persistent compilation
caching disabled, preventing one tuned case from warming another case's cache.
Global compiler settings are inherited; both tested top-level JIT levels are explicit.
No model weights or convolutions are used. Imports never initialize a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from _common import add_common_arguments, save_hlo, save_report, setup


def run_case(args):
    root, path = setup(args, persistent_cache=False)
    sys.path.insert(0, str(root))
    import numpy as np
    import jax
    import jax.numpy as jnp
    from moss_art import lora
    from vendor.moss_jax import dit
    n, level = args.case, args.autotune_level
    rng = np.random.default_rng(n)
    x = jax.device_put(rng.standard_normal((4123, 3072), dtype=np.float32)).astype(jnp.bfloat16)
    weight = jax.device_put(rng.standard_normal((n, 3072), dtype=np.float32) / np.sqrt(3072)).astype(jnp.bfloat16)
    down = jax.device_put(rng.standard_normal((4, 3072), dtype=np.float32) * .01)
    up = jax.device_put(rng.standard_normal((n, 4), dtype=np.float32) * .01)
    jax.block_until_ready((x, weight, down, up))
    if n == 3072:
        def compute(x, weight, down, up):
            return lora.linear({"linear.weight": weight},
                               {"linear" + lora.DOWN: down, "linear" + lora.UP: up}, "linear", x, 1.)
    else:
        def compute(x, weight, down, up):
            return dit.linear({"linear.weight": weight}, "linear", x)
    began = time.monotonic()
    lowered = jax.jit(compute, compiler_options={"xla_gpu_autotune_level": level}).lower(x, weight, down, up)
    exe = lowered.compile()
    compile_seconds = time.monotonic() - began
    hlo = exe.as_text()
    jax.block_until_ready(exe(x, weight, down, up))
    timings = []
    for _ in range(args.repeats):
        began = time.perf_counter()
        output = exe(x, weight, down, up)
        jax.block_until_ready(output)
        timings.append(time.perf_counter() - began)
    array = np.asarray(output)
    report = {
        "status": "passed", "physical_gpu": args.device, "n": n, "level": level,
        "m": 4123, "k": 3072, "lora_rank": 4 if n == 3072 else 0,
        "compile_seconds": compile_seconds, "median_seconds": float(np.median(timings)),
        "min_seconds": min(timings), "warm_seconds": timings,
        "output_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "finite": bool(np.isfinite(array).all()), "global_XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "runtime_environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
            "TF_FP16_CONV_USE_FP32_COMPUTE", "JAX_ENABLE_COMPILATION_CACHE")},
        "compiler_options": {"xla_gpu_autotune_level": level},
        "persistent_cache_enabled": jax.config.jax_enable_compilation_cache,
        "custom_call_targets": sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo))),
        "kernel_configs": [line.strip() for line in hlo.splitlines() if "backend_config=" in line
                           and any(word in line.lower() for word in ("gemm", "triton", "cublas"))],
        **save_hlo(args, f"{n}_autotune{level}", exe)}
    if not report["finite"]:
        report["status"] = "failed"
    save_report(path, report)
    print(json.dumps(report), flush=True)
    if not report["finite"]:
        raise FloatingPointError("Nonfinite GEMM output")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, repeats=30)
    parser.add_argument("--case", type=int, choices=(3072, 16384), help=argparse.SUPPRESS)
    parser.add_argument("--autotune-level", type=int, choices=(0, 4), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case is not None:
        if args.autotune_level is None:
            parser.error("--case requires --autotune-level")
        return run_case(args)
    if args.autotune_level is not None:
        parser.error("--autotune-level is an internal child option")
    root = Path(__file__).resolve().parents[2]
    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    path = Path(str(prefix) + ".json")
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists: {path}; choose another prefix or --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "physical_gpu": args.device,
              "isolation": "Fresh process per shape/level; persistent JAX/XLA compilation cache disabled",
              "scope": "Standalone MOSS linear operations, not full-model throughput or compile memory",
              "cases": [], "comparisons": []}
    try:
        for n in (3072, 16384):
            for level in (0, 4):
                child_prefix = str(prefix) + f"_{n}_autotune{level}"
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--device", str(args.device), "--output-prefix", child_prefix,
                           "--hlo-dir", args.hlo_dir, "--repeats", str(args.repeats),
                           "--cpu-threads", str(args.cpu_threads), "--memory-fraction", str(args.memory_fraction),
                           "--case", str(n), "--autotune-level", str(level)]
                if args.allow_busy:
                    command.append("--allow-busy")
                if args.overwrite:
                    command.append("--overwrite")
                result = subprocess.run(command, text=True, capture_output=True, check=False, cwd=root)
                if result.stderr:
                    print(result.stderr, file=sys.stderr, flush=True)
                if result.returncode:
                    raise RuntimeError(f"Child failed with code {result.returncode}: {result.stdout}")
                row = json.loads(Path(child_prefix + ".json").read_text())
                report["cases"].append(row)
                save_report(path, report)
                print(json.dumps({"event": "measured", "n": n, "level": level,
                                  "compile_seconds": row["compile_seconds"],
                                  "median_seconds": row["median_seconds"]}), flush=True)
        for n in (3072, 16384):
            zero, four = [row for row in report["cases"] if row["n"] == n]
            report["comparisons"].append({
                "n": n, "speedup": zero["median_seconds"] / four["median_seconds"],
                "output_bitwise_same": zero["output_sha256"] == four["output_sha256"]})
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save_report(path, report)
        print(json.dumps({"event": "finished", "status": report["status"], "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
