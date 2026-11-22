"""Standard-library-only setup for reproducible remote GPU diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def add_common_arguments(parser: argparse.ArgumentParser, *, repeats=20):
    parser.add_argument("--device", type=int, required=True, help="Physical nvidia-smi GPU index")
    parser.add_argument("--output-prefix", required=True, help="New JSON report prefix, relative to the project root")
    parser.add_argument("--hlo-dir", default="outputs/attention_hlo_archives",
                        help="Directory for optimized HLO and large StableHLO text")
    parser.add_argument("--repeats", type=int, default=repeats, help="Timed warm executions")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=.9)
    parser.add_argument("--allow-busy", action="store_true", help="Allow a GPU already using more than 1500 MiB")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this report prefix")


def setup(args, *, persistent_cache=True):
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if args.device < 0 or args.repeats < 1 or args.cpu_threads < 1 or not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid device, repetitions, CPU thread count, or memory fraction")
    report_path = Path(args.output_prefix + ".json").resolve()
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists: {report_path}; choose another prefix or --overwrite")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = subprocess.run(
        ["nvidia-smi", "-i", str(args.device), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    used_mib = int(inventory.stdout.strip())
    if used_mib > 1500 and not args.allow_busy:
        raise RuntimeError(f"GPU {args.device} is busy ({used_mib} MiB); no processes were changed")
    # Must happen before importing JAX, NumPy, or any moss_art module.
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.device), JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.memory_fraction),
        JAX_ENABLE_X64="true", OMP_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads),
        JAX_ENABLE_COMPILATION_CACHE="true" if persistent_cache else "false",
        JAX_COMPILATION_CACHE_DIR=str(root / ".jax_cache"))
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        count = min(len(cores), args.cpu_threads)
        offset = (args.device * count) % len(cores)
        os.sched_setaffinity(0, (cores + cores)[offset:offset + count])
    return root, report_path


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def save_hlo(args, label, executable, lowered=None):
    folder = Path(args.hlo_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output_prefix).name + "_" + label
    optimized = folder / (stem + ".hlo.txt")
    optimized.write_text(executable.as_text())
    paths = {"compiled_hlo_path": str(optimized)}
    if lowered is not None:
        stable = folder / (stem + ".stablehlo.txt")
        stable.write_text(str(lowered.compiler_ir(dialect="stablehlo")))
        paths["stablehlo_path"] = str(stable)
    return paths
