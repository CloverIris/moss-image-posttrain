#!/usr/bin/env python3
"""Isolated full-FP16 VAE decoder with FP32 reductions; no production changes."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import re
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from moss_art.config import DEFAULTS
from moss_art.runtime import preflight, configure_process
from moss_art.locking import output_lock
from scripts.benchmark_rollout import GPUMonitor
from scripts.benchmark_vae_precision import (
    STAGES, sha256, atomic_json, load_latents, parse_indices, load_reference,
    rgb_pixels, comparison_metrics, comparison_panel,
)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Physical GPU index; existing busy cards are rejected")
    parser.add_argument("--precision", choices=("fp32", "float16"), default="float16")
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--samples", default="0", help="Comma-separated indices or all")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reserve-dit-gib", type=float, default=0.0)
    parser.add_argument("--model-dir", default="$MODEL_ROOT/MOSS-Image-SFT")
    parser.add_argument("--attention-chunk-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-tiling", action="store_true")
    args = parser.parse_args(argv)
    if args.device < 0 or args.repeats < 1 or args.cpu_threads < 1 or args.attention_chunk_size < 1:
        parser.error("device must be nonnegative; repeats, threads and attention chunk must be positive")
    if not math.isfinite(args.reserve_dit_gib) or args.reserve_dit_gib < 0:
        parser.error("--reserve-dit-gib must be finite and nonnegative")
    try:
        stages = STAGES if args.precision == "float16" else ()
        latents = load_latents(args.latents)
        indices = parse_indices(args.samples, len(latents))
    except ValueError as error:
        parser.error(str(error))
    input_sha = sha256(args.latents)
    reference, reference_metadata = (None, None)
    if args.reference:
        reference, reference_metadata = load_reference(args.reference, input_sha, indices, len(latents))
    config = copy.deepcopy(DEFAULTS)
    config["runtime"].update(devices=[args.device], cpu_threads=args.cpu_threads, memory_fraction=.9)
    inventory = preflight(config)
    # This diagnostic always uses autotune 0, scoped to this subprocess only.
    # Preserve caller compiler/precision environment; do not inject global XLA policy.
    configure_process(config, args.device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    from moss_art.assets import component_identity
    from scripts.diagnostics.vae_full_fp16 import decode_latents
    from vendor.moss_jax.weights import load_config, load_host_params
    from PIL import Image

    output = Path(args.output).resolve()
    with output_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Benchmark output is nonempty; select a new --output")
        root = Path(__file__).resolve().parents[2]
        source_paths = ("scripts/diagnostics/benchmark_vae_full_fp16.py",
                        "scripts/diagnostics/vae_full_fp16.py",
                        "scripts/benchmark_vae_precision.py", "scripts/benchmark_rollout.py", "vendor/moss_jax/vae.py", "vendor/moss_jax/weights.py",
                        "moss_art/runtime.py", "moss_art/assets.py")
        report = {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]), "precision": args.precision,
                  "low_precision_stages": stages,
                  "benchmark_variant": "full-FP16 decoder with FP32 reductions" if args.precision == "float16" else "parameterized diagnostic FP32 decoder",
                  "input_path": str(Path(args.latents).resolve()),
                  "input_sha256": input_sha, "input_shape": list(latents.shape), "sample_indices": indices,
                  "model_dir": str(Path(args.model_dir).resolve()), "model_identity": component_identity(args.model_dir, "vae"),
                  "source_hashes": {name: sha256(root / name) for name in source_paths},
                  "physical_device": args.device, "inventory": inventory,
                  "reference": reference_metadata,
                  "versions": {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                               "numpy": np.__version__},
                  "policy": {"tiling": not args.no_tiling, "clamp": False, "slicing": False,
                             "attention_chunk_size": args.attention_chunk_size,
                             "weight_storage": "Decoder device weights/affine parameters FP16 when precision=float16; BN buffers FP32; explicit runtime argument to JIT",
                             "low_precision_ops": (
                                 "FP16 convolution/linear outputs, bias, residual, SiLU, QKVO and attention I/O; decoder activations stay FP16 across stages; final raw RGB converted to FP32 for storage"
                                 if args.precision == "float16" else
                                 "No reduced precision; decoder parameters, activations, convolution/linear outputs, bias, residual and SiLU are FP32"
                             ),
                             "convolution_accumulator_policy": "Native convolution follows XLA/backend defaults and any caller-supplied environment; this script does not force a global precision policy. Half-output HLO alone does not prove accumulator precision",
                             "sensitive_ops": (
                                 "BN unpack FP32 outside decoder; GN statistics/affine FP32 with stored FP16 scale/bias promoted then output FP16; QK FP16 operands with FP32 logits; softmax FP32 then probabilities FP16; PV FP16 operands/F32 accumulation then FP16 output; tiling blends FP32 then model dtype after each direction; does not claim every instruction is FP16"
                                 if args.precision == "float16" else
                                 "BN unpack, GroupNorm statistics/affine, QK logits, softmax, PV reduction and tiling blends are FP32"
                             ),
                             "timing": "per-sample H2D latent transfer + first-only compilation + synchronized decode + D2H raw FP32; excludes parameter loading, reservation, file writes and comparison metrics",
                             "compilation_cache": "persistent cache enabled; first in process is not guaranteed cold",
                             "memory": "JAX peak is process lifetime; NVIDIA readings are sampled whole-card reserved use",
                             "reservation": "device buffers only, not actual DiT weights/computation"},
                  "runtime_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_COMPILATION_CACHE_DIR", "TF_FP16_CONV_USE_FP32_COMPUTE")},
                  "repeats": []}
        atomic_json(output / "summary.json", report)
        monitor = GPUMonitor(args.device, True)
        monitor.start()
        started = time.monotonic()
        device = None
        params, reservation = None, []
        def memory():
            stats = device.memory_stats() if device is not None else {}
            return {"jax": {key: int(value) for key, value in (stats or {}).items() if isinstance(value, (int, np.integer))},
                    "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        try:
            device = jax.devices()[0]
            report["jax_device"] = str(device)
            monitor.phase = "reserve_dit_memory"
            report["before_reservation"] = memory()
            begun = time.monotonic()
            requested_bytes = round(args.reserve_dit_gib * 1024 ** 3)
            remaining = requested_bytes
            while remaining:
                chunk = min(remaining, 256 * 1024 ** 2)
                with jax.default_device(device):
                    reservation.append(jnp.zeros((chunk,), dtype=jnp.uint8))
                reservation[-1].block_until_ready()
                remaining -= chunk
            report["reservation"] = {"requested_gib": args.reserve_dit_gib, "buffer_bytes": sum(value.nbytes for value in reservation),
                                     "seconds": time.monotonic() - begun, "after": memory()}
            monitor.phase = "load_vae"
            begun = time.monotonic()
            vae_config = load_config(args.model_dir, "vae")
            host = load_host_params(args.model_dir, "vae")
            host = {key: value for key, value in host.items() if key.startswith(("decoder.", "post_quant_conv.", "bn."))}
            params = {key: jax.device_put(np.asarray(value, np.float16 if args.precision == "float16" and not key.startswith("bn.") else np.float32), device)
                      for key, value in host.items()}
            del host
            jax.block_until_ready(params)
            report["parameter_bytes"] = sum(value.nbytes for value in params.values())
            report["parameter_dtype_counts"] = {dtype: sum(str(value.dtype) == dtype for value in params.values()) for dtype in sorted({str(value.dtype) for value in params.values()})}
            report["load_seconds"] = time.monotonic() - begun
            report["after_load"] = memory()
            report["vae_config"] = vae_config
            decode = jax.jit(lambda p, z: decode_latents(p, z, vae_config,
                precision=args.precision, tiling=not args.no_tiling,
                attention_chunk_size=args.attention_chunk_size, clamp=False))
            compiled = None
            baseline = {}
            hlo_saved = False
            for repeat in range(args.repeats):
                monitor.phase = f"decode_{repeat}"
                folder = output / f"repeat_{repeat:02d}"
                folder.mkdir()
                entry = {"repeat": repeat, "label": "first_in_process" if repeat == 0 else "warm_in_process", "samples": []}
                report["repeats"].append(entry)
                for index in indices:
                    sample = {"sample_index": index}
                    entry["samples"].append(sample)
                    begun = time.monotonic()
                    tensor = jax.device_put(latents[index:index+1], device)
                    result = None
                    try:
                        if compiled is None:
                            compile_started = time.monotonic()
                            compiled = decode.lower(params, tensor).compile()
                            report["compile_seconds"] = time.monotonic() - compile_started
                            stats = compiled.memory_analysis()
                            report["compiled_memory"] = {name: int(getattr(stats, name)) for name in
                                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
                            report["compiled_memory_scope"] = "VAE executable only; excludes device reservation and allocator pool"
                        result = compiled(params, tensor)
                        result.block_until_ready()
                        raw = np.asarray(result, dtype=np.float32).copy()
                        sample["decode_seconds"] = time.monotonic() - begun
                        if not hlo_saved:
                            # Evidence collection is outside decode timing.
                            hlo = compiled.as_text()
                            (output / "optimized.hlo.txt").write_text(hlo)
                            targets = {}
                            evidence = []
                            for number, line in enumerate(hlo.splitlines(), 1):
                                match = re.search(r'custom_call_target="([^"]+)"', line)
                                if match:
                                    targets[match.group(1)] = targets.get(match.group(1), 0) + 1
                                if ("convolution(" in line or " dot(" in line or
                                    (match and any(word in match.group(1).lower() for word in ("conv", "gemm", "matmul")))):
                                    evidence.append({"line": number,
                                                     "custom_call_target": match.group(1) if match else None,
                                                     "instruction_prefix": line.strip()[:1800]})
                            report["hlo_evidence"] = {"path": "optimized.hlo.txt", "sha256": sha256(output / "optimized.hlo.txt"),
                                "custom_call_counts": targets, "convolution_and_linear_instructions": evidence,
                                "note": "Instruction types and custom-call targets are measured compiled evidence; full HLO is authoritative."}
                            report["hlo_evidence"]["fp16_activation_operations"] = [
                                {"line": number, "instruction": line.strip()[:1800]}
                                for number, line in enumerate(hlo.splitlines(), 1)
                                if re.search(r"= f16\[", line)
                                and any(operation in line for operation in (" add(", " multiply(", " logistic(", " convert(", " transpose("))
                            ]
                            atomic_json(output / "hlo_evidence.json", report["hlo_evidence"])
                            hlo_saved = True
                        expected_shape = (1, 3, latents.shape[2] * 16, latents.shape[3] * 16)
                        if raw.shape != expected_shape:
                            raise ValueError(f"Unexpected decoded RGB shape {raw.shape}; expected {expected_shape}")
                        sample["raw_shape"] = list(raw.shape)
                        sample["raw_finite"] = bool(np.isfinite(raw).all())
                        sample["raw_path"] = f"repeat_{repeat:02d}/sample-{index:06d}.raw.npy"
                        np.save(output / sample["raw_path"], raw, allow_pickle=False)
                        if not sample["raw_finite"]:
                            raise FloatingPointError(f"VAE output contains nonfinite values for sample {index}")
                        sample["raw_min"], sample["raw_max"] = float(raw.min()), float(raw.max())
                        sample["raw_sha256"] = sha256(output / sample["raw_path"])
                        sample["png_path"] = f"repeat_{repeat:02d}/sample-{index:06d}.png"
                        Image.fromarray(rgb_pixels(raw)).save(output / sample["png_path"])
                        if repeat == 0:
                            baseline[index] = raw.copy()
                        else:
                            sample["same_seed_raw_bitwise_equal"] = bool(np.array_equal(raw, baseline[index]))
                            sample["same_seed_raw_max_abs"] = float(np.max(np.abs(raw - baseline[index])))
                        if reference is not None:
                            sample["comparison"] = comparison_metrics(raw, reference[index])
                            sample["comparison_path"] = f"repeat_{repeat:02d}/sample-{index:06d}.comparison.png"
                            comparison_panel(reference[index], raw, output / sample["comparison_path"],
                                             args.precision + ":" + (",".join(stages) or "none"))
                        sample["memory_after_decode"] = memory()
                    finally:
                        tensor.delete()
                        if result is not None:
                            result.delete()
                    atomic_json(output / "summary.json", report)
                    print(json.dumps({"event": "sample_complete", "repeat": repeat, **sample}), flush=True)
                entry["decode_seconds"] = sum(sample["decode_seconds"] for sample in entry["samples"])
                entry["seconds_per_image"] = entry["decode_seconds"] / len(indices)
                entry["after_repeat"] = memory()
                atomic_json(output / "summary.json", report)
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failure_phase=monitor.phase, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report["total_seconds"] = time.monotonic() - started
            report["nvidia_smi"] = monitor.finish(output / "gpu_memory.csv")
            try:
                report["final_memory"] = memory()
            except BaseException as error:
                report["final_memory_error"] = str(error)
            atomic_json(output / "summary.json", report)
            print(json.dumps({"event": "benchmark_finished", "status": report["status"],
                              "total_seconds": report["total_seconds"]}), flush=True)

if __name__ == "__main__":
    main()
