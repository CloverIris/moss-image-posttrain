#!/usr/bin/env python3
"""Benchmark isolated native-JAX VAE precision profiles on explicit, idle GPUs.

Timing includes synchronization and GPU-to-host FP32 transfer. First decode
includes explicit compilation; warm decodes reuse the same compiled executable.
--reference accepts a completed benchmark directory or an unclipped NCHW .npy.
Reservations occupy device memory only: they do not emulate DiT computation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from moss_art.config import DEFAULTS
from moss_art.runtime import preflight, configure_process
from moss_art.locking import output_lock
from scripts.benchmark_rollout import GPUMonitor

STAGES = ("input", "mid", "up0", "up1", "up2", "up3", "output")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_indices(value, count):
    try:
        indices = list(range(count)) if value == "all" else [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--samples must be comma-separated integer indices or all") from error
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= count for index in indices):
        raise ValueError("--samples must select existing unique nonnegative indices")
    return indices


def parse_stages(value):
    stages = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(stages) != len(set(stages)) or any(stage not in STAGES for stage in stages):
        raise ValueError("--stages must select unique names from " + ",".join(STAGES))
    return stages


def load_latents(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as archive:
            key = next((name for name in ("final_latents", "latents", "endpoints") if name in archive), None)
            if key is None:
                raise ValueError("Latent NPZ needs final_latents, latents or endpoints")
            values = np.array(archive[key], dtype=np.float32, copy=True)
    else:
        values = np.asarray(loaded, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] != 128 or min(values.shape[2:]) < 1:
        raise ValueError("Expected packed latents [N,128,H,W]")
    if not np.isfinite(values).all():
        raise ValueError("Latents contain nonfinite values")
    return values


def rgb_pixels(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 4 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[0] != 3 or not np.isfinite(raw).all():
        raise ValueError("RGB conversion needs finite [3,H,W] or [1,3,H,W]")
    return np.rint(np.clip((raw.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)).astype(np.uint8)


def rgb_ssim(reference, candidate):
    """Mean RGB SSIM: Gaussian 11x11 sigma=1.5, population covariances, valid crop.

    Compute independently in float64 for each RGB channel with data_range=255,
    K1=.01, K2=.03; average over pixels and channels. No luminance transform.
    """
    from scipy.ndimage import gaussian_filter
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("SSIM needs matching HWC RGB")
    if min(reference.shape[:2]) < 11:
        raise ValueError("SSIM requires images at least 11x11")
    scores = []
    c1, c2 = (.01 * 255) ** 2, (.03 * 255) ** 2
    blur = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5, mode="reflect")
    for channel in range(3):
        a = reference[..., channel].astype(np.float64)
        b = candidate[..., channel].astype(np.float64)
        ma, mb = blur(a), blur(b)
        va, vb = blur(a * a) - ma * ma, blur(b * b) - mb * mb
        covariance = blur(a * b) - ma * mb
        score = ((2 * ma * mb + c1) * (2 * covariance + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
        scores.append(float(score[5:-5, 5:-5].mean()))
    return {"rgb_mean": float(np.mean(scores)), "per_channel": scores,
            "definition": "Gaussian11x11 sigma1.5; RGB channel mean; population covariance; valid 5px border crop; K1=.01 K2=.03 range255"}


def comparison_metrics(raw, reference):
    a, b = np.asarray(raw, np.float32), np.asarray(reference, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Reference shape differs: candidate {a.shape}, reference {b.shape}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"raw_finite": False}
    delta = a.astype(np.float64) - b.astype(np.float64)
    denominator = float(np.linalg.norm(b.astype(np.float64).ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    pixels, reference_pixels = rgb_pixels(a), rgb_pixels(b)
    pixel_delta = pixels.astype(np.float64) - reference_pixels.astype(np.float64)
    mse = float(np.mean(pixel_delta ** 2))
    return {"raw_finite": True, "raw_max_abs": float(np.max(np.abs(delta))),
            "raw_relative_l2": numerator / denominator if denominator else (0.0 if numerator == 0 else None),
            "raw_reference_l2": denominator, "raw_error_l2": numerator,
            "raw_bitwise_equal": bool(np.array_equal(a, b)),
            "pixel_mae_0_255": float(np.mean(np.abs(pixel_delta))),
            "pixel_max_abs_0_255": int(np.max(np.abs(pixel_delta))), "pixel_mse": mse,
            "pixel_psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "pixel_psnr_infinite": mse == 0, "pixels_equal": bool(np.array_equal(pixels, reference_pixels)),
            "ssim": rgb_ssim(reference_pixels, pixels)}


def load_reference(path, input_sha, indices, input_count):
    path = Path(path).resolve()
    if path.is_dir():
        report = json.loads((path / "summary.json").read_text())
        if report.get("status") != "passed":
            raise ValueError("Reference benchmark must have completed successfully")
        if report["input_sha256"] != input_sha:
            raise ValueError("Reference latent input SHA256 does not match")
        if report["precision"] != "fp32":
            raise ValueError("Reference benchmark must use precision=fp32")
        mapping = {entry["sample_index"]: entry["raw_path"] for entry in report["repeats"][0]["samples"]}
        if any(index not in mapping for index in indices):
            raise ValueError("Reference benchmark lacks a selected sample")
        arrays = {index: np.load(path / mapping[index], allow_pickle=False) for index in indices}
        return arrays, {"kind": "benchmark_directory", "path": str(path),
                        "summary_sha256": sha256(path / "summary.json"), "precision": report["precision"],
                        "source_hashes": report["source_hashes"]}
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[1] != 3:
        raise ValueError("Reference NPY must contain NCHW RGB")
    if array.shape[0] == input_count:
        arrays = {index: array[index:index+1] for index in indices}
    elif array.shape[0] == len(indices):
        arrays = {index: array[offset:offset+1] for offset, index in enumerate(indices)}
    else:
        raise ValueError("Reference batch must match the complete latent batch or selected sample count")
    return arrays, {"kind": "raw_npy", "path": str(path), "sha256": sha256(path),
                    "source_identity_verified": False}


def comparison_panel(reference, candidate, path, label):
    from PIL import Image, ImageDraw
    a, b = rgb_pixels(reference), rgb_pixels(candidate)
    # Display multiplier is a visual aid, never used for quantitative metrics.
    difference = np.minimum(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 8, 255).astype(np.uint8)
    width, height = a.shape[1], a.shape[0]
    canvas = Image.new("RGB", (width * 3, height + 32), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for column, (pixels, title) in enumerate(((a, "FP32 reference"), (b, label), (difference, "abs RGB difference x8"))):
        canvas.paste(Image.fromarray(pixels), (column * width, 32))
        draw.text((column * width + 8, 8), title, fill=(255, 255, 255))
    canvas.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True, help="Physical GPU index; existing busy cards are rejected")
    parser.add_argument("--precision", choices=("fp32", "bfloat16", "float16"), required=True)
    parser.add_argument("--stages", default="", help="Comma list: " + ",".join(STAGES) + "; empty keeps FP32")
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
    parser.add_argument("--conv-backend", choices=("xla", "cudnn_bounded", "tiled_gemm"), default="xla")
    parser.add_argument("--conv-workspace-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if args.device < 0 or args.repeats < 1 or args.cpu_threads < 1 or args.attention_chunk_size < 1:
        parser.error("device must be nonnegative; repeats, threads and attention chunk must be positive")
    if not math.isfinite(args.reserve_dit_gib) or args.reserve_dit_gib < 0:
        parser.error("--reserve-dit-gib must be finite and nonnegative")
    try:
        stages = parse_stages(args.stages)
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
    # Inherit caller/XLA defaults. Benchmark setup must not impose global
    # compiler or precision restrictions; the FFI backend declares its own compute.
    configure_process(config, args.device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    from moss_art.assets import component_identity
    from moss_art.vae_precision import decode_latents
    from vendor.moss_jax.weights import load_config, load_host_params
    from PIL import Image

    output = Path(args.output).resolve()
    with output_lock(output):
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Benchmark output is nonempty; select a new --output")
        root = Path(__file__).resolve().parents[1]
        source_paths = ("scripts/benchmark_vae_precision.py", "scripts/benchmark_rollout.py",
                        "moss_art/vae_precision.py", "vendor/moss_jax/vae.py", "vendor/moss_jax/weights.py",
                        "moss_art/runtime.py", "moss_art/assets.py")
        if args.conv_backend == "tiled_gemm":
            source_paths += ("moss_art/tiled_conv.py",)
        elif args.conv_backend == "cudnn_bounded":
            source_paths += ("moss_art/cudnn_conv.py", "moss_art/native/cudnn_conv_ffi.cc",
                             "scripts/build_cudnn_conv.py")
        report = {"status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]), "precision": args.precision,
                  "low_precision_stages": stages, "conv_backend": args.conv_backend,
                  "conv_workspace_mib": args.conv_workspace_mib, "input_path": str(Path(args.latents).resolve()),
                  "input_sha256": input_sha, "input_shape": list(latents.shape), "sample_indices": indices,
                  "model_dir": str(Path(args.model_dir).resolve()), "model_identity": component_identity(args.model_dir, "vae"),
                  "source_hashes": {name: sha256(root / name) for name in source_paths},
                  "physical_device": args.device, "inventory": inventory,
                  "reference": reference_metadata,
                  "versions": {"python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                               "numpy": np.__version__},
                  "policy": {"tiling": not args.no_tiling, "clamp": False, "slicing": False,
                             "attention_chunk_size": args.attention_chunk_size,
                             "weight_storage": "FP32 device parameters; explicit runtime argument to JIT",
                             "low_precision_ops": "selected convolution inputs/weights/output use chosen dtype, then immediate FP32 cast before bias; selected linear inputs use chosen dtype with preferred FP32 output",
                             "convolution_accumulator_request": "cudnn_bounded explicitly requests FP32 compute per plan; native XLA uses its defaults or caller environment; half-output HLO alone does not independently prove accumulator precision",
                             "sensitive_ops": "FP32 bias, normalization, nonlinearities, attention scores/softmax, residuals and final returned values; no claim that convolution output remains FP32",
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
            params = {key: jax.device_put(np.asarray(value, np.float32), device) for key, value in host.items()}
            del host
            jax.block_until_ready(params)
            report["parameter_bytes"] = sum(value.nbytes for value in params.values())
            report["load_seconds"] = time.monotonic() - begun
            report["after_load"] = memory()
            report["vae_config"] = vae_config
            compiler_options = ({"xla_gpu_autotune_level": 4} if args.conv_backend == "tiled_gemm" else None)
            report["vae_compiler_options"] = compiler_options
            decode = jax.jit(lambda p, z: decode_latents(p, z, vae_config,
                precision=args.precision, low_precision_stages=stages, tiling=not args.no_tiling,
                conv_backend=args.conv_backend, conv_workspace_mib=args.conv_workspace_mib,
                attention_chunk_size=args.attention_chunk_size, clamp=False), compiler_options=compiler_options)
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
            if args.conv_backend == "cudnn_bounded":
                from moss_art.cudnn_conv import backend_info, plan_inventory
                report["convolution_backend_info"] = backend_info()
                report["convolution_plans"] = plan_inventory()
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
