"""Independent tiny GPU FFI checks against NumPy SAME cross-correlation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", default="reports/cudnn_conv_independent_edges.json")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), JAX_PLATFORMS="cuda",
                      XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      XLA_PYTHON_CLIENT_MEM_FRACTION=".5")
    sys.path.insert(0, str(ROOT))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from moss_art.cudnn_conv import conv2d_nchw, plan_inventory, backend_info

    report = {"status": "running", "device": args.device, "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference": "NumPy float64 cross-correlation of exactly rounded input and weights, then cast to output dtype",
              "thresholds": {"fp32_rtol": 3e-5, "fp32_atol": 2e-6,
                             "fp16_max_abs": .001, "fp16_relative_l2": .002},
              "runtime_environment": {k: os.environ.get(k) for k in (
                  "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS",
                  "TF_FP16_CONV_USE_FP32_COMPUTE", "XLA_PYTHON_CLIENT_PREALLOCATE",
                  "XLA_PYTHON_CLIENT_MEM_FRACTION")},
              "cases": [], "versions": {"jax": jax.__version__}}
    output = ROOT / args.output
    try:
        rng = np.random.default_rng(20261020)
        cases = [(3, 5, 5, 7, 3, 0), (8, 12, 3, 5, 1, 0),
                 (8, 8, 1, 1, 3, 1), (4, 4, 1, 7, 3, 1)]
        for dtype in (np.float32, np.float16):
            for ci, co, h, w, kernel, cap in cases:
                x = rng.normal(0, .2, (1, ci, h, w)).astype(dtype)
                weight = rng.normal(0, .1, (co, ci, kernel, kernel)).astype(dtype)
                pad = kernel // 2
                padded = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
                expected = np.zeros((1, co, h, w), np.float64)
                for yy in range(h):
                    for xx in range(w):
                        expected[:, :, yy, xx] = np.einsum(
                            "ncrs,kcrs->nk", padded[:, :, yy:yy+kernel, xx:xx+kernel],
                            weight.astype(np.float64),
                        )
                expected = expected.astype(dtype)
                fn = jax.jit(lambda a, b: conv2d_nchw(a, b, workspace_mib=cap))
                compiled = fn.lower(jnp.asarray(x), jnp.asarray(weight)).compile()
                actual = np.asarray(compiled(jnp.asarray(x), jnp.asarray(weight)))
                difference = actual.astype(np.float64) - expected.astype(np.float64)
                max_abs = float(np.max(np.abs(difference)))
                relative = float(np.linalg.norm(difference) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30))
                passed = bool(np.isfinite(actual).all())
                if dtype == np.float32:
                    passed = passed and bool(np.allclose(actual, expected, rtol=3e-5, atol=2e-6))
                else:
                    passed = passed and max_abs <= .001 and relative <= .002
                plans = plan_inventory()
                selected = next(p for p in plans if p["input_shape"] == list(x.shape)
                                and p["weight_shape"] == list(weight.shape)
                                and p["dtype"] == str(np.dtype(dtype))
                                and p["workspace_cap_bytes"] == cap * 1024**2)
                assert selected["selected_plan"]["workspace_bytes"] <= cap * 1024**2
                report["cases"].append({
                    "dtype": str(np.dtype(dtype)), "shape": list(x.shape),
                    "weight_shape": list(weight.shape), "workspace_cap_mib": cap,
                    "max_abs": max_abs, "relative_l2": relative, "passed": passed,
                    "finite": bool(np.isfinite(actual).all()),
                    "engine": selected["selected_plan"]["engine"],
                    "plan_workspace_bytes": selected["selected_plan"]["workspace_bytes"],
                    "numerical_notes": selected["selected_plan"]["numerical_notes"],
                    "ffi_target": selected["ffi_target"],
                    "compiled_temp_bytes": compiled.memory_analysis().temp_size_in_bytes,
                })
                print(json.dumps(report["cases"][-1]), flush=True)
                if not passed:
                    raise AssertionError("Numerical tolerance failed; not widened")
        report["backend_info"] = backend_info()
        report["memory_stats"] = jax.devices()[0].memory_stats()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
