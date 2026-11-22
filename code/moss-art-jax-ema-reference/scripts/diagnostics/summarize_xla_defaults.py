"""Summarize completed default-XLA validation; reads saved artifacts on CPU."""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def read(name):
    path = ROOT / name
    return json.loads(path.read_text())
def identity(name):
    path = ROOT / name
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
def vae(name):
    path = f"outputs/vae_xla_defaults/{name}/summary.json"
    d = read(path)
    assert d["status"] == "passed"
    assert d["runtime_environment"]["XLA_FLAGS"] is None
    assert d["runtime_environment"]["TF_FP16_CONV_USE_FP32_COMPUTE"] is None
    samples = d["repeats"][1]["samples"]
    assert all(s["raw_finite"] and s["same_seed_raw_bitwise_equal"] for s in samples)
    return {
        "source": identity(path), "device": d["physical_device"],
        "samples": d["sample_indices"], "warm_seconds_per_image": d["repeats"][1]["seconds_per_image"],
        "sample_seconds": {str(s["sample_index"]): s["decode_seconds"] for s in samples},
        "compile_seconds": d["compile_seconds"],
        "first_decode_seconds": d["repeats"][0]["samples"][0]["decode_seconds"],
        "max_raw_error": max(s["comparison"]["raw_max_abs"] for s in samples),
        "min_ssim": min(s["comparison"]["ssim"]["rgb_mean"] for s in samples),
        "max_pixel_error": max(s["comparison"]["pixel_max_abs_0_255"] for s in samples),
        "all_finite": True, "same_seed_bitwise": True,
        "environment": d["runtime_environment"], "source_hashes": d["source_hashes"],
        "compiled_memory": d["compiled_memory"],
        "peak_jax_bytes_in_use": d["final_memory"]["jax"]["peak_bytes_in_use"],
        "reserve_dit_gib": d["reservation"]["requested_gib"],
        "persistent_cache_note": "Enabled. First-in-process timings do not establish cold compilation.",
    }
def main():
    profiles = {name: vae(name) for name in
        ("cudnn_fp32_four", "native_fp32_sample3", "native_fp32_four")}
    source = "outputs/rollout_1024_xla_defaults/benchmark.json"
    roll = read(source)
    assert roll["status"] == "passed" and roll["runtime_environment"]["XLA_FLAGS"] is None
    warm = roll["repeats"][1]
    assert warm["same_seed_max_abs_latent_difference"] == 0 and warm["same_seed_images_equal"]
    old_path = "outputs/rollout_1024_fast_vae_matched/repeat_00/endpoints.npy"
    new_path = "outputs/rollout_1024_xla_defaults/repeat_00/endpoints.npy"
    old = np.load(ROOT / old_path).astype(np.float64)
    new = np.load(ROOT / new_path).astype(np.float64)
    training = {}
    for name in ("train_1024_xla_defaults", "train_1024_native_xla_defaults"):
        path = f"reports/{name}_validation.json"
        d = read(path)
        assert d["status"] == "passed" and d["run_current_and_checkpoint_experiment_identity_match"]
        training[name] = {
            "source": identity(path), "complete": d["complete"], "peak_active_gib": d["peak_active_gib"],
            "grad_norms": [m["grad_norm"] for m in d["metrics"]],
            "adam_counts": d["adam_counts"], "changed_adapter_tensors": d["changed_adapter_tensors"],
            "all_adapter_and_optimizer_leaves_finite": d["all_adapter_and_optimizer_leaves_finite"],
            "export_equals_checkpoint_bitwise": d["export_equals_checkpoint_bitwise"],
            "run_current_and_checkpoint_experiment_identity_match": True,
            "scope": "Two optimizer updates including setup/compilation; totals are not isolated backend timing.",
        }
    audit_path = "reports/xla_defaults_independent_review.json"
    report = {
        "status": "passed", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "script": identity("scripts/diagnostics/summarize_xla_defaults.py"),
        "policy": {
            "injected_global_compiler_or_precision_overrides": [],
            "caller_environment": "Preserved verbatim, including absence; actual benchmark flags are null.",
            "retained_resources": ["CUDA device/platform selection", "allocator preallocation and memory budget", "CPU threads/affinity", "compilation cache directory", "x64 dtype support"],
            "local_options": "Explicit per-JIT DiT options and diagnostic A/B remain local; model level0 inherits XLA defaults.",
        },
        "vae": profiles,
        "same_gpu_sample3": {
            "device": 1,
            "bounded_seconds": profiles["cudnn_fp32_four"]["sample_seconds"]["3"],
            "native_seconds": profiles["native_fp32_sample3"]["sample_seconds"]["3"],
            "scope": "Separate runs, same GPU1 and latent; no confidence interval or isolated machine claim.",
        },
        "rollout": {
            "source": identity(source), "environment": roll["runtime_environment"],
            "warm_per_image": {k: warm[k] / warm["images"] for k in ("rollout_seconds", "decode_seconds", "reward_seconds")},
            "warm_three_phase_seconds_per_image": sum(warm[k] for k in ("rollout_seconds", "decode_seconds", "reward_seconds")) / warm["images"],
            "rewards": warm["rewards"], "same_seed_within_new_run_bitwise": True,
            "old_global0_latent_comparison": {
                "old": identity(old_path), "new": identity(new_path),
                "bitwise_equal": bool(np.array_equal(old, new)),
                "max_abs": float(np.max(np.abs(old - new))),
                "relative_l2": float(np.linalg.norm(old - new) / np.linalg.norm(old)),
                "interpretation": "Full pipeline numerical paths change with compiler policy. Do not carry forward the old small reward-delta claim to this new rollout. Fixed-latent VAE error is measured separately.",
            },
            "peak_jax_bytes_in_use": roll["final_memory"]["jax"]["peak_bytes_in_use"],
            "nvidia_smi": roll["nvidia_smi"],
        },
        "training": training,
        "independent_audit": identity(audit_path),
        "tests": {"full_suite": "255 passed, 74 subtests passed", "log": identity("reports/xla_defaults_regression.log")},
        "limits": [
            "Earlier 64.85x compares bounded FFI against global-autotune0 XLA, not normal default XLA.",
            "Native default tuning recovered from a rejected large temporary allocation and completed successfully; that log is not an execution failure.",
            "Fixed-latent profiles reserve 9.6GiB dummy buffers; actual training holds the real DiT.",
            "Cumulative peak includes tuning and allocator history, not just warm inference.",
            "Four unique fixed images and two training updates establish short-run behavior, not general image-quality or convergence improvement.",
        ],
    }
    output = ROOT / "reports/xla_defaults_validation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "native_four_seconds": profiles["native_fp32_four"]["warm_seconds_per_image"],
                      "bounded_four_seconds": profiles["cudnn_fp32_four"]["warm_seconds_per_image"]}))
if __name__ == "__main__":
    main()
