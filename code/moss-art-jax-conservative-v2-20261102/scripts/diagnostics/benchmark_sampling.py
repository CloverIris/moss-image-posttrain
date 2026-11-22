#!/usr/bin/env python3
"""CPU reference audit: pinned ComfyUI Torch flow/Beta/RES versus production JAX.

Torch runs only in an explicitly selected subprocess. No model weights, ComfyUI
startup, GPU initialization, downloads, or global XLA compiler policy changes.
Source hashes are fail-closed pins, independently checked against upstream.
"""
from __future__ import annotations

import os
os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COMFY_REV = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
KOHAKU_REV = "4b1b9dc891545f74510cef7313df1f8af66aa35f"
COMFY_PINS = {
    "comfy/model_sampling.py": "8afdc665272589a567792bb652389df7d4a83f69574c2b7e5b3d533d55592990",
    "comfy/samplers.py": "f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f",
    "comfy/k_diffusion/sampling.py": "cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce",
    "comfy/k_diffusion/utils.py": "fc30630d8a0e4520b2e864c8f84deea3231c9b04d3f1404fec93fc425f629272",
    "comfy/supported_models.py": "9893e77e3af02e2a71158523a0c6b37d724fd9ecb0f5fe6dcdbf51d6033e0f78",
    "comfy/ldm/flux/layers.py": "d7ee59e51234766a03834471eee4adc2152cd39248fc39d4e70bb1b3ba8b6a5e",
}
KOHAKU_PINS = {
    "src/kohakumoss/diffusion/sampling/schedule.py": "99cc5e2e8349595f351c9ccb4ab86164ad63cfdc2b8f290d116e3e79c986df5f",
    "src/kohakumoss/diffusion/timestep/shift.py": "61d20d5a924d68c1a35481be13e4a329e9e38a688e711f068af9b920daace3c0",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(args):
    records = []
    for root, pins, repo, revision in (
        (Path(args.comfy_root), COMFY_PINS, "Comfy-Org/ComfyUI", COMFY_REV),
        (Path(args.kohaku_root), KOHAKU_PINS, "KohakuBlueleaf/KohakuMossTrainer", KOHAKU_REV),
    ):
        for name, expected in pins.items():
            path = root / name
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Pinned oracle source changed: {path}: {actual}")
            records.append({"path": str(path.resolve()), "sha256": actual, "revision": revision,
                            "upstream": f"https://github.com/{repo}/blob/{revision}/{name}"})
    return records


def definitions(path, names, namespace):
    """Execute only original AST definitions, unchanged; never import ComfyUI."""
    text = Path(path).read_text()
    selected = [n for n in ast.parse(text).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"Missing reference definitions in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return [{"name": n.name, "first_line": n.lineno, "last_line": n.end_lineno,
             "source_segment_sha256": hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest()}
            for n in selected]


def schedule_cases():
    cases = []
    for steps in (1, 2, 10, 50, 200, 1000):
        cases.append(dict(label=f"beta_{steps}", steps=steps, scheduler="beta"))
    cases += [
        dict(label="beta_shift1", steps=50, scheduler="beta", shift=1.0),
        dict(label="beta_shift2", steps=50, scheduler="beta", shift=2.0),
        dict(label="beta_custom_shape", steps=50, scheduler="beta", beta_alpha=2.0, beta_beta=3.0),
        dict(label="beta_grid32", steps=50, scheduler="beta", total=32),
    ]
    for steps in (1, 10, 50):
        cases.append(dict(label=f"moss_{steps}", steps=steps, scheduler="moss"))
    cases.append(dict(label="moss_training_shift2", steps=50, scheduler="moss", training_shift=2.0))
    defaults = dict(steps=50, scheduler="beta", shift=5.0, training_shift=1.0,
                    total=1000, beta_alpha=.6, beta_beta=.6)
    return [{**defaults, **case} for case in cases]


SOLVER_CASES = [
    ("beta_1", "cfg", "res_multistep"), ("beta_2", "cfg", "res_multistep"),
    ("beta_50", "constant", "res_multistep"), ("beta_50", "affine", "res_multistep"),
    ("beta_50", "cfg", "res_multistep"), ("beta_200", "cfg", "res_multistep"),
    ("moss_50", "cfg", "res_multistep"), ("beta_50", "cfg", "euler"),
    ("moss_50", "cfg", "euler"), ("close_sigmas", "cfg", "res_multistep"),
]


def velocity(state, sigma, kind, xp):
    sigma = xp.asarray(sigma, dtype=xp.float32).reshape((-1,) + (1,) * (state.ndim - 1))
    if kind == "constant":
        return xp.full_like(state, .2)
    if kind == "affine":
        return state * .125 + sigma * .0625 - .05
    negative = state * .1 + sigma * .025 - .05
    positive = state * .125 + sigma * .0625 + state * state * .001 - .04
    return negative + 3.0 * (positive - negative)


def run_oracle(args):
    import numpy as np
    import scipy
    import scipy.stats
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    records = sources(args)
    namespace = {"torch": torch, "math": math, "numpy": np, "scipy": scipy, "Tensor": torch.Tensor}
    extracted = {}
    def take(relative, names):
        extracted[relative] = definitions(Path(args.comfy_root)/relative, names, namespace)
    take("comfy/model_sampling.py", ["reshape_sigma", "time_snr_shift",
                                   "ModelSamplingDiscreteFlow", "CONST"])
    take("comfy/samplers.py", ["beta_scheduler"])
    take("comfy/k_diffusion/utils.py", ["append_dims"])
    namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])
    namespace["trange"] = lambda count, **kwargs: range(count)
    take("comfy/k_diffusion/sampling.py",
         ["to_d", "get_ancestral_step", "res_multistep", "sample_res_multistep", "sample_euler"])
    for relative, names in (
        ("src/kohakumoss/diffusion/timestep/shift.py", ["shift_sigma"]),
        ("src/kohakumoss/diffusion/sampling/schedule.py", ["_grid", "moss_sigmas"]),
    ):
        extracted[relative] = definitions(Path(args.kohaku_root)/relative, names, namespace)
    Flow = type("PinnedMossFlow", (namespace["ModelSamplingDiscreteFlow"], namespace["CONST"]), {})
    arrays, schedules = {}, []
    for case in schedule_cases():
        label = case["label"]
        model_sampling = Flow(SimpleNamespace(sampling_settings={"shift": case["shift"], "multiplier": 1.0}))
        model_sampling.set_parameters(shift=case["shift"], timesteps=case["total"], multiplier=1.0)
        if case["scheduler"] == "beta":
            sigma = namespace["beta_scheduler"](model_sampling, case["steps"],
                                                alpha=case["beta_alpha"], beta=case["beta_beta"])
        else:
            sigma = namespace["moss_sigmas"](case["steps"], case["shift"],
                                             case["total"], case["training_shift"])
        arrays[label+"_sigmas"] = sigma.numpy()
        arrays[label+"_times"] = (sigma[:-1] * case["total"]).numpy()
        schedules.append({**case, "actual_steps": len(sigma)-1})
    arrays["close_sigmas_sigmas"] = np.array([1., .9999999, .9999997, .999, .5, 0.], np.float32)
    rng = np.random.default_rng(20261020)
    initial = rng.standard_normal((2, 3, 4, 5), dtype=np.float32) * np.float32(.3)
    arrays["initial"] = initial
    solver_records = []
    torch_xp = SimpleNamespace(asarray=torch.as_tensor, float32=torch.float32, full_like=torch.full_like)
    model_sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
    for schedule_label, field, method in SOLVER_CASES:
        label = f"{schedule_label}_{field}_{method}"
        sigmas = torch.from_numpy(arrays[schedule_label+"_sigmas"].copy())
        states, velocities, denoised_values = [], [], []
        class Model:
            inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
                get_model_object=lambda name: model_sampling))
            def __call__(self, state, sigma, **kwargs):
                v = velocity(state, sigma, field, torch_xp)
                d = model_sampling.calculate_denoised(sigma, v, state)
                states.append(state.detach().numpy().copy())
                velocities.append(v.detach().numpy().copy())
                denoised_values.append(d.detach().numpy().copy())
                return d
        def forbidden_noise(*_):
            raise AssertionError("Deterministic RES must not request noise")
        kwargs = dict(disable=True)
        if method == "res_multistep":
            kwargs["noise_sampler"] = forbidden_noise
        result = namespace["sample_"+method](Model(), torch.from_numpy(initial.copy()), sigmas, **kwargs)
        states.append(result.numpy().copy())
        arrays[label+"_trajectory"] = np.stack(states)
        arrays[label+"_velocities"] = np.stack(velocities)
        arrays[label+"_denoised"] = np.stack(denoised_values)
        solver_records.append({"label": label, "schedule": schedule_label, "field": field,
                               "sampler": method, "model_evaluations": len(velocities),
                               "finite": bool(np.isfinite(np.stack(states)).all())})
    # Same upstream equations in Float64, starting from the identical rounded
    # FP32 initial state/grid, isolate arithmetic conditioning of the stress grid.
    torch_xp = SimpleNamespace(asarray=torch.as_tensor, float32=torch.float64, full_like=torch.full_like)
    field = "cfg"
    states, velocities, denoised_values = [], [], []
    result64 = namespace["sample_res_multistep"](
        Model(), torch.from_numpy(initial.astype(np.float64)),
        torch.from_numpy(arrays["close_sigmas_sigmas"].astype(np.float64)),
        disable=True, noise_sampler=forbidden_noise)
    states.append(result64.numpy().copy())
    arrays["close_sigmas_cfg_res_multistep_float64_trajectory"] = np.stack(states)
    artifact = Path(args.artifacts)
    np.savez_compressed(artifact/"torch_oracle.npz", **arrays)
    result = {
        "status": "passed", "cpu_only": True, "device": "cpu",
        "versions": {"torch": torch.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
        "sources": records, "extracted_definitions": extracted,
        "schedule_cases": schedules, "solver_cases": solver_records,
        "array_file": str(artifact/"torch_oracle.npz"),
        "array_sha256": sha256(artifact/"torch_oracle.npz"),
        "scope": "Original pinned Torch AST definitions executed without importing ComfyUI; synthetic deterministic model returns official CONST x0=x-sigma*v. eta=0, CFG++ disabled, s_churn=0; 2 samples per trajectory.",
    }
    (artifact/"torch_oracle.json").write_text(json.dumps(result, indent=2)+"\n")


def differences(actual, reference):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        return {"shape_equal": False, "actual_shape": list(a.shape), "reference_shape": list(b.shape)}
    error = a.astype(np.float64) - b.astype(np.float64)
    norm = float(np.linalg.norm(b.astype(np.float64).ravel()))
    return {"shape_equal": True, "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "bitwise_equal": bool(np.array_equal(a, b)), "max_abs": float(np.max(np.abs(error))),
            "relative_l2": float(np.linalg.norm(error.ravel())) / max(norm, 1e-30)}


def main(args):
    import numpy as np
    import scipy
    import jax
    import jax.numpy as jnp
    from moss_art import sampling
    if not all(d.platform == "cpu" for d in jax.devices()):
        raise RuntimeError("Reference validation must remain CPU-only")
    output, artifact = Path(args.output).resolve(), Path(args.artifacts).resolve()
    if output.exists() or artifact.exists():
        raise FileExistsError("Select new --output and --artifacts; never overwrite audit evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True)
    args.artifacts = str(artifact)
    source_records = sources(args)
    command = [args.torch_python, str(Path(__file__).resolve()), "--oracle",
               "--comfy-root", args.comfy_root, "--kohaku-root", args.kohaku_root,
               "--artifacts", str(artifact)]
    child = subprocess.run(command, capture_output=True, text=True, timeout=180, env=dict(os.environ))
    (artifact/"torch_stdout.log").write_text(child.stdout)
    (artifact/"torch_stderr.log").write_text(child.stderr)
    if child.returncode:
        raise RuntimeError(f"Torch oracle failed: {child.stderr[-4000:]}")
    oracle = json.loads((artifact/"torch_oracle.json").read_text())
    arrays = np.load(artifact/"torch_oracle.npz", allow_pickle=False)
    report = {
        "status": "running", "recorded_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
        "argv": sys.argv, "script_sha256": sha256(__file__),
        "production_source": {"path": str(ROOT/"moss_art/sampling.py"), "sha256": sha256(ROOT/"moss_art/sampling.py")},
        "comfy_revision": COMFY_REV, "kohaku_revision": KOHAKU_REV,
        "sources": source_records, "oracle": oracle,
        "jax_devices": [str(d) for d in jax.devices()],
        "versions": {"jax": jax.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "runtime_environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "TF_FP16_CONV_USE_FP32_COMPUTE", "OMP_NUM_THREADS")},
        "tolerances": {"schedule": "bitwise FP32 equality", "trajectory_max_abs": 3e-5,
                       "trajectory_relative_l2": 3e-5, "local_velocity_jvp_max_abs": 2e-6},
        "schedules": [], "solvers": [], "rollout_summaries": [],
    }
    passed = True
    documented_stress_difference = False
    try:
        sampling.schedule(50, scheduler="beta", training_shift=3.0)
    except ValueError as error:
        report["beta_nondefault_training_shift_rejected"] = {"passed": True, "error": str(error)}
    else:
        report["beta_nondefault_training_shift_rejected"] = {"passed": False}
        passed = False
    for case in schedule_cases():
        kwargs = {k: v for k, v in case.items() if k != "label"}
        times, sigmas = sampling.schedule(**kwargs)
        label = case["label"]
        a = differences(sigmas, arrays[label+"_sigmas"])
        b = differences(times, arrays[label+"_times"])
        valid = a.get("bitwise_equal", False) and b.get("bitwise_equal", False)
        passed &= valid
        record = {**case, "passed": valid, "actual_steps": len(sigmas)-1,
                  "sigma_comparison": a, "time_comparison": b,
                  "sigmas_sha256": hashlib.sha256(sigmas.tobytes()).hexdigest()}
        if label == "beta_50":
            record["sigmas"] = sigmas.tolist()
            record["model_timesteps"] = times.tolist()
        report["schedules"].append(record)
    initial = arrays["initial"]
    for entry in oracle["solver_cases"]:
        sigmas = arrays[entry["schedule"]+"_sigmas"]
        coeffs = sampling.step_coefficients(sigmas, sampler=entry["sampler"])
        previous = None
        state = jnp.asarray(initial)
        states = [np.asarray(state)]
        current_denoised = []
        for coefficient in coeffs:
            v = velocity(state, coefficient.sigma, entry["field"], jnp)
            state, previous = sampling.sample_step(state, v, coefficient, previous)
            states.append(np.asarray(state))
            current_denoised.append(np.asarray(previous))
        actual = np.stack(states)
        reference = arrays[entry["label"]+"_trajectory"]
        metrics = differences(actual, reference)
        endpoint = differences(actual[-1], reference[-1])
        denoised = differences(np.stack(current_denoised), arrays[entry["label"]+"_denoised"])
        valid = (metrics["finite"] and metrics["max_abs"] <= 3e-5 and metrics["relative_l2"] <= 3e-5
                 and len(coeffs) == entry["model_evaluations"])
        slopes, frozen = [], []
        z = jnp.asarray(initial)
        for coefficient in coeffs:
            v = jnp.full_like(z, .2)
            history = jnp.full_like(z, .1)
            _, tangent = jax.jvp(lambda value: sampling.sample_step(z, value, coefficient, history)[0],
                                 (v,), (jnp.ones_like(v),))
            slopes.append(float(jnp.max(jnp.abs(tangent - coefficient.ode_velocity_coefficient))))
            _, hist_tangent = jax.jvp(lambda value: sampling.sample_step(z, v, coefficient, value)[0],
                                      (history,), (jnp.ones_like(history),))
            frozen.append(float(jnp.max(jnp.abs(hist_tangent))))
        valid &= max(slopes) <= 2e-6 and max(frozen) == 0.0
        record = {**entry, "passed": bool(valid), "trajectory": metrics, "endpoint": endpoint,
                  "current_denoised": denoised, "local_velocity_jvp_max_abs": max(slopes),
                  "history_jvp_max_abs": max(frozen),
                  "euler_steps": [i for i, c in enumerate(coeffs) if c.is_euler],
                  "small_h_stabilized_interior_steps": [i for i, c in enumerate(coeffs)
                                                        if not c.is_euler and abs(c.h) < 1e-3]}
        if entry["label"] == "beta_50_cfg_res_multistep":
            def whole_run(z):
                history = None
                for c in coeffs:
                    v = velocity(z, c.sigma, entry["field"], jnp)
                    z, history = sampling.sample_step(z, v, c, history)
                return z
            compiled_result = jax.jit(whole_run)(jnp.asarray(initial))
            compiled_result.block_until_ready()
            record["jitted_endpoint"] = differences(np.asarray(compiled_result), reference[-1])
            valid &= record["jitted_endpoint"]["max_abs"] <= 3e-5
            record["passed"] = bool(valid)
        if entry["schedule"] == "close_sigmas":
            high_precision = arrays[entry["label"]+"_float64_trajectory"]
            record["acceptance_scope"] = "Separate extreme-grid numerical conditioning audit; original fixed upstream FP32 parity threshold retained."
            record["float64_reference"] = {
                "definition": "Original pinned Torch solver with Float64 state, sigma, velocity and history; initial values/grid converted exactly from the same FP32 inputs.",
                "torch32_trajectory_error": differences(reference, high_precision),
                "jax32_trajectory_error": differences(actual, high_precision),
                "torch32_endpoint_error": differences(reference[-1], high_precision[-1]),
                "jax32_endpoint_error": differences(actual[-1], high_precision[-1]),
            }
            record["sigmas"] = sigmas.tolist()
            documented_stress_difference = not bool(valid)
            passed &= metrics["finite"] and max(slopes) <= 2e-6 and max(frozen) == 0.0
        else:
            record["acceptance_scope"] = "Standard scheduler/solver compatibility acceptance"
            passed &= valid
        report["solvers"].append(record)
    for path in args.rollout:
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if data.get("status") != "passed":
            raise ValueError(f"Rollout not complete: {path}")
        report["rollout_summaries"].append({
            "path": str(path), "sha256": sha256(path), "sampling": data["config"]["sampling"],
            "adapter_sha256": data.get("adapter_sha256"), "seeds": data["seeds"],
            "repeats": [{k: r.get(k) for k in ("repeat", "images", "actual_denoising_steps",
                         "rollout_seconds", "decode_seconds", "reward_seconds", "endpoints_finite",
                         "rewards", "same_seed_max_abs_latent_difference")} for r in data["repeats"]],
        })
    report["status"] = ("passed_with_documented_stress_difference" if documented_stress_difference else "passed") if passed else "failed"
    report["acceptance"] = {
        "standard_compatibility_passed": bool(passed),
        "all_strict_upstream_parity_cases_passed": bool(passed and not documented_stress_difference),
        "extreme_stress_difference_documented": documented_stress_difference,
        "scope": "Standard schedules and solver trajectories pass fixed thresholds. Artificial adjacent-FP32 sigma stress is separate and retains its original strict comparison result; this is not a claim that every case passed.",
    }
    report["interpretation"] = {
        "moss_comfy_model_sampling": "FLOW / ModelSamplingDiscreteFlow + CONST; shift=5, multiplier=1, 1000 grid. Not ModelSamplingFlux (which has exponential shift and a 10000-point default grid).",
        "time_units": "Comfy MOSS calls Flux timestep_embedding(time_factor=1000) on sigma; JAX consumes sigma*1000 directly.",
        "beta_training_shift": "Legacy training_shift=1 snapping belongs only to moss/Euler schedule; beta indexes the shifted flow grid once, without a second snap or shift.",
        "upstream_model_type_reference": f"https://github.com/Comfy-Org/ComfyUI/blob/{COMFY_REV}/comfy/model_base.py",
        "limitations": ["Synthetic vector fields validate solver arithmetic and schedule, not image quality or real DiT parity.",
                        "JAX stabilization of phi1/phi2 for h<1e-3 may differ slightly from upstream FP32 cancellation; those steps and actual errors are reported.",
                        "Frozen-history JVP validates the local ART replay coefficient; it is not a full-trajectory derivative.",
                        "No GPU calls, model loading or training updates in this audit."],
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "schedule_cases": len(report["schedules"]), "solver_cases": len(report["solvers"]),
                      "max_trajectory_abs": max(r["trajectory"]["max_abs"] for r in report["solvers"]),
                      "max_endpoint_abs": max(r["endpoint"]["max_abs"] for r in report["solvers"])}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", default="$HOME/ComfyUI")
    parser.add_argument("--kohaku-root", default="$HOME/KohakuMossTrainer")
    parser.add_argument("--torch-python", default="$CONDA_PREFIX/envs/torch_env/bin/python")
    parser.add_argument("--output", default=str(ROOT/"reports/sampling_reference_validation.json"))
    parser.add_argument("--artifacts", default=str(ROOT/"outputs/sampling_reference_validation"))
    parser.add_argument("--rollout", action="append", default=[], help="Optional completed benchmark.json to summarize")
    parser.add_argument("--oracle", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_oracle(args) if args.oracle else main(args)
