#!/usr/bin/env python3
"""Independently review a completed 1024 beta/RES generation benchmark on CPU.

The benchmark is never resumed or modified. Artifact values are recomputed,
and source/asset identities are compared with the measured snapshot. This
script uses no JAX device operations, model weights, or reward re-scoring.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="outputs/generation_res_multistep_beta_50/benchmark.json")
    parser.add_argument("--output", default="reports/res_multistep_beta_generation_review.json")
    parser.add_argument("--environment", default="outputs/sampling_source_audit/gpu_process_environment.json")
    args = parser.parse_args()
    path = Path(args.benchmark).resolve()
    benchmark = json.loads(path.read_text())
    if benchmark["status"] != "passed":
        raise RuntimeError(f"Only a completed passed benchmark can be reviewed, got {benchmark['status']}")
    folder = path.parent
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(path), "benchmark_sha256": sha(path),
        "review_script": str(Path(__file__).resolve()), "review_script_sha256": sha(__file__),
        "checks": [], "limitations": [
            "This is an artifact/source audit, not a second model inference or independent reward recomputation.",
            "Raw VAE float output is not archived by this benchmark; finite checks cover stored latents and rewards, and PNG pixel tensors.",
            "Safetensors identity covers full header plus file size, mtime and path, not full multi-GiB tensor payloads.",
            "GPU peaks are process lifetime high-water marks; sampled nvidia-smi includes allocator reservation.",
            "Timing is synchronized host latency from benchmark sections, including launch/synchronization and first-use compilation; persistent compiler cache may be reused.",
            "The 50 prediction count refers to sampler-level combined velocity evaluations; CFG and skip-guidance branches may run additional DiT forwards internally.",
            "Only the benchmark's exported coefficient arrays are persisted; in-memory ART cache lifetime is verified from the frozen worker source and existing worker tests.",
        ],
    }

    def check(name, value, **details):
        result["checks"].append({"name": name, "passed": bool(value), **details})

    config = benchmark["config"]
    sampling = config["sampling"]
    check("standard_configuration", sampling["steps"] == 50 and sampling["sampler"] == "res_multistep"
          and sampling["scheduler"] == "beta" and sampling["height"] == sampling["width"] == 1024
          and sampling["beta_alpha"] == sampling["beta_beta"] == .6
          and sampling["shift"] == 5 and sampling["training_shift"] == 1,
          sampling=sampling)
    check("no_optimizer_updates", benchmark["optimizer_updates"] == 0)
    check("four_distinct_seeds", len(benchmark["seeds"]) == len(set(benchmark["seeds"])) == 4,
          seeds=benchmark["seeds"])
    check("two_complete_repeats", len(benchmark["repeats"]) == 2)

    from moss_art.assets import experiment_identity
    from moss_art.config import load_config
    from moss_art.sampling import schedule, step_coefficients
    measured_identity = benchmark["source_asset_identity"]
    current_identity = experiment_identity(config)
    changed_sources = [name for name, entry in measured_identity["source_files"].items()
                       if current_identity["source_files"].get(name) != entry]
    check("measured_source_and_asset_identity_unchanged", current_identity == measured_identity,
          measured_fingerprint=measured_identity["fingerprint"],
          current_fingerprint=current_identity["fingerprint"], changed_sources=changed_sources)
    argv = benchmark["argv"]
    config_path = Path(argv[argv.index("--config")+1])
    current_config = load_config(config_path, {"runtime": {"output_dir": config["runtime"]["output_dir"]}})
    check("measured_configuration_unchanged", current_config == config,
          config_path=str(config_path.resolve()), measured_config_sha256=canonical_sha(config),
          current_config_sha256=canonical_sha(current_config), config_file_sha256=sha(config_path))
    script_path = ROOT / "scripts/benchmark_rollout.py"
    check("measured_benchmark_script_unchanged", sha(script_path) == benchmark["script_sha256"],
          measured_sha256=benchmark["script_sha256"], current_sha256=sha(script_path))
    if "adapter_sha256" in benchmark:
        check("adapter_unchanged", sha(benchmark["adapter"]) == benchmark["adapter_sha256"],
              adapter=benchmark["adapter"], sha256=sha(benchmark["adapter"]))

    _, expected_sigmas = schedule(sampling["steps"], scheduler=sampling["scheduler"],
                                 shift=sampling["shift"], training_shift=sampling["training_shift"],
                                 beta_alpha=sampling["beta_alpha"], beta_beta=sampling["beta_beta"])
    expected_c = np.asarray([s.ode_velocity_coefficient for s in
                             step_coefficients(expected_sigmas, sampler=sampling["sampler"])], np.float32)
    check("all_current_velocity_coefficients_negative", np.all(expected_c < 0),
          count=len(expected_c), sigma_min_nonzero=float(expected_sigmas[-2]),
          current_velocity_coefficient_min=float(expected_c.min()),
          current_velocity_coefficient_max=float(expected_c.max()))

    endpoints, image_sets, reward_sets = [], [], []
    phase_seconds = benchmark["load_seconds"]
    result["repeats"] = []
    expected_prompts = [record["prompt"] for record in benchmark["records"]
                        for _ in range(sampling["group_size"])]
    transformer_config = json.loads(Path(config["model"]["path"], "transformer/config.json").read_text())
    # Worker endpoints retain packed-channel BCHW; model tokens are flattened
    # only inside prediction, not in the persisted ART endpoint artifact.
    expected_shape = (4, transformer_config["latent_channels"],
                      sampling["height"]//16, sampling["width"]//16)
    for index, repeat in enumerate(benchmark["repeats"]):
        subfolder = folder / f"repeat_{index:02d}"
        latent_path = subfolder / "endpoints.npy"
        latent = np.load(latent_path, allow_pickle=False)
        endpoints.append(latent)
        prompts = json.loads((subfolder/"prompts.json").read_text())
        check(f"repeat_{index}_actual_schedule_and_cache_coefficients",
              repeat["actual_denoising_steps"] == repeat["model_predictions_per_image"] == 50
              and np.array_equal(np.asarray(repeat["sigmas"], np.float32), expected_sigmas)
              and np.array_equal(np.asarray(repeat["ode_velocity_coefficients"], np.float32), expected_c))
        check(f"repeat_{index}_latents_finite_and_expected_shape",
              latent.dtype == np.float32 and latent.shape == expected_shape and np.isfinite(latent).all()
              and repeat["endpoints_finite"], shape=list(latent.shape), expected_shape=list(expected_shape))
        check(f"repeat_{index}_prompt_group_order", prompts == expected_prompts)
        pixels, image_info = [], []
        for row in range(4):
            image_path = subfolder / f"sample-{row:06d}.png"
            with Image.open(image_path) as image:
                array = np.asarray(image)
                pixels.append(array)
                image_info.append({"path": str(image_path), "sha256": sha(image_path),
                                   "shape": list(array.shape), "mode": image.mode})
        check(f"repeat_{index}_four_rgb_images",
              all(p.dtype == np.uint8 and p.shape == (1024, 1024, 3) for p in pixels))
        image_sets.append(np.stack(pixels))
        rewards = np.asarray(repeat["rewards"], np.float32)
        reward_sets.append(rewards)
        check(f"repeat_{index}_finite_rewards", rewards.shape == (4,) and np.isfinite(rewards).all())
        timings = {key: float(repeat[key]) for key in ("rollout_seconds", "decode_seconds", "reward_seconds")}
        phase_seconds += sum(timings.values())
        check(f"repeat_{index}_timing_arithmetic",
              all(value > 0 and np.isfinite(value) for value in timings.values())
              and np.isclose(repeat["seconds_per_image"], timings["rollout_seconds"]/4, rtol=1e-12)
              and np.isclose(repeat["images_per_second"], 4/timings["rollout_seconds"], rtol=1e-12))
        result["repeats"].append({
            "repeat": index, **timings, "rollout_seconds_per_image": timings["rollout_seconds"]/4,
            "decode_seconds_per_image": timings["decode_seconds"]/4,
            "generation_phases_seconds_per_image": (timings["rollout_seconds"]+timings["decode_seconds"])/4,
            "generation_phases_scope": "rollout plus VAE decode host latency; excludes reward, PNG/gallery export and model load",
            "reward_seconds_per_image": timings["reward_seconds"]/4,
            "measured_phases_total_seconds_per_image": sum(timings.values())/4,
            "endpoints_path": str(latent_path), "endpoints_sha256": sha(latent_path),
            "latent_shape": list(latent.shape), "latent_dtype": str(latent.dtype),
            "latent_min": float(latent.min()), "latent_max": float(latent.max()),
            "latent_row_sha256": [hashlib.sha256(row.tobytes()).hexdigest() for row in latent],
            "images": image_info, "rewards": rewards.tolist(),
            "retained_art_numpy_bytes": repeat["retained_art_numpy_bytes"],
            "after_rollout_jax": repeat["after_rollout"]["jax"],
            "after_decode_jax": repeat["after_decode"]["jax"],
        })
    if len(endpoints) == 2:
        latent_max = float(np.max(np.abs(endpoints[1]-endpoints[0])))
        pixel_max = int(np.max(np.abs(image_sets[1].astype(np.int16)-image_sets[0].astype(np.int16))))
        reward_max = float(np.max(np.abs(reward_sets[1]-reward_sets[0])))
        check("repeated_endpoints_bitwise", np.array_equal(endpoints[0], endpoints[1]), max_abs_difference=latent_max)
        check("repeated_images_bitwise", np.array_equal(image_sets[0], image_sets[1]), max_abs_difference=pixel_max)
        check("repeated_rewards_bitwise", np.array_equal(reward_sets[0], reward_sets[1]), max_abs_difference=reward_max)
        check("recorded_repeat_metrics_match_artifacts",
              benchmark["repeats"][1]["same_seed_max_abs_latent_difference"] == latent_max
              and benchmark["repeats"][1]["same_seed_max_abs_pixel_difference"] == pixel_max
              and benchmark["repeats"][1]["same_seed_images_equal"] == np.array_equal(image_sets[0], image_sets[1]))
        check("distinct_seed_latents_are_distinct",
              len(set(result["repeats"][0]["latent_row_sha256"])) == 4)
        check("distinct_seed_images_are_distinct",
              len({row["sha256"] for row in result["repeats"][0]["images"]}) == 4)
    check("total_time_includes_measured_phases", benchmark["total_seconds"] >= phase_seconds,
          total_seconds=benchmark["total_seconds"], measured_phase_sum_seconds=phase_seconds,
          export_and_other_overhead_seconds=benchmark["total_seconds"]-phase_seconds)

    memory_path = folder / "gpu_memory.csv"
    memory = np.genfromtxt(memory_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    memory = np.atleast_1d(memory)
    max_mib = int(np.max(memory["nvidia_smi_used_mib"]))
    check("sampled_gpu_peak_matches_raw_csv", max_mib == benchmark["nvidia_smi"]["max_used_mib"]
          and len(memory) == benchmark["nvidia_smi"]["samples"]
          and np.all(np.diff(memory["seconds"]) >= 0),
          max_used_mib=max_mib, samples=len(memory), csv_sha256=sha(memory_path))
    result["memory"] = {"nvidia_smi": benchmark["nvidia_smi"], "final_memory": benchmark["final_memory"],
                        "scope": benchmark["jax_peak_memory_scope"], "note": benchmark["memory_note"]}

    env_path = Path(args.environment)
    env_report = json.loads(env_path.read_text())
    process_matches = [process for process in env_report["processes"]
                       if str(folder.relative_to(ROOT)) in process["argv"]]
    restricted = ("XLA_FLAGS", "TF_XLA_FLAGS", "TF_FP16_CONV_USE_FP32_COMPUTE",
                  "JAX_DISABLE_JIT", "JAX_DEFAULT_MATMUL_PRECISION")
    check("benchmark_launch_has_no_global_compiler_overrides",
          len(process_matches) == 1 and all(process_matches[0]["inherited_environment"].get(k) is None
                                           for k in restricted),
          environment_snapshot_path=str(env_path.resolve()), environment_snapshot_sha256=sha(env_path))
    effective = benchmark["runtime_environment"]
    check("effective_runtime_does_not_inject_xla_flags", effective.get("XLA_FLAGS") in (None, ""),
          recorded_runtime_environment=effective)
    check("physical_gpu1_single_visible_worker", effective["CUDA_VISIBLE_DEVICES"] == "1"
          and config["runtime"]["devices"] == [1])
    check("requested_cudnn_attention", benchmark["attention_backend"] == config["model"]["attention_backend"] == "cudnn")
    result["environment"] = {"launch": process_matches,
                              "recorded_effective_subset": effective,
                              "compilation_cache": benchmark["compilation_cache"],
                              "dit_autotune_level": benchmark["dit_autotune_level"],
                              "precision": config["model"]["dtype"],
                              "vae_precision": config["model"]["vae_precision"],
                              "vae_conv_backend": config["model"]["vae_conv_backend"]}
    result["status"] = "passed" if all(item["passed"] for item in result["checks"]) else "failed"
    result["passed_checks"] = sum(item["passed"] for item in result["checks"])
    result["total_checks"] = len(result["checks"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"status": result["status"], "passed_checks": result["passed_checks"],
                      "total_checks": result["total_checks"], "output": str(output),
                      "failures": [item for item in result["checks"] if not item["passed"]]}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
