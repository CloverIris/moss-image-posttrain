#!/usr/bin/env python3
"""CPU-only comparison of four VAE images using original SigLIP and real ART.

Use --reference FP32_DIRECTORY --candidate DIRECTORY [--candidate DIRECTORY]
--output REPORT.json. Inputs must contain completed benchmark_vae_precision
summaries for samples 0..3 from cat42/43, girl1051/1052. --validate-only checks
identity and PNG/raw agreement without loading either framework. No GPU use.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def cpu_environment(threads):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                       JAX_ENABLE_X64="true", OMP_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return environment


def load_profile(directory, input_sha, repeat_index, is_reference=False):
    import numpy as np
    from PIL import Image
    directory = Path(directory).resolve()
    summary_path = directory/"summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "passed":
        raise ValueError(f"{directory}: benchmark did not pass")
    if summary.get("input_sha256") != input_sha:
        raise ValueError(f"{directory}: latent SHA256 mismatch")
    if is_reference and (summary["precision"] != "fp32" or summary["low_precision_stages"]):
        raise ValueError("Reference must use FP32 with no low-precision stages")
    repeats = [r for r in summary["repeats"] if r["repeat"] == repeat_index]
    if len(repeats) != 1:
        raise ValueError(f"{directory}: missing/duplicate repeat {repeat_index}")
    samples = repeats[0]["samples"]
    if len(samples) != 4 or sorted(s["sample_index"] for s in samples) != [0, 1, 2, 3]:
        raise ValueError(f"{directory}: expected exactly samples 0,1,2,3")
    pixels, files = [], []
    for sample in sorted(samples, key=lambda s: s["sample_index"]):
        raw_path, png_path = directory/sample["raw_path"], directory/sample["png_path"]
        raw = np.load(raw_path, allow_pickle=False)
        if raw.shape != (1, 3, 1024, 1024) or not np.isfinite(raw).all():
            raise ValueError(f"{raw_path}: expected finite raw [1,3,1024,1024]")
        raw_hash = sha256(raw_path)
        if sample.get("raw_sha256") and sample["raw_sha256"] != raw_hash:
            raise ValueError(f"{raw_path}: benchmark raw SHA256 mismatch")
        raw = np.asarray(raw, np.float32)
        expected = np.rint(np.clip((raw[0].transpose(1, 2, 0)+1)*127.5, 0, 255)).astype(np.uint8)
        with Image.open(png_path) as image:
            actual = np.asarray(image.convert("RGB"))
        if not np.array_equal(actual, expected):
            raise ValueError(f"{png_path}: PNG differs from raw-RGB conversion")
        pixels.append(actual)
        files.append(dict(sample_index=sample["sample_index"], raw_path=str(raw_path),
                          raw_sha256=raw_hash, png_path=str(png_path), png_sha256=sha256(png_path)))
    return np.stack(pixels), dict(directory=str(directory), summary_sha256=sha256(summary_path),
        precision=summary["precision"], low_precision_stages=summary["low_precision_stages"],
        benchmark_variant=summary.get("benchmark_variant", "stage-selective operand precision"),
        precision_policy=summary.get("policy", {}),
        repeat=repeat_index, input_sha256=input_sha, files=files,
        model_identity=summary["model_identity"], source_hashes=summary.get("source_hashes", {}))


def reward_groups(scores, reference):
    import numpy as np
    scores, reference = np.asarray(scores, np.float32), np.asarray(reference, np.float32)
    groups = []
    for group in range(2):
        indices = [2*group, 2*group+1]
        gap = float(scores[indices[1]]-scores[indices[0]])
        original = float(reference[indices[1]]-reference[indices[0]])
        groups.append(dict(group=group, sample_indices=indices, scores=scores[indices].tolist(),
            second_minus_first=gap, reference_second_minus_first=original, gap_delta=gap-original,
            gap_relative_change=(gap-original)/abs(original) if original else None,
            ordering_sign=int(np.sign(gap)), reference_ordering_sign=int(np.sign(original)),
            ordering_preserved=bool(np.sign(gap) == np.sign(original)), tie=gap == 0))
    return groups


def art_child(request_path, output_path):
    import numpy as np
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(ROOT))
    from moss_art.art import covariance
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("ART comparison must be CPU-only")
    request = json.loads(Path(request_path).read_text())
    endpoints = np.load(request["latents"], allow_pickle=False)
    groups = np.array([0, 0, 1, 1], np.int32)
    profiles, reference_stats = [], None
    for profile in request["profiles"]:
        scores = jnp.asarray(profile["scores"], jnp.float32)
        # Exact NativeWorker.prepare_update call; direction is not reimplemented.
        stats = covariance(endpoints, scores, groups, **request["art_kwargs"])
        effective = stats.get("binary_rewards", scores)
        centered = effective-stats["group_reward_means"][groups]
        mad = stats["group_reward_abs_means"][groups]
        normalized = centered/jnp.where(mad > 0, mad, 1.0)
        stats = {k: np.asarray(v) for k, v in stats.items()}
        if reference_stats is None:
            reference_stats = stats
        cosines, errors = [], []
        for candidate, reference in zip(stats["group_directions"], reference_stats["group_directions"]):
            a, b = candidate.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
            an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cosines.append(float(np.dot(a, b)/(an*bn)) if an and bn else None)
            errors.append(float(np.linalg.norm(a-b)/bn) if bn else None)
        item = dict(label=profile["label"], effective_rewards=np.asarray(effective).tolist(),
            centered_rewards=np.asarray(centered).tolist(), centered_over_group_mad=np.asarray(normalized).tolist(),
            normalized_reward_definition="(effective reward-real ART group mean)/real ART group MAD; zero MAD uses 1, matching covariance",
            group_direction_cosine_reference=cosines, group_direction_relative_l2_reference=errors,
            direction_finite=bool(np.isfinite(stats["group_directions"]).all()),
            group_active_equal_reference=bool(np.array_equal(stats["group_active"], reference_stats["group_active"])),
            row_active_equal_reference=bool(np.array_equal(stats["row_active"], reference_stats["row_active"])))
        for key in ("group_reward_means", "group_reward_stds", "group_reward_abs_means",
                    "group_covariance_rms", "group_guidance_magnitudes", "group_active", "row_active"):
            item[key] = stats[key].tolist()
        profiles.append(item)
    atomic_json(output_path, dict(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
        function="moss_art.art.covariance", function_source_sha256=sha256(ROOT/"moss_art/art.py"),
        call_kwargs=request["art_kwargs"], profiles=profiles))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--benchmark", default=str(ROOT/"outputs/rollout_1024_cudnn_tuned/benchmark.json"))
    parser.add_argument("--latents", default=str(ROOT/"outputs/rollout_1024_cudnn_tuned/repeat_00/endpoints.npy"))
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--reward-python", default="$CONDA_PREFIX/envs/torch_env/bin/python")
    parser.add_argument("--jax-python", default='$CONDA_PREFIX/envs/jax_env/bin/python')
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--art-request", help=argparse.SUPPRESS)
    parser.add_argument("--art-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.repeat_index < 0 or args.timeout <= 0:
        parser.error("threads/timeout must be positive; repeat-index nonnegative")
    os.environ.update(cpu_environment(args.threads))
    if args.art_request:
        if not args.art_output:
            parser.error("--art-output required")
        art_child(args.art_request, args.art_output)
        return
    if not args.reference or not args.candidate or not args.output:
        parser.error("--reference, one or more --candidate, and --output required")
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; choose another output or --overwrite")
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from moss_art.rewards import score_images
    begun = time.monotonic()
    report = dict(status="running", recorded_utc=datetime.now(timezone.utc).isoformat(),
        argv=list(sys.argv if argv is None else [sys.argv[0], *argv]), script_sha256=sha256(__file__),
        cpu_only=True, profiles=[])
    try:
        benchmark_path, latents_path = Path(args.benchmark).resolve(), Path(args.latents).resolve()
        benchmark = json.loads(benchmark_path.read_text())
        config, records, seeds = benchmark["config"], benchmark["records"], benchmark["seeds"]
        if benchmark["status"] != "passed" or config["sampling"]["group_size"] != 2:
            raise ValueError("Expected passed K=2 rollout benchmark")
        if [r["sample_id"] for r in records] != ["cat", "girl"] or seeds != [42, 43, 1051, 1052]:
            raise ValueError("Expected ordered cat42/43, girl1051/1052 rollout")
        endpoints = np.load(latents_path, allow_pickle=False)
        if endpoints.shape != (4, 128, 64, 64) or not np.isfinite(endpoints).all():
            raise ValueError("Expected finite original four 1024 endpoints")
        input_sha = sha256(latents_path)
        prompts = [r["prompt"] for r in records for _ in range(2)]
        report.update(benchmark_path=str(benchmark_path), benchmark_sha256=sha256(benchmark_path),
            latents_path=str(latents_path), latents_sha256=input_sha,
            rows=[dict(index=i, group=i//2, sample_id=records[i//2]["sample_id"],
                       seed=seeds[i], prompt=prompts[i]) for i in range(4)])
        batches = []
        for index, directory in enumerate([args.reference, *args.candidate]):
            images, metadata = load_profile(directory, input_sha, args.repeat_index, index == 0)
            metadata["label"] = "reference" if index == 0 else f"candidate_{index}:{Path(directory).name}"
            if report["profiles"] and metadata["model_identity"] != report["profiles"][0]["model_identity"]:
                raise ValueError(f"{directory}: VAE model identity differs")
            batches.append(images)
            report["profiles"].append(metadata)
        if args.validate_only:
            report["status"] = "validated_inputs_only"
            return
        reward = dict(config["reward"])
        if reward["kind"] != "siglip":
            raise ValueError("Original benchmark must use SigLIP")
        reward.update(device="cpu", python=args.reward_python, threads=args.threads, timeout=args.timeout)
        report["reward_config"] = {k: reward[k] for k in
            ("kind", "model_path", "python", "device", "batch_size", "metric", "threads")}
        score_start = time.monotonic()
        scores = score_images(np.concatenate(batches), prompts*len(batches), reward).reshape(len(batches), 4)
        report["score_seconds"] = time.monotonic()-score_start
        original = np.asarray(benchmark["repeats"][0]["rewards"], np.float32)
        for index, metadata in enumerate(report["profiles"]):
            metadata.update(scores=scores[index].tolist(), score_delta_reference=(scores[index]-scores[0]).tolist(),
                max_abs_score_delta_reference=float(np.abs(scores[index]-scores[0]).max()),
                groups=reward_groups(scores[index], scores[0]))
        report["reference_score_delta_original_benchmark"] = (scores[0]-original).tolist()
        report["original_benchmark_scores"] = original.tolist()
        request = dict(latents=str(latents_path), art_kwargs=dict(mode=config["art"]["feedback"], group_size=2,
            seed=config["training"]["seed"], eps=config["art"]["eps"]),
            profiles=[dict(label=p["label"], scores=p["scores"]) for p in report["profiles"]])
        with tempfile.TemporaryDirectory(prefix="moss-vae-art-cpu-") as temporary:
            req, result_path = Path(temporary)/"request.json", Path(temporary)/"result.json"
            atomic_json(req, request)
            result = subprocess.run([args.jax_python, str(Path(__file__).resolve()), "--art-request", str(req),
                "--art-output", str(result_path), "--threads", str(args.threads)],
                env=cpu_environment(args.threads), text=True, capture_output=True, timeout=args.timeout, cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f"CPU ART failed ({result.returncode}): {result.stderr[-8000:]}")
            report["art_comparison"] = json.loads(result_path.read_text())
        report["status"] = "completed"
        report["limitations"] = ["Four fixed samples in two K=2 groups do not establish general image quality.",
            "No optimizer update or model gradient is performed.",
            "K=2 normalized ART direction largely follows reward ordering; also inspect gaps and covariance RMS.",
            "CPU SigLIP retains original metric/model/batch size; original-reference mismatch is reported."]
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["total_seconds"] = time.monotonic()-begun
        atomic_json(output, report)
        print(json.dumps(dict(status=report["status"], output=str(output))), flush=True)


if __name__ == "__main__":
    main()
