#!/usr/bin/env python3
"""Bounded, real-model fixed-replay ART precision diagnostic.

Usage: python scripts/probe_fp8_anchor.py --config /absolute/probe.json

Config wrapper (all options live in JSON, not command-line flags)::

  {"base_config": "/absolute/training-config.json", "overrides": {},
   "probe": {"output_dir": "$MOSS_ROOT/audits/anchor/probe1",
             "sample_id": "", "two_window_updates": true, "dry_run": false,
             "initial_checkpoint": "/absolute/checkpoints/step_00000006",
             "native_conv_build": "/absolute/native/build.json"}}

Uses one cached prompt, K8/S8, 768 square, 48 steps and two update windows.
The sampler, schedule, reward, LoRA, optimizer and seed come from base_config.
Starts both comparison branches from the SAME supplied checkpoint LoRA and
optimizer state, or the same seeded initial state when no checkpoint is given.
Checkpoint loading is an explicit diagnostic fork, not a training resume.
Never changes the live run or source checkpoint.
The FP8 replay is shared by raw/calibrated; no online quality gain is claimed.
Both branches optionally execute A+B updates against that SAME fixed replay.
Only this diagnostic driver deliberately restores theta0 between branches.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NVME = Path("$DATA_DISK")
DEFAULT_NATIVE = (NVME / "moss-art-jax/runs/tags40k_48step_r48_20261022/native/build.json")


def load_settings(path):
    """Validate portable inputs without importing JAX or touching a GPU."""
    from moss_art.config import load_config
    path = Path(path).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or set(source) - {"base_config", "overrides", "probe"}:
        raise ValueError("Expected base_config, optional overrides, and probe mappings")
    base_path = Path(source["base_config"]).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    probe = dict(sample_id="", initial_checkpoint="", two_window_updates=True, dry_run=False,
                 native_conv_build=str(DEFAULT_NATIVE))
    given = source.get("probe", {})
    if not isinstance(given, dict) or set(given) - {*probe, "output_dir"}:
        raise ValueError("Unknown probe option")
    probe.update(given)
    if not probe.get("output_dir"):
        raise ValueError("probe.output_dir is required and must be on NVMe")
    if not all(type(probe[k]) is bool for k in ("two_window_updates", "dry_run")):
        raise ValueError("two_window_updates and dry_run must be booleans")
    if not all(isinstance(probe[k], str) for k in ("sample_id", "initial_checkpoint", "native_conv_build")):
        raise ValueError("sample_id, initial_checkpoint and native_conv_build must be strings")
    if probe["initial_checkpoint"]:
        checkpoint = Path(probe["initial_checkpoint"]).expanduser().resolve()
        if not all((checkpoint / name).is_file() for name in ("manifest.json", "arrays.npz", "COMMITTED")):
            raise ValueError("initial_checkpoint must be a complete checkpoint directory")
        probe["initial_checkpoint"] = str(checkpoint)
    out = Path(probe["output_dir"]).expanduser().resolve()
    if NVME.resolve() not in out.parents:
        raise ValueError("probe.output_dir must be beneath $DATA_DISK")
    if probe["initial_checkpoint"]:
        checkpoint = Path(probe["initial_checkpoint"])
        if out == checkpoint or checkpoint in out.parents:
            raise ValueError("Diagnostic output cannot modify the source checkpoint")
    probe["output_dir"] = str(out)
    overrides = copy.deepcopy(source.get("overrides", {}))
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a mapping")
    forced = {
        "preparation": {"async_prefetch": False, "auto_prepare": False},
        "sampling": {"height": 768, "width": 768, "steps": 48,
                     "rollout_mode": "mxfp8_calibrated", "pair_cached_points": True},
        "training": {"prompts_per_update": 1, "rollouts_per_prompt": 8,
                     "training_points_per_trajectory": 8, "updates_per_rollout": 2,
                     "max_updates": 2, "checkpoint_every": 2, "evaluate_every": 0, "save_rollout_images": True},
        "resolution_mixing": {"enabled": False, "profiles": []},
        "runtime": {"execution_backend": "mesh", "platform": "cuda", "output_dir": str(out)},
        "logging": {"tensorboard": False, "log_dir": str(out / "tensorboard")},
        "run": {"resume": "", "init_adapter": ""},
    }
    # Probe dimensions/budget are intentionally bounded, visible in resolved config.
    for section, values in forced.items():
        overrides.setdefault(section, {}).update(values)
    c = load_config(base_path, overrides=overrides)
    if c["model"]["kind"] != "moss" or len(c["runtime"]["devices"]) != 4:
        raise ValueError("This real-model diagnostic requires MOSS and four compute GPUs")
    if c["model"]["vae_conv_backend"] == "cudnn_bounded":
        build = Path(probe["native_conv_build"]).resolve()
        if not build.is_file():
            raise ValueError("native_conv_build must point to the existing cuDNN build manifest")
        probe["native_conv_build"] = str(build)
    return c, probe, str(base_path.resolve())


def cached_record(c, requested=""):
    """Select an actual dataset row with two complete cached Qwen conditions."""
    from moss_art.data import PromptCache, load_prompts, text_key
    records = load_prompts(c["data"]["prompts"], default_negative_prompt=c["data"]["negative_prompt"])
    cache = PromptCache(c["data"]["cache"], model_dir=c["model"]["path"])
    entries = cache.manifest["entries"]
    candidates = []
    for record in records:
        if requested and record.id != requested:
            continue
        keys = [text_key(record.prompt), text_key(record.negative_prompt)]
        if not all(key in entries for key in keys):
            continue
        tokens = [int(entries[key]["token_count"]) for key in keys]
        if max(tokens) > c["data"]["context_length"]:
            continue
        candidates.append((record, tokens))
    if not candidates:
        raise ValueError("No requested dataset row has two complete cached embeddings within context_length")
    # Prefer the established short bucket to reduce first smoke-test compilation.
    short = c["data"]["short_context_length"]
    candidates.sort(key=lambda pair: max(pair[1]) > short if short else False)
    record, tokens = candidates[0]
    cache.load(record)  # Validate hashes/shapes before allocating model memory.
    result = dict(sample_id=record.id, prompt=record.prompt,
                  negative_prompt=record.negative_prompt, metadata=record.metadata)
    info = dict(sample_id=record.id, positive_tokens=tokens[0], negative_tokens=tokens[1],
                positive_sha256=text_key(record.prompt), negative_sha256=text_key(record.negative_prompt),
                eligible_cached_records=len(candidates), qwen_inference=False)
    return result, info


def summarize_by_sigma(rows):
    import numpy as np
    groups = {}
    for row in rows:
        groups.setdefault(float(row["sigma"]), []).append(row)
    result = []
    for sigma in sorted(groups, reverse=True):
        values = groups[sigma]
        active = [v for v in values if v["active"]]
        item = dict(sigma=sigma, rows=len(values), active_rows=len(active))
        for key in ("e_rms", "g_rms", "e_over_g", "cosine"):
            item[key+"_mean"] = float(np.mean([v[key] for v in active])) if active else 0.
            item[key+"_max"] = float(max((v[key] for v in active), default=0.))
        result.append(item)
    return result


def tree_comparison(left, right):
    """Device reductions only; never flatten an entire model into host RAM."""
    import jax
    import jax.numpy as jnp
    a, sa = jax.tree.flatten(left)
    b, sb = jax.tree.flatten(right)
    if sa != sb or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError("Comparison trees differ")
    aa = sum(jnp.sum(x.astype(jnp.float32)**2) for x in a)
    bb = sum(jnp.sum(x.astype(jnp.float32)**2) for x in b)
    ab = sum(jnp.sum(x.astype(jnp.float32)*y.astype(jnp.float32)) for x, y in zip(a, b))
    diff = sum(jnp.sum((x.astype(jnp.float32)-y.astype(jnp.float32))**2) for x, y in zip(a, b))
    aa, bb, ab, diff = map(float, jax.device_get((aa, bb, ab, diff)))
    na, nb = max(aa, 0.)**.5, max(bb, 0.)**.5
    return dict(left_norm=na, right_norm=nb, difference_norm=max(diff, 0.)**.5,
                cosine=ab/(na*nb) if na*nb else None,
                relative_difference_to_left=max(diff, 0.)**.5/na if na else None,
                left_zero=not bool(na), right_zero=not bool(nb))


def anchor_digest(caches):
    digest = hashlib.sha256()
    for rank, cache in enumerate(caches):
        for (row, slot), value in sorted(cache.items()):
            digest.update(f"{rank}/{row}/{slot}".encode())
            digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def prepare_environment(output, probe):
    for key, rel in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
                     ("XDG_CACHE_HOME", "cache/xdg"), ("CUDA_CACHE_PATH", "cache/cuda"),
                     ("TRITON_CACHE_DIR", "cache/triton")):
        path = output / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    if probe["native_conv_build"]:
        os.environ["MOSS_CUDNN_CONV_BUILD"] = probe["native_conv_build"]
    # configure_process selects .jax_cache relative to cwd. Keep it on NVMe.
    os.chdir(output)


def run(c, probe, base_path, record, record_info):
    from moss_art.runtime import configure_process, preflight
    from moss_art.locking import output_lock
    # No JAX import, device initialization or accelerator work until here.
    print(json.dumps(dict(event="probe_starting",sample_id=record_info["sample_id"],
                          output=probe["output_dir"],initial_checkpoint=probe["initial_checkpoint"])),flush=True)
    inventory = preflight(c)
    output = Path(probe["output_dir"])
    with output_lock(output):
        if (output / "report.json").exists():
            raise FileExistsError("Probe output already contains a completed report; use a new output_dir")
        prepare_environment(output, probe)
        configure_process(c, c["runtime"]["devices"])
        import jax
        import numpy as np
        from moss_art.art import covariance, prompt_windows
        from moss_art.mesh_runtime import MeshPool
        from moss_art.memory import memory_snapshot
        from moss_art.trainer import (write_json, make_payloads, reward_round,
            paired_replay_payloads, aggregate_gradients, apply_window_update, _optimizer)
        from moss_art.checkpoint import save_checkpoint, load_checkpoint
        from moss_art.tensorboard_log import ScalarLogger
        report = dict(schema_version=1, status="running", base_config=base_path,
                      purpose="Fixed MXFP8 replay raw/calibrated diagnostic; not image-quality evaluation",
                      initial_state=("Explicit checkpoint fork: "+probe["initial_checkpoint"] if probe["initial_checkpoint"] else
                                     "Seeded initial LoRA and fresh optimizer shared by both branches"),
                      dataset_record=record_info, gpu_inventory=inventory,
                      budget=dict(prompts=1, trajectories=8, points_per_trajectory=8,
                                  total_training_points=64, points_per_window=32,
                                  resolution=[768,768], sampling_steps=48),
                      config=c, probe=probe, timings={}, memory={})
        write_json(output / "resolved_config.json", c)
        write_json(output / "probe_config.json", dict(base_config=base_path, probe=probe))
        started = time.monotonic()
        def stage(name, fn):
            begin = time.monotonic()
            value = fn()
            report["timings"][name] = time.monotonic()-begin
            report["memory"][name] = memory_snapshot(jax.local_devices())
            report["memory"][name]["gpu_peak_allocated_gib"] = [
                (device.memory_stats() or {}).get("peak_bytes_in_use",0)/2**30
                for device in jax.local_devices()]
            write_json(output / "progress.json", report)
            print(json.dumps(dict(event="probe_stage", stage=name, seconds=report["timings"][name])), flush=True)
            return value
        try:
            restored = (load_checkpoint(probe["initial_checkpoint"],expected_metadata=None)
                        if probe["initial_checkpoint"] else None)
            with ScalarLogger(c,output) as logger, MeshPool(c,initial_lora=restored[0] if restored else None) as pool:
                pool.scalar_logger = logger
                report["timings"]["initial_model_load"] = time.monotonic()-started
                theta0 = pool.place_state(pool.worker.lora)
                pool.info[0]["lora"] = None
                tx = _optimizer(c)
                if restored:
                    source_lora, source_opt, metadata = restored
                    if metadata.get("lora_config") != c["lora"]:
                        raise ValueError("Checkpoint LoRA configuration does not match probe")
                    if metadata.get("model_config") != pool.info[0]["model_config"]:
                        raise ValueError("Checkpoint model configuration does not match probe")
                    expected = jax.eval_shape(tx.init,theta0)
                    actual_leaves, actual_tree = jax.tree.flatten(source_opt)
                    expected_leaves, expected_tree = jax.tree.flatten(expected)
                    if actual_tree != expected_tree or any(a.shape != b.shape for a,b in zip(actual_leaves,expected_leaves)):
                        raise ValueError("Checkpoint optimizer structure is incompatible with configured optimizer")
                    opt0 = pool.place_state(source_opt)
                    report["checkpoint_origin"] = dict(path=probe["initial_checkpoint"],update=metadata.get("update"),
                                                       rollout=metadata.get("rollout"),optimizer_restored=True)
                    del restored,source_lora,source_opt,metadata,actual_leaves,expected_leaves,expected
                else:
                    opt0 = pool.place_state(tx.init(theta0))
                version0 = pool.parameter_version
                outputs = stage("mxfp8_rollout", lambda: pool.call("rollout", make_payloads(c,[record],0)))
                seeds = [seed for result in outputs for seed in result["seeds"]]
                report["seeds"] = seeds
                report["rollout_precision_contract"] = outputs[0]["precision_contract"]
                rewards, images = stage("vae_and_reward", lambda: reward_round(c,pool,outputs,output/"images"))
                del images
                write_json(output / "rewards.json", dict(rewards=rewards.tolist(),
                           details=getattr(pool,"last_reward_details",None)))
                report["raw_reward"] = dict(mean=float(rewards.mean()), std=float(rewards.std()),
                                             min=float(rewards.min()), max=float(rewards.max()))
                endpoints = np.concatenate([item["endpoints"] for item in outputs])
                groups = np.zeros(len(endpoints), np.int32)
                stats = covariance(endpoints,rewards,groups,1,mode=c["art"]["feedback"],
                                   group_size=8,seed=c["training"]["seed"],eps=c["art"]["eps"])
                windows = prompt_windows(groups,c["sampling"]["train_timesteps"],seed=c["training"]["seed"])
                payloads = paired_replay_payloads(pool,groups,stats,windows)
                stage("bf16_reference_prepass", lambda: pool.call("prepare_update",payloads))
                report["anchor"] = dict(pool.anchor_metrics)
                report["anchor_by_sigma"] = summarize_by_sigma(pool.anchor_diagnostics)
                write_json(output / "anchor_rows.json", pool.anchor_diagnostics)
                digest_before = anchor_digest(pool.anchor_cache)
                report["anchor_sha256_before_updates"] = digest_before
                report["learner_precision_contract"] = pool.worker.model.precision_contract()
                def gradient(mode, window):
                    pool.config["sampling"]["rollout_mode"] = mode
                    result = pool.call("gradient",[dict(window=window) for _ in pool.workers])
                    return aggregate_gradients(result)
                # No parameter or optimizer mutation between these evaluations.
                raw_a, raw_stats = stage("raw_A_gradient", lambda: gradient("mxfp8",0))
                cal_a, cal_stats = stage("calibrated_A_gradient", lambda: gradient("mxfp8_calibrated",0))
                report["A"] = dict(raw=raw_stats, calibrated=cal_stats,
                                    gradients=tree_comparison(raw_a,cal_a))
                report["timing_note"] = "First-use JIT compilation included; not a warmed inference throughput comparison"
                if probe["two_window_updates"]:
                    finals = {}
                    for label, mode, a_grad, a_stats in (("raw","mxfp8",raw_a,raw_stats),
                                                       ("calibrated","mxfp8_calibrated",cal_a,cal_stats)):
                        # Explicit isolated fork: preserve monotonically increasing
                        # runtime versions, verify restored theta0, then stamp replay
                        # and frozen anchors with that equivalent parameter version.
                        if label != "raw":
                            pool.call("set_lora",[dict(lora=theta0) for _ in pool.workers])
                            match = tree_comparison(pool.worker.lora,theta0)
                            if match["difference_norm"] != 0.:
                                raise AssertionError("Diagnostic fork failed to restore exact theta0")
                            for shard in pool.shards:
                                shard["parameter_version"] = pool.parameter_version
                            pool.anchor_version = pool.parameter_version
                            report["branch_reset"] = dict(identical_theta0=True,
                                source_version=version0,new_version=pool.parameter_version,
                                frozen_anchor_sha256=anchor_digest(pool.anchor_cache))
                        lora, opt_state, delta, did_a = apply_window_update(tx,a_grad,opt0,theta0,a_stats["kept_count"])
                        jax.block_until_ready((lora,opt_state))
                        if did_a:
                            pool.call("set_lora",[dict(lora=lora) for _ in pool.workers])
                        b_grad, b_stats = stage(label+"_B_gradient",lambda: gradient(mode,1))
                        next_lora,next_opt,delta,did_b = apply_window_update(tx,b_grad,opt_state,lora,b_stats["kept_count"])
                        jax.block_until_ready((next_lora,next_opt))
                        if did_b:
                            pool.call("set_lora",[dict(lora=next_lora) for _ in pool.workers])
                        path = output / "checkpoints" / label
                        save_checkpoint(path,next_lora,next_opt,dict(diagnostic_only=True,branch=label,
                            initial_state=report["initial_state"],seed=c["training"]["seed"],
                            lora_config=c["lora"],model_config=pool.info[0]["model_config"],
                            update_count=int(did_a)+int(did_b),fixed_replay=True))
                        finals[label] = next_lora
                        report[label+"_updates"] = dict(A_did_update=did_a,B_did_update=did_b,B=b_stats,
                            B_gradient_norm=tree_comparison(b_grad,b_grad)["left_norm"],
                            adapter_delta=tree_comparison(theta0,next_lora),checkpoint=str(path))
                    report["final_parameter_comparison"] = tree_comparison(finals["raw"],finals["calibrated"])
                digest_after = anchor_digest(pool.anchor_cache)
                if digest_after != digest_before:
                    raise AssertionError("Frozen theta0 reference cache changed during updates")
                report["anchor_sha256_after_updates"] = digest_after
                report["anchor_cache_unchanged"] = True
                report["status"] = "completed"
                report["total_seconds"] = time.monotonic()-started
                report["quality_gain_measured"] = False
                write_json(output / "report.json", report)
                print(json.dumps(dict(event="probe_completed",report=str(output/"report.json"),
                                      A=report["A"],total_seconds=report["total_seconds"])),flush=True)
        except BaseException as exc:
            report.update(status="failed",error=f"{type(exc).__name__}: {exc}",
                          total_seconds=time.monotonic()-started)
            write_json(output / "failed.json",report)
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Probe wrapper JSON")
    args = parser.parse_args()
    c, probe, base_path = load_settings(args.config)
    record, info = cached_record(c,probe["sample_id"])
    if probe["dry_run"]:
        print(json.dumps(dict(dry_run=True,config=c,probe=probe,record=info),indent=2))
        return
    run(c,probe,base_path,record,info)


if __name__ == "__main__":
    main()
