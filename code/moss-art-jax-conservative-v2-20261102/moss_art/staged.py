"""Sequential on-policy collection and cached training in separate processes.

Module import deliberately constructs no accelerator model. Only the training
entry imports JAX; the auto controller launches one phase process at a time.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def cache_root(c):
    """Namespace an optional scratch filesystem by the durable output path."""
    output = Path(c["runtime"]["output_dir"]).resolve()
    configured = c["staging"].get("cache_dir", "")
    if not configured:
        return output/"stage_cache"
    root = Path(configured).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Stage cache root must not be a symlink")
    namespace = hashlib.sha256(str(output).encode()).hexdigest()[:20]
    return root/namespace


def collection_path(c, update):
    root = cache_root(c)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError("Stage cache namespace must not be a symlink")
    reserve = c["staging"].get("min_free_cache_gib", 0)*2**30
    if shutil.disk_usage(root).free < reserve:
        raise RuntimeError("Stage scratch filesystem has insufficient free space")
    return root/("update_%08d" % update)


def retire_collection(c, bundle):
    """Remove only this run's consumed scratch data, never durable images."""
    bundle = Path(bundle)
    if c["staging"]["keep_consumed_cache"]:
        return
    root = cache_root(c).resolve()
    if (bundle.is_symlink() or bundle.resolve().parent != root or
            not (bundle/"consumed.json").is_file()):
        raise RuntimeError("Refusing to remove an unverified stage cache")
    shutil.rmtree(bundle)


def persist_collection(c, shards, prepared, lora, optimizer, metadata, rewards,
                       rollout_metrics, resolution_label, collection_seconds):
    from .stage_cache import save_bundle
    from .trainer import write_json
    output = Path(c["runtime"]["output_dir"]).resolve()
    bundle = collection_path(c, metadata["update"])
    metadata = dict(metadata, stage_cache_id=str(bundle))
    payload = dict(config=c, shards=shards, prepared=prepared, rewards=rewards,
                   rollout_metrics=rollout_metrics, resolution_label=resolution_label,
                   collection_seconds=collection_seconds)
    started = time.monotonic()
    save_bundle(bundle, payload, lora, optimizer, metadata,
                reserve_bytes=c["staging"].get("min_free_cache_gib", 0)*2**30)
    summary = dict(bundle=str(bundle), update=metadata["update"], rollout=metadata["rollout"],
                   cache_write_seconds=time.monotonic()-started,
                   cache_bytes=sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()))
    write_json(output/"staging/ready.json", summary)
    print(json.dumps(dict(event="rollout_stage_committed", **summary)), flush=True)
    return bundle


def _publish(c, bundle, checkpoint, events, timings):
    """Publish only completed two-window updates; replaying publication is safe."""
    from .trainer import append_json, write_json
    from .tensorboard_log import ScalarLogger
    output = Path(c["runtime"]["output_dir"])
    metrics = output/"metrics.jsonl"
    seen = {int(json.loads(line)["update"]) for line in metrics.read_text().splitlines() if line.strip()} if metrics.exists() else set()
    with ScalarLogger(c, output) as logger:
        for event in events:
            if event["update"] not in seen:
                append_json(metrics, event)
                logger.write("train", event["update"], event)
                print(json.dumps(event), flush=True)
        logger.write("staging", events[-1]["update"], timings)
    result = dict(checkpoint=str(checkpoint), update=events[-1]["update"], timings=timings)
    if (Path(bundle)/"COMMITTED").is_file():
        write_json(Path(bundle)/"consumed.json", result)
    write_json(output/"staging/last_train.json", result)
    return Path(checkpoint)


def train_cached(c):
    started = time.monotonic()
    from .runtime import configure_process, preflight
    preflight(c)
    configure_process(c, c["runtime"]["devices"])
    import jax
    import numpy as np
    import optax
    from .stage_cache import load_bundle
    from .checkpoint import load_checkpoint
    from .trainer import identity_for, aggregate_gradients, apply_window_update, _save, _optimizer
    from .optimizers import learning_rate_at_step
    from .locking import output_lock
    imports_finished = time.monotonic()
    bundle = Path(c["staging"]["bundle"]).resolve()
    output = Path(c["runtime"]["output_dir"]).resolve()
    with output_lock(output):
        expected = identity_for(c)
        # A committed checkpoint lets a restart finish publication without
        # applying either ART window a second time.
        latest = output/"latest.json"
        if latest.exists():
            checkpoint = Path(json.loads(latest.read_text())["checkpoint"])
            _, _, saved = load_checkpoint(checkpoint, expected_metadata=expected)
            if saved.get("stage_cache_id") == str(bundle):
                return _publish(c, bundle, checkpoint, saved["staged_events"], saved["staged_timings"])
        cache_started = time.monotonic()
        payload, metadata = load_bundle(bundle)
        cache_seconds = time.monotonic()-cache_started
        state_started = time.monotonic()
        params, state, saved = load_checkpoint(bundle/"learner", expected_metadata=expected)
        if saved != metadata:
            raise ValueError("Stage replay and learner metadata disagree")
        if Path(payload["config"]["runtime"]["output_dir"]).resolve() != output:
            raise ValueError("Stage cache belongs to a different output directory")
        if latest.exists() and saved["update"] != json.loads(latest.read_text())["update"]:
            raise ValueError("Stage cache is stale relative to the latest checkpoint")
        state_seconds = time.monotonic()-state_started
        active_config = copy.deepcopy(payload["config"])
        # Execution phase comes from the current request, numeric sampler and
        # resolution settings come from the exact collected batch.
        active_config["staging"] = copy.deepcopy(c["staging"])
        if payload.get("phase") == "inferred":
            from .stage_pipeline import prepare_training
            payload = prepare_training(c, bundle, payload, metadata)
        from .replay import replay_identity
        version = replay_identity(payload["shards"])["parameter_version"]
        from .mesh_runtime import MeshPool
        from .tensorboard_log import ScalarLogger
        metadata = copy.deepcopy(metadata)
        update, optimizer_updates = metadata["update"], metadata["optimizer_updates"]
        timings = dict(import_seconds=imports_finished-started, cache_read_seconds=cache_seconds,
                       learner_state_read_seconds=state_seconds)
        no_signal = not any(np.any(p["stats"]["group_active"]) for p in payload["prepared"])
        events = []
        if no_signal and not c["art"]["reference_mse_beta"]:
            update += c["training"]["updates_per_rollout"]
            metadata["no_signal_collections"] = metadata.get("no_signal_collections", 0)+1
            if metadata["no_signal_collections"] >= c["training"]["max_no_signal_rollouts"]:
                raise RuntimeError("Consecutive no-signal rollout limit reached")
            events.append(dict(event="no_signal", update=update, optimizer_updates=optimizer_updates,
                optimizer_step_applied=False, rollout=metadata["rollout"], reward_mean=float(payload["rewards"].mean())))
            timings.update(model_load_seconds=0., calibration_seconds=0., startup_ready_seconds=time.monotonic()-started)
        else:
            model_started = time.monotonic()
            with MeshPool(active_config, initial_lora=params) as pool:
                params = pool.place_state(params); state = pool.place_state(state)
                jax.block_until_ready((params, state))
                pool.worker.set_lora(params)
                pool.info[0]["lora"] = None
                pool.parameter_version = version
                timings["model_load_seconds"] = time.monotonic()-model_started
                timings["startup_ready_seconds"] = time.monotonic()-started
                print(json.dumps(dict(event="training_stage_model_ready", **timings)), flush=True)
                pool.call("install_replay", [{"cache": cache} for cache in payload["shards"]])
                calibration_started = time.monotonic()
                pool.call("prepare_update", payload["prepared"])
                timings["calibration_seconds"] = time.monotonic()-calibration_started
                timings["ready_for_gradient_seconds"] = time.monotonic()-started
                if pool.anchor_metrics:
                    with ScalarLogger(c, output) as logger:
                        logger.write("anchor", metadata["rollout"], pool.anchor_metrics)
                tx = _optimizer(active_config)
                metadata["no_signal_collections"] = 0
                for window in range(c["training"]["updates_per_rollout"]):
                    gradient_started = time.monotonic()
                    result = pool.call("gradient", [{"window": window} for _ in pool.workers])
                    gradients, metrics = aggregate_gradients(result)
                    metrics["grad_norm"] = float(optax.global_norm(gradients))
                    if not np.isfinite(metrics["grad_norm"]):
                        raise FloatingPointError("Nonfinite staged gradient")
                    before = params
                    applied_learning_rate = learning_rate_at_step(active_config["optimizer"], int(optimizer_updates))
                    params, state, delta, applied = apply_window_update(tx, gradients, state, params, metrics["kept_count"])
                    jax.block_until_ready((params, state))
                    if not all(bool(np.asarray(jax.numpy.all(jax.numpy.isfinite(v)))) for v in params.values()):
                        raise FloatingPointError("Nonfinite staged adapter")
                    metrics["adapter_delta_norm"] = float(optax.global_norm(jax.tree.map(lambda n, p: n-p, params, before)))
                    optimizer_updates += int(applied); update += 1
                    metrics.update(optimizer_step_applied=bool(applied), optimizer_updates=optimizer_updates,
                                   gradient_update_seconds=time.monotonic()-gradient_started)
                    if applied:
                        metrics["learning_rate"] = applied_learning_rate
                    if window == 0:
                        timings["first_update_seconds"] = time.monotonic()-started
                    event = dict(event="update", update=update, rollout=metadata["rollout"], window=window,
                        **payload["rollout_metrics"], **metrics)
                    event.update(prompts_per_update=c["training"]["prompts_per_update"],
                        rollouts_per_prompt=c["training"]["rollouts_per_prompt"],
                        training_points_per_trajectory=c["training"]["training_points_per_trajectory"],
                        trajectories=len(payload["rewards"]))
                    events.append(event)
                    if applied:pool.call("set_lora", [{"lora": params} for _ in pool.workers])
                    del gradients, result, delta, before
                pool.call("clear_replay")
                # Checkpoint after releasing model and activation buffers, but
                # keep only the small trainable state to avoid host peaks.
            timings["training_compute_seconds"] = time.monotonic()-model_started
        metadata.update(update=update, optimizer_updates=optimizer_updates, staged_events=events, staged_timings=timings)
        save_started = time.monotonic()
        checkpoint = _save(c, output, params, state, metadata)
        timings["checkpoint_seconds"] = time.monotonic()-save_started
        timings["total_train_process_seconds"] = time.monotonic()-started
        return _publish(c, bundle, checkpoint, events, timings)


def _ready(output):
    file = Path(output)/"staging/ready.json"
    if file.exists():
        bundle = Path(json.loads(file.read_text())["bundle"])
        if (bundle/"COMMITTED").is_file() and not (bundle/"consumed.json").exists():
            return bundle
    return None


def run(c):
    phase = c["staging"]["phase"]
    if phase == "eval":
        from .stage_pipeline import evaluate_cached
        return evaluate_cached(c)
    if phase == "train":
        return train_cached(c)
    if phase in ("collect", "infer"):
        from .trainer import train
        ready = _ready(c["runtime"]["output_dir"])
        return ready if ready is not None else train(c, c["run"]["resume"] or None, c["run"]["init_adapter"] or None)
    return run_auto(c)


def run_auto(c):
    """A lightweight parent; no queue, no model, no overlapping phase processes."""
    from .trainer import write_json, append_json
    from .locking import output_lock
    output = Path(c["runtime"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with output_lock(output/"staging/controller"):
        while True:
            latest = output/"latest.json"
            current = json.loads(latest.read_text()) if latest.exists() else None
            if current and current["update"] >= c["training"]["max_updates"]:
                from .checkpoint import load_checkpoint
                from .lora import export_safetensors
                from .trainer import identity_for
                params, _, metadata = load_checkpoint(current["checkpoint"], expected_metadata=identity_for(c))
                export_safetensors(output/"adapter.safetensors", params, alpha=c["lora"]["alpha"])
                write_json(output/"complete.json", dict(update=current["update"], optimizer_updates=metadata["optimizer_updates"], staged=True))
                return output
            bundle = _ready(output)
            phase = ("eval" if bundle and (bundle/"eval_request.json").is_file()
                     and not (bundle/"evaluated.json").is_file() else "train" if bundle else "collect")
            child = copy.deepcopy(c)
            child["staging"].update(phase=phase, bundle=str(bundle) if bundle else "")
            if current:
                child["run"].update(resume=current["checkpoint"], init_adapter="")
            config = output/"staging"/(phase+".json")
            write_json(config, child)
            started = time.monotonic()
            result = subprocess.run([sys.executable, str(root/"run.py"), "train", "--config", str(config)], cwd=root)
            event = dict(phase=phase, seconds=time.monotonic()-started, returncode=result.returncode,
                         update=current["update"] if current else 0)
            append_json(output/"staging/process_times.jsonl", event)
            if result.returncode:
                raise RuntimeError("%s stage exited with code %s; committed cache/checkpoints were retained" % (phase, result.returncode))
            if phase == "train" and not c["staging"]["keep_consumed_cache"]:
                # Only retire this controller's successfully consumed cache.
                retire_collection(c, bundle)
