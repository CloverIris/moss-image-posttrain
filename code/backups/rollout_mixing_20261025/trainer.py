"""Complete ART rollout/reward/two-update/checkpoint loop, JAX+Optax on host."""
from __future__ import annotations
import copy
from contextlib import ExitStack
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from .config import fingerprint
from .assets import experiment_identity
from .locking import output_lock
from .runtime import WorkerPool, configure_process, preflight

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)

def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

def read_records(c):
    from .prefetch import enabled,plan_records
    if enabled(c):return plan_records(c)
    from .data import load_prompts
    records = load_prompts(c["data"]["prompts"], default_negative_prompt=c["data"]["negative_prompt"])
    if c["model"]["kind"]=="moss":
        from .data import PromptCache,text_key
        from .prompt_buckets import log_drop
        cache=PromptCache(c["data"]["cache"],model_dir=c["model"]["path"])
        full=c["data"]["context_length"]
        dropped={r["id"]:r for r in cache.manifest.get("dropped_records",[])}
        kept=[]
        for r in records:
            entries=cache.manifest["entries"]
            if all(text_key(t) in entries for t in (r.prompt,r.negative_prompt)):
                length=max(entries[text_key(t)]["token_count"] for t in (r.prompt,r.negative_prompt))
                if length>full: log_drop(r.id,length,full,"art");continue
                kept.append(r)
            elif r.id in dropped and dropped[r.id].get("text_keys")==[text_key(r.prompt),text_key(r.negative_prompt)]:
                log_drop(r.id,dropped[r.id]["tokens"],full,"art")
            else:
                raise ValueError(f"Missing cached condition for {r.id}; run prepare-text")
        records=kept
        if not records: raise ValueError("All prompts exceed full length")
    return [{"sample_id": r.id, "prompt": r.prompt, "negative_prompt": r.negative_prompt,
             "metadata": r.metadata} for r in records]

def data_identity(c):
    return hashlib.sha256(Path(c["data"]["prompts"]).read_bytes()).hexdigest()

def select_records(records, cursor, count, seed):
    chosen = []
    for pos in range(cursor, cursor + count):
        epoch, offset = divmod(pos, len(records))
        order = np.random.default_rng(seed + epoch).permutation(len(records))
        chosen.append(records[int(order[offset])])
    return chosen

def make_payloads(c, records, rollout, evaluation=False):
    workers, k = len(c["runtime"]["devices"]), c["sampling"]["group_size"]
    if c["sampling"].get("split_group_across_workers",False):
        if len(records)!=1 or k%workers:raise ValueError("Incomplete split prompt group")
        width=k//workers
        start=c["training"]["seed"]+(900000001 if evaluation else 1000003*rollout)
        return [dict(records=records,seeds=[(start+rank*width+i)%2**32 for i in range(width)],
                     keep_cache=not evaluation,samples_per_prompt=width) for rank in range(workers)]
    width = len(records) // workers
    result = []
    for rank in range(workers):
        chunk = records[rank * width:(rank + 1) * width]
        seeds = [(c["training"]["seed"] + (900000001 if evaluation else 1000003 * rollout)
                  + 1009 * (rank * width + p) + sample) % (2**32)
                 for p in range(width) for sample in range(k)]
        result.append({"records": chunk, "seeds": seeds, "keep_cache": not evaluation})
    return result

def reward_round(c, pool, outputs, image_dir=None, force_decode=False):
    endpoints = np.concatenate([x["endpoints"] for x in outputs])
    prompts = sum([x["prompts"] for x in outputs], [])
    images = None
    paused = False
    kind = c["reward"]["kind"]
    resident = bool(c["sampling"].get("vae_resident_rollout", False))
    if resident and (pool is None or not hasattr(pool, "ensure_rollout_vae")):
        raise ValueError("Resident VAE requires the Mesh runtime")
    if kind != "toy_latent" or force_decode:
        if pool is not None and not resident:
            pool.pause()
            paused = True
            print(json.dumps({"event": "dit_released", "phase": "vae"}), flush=True)
        if c["reward"]["cpu_decode"]:
            from .evaluate import decode_latents
            images = decode_latents(c["model"]["path"], endpoints, device="cpu:0",
                                    precision=c["model"]["vae_precision"],
                                    low_precision_stages=c["model"]["vae_low_precision_stages"],
                                conv_backend=c["model"]["vae_conv_backend"],
                                conv_workspace_mib=c["model"]["vae_conv_workspace_mib"])
        else:
            from .phases import decode_phase
            decoded = (pool.decode([x["endpoints"] for x in outputs]) if hasattr(pool,"decode") else
                       decode_phase(c, [x["endpoints"] for x in outputs]))
            images = np.concatenate([x["images"] for x in decoded])
            # Reward model/API runs before DiT is restored below.
        if resident:
            # Decode with the MXFP8 DiT and the already loaded VAE co-resident.
            # Images are CPU uint8 before releasing both models for reward/training.
            pool.pause()
            paused = True
            print(json.dumps({"event": "rollout_models_released", "phase": "reward"}), flush=True)

        if image_dir is not None:
            from PIL import Image
            image_dir = Path(image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                Image.fromarray(image).save(image_dir / f"{index:04d}.png")
            write_json(image_dir / "prompts.json", prompts)
            samples=[]
            for result in outputs:
                for index,prompt in enumerate(result["prompts"]):
                    samples.append(dict(index=len(samples),prompt=prompt,
                        sample_id=result.get("sample_ids",[None]*len(result["prompts"]))[index],
                        seed=result.get("seeds",[None]*len(result["prompts"]))[index]))
            write_json(image_dir / "samples.json",samples)
    if kind == "toy_latent":
        rewards = -np.mean(np.square(endpoints.astype(np.float32) - np.float32(.25)),
                           axis=tuple(range(1, endpoints.ndim)))
    else:
        if kind=="silva":
            from .silva_reward import score_round
            details=pool.score(images) if hasattr(pool,"score") else score_round(c,images)
            rewards=np.asarray([x["reward"] for x in details],np.float32)
            if pool is not None:pool.last_reward_details=details
        elif kind=="pixai":
            from .pixai_reward import score_round
            targets=sum([x["target_tags"] for x in outputs],[])
            details=(pool.score(images,targets) if hasattr(pool,"score") else score_round(c,images,targets))
            rewards=np.asarray([x["reward"] for x in details],np.float32)
            pool.last_reward_details=details
        else:
            from .rewards import score_images
            rewards = np.asarray(score_images(images, prompts, c["reward"]), np.float32)
    if rewards.shape != (len(endpoints),) or not np.isfinite(rewards).all():
        raise ValueError("Reward must return one finite scalar per image")
    if paused:
        pool.resume()
        print(json.dumps({"event": "vae_released", "phase": "dit_replay"}), flush=True)
    return rewards, images

def aggregate_gradients(results):
    denominator = sum(x["denominator"] for x in results)
    if denominator <= 0:
        raise ValueError("Empty global update window")
    keys = set(results[0]["gradients"])
    if any(set(x["gradients"]) != keys for x in results):
        raise ValueError("Workers returned different adapter keys")
    gradients = {}
    for key in sorted(keys):
        arrays = [x["gradients"][key] for x in results]
        if any(x.shape != arrays[0].shape for x in arrays):
            raise ValueError("Worker gradient shapes differ")
        import jax
        if isinstance(arrays[0],jax.Array):
            import jax.numpy as jnp
            gradient=arrays[0]
            for value in arrays[1:]:gradient=gradient+value
            gradient=gradient/jnp.float32(denominator)
            finite=bool(np.asarray(jnp.all(jnp.isfinite(gradient))))
        else:
            gradient = np.zeros_like(arrays[0], dtype=np.float32)
            for value in arrays:gradient += np.asarray(value, np.float32)
            gradient /= np.float32(denominator)
            finite=np.isfinite(gradient).all()
        if not finite:
            raise FloatingPointError(f"Nonfinite reduced gradient: {key}")
        gradients[key] = gradient
    return gradients, {
        "loss": sum(x["loss_sum"] for x in results) / denominator,
        "items": denominator,
        "masked_fraction": sum(x["masked"] for x in results) / denominator,
        "active_fraction": sum(x["active"] for x in results) / denominator,
        "kept_count": sum(x["active"] for x in results),
    }

def apply_window_update(tx, gradients, opt_state, lora, kept_count):
    """Skip the entire optimizer transition only when globally no row is kept."""
    import jax, optax
    if int(kept_count)==0:
        return lora, opt_state, jax.tree.map(lambda x:x*0, lora), False
    delta,next_state=tx.update(gradients,opt_state,lora)
    return optax.apply_updates(lora,delta),next_state,delta,True


def _save(c, output, lora, opt_state, metadata):
    from .checkpoint import save_checkpoint
    path = output / "checkpoints" / f"step_{metadata['update']:08d}"
    save_checkpoint(path, lora, opt_state, metadata)
    write_json(output / "latest.json", {"checkpoint": str(path.resolve()), "update": metadata["update"]})
    parents = path.parent.resolve()
    existing = sorted(path.parent.glob("step_[0-9]*"))
    for old in existing[:-c["training"]["keep_checkpoints"]]:
        if old.is_dir() and not old.is_symlink() and old.resolve().parent == parents:
            shutil.rmtree(old)
    return path

def _optimizer(c):
    from .optimizers import build_optimizer
    return build_optimizer(c["optimizer"])


def resolve_checkpoint(path):
    path = Path(path)
    if (path / "latest.json").exists():
        return Path(json.loads((path / "latest.json").read_text())["checkpoint"])
    return path

def identity_for(c):
    return {"config_fingerprint": fingerprint(c), "data_sha256": data_identity(c),
            "experiment": experiment_identity(c)}

def paired_replay_payloads(pool, groups, stats, windows, caches=None):
    """Keep ART windows unchanged; split each trajectory's slots over two workers."""
    count = len(pool.workers)
    if count < 2 or count % 2:
        raise ValueError("Paired replay requires an even worker count")
    caches = pool.call("export_replay") if caches is None else caches
    first = caches[0]
    slots = len(first["selected"])
    for cache in caches[1:]:
        for key in ("sigmas", "selected", "ode_velocity_coefficients"):
            if not np.array_equal(cache[key], first[key]):
                raise ValueError("Inconsistent replay sampler metadata")
    combined = {key: first[key] for key in ("sigmas", "selected", "ode_velocity_coefficients")}
    from .replay import replay_identity
    combined.update(replay_identity(caches))
    for key in ("states", "old", "endpoints"):
        combined[key] = np.concatenate([cache[key] for cache in caches])
    combined["groups"] = groups
    combined["contexts"] = [pair for cache in caches for pair in cache["contexts"]]
    lengths = {}
    offset = 0
    for cache in caches:
        for row, local_group in enumerate(cache["groups"]):
            group = int(groups[offset + row])
            value = cache["condition_lengths"][int(local_group)]
            if group in lengths and lengths[group] != value:
                raise ValueError("Inconsistent replay condition lengths")
            lengths[group] = value
        offset += len(cache["endpoints"])
    combined["condition_lengths"] = [lengths[i] for i in range(len(lengths))]
    assigned = [[[] for _ in windows] for _ in pool.workers]
    for wi, window in enumerate(windows):
        rows = list(dict.fromkeys(int(row) for row, slot in window))
        row_position = {row: index for index, row in enumerate(rows)}
        for row, slot in window:
            rank = (2 * (row_position[int(row)] % (count // 2)) + int(slot) % 2
                    if slots % 2 == 0 else (row_position[int(row)] * slots + int(slot)) % count)
            assigned[rank][wi].append((int(row), int(slot)))
        actual = sorted(item for worker in assigned for item in worker[wi])
        if actual != sorted(map(tuple, window.tolist())):
            raise ValueError("Replay distribution omitted or duplicated training points")
    from .replay import canonical_contexts,shard_replay
    combined["contexts"]=canonical_contexts(combined["contexts"],groups)
    local_caches,payloads=[],[]
    for worker in assigned:
        cache,payload=shard_replay(combined,stats,worker)
        local_caches.append({"cache":cache});payloads.append(payload)
    pool.call("install_replay",local_caches)
    print(json.dumps({"event": "paired_replay", "items_per_worker_per_window":
                      [[len(w) for w in worker] for worker in assigned]}), flush=True)
    return payloads

def merge_rollout_rounds(outputs_by_round, caches_by_round, prompts_per_round, split_groups):
    """Merge host replay buffers, preserving independent covariance prompt groups."""
    workers = len(outputs_by_round[0])
    outputs, caches, global_groups = [], [], []
    for rank in range(workers):
        pieces = [value[rank] for value in outputs_by_round]
        replay = [value[rank] for value in caches_by_round]
        local_prompts = len(replay[0]["condition_lengths"])
        out = dict(pieces[0])
        out["endpoints"] = np.concatenate([piece["endpoints"] for piece in pieces])
        for key in ("seeds", "prompts", "sample_ids", "target_tags"):
            out[key] = [item for piece in pieces for item in piece[key]]
        out["groups"] = np.concatenate([np.asarray(piece["groups"], np.int32) + index * prompts_per_round
                                        + (0 if split_groups else rank * local_prompts)
                                        for index, piece in enumerate(pieces)])
        cache = {key: replay[0][key] for key in ("sigmas", "selected", "ode_velocity_coefficients")}
        from .replay import replay_identity
        identity=replay_identity(replay)
        for piece in replay[1:]:
            for key in cache:
                if not np.array_equal(piece[key], cache[key]):
                    raise ValueError("Rollout rounds have inconsistent sampler metadata")
        cache.update(identity)
        for key in ("states", "old", "endpoints"):
            cache[key] = np.concatenate([piece[key] for piece in replay])
        cache["groups"] = np.concatenate([np.asarray(piece["groups"], np.int32) + index * local_prompts
                                          for index, piece in enumerate(replay)])
        for key in ("contexts", "condition_lengths"):
            cache[key] = [item for piece in replay for item in piece[key]]
        outputs.append(out); caches.append(cache); global_groups.append(out["groups"])
    return outputs, caches, np.concatenate(global_groups)

def train(c, resume=None, init_adapter=None):
    if resume and init_adapter:
        raise ValueError("--resume and --init-adapter are mutually exclusive")
    with output_lock(c["runtime"]["output_dir"]):
        return _train(c, resume=resume, init_adapter=init_adapter)

def _train(c, resume=None, init_adapter=None):
    c = copy.deepcopy(c)
    from .prepare import ensure_cache
    from .prefetch import enabled,DoubleBuffer
    if not enabled(c):ensure_cache(c)
    preflight(c)
    configure_process(c,c["runtime"]["devices"] if c["runtime"]["execution_backend"]=="mesh" else None)
    import jax
    import optax
    from .checkpoint import load_checkpoint
    output = Path(c["runtime"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() and not resume:
        raise FileExistsError("Output already contains metrics; use --resume or a new output directory")
    records = read_records(c)
    identity = identity_for(c)
    update, rollout, cursor, skipped = 0, 0, 0, 0
    optimizer_updates=0
    tx = _optimizer(c)
    restored = None
    initial_adapter = None
    if init_adapter:
        adapter_path = Path(init_adapter).resolve()
        from .data import sha256_file
        initial_adapter = {"path": str(adapter_path), "sha256": sha256_file(adapter_path)}
    if resume:
        restored = load_checkpoint(resolve_checkpoint(resume), expected_metadata=identity)
        metadata = restored[2]
        initial_adapter = metadata.get("initial_adapter")
        update, rollout, cursor = [int(metadata[x]) for x in ("update", "rollout", "prompt_cursor")]
        optimizer_updates=int(metadata.get("optimizer_updates",update))
        if update % c["training"]["updates_per_rollout"] or update > c["training"]["max_updates"]:
            raise ValueError("Resume needs a completed rollout collection within requested budget")
        if (output / "metrics.jsonl").exists():
            lines = [json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines() if x.strip()]
            if any(x.get("update", 0) > update for x in lines):
                raise ValueError("Output history is newer than checkpoint; resume into a new output directory")
    from .resolution_mixing import ResolutionPlan, apply_profile
    mix_plan = ResolutionPlan(c, records) if c["resolution_mixing"]["enabled"] else None
    mix_cursors = {} if restored is None else dict(restored[2].get("resolution_cursors", {}))
    from . import batch_plan
    logical_batch = batch_plan.enabled(c)
    if logical_batch:
        pools=[records] if mix_plan is None else [mix_plan.buckets[i] for i in mix_plan.active]
        if any(len(batch_plan.unique_pool(rows))<c["training"]["prompts_per_update"] for rows in pools):
            raise ValueError("Not enough distinct prompts per active resolution bucket")
        print(json.dumps({"event":"logical_batch", "prompts_per_update":c["training"]["prompts_per_update"],
                          "rollouts_per_prompt":c["training"]["rollouts_per_prompt"],
                          "training_points_per_trajectory":c["training"]["training_points_per_trajectory"],
                          "training_points_per_collection":c["training"]["prompts_per_update"]*c["training"]["rollouts_per_prompt"]*c["training"]["training_points_per_trajectory"],
                          "update_windows":c["training"]["updates_per_rollout"]}),flush=True)
    if mix_plan is not None:
        if restored is not None and "resolution_cursors" not in restored[2]:
            raise ValueError("Mixed-resolution resume requires bucket cursors")
        print(json.dumps({"event":"resolution_buckets", "counts":
                          {label:len(rows) for label,rows in zip(mix_plan.labels,mix_plan.buckets)}}), flush=True)
    started = time.monotonic()
    write_json(output / "config.json", c)
    print(json.dumps({"event": "starting", "output": str(output), "workers": len(c["runtime"]["devices"]),
                      "model": c["model"]["kind"], "reward": c["reward"]["kind"]}), flush=True)
    with ExitStack() as stack:
        prefetch=stack.enter_context(DoubleBuffer(c,records)) if enabled(c) else None
        pending=None
        if prefetch is not None and update<c["training"]["max_updates"]:
            first=(batch_plan.select(c,records,update,cursor,mix_plan,mix_cursors)[:1] if logical_batch else
                   mix_plan.select(update,0,c["sampling"]["prompts_per_rollout"],mix_cursors) if mix_plan is not None else
                   select_records(records,cursor,c["sampling"]["prompts_per_rollout"],c["training"]["seed"]))
            pending=prefetch.submit(first)
        from .tensorboard_log import ScalarLogger
        logger=stack.enter_context(ScalarLogger(c,output))
        if c["runtime"]["execution_backend"]=="mesh":
            from .mesh_runtime import MeshPool
            pool=stack.enter_context(MeshPool(c,initial_lora=restored[0] if restored is not None else None))
        else:pool=stack.enter_context(WorkerPool(c))
        pool.scalar_logger=logger
        lora = pool.info[0]["lora"]
        pool.info[0]["lora"]=None
        if init_adapter:
            from .lora import import_safetensors
            lora = import_safetensors(init_adapter, lora, alpha=c["lora"]["alpha"])
        if restored is None:opt_state=tx.init(lora)
        else:lora,opt_state,_=restored
        if c["runtime"]["execution_backend"]=="mesh":
            lora=pool.place_state(lora);opt_state=pool.place_state(opt_state)
            jax.block_until_ready((lora,opt_state))
        pool.call("set_lora", [{"lora": lora} for _ in pool.workers])
        # The checkpoint's CPU arrays must not stay pinned by this tuple all run.
        restored=None
        if hasattr(pool,"memory_event"):pool.memory_event("training_state_loaded")
        write_json(output / "run_info.json", {"worker_devices": [x["device"] for x in pool.info],
                   "model_config": pool.info[0]["model_config"], "adapter_parameters": sum(x.size for x in lora.values()),
                   "identity": identity, "initial_adapter": initial_adapter, "host_reduction": c["runtime"]["execution_backend"]!="mesh",
                   "note": "Toy rewards only validate mechanics and are not evidence of image quality gains."})
        if c["training"]["evaluate_every"] and update == 0:
            evaluate(c, pool, records, update, output, prefetch=prefetch)
        while update < c["training"]["max_updates"]:
            resolution_label = None
            if mix_plan is not None:
                profile_index = mix_plan.index(update)
                resolution_label = apply_profile(c,pool,mix_plan.profiles[profile_index])
                print(json.dumps({"event":"resolution_profile", "update":update, "resolution":resolution_label,
                                  "group_size":c["sampling"]["group_size"], "rollout_batch_size":c["sampling"]["rollout_batch_size"],
                                  "microbatch_size":c["training"]["microbatch_size"], "rollouts_per_update":c["training"]["rollouts_per_update"]}),flush=True)
            free=shutil.disk_usage(output).free
            required=c["runtime"].get("min_free_disk_gib",0)*2**30
            if free<required:
                if update and update%c["training"]["updates_per_rollout"]==0:
                    _save(c,output,lora,opt_state,{**identity,"update":update,"optimizer_updates":optimizer_updates,"rollout":rollout,"prompt_cursor":cursor,
                        "seed":c["training"]["seed"],"initial_adapter":initial_adapter,"resolution_cursors":dict(mix_cursors),"model_config":pool.info[0]["model_config"],
                        "lora_config":c["lora"],"runtime_devices":c["runtime"]["devices"]})
                write_json(output/"stopped_low_disk.json",dict(update=update,rollout=rollout,free_bytes=free,required_bytes=required))
                raise RuntimeError("Training stopped at a rollout boundary: free disk below configured minimum; saved images were retained")
            beginning = time.monotonic()
            rounds = c["training"]["rollouts_per_update"]
            count = c["sampling"]["prompts_per_rollout"]
            round_outputs, round_caches = [], []
            update_prompts = batch_plan.select(c,records,update,cursor,mix_plan,mix_cursors) if logical_batch else None
            for round_index in range(rounds):
                batch = ([update_prompts[round_index]] if logical_batch else
                         mix_plan.select(update,round_index*count,count,mix_cursors) if mix_plan is not None
                         else select_records(records, cursor + round_index * count, count, c["training"]["seed"]))
                if prefetch is not None:
                    handoff = prefetch.bind(pool, batch, pending)
                    following = (batch_plan.following(c,records,update,cursor,update_prompts,round_index,mix_plan,mix_cursors) if logical_batch else
                                 mix_plan.following(update,round_index,count,rounds,mix_cursors) if mix_plan is not None
                                 else select_records(records, cursor + (round_index + 1) * count, count, c["training"]["seed"]))
                    pending = prefetch.submit(following)
                    append_json(output/"prefetch.jsonl", dict(handoff, rollout=rollout+round_index))
                compute_started = time.monotonic()
                current = pool.call("rollout", make_payloads(c, batch, rollout + round_index))
                round_outputs.append(current)
                if rounds > 1:
                    round_caches.append(pool.call("export_replay"))
                timing = dict(event="dit_rollout_interval", rollout=rollout+round_index,
                              started=compute_started, finished=time.monotonic())
                append_json(output/"prefetch.jsonl", timing)
                print(json.dumps(timing), flush=True)
                if rounds > 1:
                    print(json.dumps(dict(event="rollout_collection", completed_rounds=round_index+1,
                                          target_rounds=rounds, update=update,
                                          trajectories=sum(len(x["endpoints"]) for x in current))), flush=True)
            replay_caches = None
            collection_groups = None
            if rounds > 1:
                outputs, replay_caches, collection_groups = merge_rollout_rounds(
                    round_outputs, round_caches, count, c["sampling"]["split_group_across_workers"])
                del round_caches, round_outputs
            else:
                outputs = round_outputs[0]
            image_dir=output/"rollout_images"/f"rollout_{rollout:06d}" if c["training"].get("save_rollout_images",False) else None
            rewards, _ = reward_round(c, pool, outputs,image_dir=image_dir)
            rollout_metrics=dict(reward_mean=float(rewards.mean()),reward_std=float(rewards.std()),reward_min=float(rewards.min()),reward_max=float(rewards.max()))
            if c["reward"]["kind"]=="pixai":
                details=pool.last_reward_details
                rollout_metrics.update(coverage=float(np.mean([x["coverage"] for x in details])),iou=float(np.mean([x["iou"] for x in details])))
                write_json(output/"rewards"/f"rollout_{rollout:06d}.json",dict(rollout=rollout,update=update,details=details))
            if c["reward"]["kind"]=="silva":
                details=pool.last_reward_details
                rollout_metrics.update(aesthetic=float(np.mean([x["reward"] for x in details])),raw_score=float(np.mean([x["raw_score"] for x in details])))
                write_json(output/"rewards"/f"rollout_{rollout:06d}.json",dict(rollout=rollout,update=update,details=details))
            if resolution_label is not None:
                rollout_metrics.update(height=c["sampling"]["height"],width=c["sampling"]["width"])
                logger.write("rollout/"+resolution_label,rollout,rollout_metrics)
            logger.write("rollout",rollout,rollout_metrics)
            append_json(output/"rollouts.jsonl",dict(rollout=rollout,update=update,**rollout_metrics))
            sizes = [len(x["endpoints"]) for x in outputs]
            offsets = np.cumsum([0] + sizes)
            from .art import covariance, prompt_windows
            endpoints = np.concatenate([x["endpoints"] for x in outputs])
            k = c["sampling"]["group_size"]
            groups = (np.arange(len(batch), dtype=np.int32).repeat(k)
                      if collection_groups is None else collection_groups)
            seed = c["training"]["seed"] + rollout * 7919
            stats = covariance(endpoints, rewards, groups, mode=c["art"]["feedback"],
                               group_size=k, seed=seed, eps=c["art"]["eps"])
            stats = {key: np.asarray(value) for key, value in stats.items()}
            if c["art"]["feedback"]=="scalar":
                from .reward_diagnostics import record_training_rewards
                record_training_rewards(c,output,logger,rollout,update,rewards,groups,outputs)
            windows = prompt_windows(groups, len(c["sampling"]["train_timesteps"]), seed=seed)
            if c["training"]["updates_per_rollout"] == 1:
                windows = [np.concatenate(windows)]
            payloads = []
            for i in range(len(sizes)):
                lo, hi = offsets[i], offsets[i+1]
                group_ids=np.unique(groups[lo:hi])
                local_stats = {key: value[group_ids] if key.startswith("group_")
                               else value[lo:hi] if key.startswith("row_") else value
                               for key, value in stats.items()}
                local_windows = [np.asarray([(int(row)-lo, int(slot)) for row, slot in w
                                              if lo <= row < hi], dtype=np.int64) for w in windows]
                payloads.append({"stats": local_stats, "windows": local_windows})
            if c["sampling"].get("pair_cached_points", False):
                payloads = paired_replay_payloads(pool, groups, stats, windows, caches=replay_caches)
            elif replay_caches is not None:
                pool.call("install_replay", [{"cache": cache} for cache in replay_caches])
            prepared = pool.call("prepare_update", payloads)
            del replay_caches
            cursor += count * rounds
            if resolution_label is not None:
                mix_cursors[resolution_label] = mix_cursors.get(resolution_label,0) + count*rounds
            rollout += rounds
            if hasattr(pool,"anchor_metrics") and pool.anchor_metrics:
                logger.write("anchor",rollout-rounds,pool.anchor_metrics)
                write_json(output/"anchor_diagnostics"/f"rollout_{rollout-rounds:06d}.json",dict(rollout=rollout-rounds,update=update,**pool.anchor_metrics,rows=pool.anchor_diagnostics))
            if sum(x["active_groups"] for x in prepared) == 0:
                skipped += 1
                update += c["training"]["updates_per_rollout"]
                event = {"event": "no_signal", "rollout": rollout, "update": update, "optimizer_updates":optimizer_updates, "optimizer_step_applied":False,
                         "reward_mean": float(rewards.mean()), "elapsed_seconds": time.monotonic() - beginning}
                if logical_batch:
                    event.update(prompts_per_update=c["training"]["prompts_per_update"],
                                 rollouts_per_prompt=c["training"]["rollouts_per_prompt"],
                                 training_points_per_trajectory=c["training"]["training_points_per_trajectory"],
                                 trajectories=len(rewards),collected_images=rollout*k)
                append_json(output / "metrics.jsonl", event)
                print(json.dumps(event), flush=True)
                if prefetch is not None and pending is not None:
                    prefetch.acquire(pending);prefetch.release(pending);pending=None
                pool.call("clear_replay")
                if skipped >= c["training"]["max_no_signal_rollouts"]:
                    raise RuntimeError("Consecutive no-signal rollout limit reached; choose a discriminating reward")
                if update==c["training"]["max_updates"] or (c["training"]["checkpoint_every"] and update%c["training"]["checkpoint_every"]==0):
                    _save(c,output,lora,opt_state,{**identity,"update":update,"optimizer_updates":optimizer_updates,
                        "rollout":rollout,"prompt_cursor":cursor,"seed":c["training"]["seed"],"initial_adapter":initial_adapter,
                        "resolution_cursors":dict(mix_cursors),"model_config":pool.info[0]["model_config"],
                        "lora_config":c["lora"],"runtime_devices":c["runtime"]["devices"]})
                continue
            skipped = 0
            for window in range(c["training"]["updates_per_rollout"]):
                result = pool.call("gradient", [{"window": window} for _ in pool.workers])
                gradients, metrics = aggregate_gradients(result)
                metrics["grad_norm"] = float(optax.global_norm(gradients))
                metrics["worker_peak_bytes_in_use"] = [x.get("memory_stats", {}).get("peak_bytes_in_use", 0) for x in result]
                if not np.isfinite(metrics["grad_norm"]):
                    raise FloatingPointError("Nonfinite global gradient norm")
                next_lora,opt_state,delta,applied=apply_window_update(tx,gradients,opt_state,lora,metrics["kept_count"])
                optimizer_updates+=int(applied)
                metrics.update(optimizer_step_applied=applied,optimizer_updates=optimizer_updates)
                metrics["adapter_delta_norm"] = float(optax.global_norm(jax.tree.map(lambda n, p: n-p, next_lora, lora)))
                if c["runtime"]["execution_backend"]=="mesh":
                    lora=next_lora
                    finite=all(bool(np.asarray(jax.numpy.all(jax.numpy.isfinite(x)))) for x in lora.values())
                else:
                    lora = {k: np.asarray(v, dtype=np.float32) for k, v in next_lora.items()}
                    finite=all(np.isfinite(x).all() for x in lora.values())
                if not finite:raise FloatingPointError("Optimizer produced nonfinite adapter weights")
                if applied:pool.call("set_lora", [{"lora": lora} for _ in pool.workers])
                update += 1
                event = {"event": "update", "update": update, "rollout": rollout, "window": window,
                         "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
                         "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
                         "elapsed_seconds": time.monotonic()-beginning, **metrics}
                if logical_batch:
                    event.update(prompts_per_update=c["training"]["prompts_per_update"],
                                 rollouts_per_prompt=c["training"]["rollouts_per_prompt"],
                                 training_points_per_trajectory=c["training"]["training_points_per_trajectory"],
                                 trajectories=len(rewards),collected_images=rollout*k)
                append_json(output / "metrics.jsonl", event)
                print(json.dumps(event), flush=True)
                if resolution_label is not None:
                    logger.write("train/"+resolution_label,update,metrics)
                logger.write("train",update,metrics)
            metadata = {**identity, "update": update, "optimizer_updates":optimizer_updates,"rollout": rollout, "prompt_cursor": cursor,
                        "seed": c["training"]["seed"], "model_config": pool.info[0]["model_config"],
                        "lora_config": c["lora"], "initial_adapter": initial_adapter,
                        "runtime_devices": c["runtime"]["devices"],
                        "elapsed_seconds": time.monotonic()-started}
            if mix_plan is not None:
                metadata["resolution_cursors"] = dict(mix_cursors)
            # Replay and update temporaries are retired BEFORE checkpoint CPU staging.
            pool.call("clear_replay")
            del outputs,endpoints,stats,payloads,windows,result,gradients,delta,next_lora
            import gc
            gc.collect()
            interval = c["training"]["checkpoint_every"]
            if update == c["training"]["max_updates"] or (interval and update % interval == 0):
                path = _save(c, output, lora, opt_state, metadata)
                print(json.dumps({"event":"checkpoint", "path":str(path)}), flush=True)
                if hasattr(pool,"memory_event"):pool.memory_event("after_checkpoint")
            interval = c["training"]["evaluate_every"]
            if interval and update % interval == 0:
                evaluate(c, pool, records, update, output, prefetch=prefetch)
        from .lora import export_safetensors
        export_safetensors(output / "adapter.safetensors", lora, alpha=c["lora"]["alpha"])
    write_json(output / "complete.json", {"update": update, "optimizer_updates":optimizer_updates,"rollout": rollout,
                "elapsed_seconds": time.monotonic()-started, "adapter": str(output/"adapter.safetensors")})
    return output

def evaluate(c, pool, records, update, output, force_decode=False, prefetch=None):
    if not c["resolution_mixing"]["enabled"]:
        return _evaluate_single(c,pool,records,update,output,force_decode,prefetch)
    from .resolution_mixing import ResolutionPlan,apply_profile
    plan=ResolutionPlan(c,records)
    original=copy.deepcopy(c)
    result={}
    try:
        for index in plan.active:
            label=apply_profile(c,pool,plan.profiles[index])
            result[label]=_evaluate_single(c,pool,plan.buckets[index],update,output,
                                           force_decode,prefetch,resolution_label=label)
    finally:
        c.clear();c.update(original)
        apply_profile(c,pool,{**c["sampling"],**{k:c["training"][k] for k in ("microbatch_size","rollouts_per_update")}})
    return result


def _evaluate_single(c, pool, records, update, output, force_decode=False, prefetch=None, resolution_label=None):
    count = c["sampling"]["prompts_per_rollout"]
    batch = [records[i % len(records)] for i in range(count)]
    pool.call("clear_replay")
    if prefetch is not None:prefetch.bind(pool,batch)
    outputs = pool.call("rollout", make_payloads(c, batch, 0, evaluation=True))
    directory = Path(output) / "evaluation" / f"step_{update:08d}"
    if resolution_label is not None:directory = directory / resolution_label
    rewards, images = reward_round(c, pool, outputs, directory, force_decode=force_decode)
    if images is not None:
        from .evaluate import write_gallery
        prompts = sum([x["prompts"] for x in outputs], [])
        write_gallery([{"image": str(directory/f"{i:04d}.png"), "prompt": prompt, "score": float(score)}
                       for i, (prompt, score) in enumerate(zip(prompts, rewards))], directory,
                      title=f"MOSS DiffusionART JAX - update {update}")
    result = {"event":"evaluation", "update":update, "reward_mean":float(rewards.mean()),
              "reward_std":float(rewards.std()), "rewards":rewards.tolist(), "fixed_seeds":True}
    if c["reward"]["kind"]=="pixai":
        details=pool.last_reward_details
        result.update(coverage=float(np.mean([x["coverage"] for x in details])),iou=float(np.mean([x["iou"] for x in details])))
        write_json(directory/"tag_scores.json",details)
    if c["reward"]["kind"]=="silva":
        details=pool.last_reward_details
        result.update(aesthetic=float(np.mean([x["reward"] for x in details])),raw_score=float(np.mean([x["raw_score"] for x in details])))
        write_json(directory/"aesthetic_scores.json",details)
    logger=getattr(pool,"scalar_logger",None)
    if resolution_label is not None:
        result.update(height=c["sampling"]["height"],width=c["sampling"]["width"])
    if logger is not None:logger.write("eval"+("/"+resolution_label if resolution_label is not None else ""),update,result)
    write_json(directory/"metrics.json", result)
    print(json.dumps(result), flush=True)
    return result

def evaluate_checkpoint(c, checkpoint, force_decode=False):
    with output_lock(c["runtime"]["output_dir"]):
        return _evaluate_checkpoint(c, checkpoint, force_decode)

def _evaluate_checkpoint(c, checkpoint, force_decode=False):
    preflight(c)
    configure_process(c)
    from .checkpoint import load_checkpoint
    lora, _, metadata = load_checkpoint(resolve_checkpoint(checkpoint),
        expected_metadata=identity_for(c))
    from .prefetch import enabled,DoubleBuffer
    records=read_records(c)
    with ExitStack() as stack:
        prefetch=stack.enter_context(DoubleBuffer(c,records)) if enabled(c) else None
        pool=stack.enter_context(WorkerPool(c))
        pool.call("set_lora", [{"lora":lora} for _ in pool.workers])
        return evaluate(c, pool, records, metadata["update"],
                        Path(c["runtime"]["output_dir"]), force_decode, prefetch=prefetch)

def evaluate_adapter(c, adapter, force_decode=False):
    """Evaluate an exported adapter with a fresh model, without optimizer state."""
    with output_lock(c["runtime"]["output_dir"]):
        preflight(c)
        configure_process(c)
        from .data import sha256_file
        from .lora import import_safetensors
        source = {"path": str(Path(adapter).resolve()), "sha256": sha256_file(adapter)}
        identity = identity_for(c)
        from .prefetch import enabled,DoubleBuffer
        records=read_records(c)
        with ExitStack() as stack:
            prefetch=stack.enter_context(DoubleBuffer(c,records)) if enabled(c) else None
            pool=stack.enter_context(WorkerPool(c))
            lora = import_safetensors(adapter, pool.info[0]["lora"], alpha=c["lora"]["alpha"])
            pool.call("set_lora", [{"lora": lora} for _ in pool.workers])
            result = evaluate(c, pool, records, 0, Path(c["runtime"]["output_dir"]), force_decode, prefetch=prefetch)
        write_json(Path(c["runtime"]["output_dir"]) / "adapter_evaluation.json",
                   {"source_adapter": source, "identity": identity, "result": result})
        return result
