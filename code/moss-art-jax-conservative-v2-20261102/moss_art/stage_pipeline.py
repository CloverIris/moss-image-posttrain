"""File-based infer -> eval -> train handoff; no resident controller model."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time


def _write_step_exports(c, bundle, metadata, request, source_records):
    """Keep original conditions and exported-file inventories on durable storage."""
    from .trainer import write_json
    step = Path(c["runtime"]["output_dir"]).resolve()/"steps"/("step_%08d" % metadata["update"])
    step.mkdir(parents=True, exist_ok=True)
    # This is the dataset text/metadata, before adding any Qwen chat template.
    # Do not strip, unescape, reformat or split its MOSS-level contents here.
    write_json(step/"prompts.json", [dict(record, prompt_index=i) for i, record in enumerate(source_records)])
    write_json(step/"samples.json", request["items"])
    write_json(step/"config.json", c)
    write_json(step/"eval_request.json", request)
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    images = [row["image"] for row in request["items"] if Path(row["image"]).is_file()]
    write_json(step/"exported_files.json", dict(version=1,
        images=images, replay_bundle=str(bundle),
        replay_files=[dict(path=str(p), relative_path=str(p.relative_to(bundle)), bytes=p.stat().st_size) for p in files],
        replay_retention="temporary; removed only after a successful training commit"))
    # Preserve the original tensor dtype/shape/checksum descriptors even after
    # RAM replay arrays are retired; never duplicate the arrays themselves.
    temporary = step/"replay_manifest.json.tmp"
    shutil.copyfile(bundle/"manifest.json", temporary)
    temporary.replace(step/"replay_manifest.json")
    write_json(step/"manifest.json", dict(version=1, step=metadata["update"],
        planned_update_steps=list(range(metadata["update"]+1, metadata["update"]+c["training"]["updates_per_rollout"]+1)),
        rollout_start=request["rollout"], rollout_next=metadata["rollout"],
        prompt_count=len(source_records), trajectory_count=len(request["items"]), exported_image_count=len(images),
        source_prompts=dict(path=str(Path(c["data"]["prompts"]).resolve()), sha256=metadata["data_sha256"]),
        source_checkpoint=c["run"]["resume"] or None,
        config_fingerprint=metadata["config_fingerprint"],
        source_commit=(bundle/"COMMITTED").read_text().strip(),
        raw_prompt_format="Original selected dataset strings and metadata; no Qwen chat shell",
        files=["prompts.json", "samples.json", "config.json", "eval_request.json", "exported_files.json", "replay_manifest.json"]))
    return step


def persist_inference(c, shards, outputs, groups, lora, optimizer, metadata,
                      rollout, image_dir, resolution_label, collection_seconds, *, source_records):
    from .stage_cache import save_bundle
    from .staged import collection_path
    from .trainer import write_json
    output = Path(c["runtime"]["output_dir"]).resolve()
    bundle = collection_path(c, metadata["update"])
    metadata = dict(metadata, stage_cache_id=str(bundle))
    # Keep row order identical to the rollout outputs and replay arrays.
    rows = []
    repeats = {}
    for result in outputs:
        for index, prompt in enumerate(result["prompts"]):
            group = int(groups[len(rows)])
            record = source_records[group]
            if record["prompt"] != prompt or record["sample_id"] != result["sample_ids"][index]:
                raise ValueError("Exported trajectory does not match its original prompt record")
            row = dict(index=len(rows), prompt=prompt,
                prompt_index=group, rollout_id=repeats.get(group, 0),
                negative_prompt=record.get("negative_prompt", ""),
                image=str((Path(image_dir)/("%04d.png" % len(rows))).resolve()))
            repeats[group] = row["rollout_id"]+1
            for source, target in (("sample_ids", "sample_id"), ("seeds", "seed"),
                    ("target_tags", "target_tags"), ("sampling_methods", "sampling_method"),
                    ("dit_full_steps", "dit_full_steps"), ("dit_forecast_steps", "dit_forecast_steps")):
                if source in result:
                    row[target] = result[source][index]
            rows.append(row)
    payload = dict(phase="inferred", config=c, shards=shards, groups=groups,
        outputs=[{k: v for k, v in result.items() if k != "endpoints"} for result in outputs],
        rollout_start=rollout, resolution_label=resolution_label, collection_seconds=collection_seconds)
    request = dict(version=1, config=c, config_fingerprint=metadata["config_fingerprint"],
                   items=rows, rollout=rollout, update=metadata["update"])
    started = time.monotonic()
    save_bundle(bundle, payload, lora, optimizer, metadata,
        attachments={"eval_request.json": request},
        reserve_bytes=c["staging"]["min_free_cache_gib"]*2**30)
    step_dir = _write_step_exports(c, bundle, metadata, request, source_records)
    summary = dict(bundle=str(bundle), update=metadata["update"], rollout=metadata["rollout"],
        step_dir=str(step_dir),
        cache_write_seconds=time.monotonic()-started,
        cache_bytes=sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()))
    write_json(output/"staging/ready.json", summary)
    print(json.dumps(dict(event="inference_stage_committed", **summary)), flush=True)
    return bundle


def evaluation_result(bundle, count=None):
    import numpy as np
    bundle = Path(bundle)
    result = json.loads((bundle/"evaluated.json").read_text())
    if result.get("source_commit") != (bundle/"COMMITTED").read_text().strip():
        raise ValueError("Evaluation belongs to a different inference bundle")
    scores = np.asarray(result["scores"], np.float32)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (count is not None and len(scores) != count):
        raise ValueError("Evaluation must provide one finite score per trajectory")
    return result, scores


def _score_request(c, request, bundle):
    """Only this process loads reward models; it never constructs a DiT."""
    import numpy as np
    kind, reward = c["reward"]["kind"], c["reward"]
    rows = request["items"]
    if kind == "toy_latent":
        from .stage_cache import load_bundle
        payload, _ = load_bundle(bundle)
        endpoints = np.concatenate([x["endpoints"] for x in payload["shards"]])
        scores = -np.mean(np.square(endpoints.astype(np.float32)-np.float32(.25)),
                          axis=tuple(range(1, endpoints.ndim)))
        return scores, None
    if kind == "command":
        # Reuse saved PNGs: external vLLM/multi-model evaluation needs no replay
        # reads, image round-trip or JAX/CUDA initialization in this process.
        import subprocess
        from .rewards import _command_argv
        from .trainer import write_json
        from .reward_audit import collection_context, new_command_attempt, log_tail
        worker = {k: reward[k] for k in ("model_path", "device", "batch_size", "metric", "threads")}
        context = collection_context(c, request, bundle)
        if c["art"].get("reward_contract")=="conservative_v2":
            from .art_options import declared_contract
            context["reward_contract"]=declared_contract(c["art"])
        attempt = new_command_attempt(context)
        input_path, output_path = attempt/"request.json", attempt/"response.json"
        stdout_path, stderr_path = attempt/"stdout.log", attempt/"stderr.log"
        write_json(input_path, dict(version=1, items=rows, config=worker, context=context))
        # These legacy scratch paths remain available to older diagnostics.
        shutil.copyfile(input_path, bundle/"reward_request.json")
        argv = _command_argv(reward["command"], input_path, output_path)
        started = time.monotonic()
        write_json(attempt/"command.json", dict(argv=argv, context=context, status="running", started=time.time()))
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.run(argv, text=True, stdout=stdout, stderr=stderr, timeout=reward["timeout"])
        except (subprocess.TimeoutExpired, OSError) as exc:
            write_json(attempt/"status.json", dict(status="failed", error=str(exc), seconds=time.monotonic()-started))
            raise RuntimeError("Reward command failed: %s; logs: %s\n%s" % (exc, attempt, log_tail(stderr_path))) from exc
        write_json(attempt/"status.json", dict(status="completed" if process.returncode == 0 else "failed",
                   returncode=process.returncode, seconds=time.monotonic()-started))
        if process.returncode:
            raise RuntimeError("Reward command failed (%s); logs: %s\n%s" % (process.returncode, attempt, log_tail(stderr_path)))
        response = json.loads(output_path.read_text())
        shutil.copyfile(output_path, bundle/"reward_response.json")
        return np.asarray(response["scores"], np.float32), response.get("details")
    from PIL import Image
    pixels = []
    for row in rows:
        with Image.open(row["image"]) as image:
            pixels.append(np.asarray(image.convert("RGB")))
    images = np.stack(pixels)
    if kind in ("silva", "pixai"):
        from .runtime import configure_process, preflight
        preflight(c)
        configure_process(c, c["runtime"]["devices"])
        if kind == "silva":
            from .silva_jax import SilvaScorerJax
            model = SilvaScorerJax.from_pretrained(reward["model_path"], reward["backbone_path"],
                dtype=reward["dtype"], attention_backend=reward["attention_backend"],
                data_parallel_size=len(c["runtime"]["devices"]))
            try:
                details = model.score(images, batch_size=reward["batch_size"])
            finally:
                model.close()
        else:
            from .pixai_tagger import PixAITagger
            from .tag_reward import hard_scores
            model = PixAITagger.from_pretrained(reward["model_path"], metadata_dir=reward["metadata_dir"],
                dtype=reward["dtype"], attention_backend=reward["attention_backend"],
                data_parallel_size=len(c["runtime"]["devices"]))
            try:
                probabilities = model.probabilities(images, batch_size=reward["batch_size"])
                details = hard_scores(probabilities, [r.get("target_tags") for r in rows],
                                      reward["metadata_dir"], reward["metric"])
            finally:
                model.close()
        return np.asarray([x["reward"] for x in details], np.float32), details
    from .rewards import score_images
    return score_images(images, [r["prompt"] for r in rows], reward), None


def evaluate_cached(c):
    from .config import fingerprint
    from .locking import output_lock
    from .stage_cache import load_attachment
    from .trainer import write_json
    import numpy as np
    bundle = Path(c["staging"]["bundle"]).resolve()
    with output_lock(c["runtime"]["output_dir"]):
        request = load_attachment(bundle, "eval_request.json")
        if request["config_fingerprint"] != fingerprint(c):
            raise ValueError("Evaluator configuration differs from the collected batch")
        if Path(request["config"]["runtime"]["output_dir"]).resolve() != Path(c["runtime"]["output_dir"]).resolve():
            raise ValueError("Evaluation bundle belongs to another run")
        if (bundle/"evaluated.json").is_file():
            evaluation_result(bundle, len(request["items"]))
            return bundle/"evaluated.json"
        started = time.monotonic()
        scores, details = _score_request(request["config"], request, bundle)
        scores = np.asarray(scores, np.float32)
        if scores.shape != (len(request["items"]),) or not np.isfinite(scores).all():
            raise ValueError("Reward must return one finite scalar per saved image")
        from .reward_audit import collection_context, persist_reward_evaluation
        result = dict(version=2, source_commit=(bundle/"COMMITTED").read_text().strip(),
            update=request["update"], rollout=request["rollout"],
            context=collection_context(request["config"], request, bundle),
            scores=scores.tolist(), details=details, seconds=time.monotonic()-started)
        # Keep scores with durable images; only the handoff copy is temporary.
        output = Path(c["runtime"]["output_dir"])
        persist_reward_evaluation(request["config"], request, bundle, result)
        write_json(output/"rewards"/("rollout_%06d.json" % request["rollout"]), result)
        write_json(bundle/"evaluated.json", result)
        print(json.dumps(dict(event="evaluation_stage_committed", images=len(scores),
            reward_mean=float(scores.mean()), seconds=result["seconds"])), flush=True)
        return bundle/"evaluated.json"


def prepare_training(c, bundle, payload, metadata):
    """Rebuild cheap reward-derived arrays after eval, never dump row directions."""
    import numpy as np
    from .art import covariance, prompt_windows
    from .trainer import paired_replay_payloads, append_json, write_json
    from .tensorboard_log import ScalarLogger
    from .rollout_mixing import enabled, collection_stats
    active = payload["config"]
    shards, groups = payload["shards"], payload["groups"]
    result, rewards = evaluation_result(bundle, sum(len(s["endpoints"]) for s in shards))
    endpoints = np.concatenate([s["endpoints"] for s in shards])
    seed = active["training"]["seed"]+payload["rollout_start"]*7919
    from .art_options import covariance_options,validate_details,validate_stats
    options=covariance_options(active["art"])
    validate_details(active["art"],result.get("details"),rewards)
    stats = covariance(endpoints, rewards, groups, **options,
        group_size=active["sampling"]["group_size"], seed=seed, eps=active["art"]["eps"])
    stats = {k: np.asarray(v) for k, v in stats.items()}
    validate_stats(active["art"],stats)
    windows = prompt_windows(groups, len(active["sampling"]["train_timesteps"]), seed=seed)
    if active["training"]["updates_per_rollout"] == 1:
        windows = [np.concatenate(windows)]
    if active["sampling"]["pair_cached_points"]:
        shards, prepared = paired_replay_payloads(len(shards), groups, stats, windows, caches=shards, install=False)
    else:
        prepared, offset = [], 0
        for shard in shards:
            end = offset+len(shard["endpoints"])
            ids = np.unique(groups[offset:end])
            local = {k: v[ids] if k.startswith("group_") else v[offset:end] if k.startswith("row_") else v
                     for k, v in stats.items()}
            local_windows = [np.asarray([(int(r)-offset, int(s)) for r, s in w if offset <= r < end],
                                       np.int64).reshape(-1, 2) for w in windows]
            prepared.append(dict(stats=local, windows=local_windows))
            offset = end
    metrics = dict(reward_mean=float(rewards.mean()), reward_std=float(rewards.std()),
                   reward_min=float(rewards.min()), reward_max=float(rewards.max()))
    kind, details = active["reward"]["kind"], result["details"]
    if kind == "silva":
        metrics.update(aesthetic=float(rewards.mean()), raw_score=float(np.mean([x["raw_score"] for x in details])))
    elif kind == "pixai":
        metrics.update(**{k: float(np.mean([x[k] for x in details])) for k in ("coverage", "iou")})
    label = payload["resolution_label"]
    if label is not None:
        metrics.update(height=active["sampling"]["height"], width=active["sampling"]["width"])
    output, rollout = Path(c["runtime"]["output_dir"]), payload["rollout_start"]
    with ScalarLogger(c, output) as logger:
        from .reward_audit import persist_reward_evaluation, log_reward_components
        # Also materialize audits for old committed evaluated bundles resumed
        # after this feature was added. This never changes their saved scores.
        request_path = Path(bundle)/"eval_request.json"
        if request_path.is_file():
            request = json.loads(request_path.read_text())
            analysis = persist_reward_evaluation(active, request, bundle, result)
            log_reward_components(logger, rollout, analysis)
        if enabled(active):
            methods = collection_stats(payload["outputs"], rewards)
            for mode, values in methods.items():
                logger.write("rollout/"+mode, rollout, values)
            metrics.update({mode+"_"+key: methods.get(mode, {}).get(key, 0)
                for mode in ("full_dit", "spectrum_default") for key in ("prompts", "images")})
            metrics["full_dit_prompt_fraction"] = methods.get("full_dit", {}).get("prompts", 0)/max(1, sum(x["prompts"] for x in methods.values()))
            write_json(output/"sampling"/("rollout_%06d.json" % rollout), dict(rollout=rollout, methods=methods))
        if active["art"]["feedback"] == "scalar":
            from .reward_diagnostics import record_training_rewards
            record_training_rewards(active, output, logger, rollout, metadata["update"], rewards, groups, payload["outputs"])
        if active["art"].get("reward_contract")=="conservative_v2":
            logger.write("art_guidance",rollout,dict(reward_mad_mean=float(np.mean(stats["group_reward_abs_means"])),guidance_magnitude_mean=float(np.mean(stats["group_guidance_magnitudes"])),active_group_fraction=float(np.mean(stats["group_active"])),guidance_gain=options["guidance_gain"],reference_mse_beta=active["art"]["reference_mse_beta"]))
        logger.write("rollout", rollout, metrics)
        if label is not None:
            logger.write("rollout/"+label, rollout, metrics)
    history = output/"rollouts.jsonl"
    seen = {json.loads(x)["rollout"] for x in history.read_text().splitlines() if x.strip()} if history.exists() else set()
    if rollout not in seen:
        append_json(history, dict(rollout=rollout, update=metadata["update"], **metrics))
    return dict(payload, shards=shards, prepared=prepared, rewards=rewards, rollout_metrics=metrics, covariance_options=options)


def stage_main(phase, argv=None):
    from .config import load_config, validate
    from .staged import _ready, run, retire_collection
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    c = load_config(args.config)
    if phase in ("eval", "train") and not c["staging"]["bundle"]:
        bundle = _ready(c["runtime"]["output_dir"])
        if bundle is None:
            raise RuntimeError("No committed inference bundle; run infer first")
        c["staging"]["bundle"] = str(bundle)
    c["staging"].update(enabled=True, phase=phase)
    latest = Path(c["runtime"]["output_dir"])/"latest.json"
    if latest.is_file():
        c["run"].update(resume=json.loads(latest.read_text())["checkpoint"], init_adapter="")
    validate(c)
    result = run(c)
    if phase == "train":
        retire_collection(c, c["staging"]["bundle"])
    return result


def main(argv=None):
    """Small shell helper: select a phase, write its JSON, or finalize a run."""
    from .config import load_config
    from .staged import _ready
    from .trainer import write_json, append_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("prepare", "next", "record"), default="next")
    parser.add_argument("--phase", default="")
    parser.add_argument("--seconds", type=float, default=0.)
    args = parser.parse_args(argv)
    c = load_config(args.config)
    output = Path(c["runtime"]["output_dir"]).resolve()
    controller = output/"staging/controller"
    controller.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        print(controller)
        return
    if args.action == "record":
        append_json(output/"staging/process_times.jsonl", dict(phase=args.phase, seconds=args.seconds, returncode=0))
        return
    latest = output/"latest.json"
    current = json.loads(latest.read_text()) if latest.is_file() else None
    if current and current["update"] >= c["training"]["max_updates"]:
        from .runtime import configure_process
        configure_process(c)
        from .checkpoint import load_checkpoint
        from .lora import export_safetensors
        from .trainer import identity_for
        params, _, metadata = load_checkpoint(current["checkpoint"], expected_metadata=identity_for(c))
        export_safetensors(output/"adapter.safetensors", params, alpha=c["lora"]["alpha"])
        write_json(output/"complete.json", dict(update=current["update"], optimizer_updates=metadata["optimizer_updates"], staged=True))
        print("done")
        return
    bundle = _ready(output)
    phase = "infer" if bundle is None else "eval" if (bundle/"eval_request.json").is_file() and not (bundle/"evaluated.json").is_file() else "train"
    child = copy.deepcopy(c)
    child["staging"].update(enabled=True, phase=phase, bundle=str(bundle) if bundle else "")
    if current:
        child["run"].update(resume=current["checkpoint"], init_adapter="")
    write_json(controller/(phase+".json"), child)
    print(phase)


if __name__ == "__main__":
    main()
