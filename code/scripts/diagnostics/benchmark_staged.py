"""Benchmark real MOSS stages with synthetic cached conditions, no Qwen/VAE.

Only accepts --config. This measures startup and shape costs, not image quality.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    request = json.loads(Path(parser.parse_args().config).read_text())
    from moss_art.config import load_config, validate
    from moss_art.trainer import write_json
    if request["action"] == "clone":
        # No CUDA context is needed to duplicate an initial learner/cache pair.
        os.environ["JAX_PLATFORMS"] = "cpu"
        from moss_art.stage_cache import load_bundle, save_bundle
        from moss_art.checkpoint import load_checkpoint
        c = load_config(request["source_config"])
        source = Path(c["staging"]["bundle"])
        payload, metadata = load_bundle(source)
        params, optimizer, _ = load_checkpoint(source/"learner")
        output = Path(request["output_dir"]).resolve()
        c["runtime"]["output_dir"] = str(output)
        c["logging"]["log_dir"] = str(output/"tensorboard")
        destination = output/"stage_cache/update_00000000"
        metadata["stage_cache_id"] = str(destination)
        payload["config"] = c
        c["staging"]["bundle"] = str(destination)
        save_bundle(destination, payload, params, optimizer, metadata)
        write_json(output/"train.json", c)
        print(json.dumps(dict(config=str(output/"train.json"), bundle=str(destination))), flush=True)
        return
    c = load_config(request["training_config"])
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = output/"synthetic.jsonl"
    data.write_text(json.dumps(dict(id="synthetic", prompt="synthetic cached feature benchmark"))+"\n")
    c["name"] = "staged_startup_%d" % request["resolution"]
    c["data"]["prompts"] = str(data)
    c["preparation"]["async_prefetch"] = True
    c["resolution_mixing"]["enabled"] = False
    c["sampling"].update(height=request["resolution"], width=request["resolution"], steps=48,
        train_timesteps=[0, 15, 31, 47], group_size=4, rollout_batch_size=1,
        prompts_per_rollout=1, pair_cached_points=True, split_group_across_workers=True)
    c["sampling"]["rollout_mixing"]["enabled"] = False
    c["sampling"]["spectrum"]["enabled"] = False
    c["training"].update(max_updates=2, prompts_per_update=1, rollouts_per_prompt=4,
        training_points_per_trajectory=4, microbatch_size=1, evaluate_every=0)
    c["reward"]["kind"] = "toy_latent"
    c["runtime"]["output_dir"] = str(output)
    c["logging"]["log_dir"] = str(output/"tensorboard")
    c["staging"].update(enabled=True, phase="collect")
    validate(c)
    from moss_art.runtime import configure_process, preflight
    preflight(c); configure_process(c, c["runtime"]["devices"])
    import jax
    import numpy as np
    from moss_art.mesh_runtime import MeshPool
    from moss_art.art import covariance, prompt_windows
    from moss_art.trainer import make_payloads, paired_replay_payloads, identity_for, _optimizer
    from moss_art.staged import persist_collection
    identity = identity_for(c)
    started = time.monotonic()
    with MeshPool(c, base_precision="mxfp8") as pool:
        load_seconds = time.monotonic()-started
        rng = np.random.default_rng(123)
        context = rng.normal(0, .1, (1, request.get("context_tokens", 288), pool.worker.model.text_dim)).astype(np.float32)
        negative = np.zeros((1, 64, pool.worker.model.text_dim), np.float32)
        pool.worker.bound_conditions = {"synthetic": (context, negative)}
        record = dict(sample_id="synthetic", prompt="synthetic cached feature benchmark", negative_prompt="")
        rollout_started = time.monotonic()
        outputs = pool.call("rollout", make_payloads(c, [record], 0))
        rollout_seconds = time.monotonic()-rollout_started
        caches = pool.call("export_replay")
        endpoints = np.concatenate([o["endpoints"] for o in outputs])
        rewards = np.array([-.3, .1, .5, .7], np.float32)
        groups = np.zeros(4, np.int32)
        stats = {k: np.asarray(v) for k, v in covariance(endpoints, rewards, groups, group_size=4).items()}
        windows = prompt_windows(groups, 4, seed=42)
        prepared = paired_replay_payloads(pool, groups, stats, windows, caches=caches)
        params = pool.worker.lora
        optimizer = jax.tree.map(lambda value: np.array(value, copy=True), _optimizer(c).init(params))
        metadata = dict(identity, update=0, optimizer_updates=0, rollout=1, prompt_cursor=1,
            seed=42, initial_adapter=None, resolution_cursors={}, model_config=pool.info[0]["model_config"],
            lora_config=c["lora"], runtime_devices=c["runtime"]["devices"])
        pool.pause()
        metrics = dict(reward_mean=float(rewards.mean()), reward_std=float(rewards.std()),
                       reward_min=float(rewards.min()), reward_max=float(rewards.max()))
        bundle = persist_collection(c, pool.shards, prepared, params, optimizer, metadata, rewards,
                                    metrics, None, rollout_seconds)
        result = dict(dit_load_seconds=load_seconds, rollout_seconds=rollout_seconds,
            resolution=request["resolution"], context_capacity=request.get("context_tokens", 288),
            steps=48, trajectories=4, training_points=4, devices=c["runtime"]["devices"],
            lora=c["lora"], synthetic_features=True, qwen_vae_reward_models_loaded=False)
    c["staging"].update(phase="train", bundle=str(bundle))
    write_json(output/"train.json", c)
    write_json(output/"collect_benchmark.json", result)
    print(json.dumps(dict(config=str(output/"train.json"), **result)), flush=True)


if __name__ == "__main__":
    main()
