"""CPU integration oracle against the public Torch ART pipeline.

Shares explicit noise, sigma grid and row order, rather than pretending NumPy,
JAX and Torch RNG streams are identical. Two prompt groups have K=4; one group
has constant rewards. The second update masks only part of the active rows'
selected steps, exercising the original denominator across two worker results.
"""
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from moss_art.art import covariance, prompt_windows
from moss_art.config import DEFAULTS
from moss_art.optimizers import build_optimizer
from moss_art.trainer import aggregate_gradients
from moss_art.worker import NativeWorker


TORCH_ORACLE = r'''
import json, sys, torch
q = json.load(sys.stdin)
sys.path.insert(0, q["upstream"])
from visual_rl.algorithms import art_scalar_covariance
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.objectives import ARTObjective, prompt_disjoint_windows

torch.set_num_threads(1)
theta = torch.nn.Parameter(torch.tensor(q["theta"], dtype=torch.float32))
def predict(state, sigma, rows):
    del rows
    time = sigma.reshape((-1,) + (1,) * (state.ndim - 1))
    return 0.15 * state + 0.03 * time.square() + theta
noise = torch.tensor(q["noise"], dtype=torch.float32)
sigmas = torch.tensor(q["sigmas"], dtype=torch.float32)
groups = torch.tensor(q["groups"], dtype=torch.long)
cache = collect_rollout(predict, noise, sigmas, selected=q["selected"], microbatch=1)
stats = art_scalar_covariance(cache.endpoints, torch.tensor(q["rewards"]), groups,
                             2, zero_signal_mode="exact")
windows = prompt_disjoint_windows(groups, len(q["selected"]), order=q["order"])
objective = ARTObjective(cache, predict, stats["group_directions"][groups],
                         stats["group_guidance_magnitudes"][groups],
                         torch.zeros(len(groups), dtype=torch.bool),
                         torch.device("cpu"))
o = q["optimizer"]
optimizer = torch.optim.AdamW([theta], lr=o["learning_rate"], betas=o["betas"],
                              eps=o["epsilon"], weight_decay=o["weight_decay"],
                              foreach=False)
updates=[]
for items in windows:
    optimizer.zero_grad(set_to_none=True)
    losses = objective(items)
    loss = losses.sum() / len(items)
    loss.backward()
    gradient = theta.grad.detach().clone()
    # Call the public loss on each cached step to inspect the exact mask/target.
    from visual_rl.algorithms import art_loss
    rows, slots = torch.tensor(items).unbind(1)
    steps = cache.indices(rows, slots)
    state, old = cache.states[rows, slots], cache.old_velocities[rows, slots]
    velocity = predict(state, cache.sigmas[steps], rows)
    terms = art_loss(velocity, old, cache.sigmas[steps],
                     stats["group_directions"][groups[rows]],
                     stats["group_guidance_magnitudes"][groups[rows]],
                     ode_velocity_coefficient=cache.sigmas[steps+1]-cache.sigmas[steps])
    torch.nn.utils.clip_grad_norm_([theta], o["max_grad_norm"])
    clipped_gradient = theta.grad.detach().clone()
    optimizer.step()
    updates.append(dict(loss=float(loss.detach()), gradient=float(gradient),
                        clipped_gradient=float(clipped_gradient), theta=float(theta.detach()),
                        denominator=len(items), masked=int(terms["masked"].sum()),
                        active=int(terms["keep_mask"].sum()),
                        targets=terms["target"].detach().tolist()))
def values(d):
    return {key:value.detach().tolist() for key,value in d.items() if key.startswith("group_")}
json.dump(dict(endpoints=cache.endpoints.tolist(), states=cache.states.tolist(),
               old=cache.old_velocities.tolist(), stats=values(stats),
               windows=windows, updates=updates), sys.stdout)
'''


class AffinePolicy:
    """Same differentiable field as the independent Torch oracle."""
    config = {"latent_channels": 2}

    @staticmethod
    def predict(adapters, state, sigma, context, **kwargs):
        del context, kwargs
        return (jnp.float32(.15) * state + jnp.float32(.03) *
                jnp.square(jnp.float32(sigma)) + adapters["theta"])

    def predict_with_pullback(self, adapters, state, sigma, context, **kwargs):
        value = self.predict(adapters, state, sigma, context, **kwargs)
        return value, lambda cotangent: {"theta": jnp.sum(cotangent)}


def bare_worker(config, theta):
    worker = NativeWorker.__new__(NativeWorker)
    worker.c = copy.deepcopy(config)
    worker.model = AffinePolicy()
    worker.device = jax.devices("cpu")[0]
    worker.lora = {"theta": jnp.asarray(theta, jnp.float32)}
    worker.cache = None
    worker.text_cache = OrderedDict()
    worker._compiled_text_lengths = set()
    worker._conditioning = lambda record: (
        jnp.zeros((1, 1, 1), jnp.float32), jnp.zeros((1, 1, 1), jnp.float32))
    return worker


def test_k4_euler_two_worker_two_update_pipeline_matches_upstream(monkeypatch):
    python = Path(os.environ.get("MOSS_ART_TORCH_PYTHON",
                                 "$CONDA_PREFIX/envs/torch_env/bin/python"))
    upstream = Path(os.environ.get("MOSS_ART_UPSTREAM", "$HOME/DiffusionART/src"))
    if not python.is_file() or not upstream.is_dir():
        pytest.skip("optional upstream Torch oracle environment unavailable")
    config = copy.deepcopy(DEFAULTS)
    selected = [0, 1, 2]
    sigmas = np.asarray([1., .6, .25, 0.], np.float32)
    config["sampling"].update(height=32, width=32, steps=3, group_size=4,
                               prompts_per_rollout=2, sampler="euler", scheduler="moss",
                               train_timesteps=selected, guidance_scale=1., skip_layers=[])
    config["optimizer"].update(name="adamw", learning_rate=.004, weight_decay=.0001,
                                max_grad_norm=.2)
    import moss_art.sampling as sampling
    monkeypatch.setattr(sampling, "schedule", lambda *args, **kwargs:
                        (sigmas[:-1]*np.float32(1000), sigmas.copy()))

    base = np.linspace(-.2, .2, 8, dtype=np.float32).reshape(1, 2, 2, 2)
    noise = np.concatenate([base + np.float32(offset)
                            for offset in (-.9, -.3, .3, .9, -.7, -.1, .5, 1.1)])
    seeds = [11, 12, 13, 14, 21, 22, 23, 24]
    shared_noise = {seed: value[None] for seed, value in zip(seeds, noise)}
    def explicit_noise(key, shape, dtype=jnp.float32, **kwargs):
        del kwargs
        value = shared_noise[int(np.asarray(key)[-1])]
        assert value.shape == shape
        return jnp.asarray(value, dtype)
    monkeypatch.setattr(jax.random, "normal", explicit_noise)
    groups = np.repeat(np.arange(2, dtype=np.int32), 4)
    rewards = np.asarray([0., 1., 2., 3., 2., 2., 2., 2.], np.float32)
    window_seed = 31
    order = np.random.default_rng(window_seed).permutation(8).tolist()
    initial_theta = .02
    request = dict(upstream=str(upstream), noise=noise.tolist(), sigmas=sigmas.tolist(),
                   groups=groups.tolist(), rewards=rewards.tolist(), selected=selected,
                   order=order, theta=initial_theta, optimizer=config["optimizer"])
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run([str(python), "-c", TORCH_ORACLE], input=json.dumps(request),
                            text=True, capture_output=True, timeout=90, env=environment)
    assert result.returncode == 0, result.stderr
    oracle = json.loads(result.stdout)

    workers = [bare_worker(config, initial_theta) for _ in range(2)]
    outputs = [worker.rollout([{"sample_id": str(rank), "prompt": f"group {rank}"}],
                              seeds[4*rank:4*rank+4]) for rank, worker in enumerate(workers)]
    close = lambda actual, expected: np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-5, atol=3e-7)
    endpoints = np.concatenate([output["endpoints"] for output in outputs])
    close(endpoints, oracle["endpoints"])
    close(np.concatenate([worker.cache["states"] for worker in workers]), oracle["states"])
    close(np.concatenate([worker.cache["old"] for worker in workers]), oracle["old"])
    stats = covariance(endpoints, rewards, groups, group_size=4)
    for name, expected in oracle["stats"].items():
        close(stats[name], expected)
    windows = prompt_windows(groups, len(selected), seed=window_seed)
    assert [window.tolist() for window in windows] == [list(map(list, window)) for window in oracle["windows"]]
    for rank, worker in enumerate(workers):
        local_stats = {name: np.asarray(value)[rank:rank+1] if name.startswith("group_")
                       else np.asarray(value)[4*rank:4*rank+4] if name.startswith("row_")
                       else np.asarray(value) for name, value in stats.items()}
        local_windows = [np.asarray([(int(row)-4*rank, int(slot)) for row, slot in window
                                     if 4*rank <= row < 4*rank+4], np.int64)
                         for window in windows]
        worker.prepare_update(stats=local_stats, windows=local_windows)

    parameters = {"theta": jnp.asarray(initial_theta, jnp.float32)}
    tx = build_optimizer(config["optimizer"])
    state = tx.init(parameters)
    from moss_art.art import art_loss
    for index, expected in enumerate(oracle["updates"]):
        results = [worker.gradient(index) for worker in workers]
        gradients, metrics = aggregate_gradients(results)
        assert metrics["items"] == expected["denominator"] == 12
        assert sum(value["denominator"] for value in results) == 12
        assert results[1]["active"] == 0 and results[1]["denominator"] == 6
        assert sum(value["masked"] for value in results) == expected["masked"]
        assert sum(value["active"] for value in results) == expected["active"]
        close(metrics["loss"], expected["loss"])
        close(gradients["theta"], expected["gradient"])
        # Independent target check in exactly the global window's row/slot order.
        states = np.concatenate([worker.cache["states"] for worker in workers])
        old = np.concatenate([worker.cache["old"] for worker in workers])
        targets = []
        for row, slot in windows[index]:
            sigma = sigmas[selected[slot]]
            velocity = workers[row//4].model.predict(parameters, jnp.asarray(states[row:row+1,slot]),
                                                     sigma, None)
            terms = art_loss(velocity, old[row:row+1,slot], [sigma],
                             np.asarray(stats["group_directions"])[groups[row]][None],
                             [float(stats["group_guidance_magnitudes"][groups[row]])],
                             ode_velocity_coefficient=[sigmas[selected[slot]+1]-sigma])
            targets.append(np.asarray(terms["target"])[0])
        close(targets, expected["targets"])
        clipped = optax.clip_by_global_norm(config["optimizer"]["max_grad_norm"])
        clipped_gradients, _ = clipped.update(gradients, clipped.init(parameters))
        close(clipped_gradients["theta"], expected["clipped_gradient"])
        delta, state = tx.update(gradients, state, parameters)
        parameters = optax.apply_updates(parameters, delta)
        close(parameters["theta"], expected["theta"])
        for worker in workers:
            worker.set_lora({name: np.asarray(value) for name, value in parameters.items()})
    assert oracle["updates"][0]["masked"] == 0
    assert 0 < oracle["updates"][1]["masked"] < oracle["updates"][1]["denominator"]
    assert 0 < oracle["updates"][1]["active"] < oracle["updates"][0]["active"]