"""CPU integration oracles for scheduler history and frozen ART replay.

The analytic field avoids a 5B model while exercising the actual worker loops,
cache slot mapping, and ART loss/gradient. NFE here means calls to predict;
real CFG performs its positive/negative network branches inside that call.
"""
from collections import OrderedDict
import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import DEFAULTS
from moss_art.worker import NativeWorker
from vendor.moss_jax.scheduler import schedule as legacy_schedule


def field(theta, state, sigma):
    return (jnp.float32(.15) * state + jnp.float32(.03) *
            jnp.square(jnp.float32(sigma)) + theta)


class AnalyticModel:
    config = {"latent_channels": 2}

    def __init__(self):
        self.calls = []
        self.replay_calls = []
        self.pullbacks = 0

    def predict(self, adapters, state, sigma, context, **kwargs):
        value = field(adapters["theta"], state, sigma)
        self.calls.append((np.asarray(state).copy(), np.asarray(value).copy(),
                           float(sigma)))
        return value

    def predict_with_pullback(self, adapters, state, sigma, context, **kwargs):
        value = field(adapters["theta"], state, sigma)
        self.replay_calls.append((np.asarray(state).copy(), float(sigma)))

        def pullback(cotangent):
            self.pullbacks += 1
            return {"theta": jnp.sum(cotangent)}
        return value, pullback


def bare_worker(*, sampler="res_multistep", scheduler="beta", steps=50,
                selected=(0, 17, 35), group_size=2):
    worker = NativeWorker.__new__(NativeWorker)
    worker.c = copy.deepcopy(DEFAULTS)
    worker.c["sampling"].update(
        sampler=sampler, scheduler=scheduler, steps=steps,
        height=32, width=32, group_size=group_size, prompts_per_rollout=2,
        train_timesteps=list(selected), guidance_scale=3., skip_layers=[])
    worker.model = AnalyticModel()
    worker.device = jax.devices("cpu")[0]
    worker.lora = {"theta": jnp.asarray(0., jnp.float32)}
    worker.cache = None
    worker.text_cache = OrderedDict()
    worker._compiled_text_lengths = set()
    worker._conditioning = lambda record: (
        jnp.zeros((1, 1, 1), jnp.float32), jnp.zeros((1, 1, 1), jnp.float32))
    return worker


RECORDS = [{"sample_id": "first", "prompt": "one"},
           {"sample_id": "second", "prompt": "two"}]
NONUNIFORM = np.asarray([1., .83, .69, .41, .18, 0.], np.float32)


def inject_schedule(monkeypatch, sigmas):
    import moss_art.sampling as sampling
    def schedule(*args, **kwargs):
        return sigmas[:-1] * np.float32(1000), sigmas.copy()
    monkeypatch.setattr(sampling, "schedule", schedule)


def res_velocity_coefficient(sigmas, index):
    """Independent closed-form derivative of the official deterministic step."""
    sigma, next_sigma = map(float, sigmas[index:index + 2])
    if index == 0 or next_sigma == 0:
        return next_sigma - sigma
    h = math.log(sigma / next_sigma)
    h_previous = math.log(float(sigmas[index - 1]) / sigma)
    phi1 = -math.expm1(-h) / h
    phi2 = (1. - phi1) / h
    b1 = phi1 + phi2 * h / h_previous
    return -sigma * h * b1


def test_beta50_nfe_raw_cache_and_history_reset_per_image_and_rollout():
    worker = bare_worker()
    # Same seeds in different groups reveal a history leak across image boundaries.
    seeds = [11, 12, 11, 12]
    result = worker.rollout(RECORDS, seeds)
    cache = worker.cache
    actual = len(result["sigmas"]) - 1
    assert actual == result["actual_steps"] == 50
    assert result["sampler"] == "res_multistep"
    assert result["scheduler"] == "beta"
    assert len(worker.model.calls) == 4 * actual
    assert len(cache["ode_velocity_coefficients"]) == actual
    assert cache["ode_velocity_coefficients"].dtype == np.float32
    np.testing.assert_array_equal(result["endpoints"][:2], result["endpoints"][2:])
    for row in range(4):
        for slot, step in enumerate(cache["selected"]):
            state, raw_velocity, sigma = worker.model.calls[row * actual + step]
            np.testing.assert_array_equal(cache["states"][row, slot], state[0])
            np.testing.assert_array_equal(cache["old"][row, slot], raw_velocity[0])
            assert sigma == result["sigmas"][step]
    # A returned metadata array must not let its consumer change frozen replay.
    saved_coefficients = cache["ode_velocity_coefficients"].copy()
    result["ode_velocity_coefficients"][0] = -123.
    np.testing.assert_array_equal(cache["ode_velocity_coefficients"], saved_coefficients)

    original = result["endpoints"].copy()
    worker.rollout(RECORDS, [31, 32, 33, 34])
    repeated = worker.rollout(RECORDS, seeds)
    np.testing.assert_array_equal(repeated["endpoints"], original)
    assert len(worker.model.calls) == 3 * 4 * actual


@pytest.mark.parametrize("sampler", ["euler", "res_multistep"])
def test_evaluation_without_art_cache_keeps_same_trajectory(monkeypatch, sampler):
    inject_schedule(monkeypatch, NONUNIFORM)
    worker = bare_worker(sampler=sampler, steps=5, selected=(3, 1, 0, 4))
    seeds = [7, 8, 9, 10]
    trained = worker.rollout(RECORDS, seeds)
    original_cache = worker.cache
    evaluated = worker.rollout(RECORDS, seeds, keep_cache=False)
    assert worker.cache is original_cache
    np.testing.assert_array_equal(trained["endpoints"], evaluated["endpoints"])
    assert len(worker.model.calls) == 2 * 4 * 5


def test_euler_preserves_original_schedule_and_update_bitwise(monkeypatch):
    _, sigmas = legacy_schedule(steps=6, shift=5., training_shift=1.)
    inject_schedule(monkeypatch, sigmas)
    selected = (4, 1, 0, 5)  # Deliberately not sorted: slots are not step indices.
    worker = bare_worker(sampler="euler", scheduler="moss", steps=6,
                         selected=selected)
    seeds = [21, 22, 23, 24]
    result = worker.rollout(RECORDS, seeds)
    expected_endpoints, expected_states, expected_old = [], [], []
    for seed in seeds:
        z = jax.random.normal(jax.random.PRNGKey(seed), (1, 2, 2, 2),
                              dtype=jnp.float32)
        states, old = {}, {}
        for step in range(len(sigmas) - 1):
            velocity = field(jnp.float32(0), z, sigmas[step])
            if step in selected:
                states[step] = np.asarray(z[0]).copy()
                old[step] = np.asarray(velocity[0]).copy()
            # Exact pre-multistep worker expression, not the new sampling helper.
            z = z + jnp.float32(sigmas[step + 1] - sigmas[step]) * velocity
        expected_endpoints.append(np.asarray(z[0]))
        expected_states.append([states[i] for i in selected])
        expected_old.append([old[i] for i in selected])
    assert result["endpoints"].tobytes() == np.asarray(expected_endpoints).tobytes()
    assert worker.cache["states"].tobytes() == np.asarray(expected_states).tobytes()
    assert worker.cache["old"].tobytes() == np.asarray(expected_old).tobytes()
    np.testing.assert_array_equal(worker.cache["ode_velocity_coefficients"],
                                  np.diff(sigmas))


def test_nonuniform_res_coefficients_are_frozen_full_grid(monkeypatch):
    inject_schedule(monkeypatch, NONUNIFORM)
    worker = bare_worker(steps=5, selected=(3, 1, 0, 4))
    result = worker.rollout(RECORDS, [51, 52, 53, 54])
    expected = np.asarray([res_velocity_coefficient(NONUNIFORM, i)
                           for i in range(len(NONUNIFORM) - 1)], np.float32)
    np.testing.assert_allclose(worker.cache["ode_velocity_coefficients"], expected,
                               atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(result["ode_velocity_coefficients"],
                                  worker.cache["ode_velocity_coefficients"])
    assert np.all(expected < 0)
    assert np.all(expected[1:-1] < np.diff(NONUNIFORM)[1:-1])
    assert worker.cache["selected"] == (3, 1, 0, 4)


def test_art_replay_uses_selected_step_local_coefficient_and_raw_velocity(monkeypatch):
    inject_schedule(monkeypatch, NONUNIFORM)
    worker = bare_worker(steps=5, selected=(3, 1, 0, 4))
    worker.rollout(RECORDS, [61, 62, 63, 64])
    row, slot, step = 0, 1, 1
    state = jnp.asarray(worker.cache["states"][row:row + 1, slot])
    old = worker.cache["old"][row:row + 1, slot]
    sigma = float(NONUNIFORM[step])
    coefficient = res_velocity_coefficient(NONUNIFORM, step)
    delta_sigma = float(NONUNIFORM[step + 1] - NONUNIFORM[step])
    worker.lora["theta"] = jnp.float32(-.01)
    current = np.asarray(field(worker.lora["theta"], state, sigma))
    movement_mse = np.mean((current - old).astype(np.float64) ** 2)
    noise = worker.c["art"]["noise_level"]

    def metric(c):
        variance = noise**2 * sigma / (1. - sigma) * (-c)
        return (c * (1. + .5 * noise**2))**2 * movement_mse / (2. * variance)

    old_euler_energy, res_energy = metric(delta_sigma), metric(coefficient)
    assert res_energy > old_euler_energy
    # Under this threshold the old delta_sigma implementation fails to mask.
    worker.c["art"]["kl_threshold"] = (old_euler_energy + res_energy) / 2.
    worker.stats = {
        "group_directions": np.ones((2, 2, 2, 2), np.float32),
        "group_guidance_magnitudes": np.ones(2, np.float32),
    }
    worker.windows = [np.asarray([[row, slot]], np.int64)]
    # Replay must use saved coefficients even if later caller config changes.
    worker.c["sampling"]["sampler"] = "euler"
    worker.c["sampling"]["shift"] = 2.
    masked = worker.gradient(0)
    assert masked["denominator"] == 1
    assert masked["masked"] == 1 and masked["active"] == 0
    assert masked["loss_sum"] == 0.
    assert float(masked["gradients"]["theta"]) == 0.
    assert worker.model.pullbacks == 0
    np.testing.assert_array_equal(worker.model.replay_calls[-1][0], state)
    assert worker.model.replay_calls[-1][1] == sigma

    worker.c["art"]["kl_threshold"] = res_energy * 1.1
    active = worker.gradient(0)
    assert active["masked"] == 0 and active["active"] == 1
    assert worker.model.pullbacks == 1
    # Velocity-space ART derivative: no extra multistep coefficient/history factor.
    expected_gradient = 2. * sigma * np.mean(current - old + 1.)
    np.testing.assert_allclose(active["gradients"]["theta"], expected_gradient,
                               atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(active["loss_sum"],
                               sigma * np.mean((current - old + 1.)**2),
                               atol=2e-7, rtol=2e-6)
