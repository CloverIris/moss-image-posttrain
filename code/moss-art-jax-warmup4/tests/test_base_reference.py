"""Frozen-base regularization: real forwards, independent masks and TB events."""
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from moss_art.art import art_loss, covariance, prompt_windows
from moss_art.config import DEFAULTS, fingerprint, load_config, validate
from moss_art.mesh_runtime import MeshPool, _loss_vjp, _loss_vjp_items
from moss_art.model import MossModel
from moss_art.trainer import aggregate_gradients, make_payloads, paired_replay_payloads
from moss_art.worker import NativeWorker


@pytest.mark.parametrize("cfg_scale", [1., 3.])
def test_base_shares_buffers_matches_zero_delta_and_never_records_tape(cfg_scale, monkeypatch):
    model = MossModel.tiny(seed=17, dtype=jnp.bfloat16, offload_activations=False)
    rng = np.random.default_rng(18)
    initial = model.init_lora()
    trained = {k: v + jnp.asarray(rng.normal(0, .03, v.shape), jnp.float32)
               for k, v in initial.items()}
    before = {k: np.asarray(v).copy() for k, v in trained.items()}
    base_ids = {k: id(v) for k, v in model.base.items()}
    state = jnp.asarray(rng.normal(size=(1, 8, 2, 2)), jnp.float32)
    positive = jnp.asarray(rng.normal(size=(1, 5, 24)), jnp.bfloat16)
    negative = jnp.asarray(rng.normal(size=(1, 3, 24)), jnp.bfloat16)
    kwargs = dict(cfg_scale=cfg_scale, negative_context=negative, negative_skip_layers=(1,),
                  context_lengths=np.array([4], np.int32), negative_context_lengths=np.array([2], np.int32))
    expected = model.predict(initial, state, .61, positive, **kwargs)
    trained_prediction = model.predict(trained, state, .61, positive, **kwargs)
    assert not np.array_equal(expected, trained_prediction)
    monkeypatch.setattr(model, "_checkpoint", lambda value: pytest.fail("base recorded an activation tape"))
    actual = model.predict_base(state, .61, positive, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert {k: id(v) for k, v in model.base.items()} == base_ids
    for k in trained:
        np.testing.assert_array_equal(trained[k], before[k])
    # A different adapter state never changes the base reference.
    model.predict({k: 2*v for k, v in trained.items()}, state, .61, positive, **kwargs)
    np.testing.assert_array_equal(actual, model.predict_base(state, .61, positive, **kwargs))


def loss_inputs():
    rng = np.random.default_rng(43)
    direction = rng.normal(size=(4, 2, 3, 2)).astype(np.float32)
    norm = direction / np.sqrt(np.mean(direction**2, axis=(1, 2, 3), keepdims=True))
    current = np.zeros_like(direction)
    current[0] = -.5*norm[0]  # ART masked: continuing outward.
    current[1] = .3          # No reward signal; reference MUST still train it.
    current[2] = .25         # Explicit ART restoration anchor.
    current[3] = 100.        # Finite padding; NEVER regularize it.
    return [jnp.asarray(x) for x in (current, np.zeros_like(current), np.full(4, .5, np.float32),
        direction, np.array([1., 0., 0., 1.], np.float32), np.full(4, -.05, np.float32),
        np.array([False, False, True, False]), np.array([True, True, True, False]))]


@pytest.mark.parametrize("mode", ["cached", "self", "calibrated"])
def test_regularizer_gradient_survives_art_mask_and_no_signal_but_excludes_padding(mode):
    args = loss_inputs()
    base = jnp.full_like(args[0], .1)
    beta = .07
    options = dict(noise=.8, kl=1e-5, eps=1e-5, anchor_mode=mode)
    (_, _), art_gradient = _loss_vjp(*args, **options)
    (value, counts), gradient, details = _loss_vjp(*args, **options,
        reference_prediction=base, reference_beta=beta, return_details=True)
    correction = 2*beta*(args[0]-base)/np.prod(args[0].shape[1:])
    correction = jnp.where(args[-1].reshape(-1, 1, 1, 1), correction, 0.)
    np.testing.assert_allclose(gradient, art_gradient+correction, rtol=2e-6, atol=1e-7)
    assert counts[1] == 3
    assert details["policy_active"] < 3
    np.testing.assert_array_equal(gradient[3], 0.)
    assert np.any(np.asarray(gradient[1]) != 0)
    np.testing.assert_allclose(value, details["policy_loss_sum"]+beta*details["reference_mse_sum"], rtol=2e-6)
    # Two microbatches have exactly the same loss/gradient numerator as the full batch.
    parts = [_loss_vjp(*[a[i:i+2] for a in args], **options,
        reference_prediction=base[i:i+2], reference_beta=beta) for i in (0, 2)]
    np.testing.assert_allclose(sum(p[0][0] for p in parts)/3, value/3, rtol=2e-6)
    np.testing.assert_allclose(jnp.concatenate([p[1] for p in parts])/3, gradient/3, rtol=2e-6, atol=1e-7)


def test_reference_prediction_has_no_gradient():
    current, old, sigma, direction, magnitude, coef, anchor, valid = loss_inputs()
    def objective(reference):
        return art_loss(current, old, sigma, direction, magnitude,
            ode_velocity_coefficient=coef, restore_anchor_mask=anchor, valid_mask=valid,
            reference_prediction=reference, reference_mse_beta=.1)["per_item_loss"].sum()
    np.testing.assert_array_equal(jax.grad(objective)(jnp.zeros_like(current)), 0.)


@pytest.mark.parametrize("invalid", [-1., float("nan"), float("inf"), True, "0.1"])
def test_invalid_reference_coefficient_rejected(invalid):
    config = copy.deepcopy(DEFAULTS)
    config["art"]["reference_mse_beta"] = invalid
    with pytest.raises(ValueError, match="reference_mse_beta"):
        validate(config)


def test_disabled_reference_keeps_old_fingerprint_and_enabled_reference_changes_it():
    config = copy.deepcopy(DEFAULTS)
    old = copy.deepcopy(config)
    old["art"].pop("reference_mse_beta")
    assert fingerprint(config) == fingerprint(old)
    config["art"]["reference_mse_beta"] = .01
    assert fingerprint(config) != fingerprint(old)


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four logical CPU devices")
@pytest.mark.parametrize("feedback", ["scalar", "pairwise"])
@pytest.mark.parametrize("microbatch", [1, 2])
@pytest.mark.parametrize("mode", ["bf16", "mxfp8_calibrated"])
def test_real_mesh_and_serial_reference_gradients_agree_in_both_windows(feedback, microbatch, mode):
    from test_mesh_dp import configuration
    config = configuration()
    config["sampling"].update(rollout_mode="bf16", sampler="euler", scheduler="moss")
    config["training"].update(updates_per_rollout=2, max_updates=2, microbatch_size=microbatch)
    config["art"].update(feedback=feedback, reference_mse_beta=.01)
    validate(config)
    serial = NativeWorker(config, 0)
    rng = np.random.default_rng(93)
    params = {k: v+jnp.asarray(rng.normal(0, .015, v.shape), jnp.float32) for k, v in serial.lora.items()}
    serial.set_lora(params)
    record = dict(sample_id="fixture", prompt="synthetic reference fixture", negative_prompt="")
    with MeshPool(config) as pool:
        pool.call("set_lora", [{"lora": params}]*4)
        outputs = pool.call("rollout", make_payloads(config, [record], 0))
        serial.rollout([record], list(range(config["training"]["seed"], config["training"]["seed"]+4)))
        endpoints = np.concatenate([o["endpoints"] for o in outputs])
        groups = np.zeros(4, np.int32)
        # All equal rewards exercise the formerly unconditional no-signal skip.
        stats = {k: np.asarray(v) for k, v in covariance(endpoints, np.full(4, .5), groups,
            mode=feedback, group_size=4).items()}
        windows = prompt_windows(groups, 3, seed=7)
        payloads = paired_replay_payloads(pool, groups, stats, windows)
        # CPU oracle: zero-quantization-mismatch rollout, independent of anchor mode.
        pool.config["sampling"]["rollout_mode"] = mode
        pool.call("prepare_update", payloads)
        serial.prepare_update(stats=stats, windows=windows)
        for window in range(2):
            actual, metrics = aggregate_gradients(pool.call("gradient", [{"window": window}]*4))
            expected, reference = aggregate_gradients([serial.gradient(window)])
            assert metrics["items"] == reference["items"] == 6
            assert metrics["active_fraction"] == 1.
            assert metrics["policy_active_fraction"] == 0.
            assert metrics["base_reference_mse"] > 0
            assert metrics["art_loss"] == 0.
            np.testing.assert_allclose(metrics["loss"], .01*metrics["base_reference_mse"], rtol=1e-6)
            np.testing.assert_allclose(metrics["base_reference_mse"], reference["base_reference_mse"], rtol=2e-4)
            for k in actual:
                np.testing.assert_allclose(actual[k], expected[k], rtol=5e-4, atol=3e-6)
            # This oracle intentionally drives a serial and a distributed
            # model together; hand off host arrays to their own placements.
            params = {k: np.asarray(params[k])-.01*np.asarray(actual[k]) for k in params}
            pool.call("set_lora", [{"lora": params}]*4)
            serial.set_lora(params)


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four logical CPU devices")
def test_regularized_mesh_loss_vectors_preserve_sharding_without_scalar_collectives():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.asarray(jax.devices()), ("dp",))
    args = [jax.device_put(v, NamedSharding(mesh, P("dp", *(None for _ in range(v.ndim-1))))) for v in loss_inputs()]
    reference = jax.device_put(np.zeros(args[0].shape, np.float32), args[0].sharding)
    options = dict(noise=.8, kl=1e-5, eps=1e-5, reference_prediction=reference,
                   reference_beta=.01, return_details=True)
    (losses, aux), gradient = _loss_vjp_items(*args, **options)
    assert losses.shape == (4,) and all(v.shape == (4,) for v in aux)
    assert gradient.sharding.is_equivalent_to(args[0].sharding, args[0].ndim)
    hlo = _loss_vjp_items.lower(*args, **options).compile().as_text().lower()
    assert "all-reduce" not in hlo and "all-gather" not in hlo


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four logical CPU devices")
def test_no_signal_training_still_regularizes_and_writes_separate_tensorboard_curves(tmp_path, monkeypatch):
    from test_anchor_integration import configuration, records
    import moss_art.trainer as trainer
    from moss_art.lora import export_safetensors
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    config = configuration()
    config["art"]["reference_mse_beta"] = .01
    config["training"].update(max_updates=2, evaluate_every=0)
    config["runtime"]["output_dir"] = str(tmp_path/"training")
    config["logging"].update(tensorboard=True, log_dir=str(tmp_path/"tensorboard"))
    data = tmp_path/"synthetic.jsonl"
    data.write_text("".join(json.dumps(row)+"\n" for row in records(6)))
    config["data"]["prompts"] = str(data)
    validate(config)
    model = MossModel.tiny(seed=config["model"]["tiny_seed"], rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"], targets="attention", offload_activations=False)
    rng = np.random.default_rng(33)
    params = {k: v+jnp.asarray(rng.normal(0, .02, v.shape), jnp.float32) for k, v in model.init_lora().items()}
    adapter = tmp_path/"initial.safetensors"
    export_safetensors(adapter, params, alpha=config["lora"]["alpha"])
    def constant_rewards(c, pool, outputs, **kwargs):
        return np.full(sum(len(piece["endpoints"]) for piece in outputs), .5, np.float32), None
    monkeypatch.setattr(trainer, "reward_round", constant_rewards)
    output = trainer.train(config, init_adapter=str(adapter))
    rows = [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [1, 2]
    assert all(row["event"] == "update" and row["optimizer_step_applied"] for row in rows)
    assert all(row["art_loss"] == 0. and row["base_reference_mse"] > 0 for row in rows)
    events = EventAccumulator(str(tmp_path/"tensorboard"))
    events.Reload()
    for key in ("loss", "art_loss", "base_reference_mse", "base_reference_rms", "base_reference_loss",
                "base_reference_beta", "base_reference_seconds", "policy_active_fraction"):
        scalars = events.Scalars("train/"+key)
        assert [s.step for s in scalars] == [1, 2]
        np.testing.assert_allclose([s.value for s in scalars], [row[key] for row in rows], rtol=2e-6, atol=1e-9)
