"""CPU regression coverage for the moving, detached EMA teacher runtime."""
import copy
import types
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import DEFAULTS
from moss_art.worker import NativeWorker
from moss_art.mesh_runtime import MeshPool
from moss_art.art import covariance, prompt_windows
from moss_art.trainer import aggregate_gradients, make_payloads, paired_replay_payloads


def config(micro=1):
    c = copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny", dtype="float32", activation_storage="gpu", checkpoint_interval=2)
    c["runtime"].update(platform="cpu", devices=[0], execution_backend="mesh")
    c["sampling"].update(height=32, width=32, steps=4, train_timesteps=[0, 1, 2], group_size=4,
        prompts_per_rollout=1, rollout_batch_size=1, guidance_scale=1.5, skip_layers=[1],
        sampler="euler", scheduler="moss")
    c["training"].update(microbatch_size=micro, max_updates=2, updates_per_rollout=2, evaluate_every=0)
    c["reward"]["kind"] = "toy_latent"
    c["art"].update(reference_mode="ema", reference_mse_beta=.1, reference_ema_decay=.99)
    return c


def perturbed(tree, seed, scale=.03):
    rng = np.random.default_rng(seed)
    return {k: np.asarray(v, np.float32)+rng.normal(0, scale, v.shape).astype(np.float32)
            for k, v in tree.items()}


def make_worker(micro=1):
    worker = NativeWorker(config(micro), 0)
    worker.set_lora(perturbed(worker.lora, 11))
    return worker


def install_replay(worker):
    record = dict(sample_id="ema-fixture", prompt="synthetic fixture", negative_prompt="")
    worker.rollout([record], [2, 3, 4, 5])
    worker.prepare_update(rewards=np.full(4, .5, np.float32), seed=7)


def test_reference_routes_teacher_without_mutating_learner_and_host_inputs():
    worker = make_worker()
    learner = {k: np.array(v, copy=True) for k, v in worker.lora.items()}
    teacher = perturbed(worker.lora, 12)
    frozen = {k: v.copy() for k, v in teacher.items()}
    worker.set_reference_lora(teacher)
    for value in teacher.values():
        value.fill(99)
    rng = np.random.default_rng(8)
    state = jnp.asarray(rng.normal(size=(2, 8, 2, 2)), jnp.float32)
    pos = jnp.asarray(rng.normal(size=(2, 6, 24)), jnp.float32)
    neg = jnp.asarray(rng.normal(size=(2, 4, 24)), jnp.float32)
    kwargs = dict(cfg_scale=1.5, negative_context=neg, negative_skip_layers=[1],
                  context_lengths=np.array([3, 6]), negative_context_lengths=np.array([2, 4]))
    actual = worker._reference_prediction(state, np.array([.3, .7]), pos, **kwargs)
    expected = worker.model.predict(frozen, state, np.array([.3, .7]), pos, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert not np.allclose(actual, worker.model.predict_base(state, np.array([.3, .7]), pos, **kwargs))
    for k in learner:
        np.testing.assert_array_equal(worker.lora[k], learner[k])
        np.testing.assert_array_equal(worker.reference_lora[k], frozen[k])
    exported = worker.export_state()
    worker.set_reference_lora(None)
    worker.import_state(exported)
    for k in frozen:
        np.testing.assert_array_equal(worker.reference_lora[k], frozen[k])


def test_teacher_output_is_detached_but_learner_gradient_survives():
    worker = object.__new__(NativeWorker)
    worker.c = {"art": {"reference_mode": "ema"}}
    worker.model = types.SimpleNamespace(base_precision="bf16",
        predict=lambda lora, state, sigma, positive, **kw: state+lora["w"])
    def loss(learner, teacher):
        worker.reference_lora = {"w": teacher}
        target = worker._reference_prediction(jnp.array(0.), .5, None)
        return jnp.square(learner-target)
    student_grad, teacher_grad = jax.grad(loss, argnums=(0, 1))(jnp.array(3.), jnp.array(1.))
    np.testing.assert_allclose(student_grad, 4.)
    np.testing.assert_array_equal(teacher_grad, 0.)


def test_ema_fails_closed_and_rejects_invalid_teacher():
    worker = make_worker()
    with pytest.raises(RuntimeError, match="must be installed"):
        worker._reference_prediction(jnp.zeros((1,8,2,2)), .5, jnp.zeros((1,4,24)))
    with pytest.raises(ValueError, match="keys"):
        worker.set_reference_lora({})
    bad = {k: np.asarray(v, np.float16) for k,v in worker.lora.items()}
    with pytest.raises(ValueError, match="FP32"):
        worker.set_reference_lora(bad)
    bad = {k: np.asarray(v).copy() for k,v in worker.lora.items()}
    next(iter(bad.values())).flat[0] = np.nan
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        worker.set_reference_lora(bad)


def test_base_mode_uses_only_existing_base_prediction():
    worker = object.__new__(NativeWorker)
    worker.c = {"art": {"reference_mode": "base"}}
    sentinel = object()
    worker.model = types.SimpleNamespace(predict_base=lambda *a, **kw: sentinel,
        predict=lambda *a, **kw: pytest.fail("EMA path used in base mode"))
    assert worker._reference_prediction(None, .5, None) is sentinel


@pytest.mark.parametrize("micro", [1, 2])
def test_actual_worker_gradient_uses_teacher_in_both_microbatch_paths(micro, monkeypatch):
    worker = make_worker(micro)
    install_replay(worker)
    student = {k: np.array(v, copy=True) for k,v in worker.lora.items()}
    worker.set_reference_lora(perturbed(student, 17))
    teacher = {k: np.array(v, copy=True) for k,v in worker.reference_lora.items()}
    monkeypatch.setattr(worker.model, "predict_base", lambda *a, **kw: pytest.fail("EMA fell back to Base"))
    grads, metrics = aggregate_gradients([worker.gradient(0)])
    assert metrics["base_reference_mse"] > 0
    assert metrics["art_loss"] == 0.
    np.testing.assert_allclose(metrics["loss"], .1*metrics["base_reference_mse"], rtol=2e-6)
    assert any(np.any(np.asarray(v) != 0) for v in grads.values())
    for k in student:
        np.testing.assert_array_equal(worker.lora[k], student[k])
        np.testing.assert_array_equal(worker.reference_lora[k], teacher[k])
    # If teacher exactly equals student, the reference-only objective has no
    # gradient. This distinguishes an actual teacher target from Base routing.
    worker.set_reference_lora(student)
    same, same_metrics = aggregate_gradients([worker.gradient(1)])
    np.testing.assert_allclose(same_metrics["loss"], 0., atol=1e-12)
    for value in same.values():
        np.testing.assert_allclose(value, 0., atol=1e-9)


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four logical CPU devices")
@pytest.mark.parametrize("micro", [1, 2])
@pytest.mark.parametrize("art_anchor", ["cached", "calibrated"])
def test_mesh_teacher_agrees_with_serial_and_survives_pause_resume(micro, art_anchor, monkeypatch):
    from test_mesh_dp import configuration
    c = configuration()
    c["sampling"].update(rollout_mode="bf16", sampler="euler", scheduler="moss")
    c["training"].update(microbatch_size=micro, max_updates=2, updates_per_rollout=2)
    c["art"].update(reference_mode="ema", reference_mse_beta=.1, reference_ema_decay=.99)
    serial = NativeWorker(c, 0)
    params = perturbed(serial.lora, 19)
    teacher = perturbed(serial.lora, 23)
    serial.set_lora(params); serial.set_reference_lora(teacher)
    record = dict(sample_id="ema-fixture", prompt="synthetic fixture", negative_prompt="")
    with MeshPool(c) as pool:
        pool.call("set_lora", [{"lora": params}]*4)
        version = pool.parameter_version
        pool.call("set_reference_lora", [{"lora": teacher}]*4)
        assert pool.parameter_version == version
        assert pool.worker.reference_lora is not pool.worker.lora
        outputs = pool.call("rollout", make_payloads(c, [record], 0))
        serial.rollout([record], list(range(c["training"]["seed"], c["training"]["seed"]+4)))
        groups = np.zeros(4, np.int32)
        endpoints = np.concatenate([o["endpoints"] for o in outputs])
        stats = {k: np.asarray(v) for k,v in covariance(endpoints, np.full(4,.5), groups, group_size=4).items()}
        windows = prompt_windows(groups, 3, seed=7)
        # This only selects the existing ART local anchor contract. It must
        # remain separate from the moving EMA regularization target.
        if art_anchor == "calibrated":
            pool.config["sampling"]["rollout_mode"] = "mxfp8_calibrated"
        pool.call("prepare_update", paired_replay_payloads(pool, groups, stats, windows))
        serial.prepare_update(stats=stats, windows=windows)
        monkeypatch.setattr(pool.worker.model, "predict_base", lambda *a, **kw: pytest.fail("mesh EMA used Base"))
        for window in range(2):
            actual, am = aggregate_gradients(pool.call("gradient", [{"window":window}]*4))
            expected, em = aggregate_gradients([serial.gradient(window)])
            np.testing.assert_allclose(am["loss"], em["loss"], rtol=5e-4, atol=2e-7)
            for k in actual:
                np.testing.assert_allclose(actual[k], expected[k], rtol=8e-4, atol=3e-6)
            params = {k: np.asarray(params[k])-.01*np.asarray(actual[k]) for k in params}
            teacher = {k: .99*np.asarray(teacher[k])+.01*params[k] for k in teacher}
            pool.call("set_lora", [{"lora": params}]*4);serial.set_lora(params)
            version = pool.parameter_version
            pool.call("set_reference_lora", [{"lora": teacher}]*4);serial.set_reference_lora(teacher)
            assert pool.parameter_version == version
        pool.call("clear_replay")
        teacher_before = {k: np.array(v, copy=True) for k,v in pool.worker.reference_lora.items()}
        learner_before = {k: np.array(v, copy=True) for k,v in pool.worker.lora.items()}
        version = pool.parameter_version
        pool.pause();pool.resume()
        assert pool.parameter_version == version
        for k in teacher_before:
            np.testing.assert_array_equal(pool.worker.reference_lora[k], teacher_before[k])
            np.testing.assert_array_equal(pool.worker.lora[k], learner_before[k])
