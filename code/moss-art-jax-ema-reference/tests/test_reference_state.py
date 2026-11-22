"""EMA follows applied optimizer steps and never receives learner gradients."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moss_art.config import DEFAULTS, fingerprint, validate
from moss_art.reference import (initialize_reference, validate_reference_state,
                                update_reference, ema_parameters)


def config(decay=.75, mode="ema"):
    c = copy.deepcopy(DEFAULTS)
    c["art"].update(reference_mode=mode, reference_ema_decay=decay)
    return c


def parameters(value=0., numpy=True):
    xp = np if numpy else jnp
    return {"A": xp.full((2, 3), value, xp.float32),
            "B": xp.full((3, 2), value + 1., xp.float32)}


@pytest.mark.parametrize("numpy", [True, False])
def test_exact_recurrence_is_independent_and_skipped_steps_are_noop(numpy):
    c = config(.75)
    initial = parameters(0., numpy)
    state = initialize_reference(c, initial)
    assert state["updates"] == 0
    assert state["lora"] is not initial
    if numpy:
        for k in initial:
            assert not np.shares_memory(initial[k], state["lora"][k])
    assert update_reference(c, state, parameters(123., numpy), False) is state
    first = update_reference(c, state, parameters(4., numpy), True)
    second = update_reference(c, first, parameters(8., numpy), True)
    for k in initial:
        np.testing.assert_array_equal(first["lora"][k], .75 * initial[k] + .25 * parameters(4., numpy)[k])
        np.testing.assert_array_equal(second["lora"][k], .75 * first["lora"][k] + .25 * parameters(8., numpy)[k])
        np.testing.assert_array_equal(state["lora"][k], initial[k])
        assert str(second["lora"][k].dtype) == "float32"
    assert second["updates"] == 2
    assert first["updates"] == 1


def test_zero_decay_tracks_current_learner_but_is_still_a_distinct_state():
    c = config(0.)
    state = initialize_reference(c, parameters())
    learner = parameters(3.)
    state = update_reference(c, state, learner, True)
    assert state["updates"] == 1
    for k in learner:
        np.testing.assert_array_equal(state["lora"][k], learner[k])
        assert not np.shares_memory(state["lora"][k], learner[k])


def test_initialization_casts_to_fp32_and_does_not_mutate_input():
    values = {"A": np.ones((2, 2), np.float16)}
    state = initialize_reference(config(), values)
    assert state["lora"]["A"].dtype == np.float32
    values["A"][:] = 9
    np.testing.assert_array_equal(state["lora"]["A"], 1.)


def test_numeric_ema_blocks_gradients_to_both_parameter_sets_and_jits():
    reference = parameters(1., False)
    learner = parameters(5., False)
    def objective(a, b):
        return sum(jnp.sum(v) for v in ema_parameters(a, b, .8).values())
    ga, gb = jax.grad(objective, argnums=(0, 1))(reference, learner)
    for values in [ga, gb]:
        for v in values.values():
            np.testing.assert_array_equal(v, 0.)
    actual = jax.jit(lambda a, b: ema_parameters(a, b, .8))(reference, learner)
    for k in reference:
        np.testing.assert_allclose(actual[k], .8 * reference[k] + .2 * learner[k], rtol=1e-6)


def test_base_mode_needs_no_state_and_rejects_ema_resume():
    c = config(mode="base")
    assert initialize_reference(c, parameters()) is None
    assert validate_reference_state(c, None, parameters()) is None
    assert update_reference(c, None, parameters(), True) is None
    with pytest.raises(ValueError, match="Base reference"):
        validate_reference_state(c, initialize_reference(config(), parameters()), parameters())


@pytest.mark.parametrize("change", ["missing", "decay", "mode", "keys", "shape", "dtype", "nan", "updates", "extra"])
def test_resume_reference_mismatch_fails_closed(change):
    c = config()
    lora = parameters()
    state = initialize_reference(c, lora)
    if change == "missing": state = None
    elif change == "decay": state["decay"] = .5
    elif change == "mode": state["mode"] = "base"
    elif change == "keys": state["lora"].pop("B")
    elif change == "shape": state["lora"]["A"] = np.zeros((1,), np.float32)
    elif change == "dtype": state["lora"]["A"] = state["lora"]["A"].astype(np.float16)
    elif change == "nan": state["lora"]["A"][0, 0] = np.nan
    elif change == "updates": state["updates"] = True
    else: state["unexpected"] = 1
    with pytest.raises(ValueError):
        validate_reference_state(c, state, lora)


@pytest.mark.parametrize("decay", [-.1, 1., 2., float("nan"), float("inf"), True, "0.9"])
def test_invalid_decay_config_fails(decay):
    c = config(decay)
    with pytest.raises(ValueError, match="reference_ema_decay"):
        validate(c)


def test_mode_validation_and_checkpoint_identity_compatibility():
    c = config(mode="base")
    old = copy.deepcopy(c)
    old["art"].pop("reference_mode")
    old["art"].pop("reference_ema_decay")
    assert fingerprint(c) == fingerprint(old)
    c["art"]["reference_ema_decay"] = .01
    assert fingerprint(c) == fingerprint(old)
    c["art"]["reference_mode"] = "ema"
    first = fingerprint(c)
    assert first != fingerprint(old)
    c["art"]["reference_ema_decay"] = .99
    assert fingerprint(c) != first
    c["art"]["reference_mode"] = "oops"
    with pytest.raises(ValueError, match="reference_mode"):
        validate(c)
