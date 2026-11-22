"""Independent scheduler/solver oracles and frozen-history differentiation."""
import os
import subprocess
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.sampling import (
    COMFYUI_REVISION, SAMPLERS, schedule, sample_step, step_coefficients,
)


def _reference_res(state, velocity, sigma, sigma_next, sigma_previous, history):
    """Float64 equation oracle, deliberately independent of production fields."""
    x = np.asarray(state, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    denoised = x-float(sigma)*v
    if history is None or sigma_next == 0:
        return x+(float(sigma_next)-float(sigma))*v, denoised
    h = np.log(float(sigma)/float(sigma_next))
    h_previous = np.log(float(sigma_previous)/float(sigma))
    phi1 = -np.expm1(-h)/h
    phi2 = (1-phi1)/h
    b2 = -phi2*h/h_previous
    return (np.exp(-h)*x+h*((phi1-b2)*denoised+b2*np.asarray(history)),
            denoised)


def test_default_beta_grid_endpoints_nfe_and_single_shift():
    times, sigmas = schedule()
    assert times.dtype == sigmas.dtype == np.float32
    assert sigmas.shape == (51,) and times.shape == (50,)
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert np.all(np.diff(sigmas) < 0)
    np.testing.assert_array_equal(times, sigmas[:-1]*np.float32(1000))
    # Official beta=.6 indices include the low endpoint index 3, not index 0.
    np.testing.assert_array_equal(
        sigmas[[0, 1, 25, 48, 49, 50]],
        np.asarray([1, .9993985, .8338881, .048076920211315155, .019685039296746254, 0], np.float32))
    # Reject an extra legacy shift rather than silently changing its meaning.
    with pytest.raises(ValueError, match="training_shift=1"):
        schedule(training_shift=9)
    assert np.min(np.abs(sigmas[1:-1]*1000-np.rint(sigmas[1:-1]*1000))) > 0
    assert COMFYUI_REVISION == "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"


@pytest.mark.parametrize("shift", [1.0, 2.5, 5.0])
def test_uniform_beta_quantiles_choose_shifted_training_grid_once(shift):
    # alpha=beta=1 gives exact uniform quantiles; five steps on five grid points.
    _, sigmas = schedule(5, shift=shift, total=5, beta_alpha=1, beta_beta=1)
    t = np.asarray([1, .8, .6, .4], np.float32)  # rint(4*[1,.8,.6,.4,.2]) -> 4,3,2,2,1
    factor = np.float32(shift)
    expected = factor*t/(np.float32(1)+(factor-np.float32(1))*t)
    np.testing.assert_array_equal(sigmas, np.r_[expected, np.float32(0)])


def test_quantile_deduplication_reports_actual_nfe_without_zero_length_steps():
    _, sigmas = schedule(100, total=4)
    assert len(sigmas) == 5
    assert np.all(np.diff(sigmas) < 0)
    assert len(step_coefficients(sigmas)) == 4
    for count in (1, 50):
        np.testing.assert_array_equal(schedule(count, total=1)[1], [1, 0])


@pytest.mark.parametrize("steps,shift,training_shift", [(1, 5., 1.), (10, 5., 1.), (50, 2., 3.), (1500, 5., 1.)])
def test_legacy_schedule_is_bitwise_unchanged(steps, shift, training_shift):
    from moss_jax.scheduler import schedule as old_schedule
    got = schedule(steps, scheduler="moss", shift=shift, training_shift=training_shift)
    expected = old_schedule(steps, shift, training_shift)
    for actual, old in zip(got, expected):
        np.testing.assert_array_equal(actual, old)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_first_and_terminal_steps_use_euler_and_do_not_read_history(sampler):
    sigmas = np.asarray([1, .8, .25, 0], np.float32)
    coeffs = step_coefficients(sigmas, sampler=sampler)
    for index in (0, len(coeffs)-1):
        c = coeffs[index]
        assert c.is_euler
        assert c.ode_velocity_coefficient == float(np.float32(sigmas[index+1]-sigmas[index]))
        assert c.state_coefficient == 1 and c.history_coefficient == 0
        state = jnp.asarray([[1., -2.]], dtype=jnp.float32)
        velocity = jnp.asarray([[.4, -.3]], dtype=jnp.float32)
        actual, denoised = sample_step(state, velocity, c, jnp.full_like(state, jnp.nan))
        np.testing.assert_array_equal(actual, state+np.float32(sigmas[index+1]-sigmas[index])*velocity)
        np.testing.assert_array_equal(denoised, state-sigmas[index]*velocity)


def test_euler_preserves_original_moss_arithmetic_through_all_steps():
    from moss_jax.scheduler import euler_step
    _, sigmas = schedule(13, scheduler="moss")
    x = jnp.asarray(np.random.default_rng(3).normal(size=(1, 5, 2)), jnp.float32)
    legacy = x
    history = None
    for c in step_coefficients(sigmas, sampler="euler"):
        v = jnp.sin(x)*jnp.float32(.2)
        legacy = euler_step(legacy, v, c.sigma, c.sigma_next)
        x, history = sample_step(x, v, c, history)
        np.testing.assert_array_equal(x, legacy)
    assert x.dtype == jnp.float32


def test_nonuniform_res_matches_independent_float64_equations():
    sigmas = np.asarray([1, .93, .61, .59, .2, .03, 0], np.float32)
    rng = np.random.default_rng(5)
    initial = rng.normal(size=(2, 3)).astype(np.float32)
    velocities = rng.normal(size=(len(sigmas)-1, 2, 3)).astype(np.float32)*.1
    x, history = jnp.asarray(initial), None
    expected, expected_history = initial.astype(np.float64), None
    for i, c in enumerate(step_coefficients(sigmas)):
        v = velocities[i]
        x, history = sample_step(x, v, c, history)
        expected, expected_history = _reference_res(expected, v, sigmas[i], sigmas[i+1],
                                                   sigmas[max(i-1, 0)], expected_history)
        np.testing.assert_allclose(x, expected, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(history, expected_history, rtol=3e-6, atol=3e-6)


def test_local_current_velocity_jacobian_is_exposed_coefficient_and_history_frozen():
    c = step_coefficients([1, .93, .61, .08, 0])[1]
    assert not c.is_euler
    state, velocity, history = (jnp.asarray(v, jnp.float32) for v in
                               ([.2, -.5, .8], [.1, .4, -.7], [.3, -.2, .1]))
    def run(v, previous):
        return sample_step(state, v, c, previous)[0]
    dv, dhistory = jax.jit(jax.jacrev(run, argnums=(0, 1)))(velocity, history)
    np.testing.assert_allclose(dv, np.eye(3)*c.ode_velocity_coefficient, atol=1e-7, rtol=2e-7)
    np.testing.assert_array_equal(dhistory, np.zeros((3, 3), np.float32))
    ordinary_dt = c.sigma_next-c.sigma
    assert abs(c.ode_velocity_coefficient-ordinary_dt) > .01
    # The public affine coefficients describe exactly the same frozen-history update.
    actual, _ = run(velocity, history), None
    affine = (c.state_coefficient*state+c.ode_velocity_coefficient*velocity
              +c.history_coefficient*history)
    np.testing.assert_allclose(actual, affine, rtol=2e-6, atol=2e-7)


def test_history_reset_makes_multiple_trajectories_independent():
    _, sigmas = schedule(10)
    coefficients = step_coefficients(sigmas)
    def trajectory(initial):
        x, previous = jnp.asarray(initial, jnp.float32), None
        for c in coefficients:
            v = jnp.tanh(x)*.1+c.sigma*.01
            x, previous = sample_step(x, v, c, previous)
        return x
    initial = np.asarray([.1, -.6, .2], np.float32)
    first = trajectory(initial)
    trajectory(initial+10)
    np.testing.assert_array_equal(trajectory(initial), first)


def test_adjacent_float32_sigmas_stay_finite_and_constant_velocity_is_consistent():
    values = [np.float32(1)]
    for _ in range(4):
        values.append(np.nextafter(values[-1], np.float32(0)))
    coefficients = step_coefficients([*values, 0])
    x, previous = jnp.asarray([1., -1.]), None
    for c in coefficients:
        assert np.isfinite([c.ode_velocity_coefficient, c.history_coefficient]).all()
        assert c.ode_velocity_coefficient < 0
        x, previous = sample_step(x, jnp.asarray([.25, -.25]), c, previous)
    np.testing.assert_allclose(x, [.75, -.75], rtol=1e-6, atol=1e-6)


def test_step_is_jittable_with_explicit_static_coefficients():
    c = step_coefficients([1, .7, .2, 0])[1]
    state = jnp.asarray([1., 2.])
    args = (state, state*.1, c, state*.9)
    got = jax.jit(sample_step, static_argnums=(2,))(*args)
    expected = sample_step(*args)
    for actual, reference in zip(got, expected):
        np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("option,value", [
    ("steps", 0), ("steps", True), ("steps", 2.5), ("total", 0),
    ("shift", 0), ("shift", float("nan")), ("shift", "5"),
    ("training_shift", -1), ("beta_alpha", 0), ("beta_beta", float("inf")),
    ("scheduler", "unknown"),
])
def test_invalid_schedule_values_fail_before_device_work(option, value):
    with pytest.raises(ValueError):
        schedule(**{option: value})


@pytest.mark.parametrize("sigmas", [
    [], [1], [[1, 0]], [1, .5], [1, 0, 0], [1, .5, .5, 0],
    [1, .4, .5, 0], [1, np.nan, 0], [1, np.inf, 0], [1, -.2, 0], [1.1, .4, 0],
])
def test_invalid_sigma_grids_rejected(sigmas):
    with pytest.raises(ValueError):
        step_coefficients(sigmas)


def test_unknown_sampler_and_missing_or_mismatched_history_rejected():
    with pytest.raises(ValueError, match="sampler"):
        step_coefficients([1, 0], sampler="automatic")
    c = step_coefficients([1, .8, .2, 0])[1]
    with pytest.raises(ValueError, match="previous_denoised"):
        sample_step(jnp.ones(2), jnp.ones(2), c)
    with pytest.raises(ValueError, match="same shape"):
        sample_step(jnp.ones(2), jnp.ones(2), c, jnp.ones(3))
    with pytest.raises(ValueError, match="same shape"):
        sample_step(jnp.ones(2), jnp.ones(3), c, jnp.ones(2))


def test_host_schedule_import_does_not_import_torch_or_initialize_jax():
    script = textwrap.dedent("""
        import sys
        from moss_art.sampling import schedule, step_coefficients
        _, sigmas = schedule()
        assert len(step_coefficients(sigmas)) == 50
        assert 'torch' not in sys.modules
        assert 'jax' not in sys.modules
    """)
    import sys
    completed = subprocess.run([sys.executable, "-c", script], text=True,
                               capture_output=True, timeout=30,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    assert completed.returncode == 0, completed.stderr
