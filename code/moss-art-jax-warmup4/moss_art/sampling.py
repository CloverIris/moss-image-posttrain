"""MOSS flow schedules and deterministic one-evaluation-per-step solvers.

The beta grid and RES multistep equations follow Comfy-Org/ComfyUI commit
b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c:
  comfy/samplers.py: beta_scheduler
  comfy/model_sampling.py: ModelSamplingDiscreteFlow, CONST, time_snr_shift
  comfy/k_diffusion/sampling.py: res_multistep (eta=0, cfg_pp=False)
Source identity is recorded in outputs/sampling_source_audit/upstream_comfy_identity.json.
MOSS uses CONST/flow x0 = x - sigma * velocity; RES still uses t=-log(sigma),
not a logit/Flux or stochastic RF variant. Sampling history belongs to one
trajectory and must start as None for each new image.

The host schedule and coefficient construction require NumPy/SciPy only.
JAX is imported by sample_step so importing this module does not select a device.
"""
from dataclasses import dataclass
from numbers import Real
import operator

import numpy as np

COMFYUI_REVISION = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
SAMPLERS = ("euler", "res_multistep")
SCHEDULERS = ("moss", "beta")


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def schedule(steps=50, *, scheduler="beta", shift=5.0, training_shift=1.0,
             total=1000, beta_alpha=0.6, beta_beta=0.6):
    """Return (model timesteps, descending sigmas including terminal zero).

    beta selects quantized beta-distribution percentiles from the ascending
    shifted flow training grid, [1/total, ..., 1], exactly once. Consecutive
    duplicate grid indices are removed, so len(sigmas)-1 is the actual NFE.
    training_shift must remain 1 for beta, which never performs the legacy
    nearest-training-grid snap. Timesteps are sigma*total;
    the default MOSS model uses total=1000.

    moss delegates to the unchanged vendored scheduler, preserving old grids.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; expected {SCHEDULERS}")
    steps = _positive_integer(steps, "steps")
    total = _positive_integer(total, "total")
    shift = _positive_real(shift, "shift")
    training_shift = _positive_real(training_shift, "training_shift")
    beta_alpha = _positive_real(beta_alpha, "beta_alpha")
    beta_beta = _positive_real(beta_beta, "beta_beta")
    if scheduler == "moss":
        from moss_jax.scheduler import schedule as legacy_schedule
        return legacy_schedule(steps, shift, training_shift, total)

    if training_shift != 1.0:
        raise ValueError("beta scheduler requires training_shift=1; use shift for the flow grid")
    from scipy.stats import beta
    percentiles = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False)
    indices = np.rint(beta.ppf(percentiles, beta_alpha, beta_beta) * (total - 1)).astype(np.int64)
    indices = indices[np.r_[True, indices[1:] != indices[:-1]]]
    grid = np.arange(1, total + 1, dtype=np.float32) / np.float32(total)
    if shift != 1.0:
        factor = np.float32(shift)
        grid = factor * grid / (np.float32(1.0) + (factor - np.float32(1.0)) * grid)
    selected = grid[indices]
    sigmas = np.r_[selected, np.float32(0.0)].astype(np.float32)
    _validate_sigmas(sigmas)
    return selected * np.float32(total), sigmas


@dataclass(frozen=True)
class StepCoefficients:
    """One host-precomputed step, with a frozen-history local velocity slope.

    next_state = state_coefficient*state + ode_velocity_coefficient*velocity
                 + history_coefficient*previous_denoised
    mathematically. sample_step retains the upstream factored update instead
    of expanding it, avoiding extra cancellation/rounding. Scalars are FP32
    values represented as Python floats, suitable for JSON or static JIT args.
    """
    sigma: float
    sigma_next: float
    state_coefficient: float
    ode_velocity_coefficient: float
    history_coefficient: float
    is_euler: bool
    decay: float = 1.0
    h: float = 0.0
    b1: float = 0.0
    b2: float = 0.0


def _validate_sigmas(sigmas):
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if sigmas.ndim != 1 or sigmas.size < 2:
        raise ValueError("sigmas must be a one-dimensional grid with at least one step and terminal zero")
    if not np.isfinite(sigmas).all():
        raise ValueError("sigmas must be finite")
    if sigmas[-1] != 0 or np.any(sigmas[:-1] <= 0) or sigmas[0] > 1:
        raise ValueError("MOSS flow sigmas must be in (0, 1], followed by terminal zero")
    if np.any(sigmas[1:] >= sigmas[:-1]):
        raise ValueError("sigmas must be strictly decreasing in float32; duplicate steps are invalid")
    return sigmas


def _phi_minus_h(h):
    """Stable phi1(-h), phi2(-h), using the upstream FP32 path normally."""
    f = np.float32
    z = f(-h)
    if abs(float(z)) < 1e-3:
        # Avoid subtracting nearly equal FP32 values in (phi1(z)-1)/z.
        # These analytic limits also cover adjacent representable sigmas.
        zd = float(z)
        p1 = f(1 + zd/2 + zd*zd/6 + zd**3/24 + zd**4/120)
        p2 = f(0.5 + zd/6 + zd*zd/24 + zd**3/120 + zd**4/720)
    else:
        p1 = f(np.expm1(z) / z)
        p2 = f((p1 - f(1)) / z)
    return p1, p2


def step_coefficients(sigmas, *, sampler="res_multistep"):
    """Precompute steps and their current-velocity ART coefficients.

    RES uses Euler at the first step (no history) and the terminal step
    (sigma_next=0). Interior steps use h=log(sigma/sigma_next),
    c2=-h_previous/h, b2=phi2(-h)/c2, b1=phi1(-h)-b2.
    Thus the local derivative with respect to current raw velocity is
    -sigma*h*b1, with state and previous denoised prediction held fixed.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler {sampler!r}; expected {SAMPLERS}")
    sigmas = _validate_sigmas(sigmas)
    f = np.float32
    result = []
    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        if sampler == "euler" or i == 0 or sigma_next == 0:
            result.append(StepCoefficients(float(sigma), float(sigma_next),
                                          1.0, float(f(sigma_next-sigma)), 0.0, True))
            continue
        t, t_next, t_previous = -np.log(sigma), -np.log(sigma_next), -np.log(sigmas[i-1])
        h = f(t_next - t)
        c2 = f((t_previous - t) / h)
        p1, p2 = _phi_minus_h(h)
        b2 = f(p2 / c2)
        b1 = f(p1 - b2)
        decay = f(np.exp(-h))
        current_weight = f(h*b1)
        history_weight = f(h*b2)
        slope = f(-sigma*current_weight)
        scalars = (h, c2, b1, b2, decay, slope, history_weight)
        if not np.isfinite(scalars).all() or h <= 0 or slope >= 0:
            raise ValueError("RES coefficients are not finite/decreasing for this float32 sigma grid")
        result.append(StepCoefficients(
            float(sigma), float(sigma_next), float(f(decay+current_weight)),
            float(slope), float(history_weight), False, float(decay), float(h),
            float(b1), float(b2)))
    return tuple(result)


def sample_step(state, velocity, coefficients, previous_denoised=None):
    """Return (next_state, current_denoised) using FP32 flow predictions.

    previous_denoised is frozen for local ART replay: gradients do not flow
    through another sampling step. The current velocity and state retain their
    normal gradients. No model evaluation occurs here; callers make one per
    step and pass this step's returned denoised value into the next step.
    """
    import jax
    import jax.numpy as jnp
    if not isinstance(coefficients, StepCoefficients):
        raise TypeError("coefficients must be a StepCoefficients from step_coefficients")
    state = jnp.asarray(state, dtype=jnp.float32)
    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have the same shape")
    denoised = state - jnp.float32(coefficients.sigma) * velocity
    if coefficients.is_euler:
        # Preserve the original MOSS Euler arithmetic, including FP32 dt.
        updated = state + jnp.float32(coefficients.ode_velocity_coefficient) * velocity
    else:
        if previous_denoised is None:
            raise ValueError("RES interior steps require previous_denoised from the same trajectory")
        history = jax.lax.stop_gradient(jnp.asarray(previous_denoised, dtype=jnp.float32))
        if history.shape != state.shape:
            raise ValueError("previous_denoised and state must have the same shape")
        updated = (jnp.float32(coefficients.decay) * state
                   + jnp.float32(coefficients.h)
                   * (jnp.float32(coefficients.b1) * denoised
                      + jnp.float32(coefficients.b2) * history))
    return updated, denoised
