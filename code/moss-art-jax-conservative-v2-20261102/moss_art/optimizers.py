"""LoRA optimizers. Muon follows KohakuMossTrainer's adapter grouping/scaling.

Apply once to globally averaged DP gradients. All master/momentum arithmetic,
including Newton-Schulz, is FP32 on the CPU coordinator; Kohaku uses BF16 for
NS only. No full frozen model or reconstructed delta-W is optimized.
"""
import math


def validate_optimizer(o):
    if o["name"] not in ("adamw", "muon"):
        raise ValueError("optimizer.name must be adamw or muon")
    if any(o[k] <= 0 for k in ("learning_rate", "epsilon", "max_grad_norm")) or o["weight_decay"] < 0:
        raise ValueError("Invalid optimizer rates/epsilon/gradient clipping")
    if len(o["betas"]) != 2 or any(type(b) not in (float, int) or not 0 <= b < 1 for b in o["betas"]):
        raise ValueError("optimizer.betas must contain two numbers in [0,1)")
    if not 0 <= o["momentum"] < 1 or o["ns_steps"] < 1 or o["rms_match"] <= 0:
        raise ValueError("Invalid Muon momentum/ns_steps/rms_match")
    warmup = o.get("warmup_steps", 0)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("optimizer.warmup_steps must be a nonnegative integer")


def learning_rate_at_step(o, completed_updates):
    """Rate for the next applied optimizer update (zero-based completed count).

    A four-step warmup uses 1/4, 2/4, 3/4 and 4/4 of the peak rate. Counts
    advance only when the optimizer transformation is actually updated.
    This stateless Python helper also serves training logs and inspection.
    """
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a nonnegative integer")
    warmup = o.get("warmup_steps", 0)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("optimizer.warmup_steps must be a nonnegative integer")
    fraction = min((completed_updates + 1) / warmup, 1.) if warmup else 1.
    return float(o["learning_rate"]) * fraction


def parameter_labels(params):
    """Rank-one factors and scalar/norm-like parameters retain AdamW."""
    import jax
    excluded = ("dora_scale", "rescale", "scalar", "lambda_layer", "alpha")
    def label(path, value):
        name = jax.tree_util.keystr(path)
        eligible = value.ndim == 2 and min(value.shape) > 1 and not any(x in name for x in excluded)
        return "muon" if eligible else "adamw"
    return jax.tree_util.tree_map_with_path(label, params)


def _orthogonalizer(steps):
    import jax
    import jax.numpy as jnp
    @jax.jit
    def orthogonalize(g):
        x = g.astype(jnp.float32)
        transpose = x.shape[0] > x.shape[1]
        if transpose:
            x = x.T
        # Transpose first: for LoRA the Gram matrix is rank x rank.
        x = x / (jnp.linalg.norm(x) + jnp.float32(1e-7))
        def iteration(_, x):
            a = x @ x.T
            b = -4.775 * a + 2.0315 * (a @ a)
            return 3.4445 * x + b @ x
        x = jax.lax.fori_loop(0, steps, iteration, x)
        return x.T if transpose else x
    return orthogonalize


def build_optimizer(o):
    import jax
    import jax.numpy as jnp
    import optax
    validate_optimizer(o)
    rate = o["learning_rate"]
    warmup = o.get("warmup_steps", 0)
    if warmup:
        peak_rate = o["learning_rate"]
        def rate(count):
            fraction = jnp.minimum((jnp.asarray(count, jnp.float32) + 1.) / warmup, 1.)
            return jnp.asarray(peak_rate, jnp.float32) * fraction
    adam = optax.adamw(rate, b1=o["betas"][0], b2=o["betas"][1],
                       eps=o["epsilon"], weight_decay=o["weight_decay"])
    if o["name"] == "adamw":
        inner = adam
    else:
        orthogonalize = _orthogonalizer(o["ns_steps"])
        def scale(updates, state, params=None):
            del params
            updates = jax.tree.map(lambda g: orthogonalize(g) *
                (o["rms_match"] * math.sqrt(max(g.shape))), updates)
            return updates, state
        # Keep the legacy constant-rate transformation and state tree exactly
        # unchanged when warmup is disabled.
        rate_scale = (optax.scale_by_schedule(lambda count: -rate(count))
                      if warmup else optax.scale(-o["learning_rate"]))
        muon = optax.chain(
            # Unnormalized v = momentum*v + g; Nesterov u = g + momentum*v.
            optax.trace(decay=o["momentum"], nesterov=o["nesterov"], accumulator_dtype=jnp.float32),
            optax.GradientTransformation(optax.init_empty_state, scale),
            optax.add_decayed_weights(o["weight_decay"]),
            rate_scale)
        inner = optax.partition({"muon": muon, "adamw": adam}, parameter_labels)
    # Clipping precedes momentum/orthogonalization, and follows DP reduction.
    return optax.chain(optax.clip_by_global_norm(o["max_grad_norm"]), inner)
