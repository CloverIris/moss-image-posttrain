"""Numerical ART oracles, gradient isolation and exact update-budget tests."""
import json
import os
from pathlib import Path
import subprocess
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from moss_art.art import covariance, art_loss, prompt_windows, reduce_window_loss


class ARTTests(unittest.TestCase):
    def assertClose(self, a, b, atol=2e-6, rtol=2e-5):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)

    def test_k2_closed_form_unbiased_direction(self):
        x = np.array([[1., 2.], [5., 8.]], np.float32)
        got = covariance(x, [1., 3.], [0, 0])
        self.assertClose(got["group_covariance"], [[4., 6.]])
        self.assertClose(got["group_directions"], [[4., 6.]]/np.sqrt(26.))
        self.assertClose(got["group_guidance_magnitudes"], [1.])
        self.assertClose(got["group_reward_abs_means"], [1.])

    def test_full_group_numpy_oracle(self):
        rng = np.random.default_rng(77)
        x, r = rng.normal(size=(12, 2, 3)).astype("float32"), rng.normal(size=12).astype("float32")
        groups = np.array([1, 0, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0])
        got = covariance(x, r, groups)
        for group in range(3):
            rows = groups == group
            centered = r[rows].astype("float64")-np.mean(r[rows].astype("float64"))
            expected = np.sum(centered[:, None, None]*x[rows], axis=0)/(sum(rows)-1)
            self.assertClose(got["group_covariance"][group], expected)
            self.assertClose(got["group_directions"][group], expected/np.sqrt(np.mean(expected**2)))

    def test_zero_signal_exact_tiny_floor_and_singleton(self):
        x = np.arange(27, dtype=np.float32).reshape(9, 3)
        got = covariance(x, np.full(9, .1, np.float32), np.zeros(9, np.int32))
        self.assertFalse(np.asarray(got["group_active"]).any())
        self.assertClose(got["group_covariance"], 0.)
        self.assertClose(got["group_guidance_magnitudes"], 0.)
        singleton = covariance([[1., 2.]], [3.], [0])
        self.assertFalse(bool(singleton["group_active"][0]))
        tiny = covariance([[0., 0.], [1e-7, 1e-7]], [0., 1.], [0, 0])
        self.assertTrue(bool(tiny["group_active"][0]))
        self.assertClose(tiny["group_directions"], [[.01, .01]], atol=1e-8)
        aggregate = covariance([[0., 0.], [1e-7, 1e-7]], [0., 1.], [0, 0], direction_mode="normalized_aggregate")
        self.assertClose(aggregate["group_directions"], 1.)

    def test_closed_form_loss_sigma_and_frozen_inputs(self):
        v = jnp.zeros((2, 2))
        old, direction = jnp.zeros_like(v), jnp.ones_like(v)
        s, m, dt = jnp.array([.5, .25]), jnp.array([1., 0.]), jnp.array([-.1, -.1])
        def objective(current, previous, guidance, scale, sigma, delta):
            terms = art_loss(current, previous, sigma, guidance, scale, delta)
            return reduce_window_loss(terms["per_item_loss"], 2)
        loss = objective(v, old, direction, m, s, dt)
        self.assertClose(loss, .25)
        grads = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3, 4, 5)))(v, old, direction, m, s, dt)
        self.assertClose(grads[0], [[.25, .25], [0., 0.]])
        for gradient in grads[1:]:
            self.assertClose(gradient, 0.)
        covariance_gradient = jax.grad(lambda x: jnp.sum(covariance(x, [0., 1.], [0, 0], 1)["group_directions"]))(v)
        self.assertClose(covariance_gradient, 0.)

    def test_mask_outward_vs_restore_and_inactive(self):
        current = jnp.array([[-.25, -.25], [-2., -2.], [4., 4.]])
        got = art_loss(current, jnp.zeros_like(current), jnp.full(3, .5),
                       jnp.ones_like(current), jnp.array([1., 1., 0.]), jnp.full(3, -.1))
        self.assertEqual(np.asarray(got["masked"]).tolist(), [True, False, False])
        self.assertEqual(np.asarray(got["keep_mask"]).tolist(), [False, True, False])
        self.assertClose(got["per_item_loss"], [0., .5, 0.])
        anchor = art_loss(current, jnp.zeros_like(current), jnp.full(3, .5),
                         jnp.ones_like(current), jnp.array([1., 1., 0.]), jnp.full(3, -.1),
                         restore_anchor_mask=[False, False, True])
        self.assertClose(anchor["per_item_loss"], [0., .5, 8.])
        self.assertClose(reduce_window_loss(got["per_item_loss"], 3), 1./6.)

    def test_scheduler_reference_score_initial_and_interior(self):
        current = jnp.array([[.1, .1], [.1, .1]])
        got = art_loss(current, jnp.zeros_like(current), [1., .5], jnp.ones_like(current), [1., 1.], [-.1, -.1])
        expected_initial = .1**2*.1**2/(2*.8**2)
        expected_interior = (.1*(1+.5*.8**2))**2*.1**2/(2*(.8**2*.5/(1-.5)*.1))
        self.assertClose(got["movement_score"], [expected_initial, expected_interior], atol=1e-9)

    def test_windows_disjoint_balanced_and_denominator(self):
        groups = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        windows = prompt_windows(groups, [2, 5, 7], seed=93)
        self.assertEqual([len(w) for w in windows], [12, 12])
        rows = [set(w[:, 0]) for w in windows]
        self.assertFalse(rows[0] & rows[1])
        self.assertEqual(rows[0] | rows[1], set(range(8)))
        for window in windows:
            unique = np.unique(window[:, 0])
            self.assertEqual(np.bincount(groups[unique]).tolist(), [2, 2])
            for row in unique:
                self.assertEqual(window[window[:, 0] == row, 1].tolist(), [0, 1, 2])
        terms = jnp.array([0., 0., 2., 0., 4., 0.])
        whole = reduce_window_loss(terms, 6)
        divided = sum(reduce_window_loss(chunk, 6) for chunk in jnp.split(terms, 3))
        self.assertClose(whole, divided)
        self.assertClose(whole, 1.)
        with self.assertRaises(ValueError):
            prompt_windows([0, 0, 0], 2)

    def test_pairwise_ties_global_anchors_and_seed(self):
        x = np.arange(16, dtype=np.float32).reshape(8, 2)
        groups = np.repeat(np.arange(2), 4)
        scores = np.array([1., 1., 1., 1., 0., 1., 2., 3.])
        result = covariance(x, scores, groups, mode="pairwise", seed=19)
        self.assertClose(result["binary_rewards"][:4], .5)
        self.assertTrue(np.asarray(result["row_restore_anchor"][:4]).all())
        np.testing.assert_array_equal(result["pair_rows"], covariance(x, scores, groups, mode="pairwise", seed=19)["pair_rows"])
        for pair in np.asarray(result["pair_rows"]):
            self.assertClose(sum(np.asarray(result["binary_rewards"])[pair]), 1.)
        all_ties = covariance(x[:4], scores[:4], groups[:4], mode="pairwise")
        self.assertFalse(np.asarray(all_ties["row_trainable"]).any())
        external_active = covariance(x[:4], scores[:4], groups[:4], mode="pairwise", any_preference_active=True)
        self.assertTrue(np.asarray(external_active["row_restore_anchor"]).all())

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            covariance([[1., 2.], [3., 4.]], [1., np.nan], [0, 0])
        with self.assertRaises(ValueError):
            covariance([[1., 2.], [3., 4.]], [1., 2.], [0, 1], 1)
        with self.assertRaises(ValueError):
            art_loss(np.zeros((2, 2)), np.zeros((2, 2)), [1., 1.], np.ones((2, 2)), [1., 1.], [0., -.1])

    def test_upstream_torch_numerical_and_gradient_parity(self):
        python = os.environ.get("MOSS_ART_TORCH_PYTHON", "$CONDA_PREFIX/envs/torch_env/bin/python")
        upstream = Path(os.environ.get("MOSS_ART_UPSTREAM", "$HOME/DiffusionART/src"))
        if not Path(python).is_file() or not upstream.is_dir():
            self.skipTest("optional upstream Torch oracle environment unavailable")
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(8, 2, 3)).astype("float32")
        r = rng.normal(size=8).astype("float32")
        g = np.array([0, 1, 0, 1, 0, 1, 0, 1], np.int32)
        v = rng.normal(size=(8, 2, 3)).astype("float32")*.2
        old, direction = rng.normal(size=v.shape).astype("float32")*.05, rng.normal(size=v.shape).astype("float32")
        magnitude = np.array([1., .5, 0., 1., 2., 1., 0., .2], np.float32)
        s, dt = np.array([1., .8, .7, .6, .5, .4, .2, 0.], np.float32), np.full(8, -.05, np.float32)
        cov_options = [{}, {"reduction": "population"}, {"scale_mode": "reward_mad"},
                       {"scale_mode": "raw_covariance"}, {"scale_mode": "reward_std_normalized_covariance"},
                       {"direction_mode": "normalized_aggregate"},
                       {"zero_signal_mode": "threshold", "zero_signal_atol": 1e-4}]
        loss_options = [{}, {"corrective_mask_enabled": False}, {"loss_space": "velocity_uniform"},
                        {"loss_space": "x0"}, {"velocity_mask_metric": "fixed_mse"},
                        {"velocity_mask_metric": "noise_normalized_v"}, {"direction_normalization": "none"},
                        {"reference_mse_beta": .05}, {"loss_space": "x0", "target_projection_radius": .1},
                        {"mask_space": "ode_mean", "velocity_mask_metric": "fixed_mse"}]
        request = dict(upstream=str(upstream), x=x.tolist(), r=r.tolist(), g=g.tolist(), v=v.tolist(),
                       old=old.tolist(), direction=direction.tolist(), magnitude=magnitude.tolist(),
                       s=s.tolist(), dt=dt.tolist(), cov_options=cov_options, loss_options=loss_options)
        script = r"""
import json, sys, torch
q = json.load(sys.stdin)
sys.path.insert(0, q["upstream"])
from visual_rl.algorithms import art_scalar_covariance, art_loss
t = {k: torch.tensor(q[k], dtype=torch.long if k == "g" else torch.float32) for k in ("x","r","g","v","old","direction","magnitude","s","dt")}
def conv(d):
    return {k: v.detach().cpu().tolist() for k,v in d.items()}
cov=[]
for options in q["cov_options"]:
    kwargs={"zero_signal_mode":"exact", **options}
    cov.append(conv(art_scalar_covariance(t["x"],t["r"],t["g"],2,**kwargs)))
loss=[]
for options in q["loss_options"]:
    v=t["v"].clone().requires_grad_(True)
    result=art_loss(v,t["old"],t["s"],t["direction"],t["magnitude"],ode_velocity_coefficient=t["dt"],
                    state=t["x"],current_ode_mean=t["x"]+t["dt"][:,None,None]*v,
                    old_ode_mean=t["x"]+t["dt"][:,None,None]*t["old"],
                    reference_prediction=t["old"],restore_anchor_mask=torch.tensor([0,0,1,0,0,0,0,0]),**options)
    result["per_item_loss"].sum().backward()
    loss.append({"terms":conv(result),"gradient":v.grad.tolist()})
json.dump({"cov":cov,"loss":loss},sys.stdout)
"""
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        process = subprocess.run([python, "-c", script], input=json.dumps(request), text=True,
                                 capture_output=True, timeout=90, env=env)
        self.assertEqual(process.returncode, 0, process.stderr)
        oracle = json.loads(process.stdout)
        for options, expected in zip(cov_options, oracle["cov"]):
            got = covariance(x, r, g, 2, **options)
            for key, value in expected.items():
                self.assertClose(got[key], value)
        for options, expected in zip(loss_options, oracle["loss"]):
            def forward(current):
                return art_loss(current, old, s, direction, magnitude, dt, state=x,
                                current_ode_mean=x+dt[:,None,None]*current,
                                old_ode_mean=x+dt[:,None,None]*old, reference_prediction=old,
                                restore_anchor_mask=[False,False,True,False,False,False,False,False], **options)
            got = jax.jit(forward)(jnp.asarray(v))
            for key, value in expected["terms"].items():
                self.assertClose(got[key], value, atol=3e-6)
            gradient = jax.jit(jax.grad(lambda current: jnp.sum(forward(current)["per_item_loss"])))(jnp.asarray(v))
            self.assertClose(gradient, expected["gradient"], atol=3e-6)


if __name__ == "__main__":
    unittest.main()
