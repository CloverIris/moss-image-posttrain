"""CPU ART anchor-calibration oracles; no FP8/image-quality claims."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from moss_art.art import art_loss, covariance


class AnchorLossTests(unittest.TestCase):
    def close(self, actual, expected, **kwargs):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                   rtol=kwargs.pop("rtol", 2e-6),
                                   atol=kwargs.pop("atol", 2e-6), **kwargs)

    def loss(self, v, old, G, *, anchor_mode="calibrated", **options):
        n = np.shape(v)[0]
        return art_loss(v, old, options.pop("sigma", np.ones(n, np.float32)),
                        G, options.pop("magnitude", np.ones(n, np.float32)),
                        direction_normalization="none", velocity_mask_metric="fixed_mse",
                        movement_mask_threshold=options.pop("threshold", 1.),
                        anchor_mode=anchor_mode, **options)

    def test_offset_removed_and_self_anchor_gradient_preserved(self):
        old = jnp.asarray([[1.3]])
        v = jnp.asarray([[1.]])
        G = jnp.asarray([[.1]])
        gradients = {}
        for mode in ("cached", "self"):
            gradients[mode] = jax.jit(jax.grad(lambda current:
                self.loss(current, old, G, anchor_mode=mode)["per_item_loss"].sum()))(v)
        self.close(gradients["cached"], [[-.4]])
        self.close(gradients["self"], [[.2]])
        frozen_grad = jax.grad(lambda current:
            self.loss(current, v, G)["per_item_loss"].sum())(v)
        self.close(gradients["self"], frozen_grad)
        h = 1e-3
        fd = (self.loss(v+h, v, G)["per_item_loss"].sum()
              - self.loss(v-h, v, G)["per_item_loss"].sum())/(2*h)
        self.close(fd, .2, rtol=1e-4)

    def test_b_keeps_theta0_restoration(self):
        theta0 = jnp.asarray([[1.]])
        G = jnp.asarray([[.1]])
        grad_a = jax.grad(lambda v:
            self.loss(v, jnp.asarray([[1.3]]), G, anchor_mode="self")["per_item_loss"].sum())(theta0)
        theta1 = theta0 - .5*grad_a
        self.close(theta1, [[.9]])
        grad_b = jax.grad(lambda v:
            self.loss(v, theta0, G)["per_item_loss"].sum())(theta1)
        self.close(grad_b, [[0.]])
        wrong = jax.grad(lambda v:
            self.loss(v, theta0, G, anchor_mode="self")["per_item_loss"].sum())(theta1)
        self.close(wrong, [[.2]])
        self.close(theta0, [[1.]])

    def test_target_and_velocity_mask_share_calibrated_old(self):
        v = jnp.asarray([[1.]])
        G = jnp.asarray([[.4]])
        raw = self.loss(v, jnp.asarray([[1.3]]), G, anchor_mode="cached", threshold=.01)
        calibrated = self.loss(v, v, G, threshold=.01)
        self.assertTrue(bool(raw["masked"][0]))
        self.assertFalse(bool(calibrated["masked"][0]))
        self.close(calibrated["target"], [[.6]])
        self.close(calibrated["movement_score"], [0.])

    def test_ode_mean_cannot_override_calibrated_reference(self):
        v = jnp.asarray([[.8], [1.]])
        old = jnp.ones_like(v)
        G = jnp.full_like(v, .4)
        opts = dict(mask_space="ode_mean", dt=[-.1, -.2], threshold=.0001,
                    current_ode_mean=jnp.full_like(v, 300.),
                    old_ode_mean=jnp.full_like(v, -700.))
        got = self.loss(v, old, G, **opts)
        self.close(got["movement_score"], [.0004, 0.])
        self.close(got["outward_alignment"], [.0004, 0.])
        np.testing.assert_array_equal(got["masked"], [True, False])
        self_got = self.loss(v, old, G, anchor_mode="self", **opts)
        self.close(self_got["movement_score"], [0., 0.])
        np.testing.assert_array_equal(self_got["masked"], [False, False])

    def test_scheduler_metric_retained_with_calibrated_reference(self):
        v, old = jnp.full((2, 3), .1), jnp.zeros((2, 3))
        got = art_loss(v, old, [1., .5], jnp.ones_like(v), [1., 1.], [-.1, -.1],
                       anchor_mode="calibrated")
        expected = [.1**2*.1**2/(2*.8**2),
                    (.1*(1+.5*.8**2))**2*.1**2/(2*(.8**2*.1))]
        self.close(got["movement_score"], expected, atol=1e-8)

    def test_inputs_frozen_and_residual_is_fp32(self):
        v = jnp.asarray([[1., 2.]], jnp.bfloat16)
        old = jnp.asarray([[1., 2.]])
        G, sigma, mag = jnp.asarray([[.1, -.2]]), jnp.asarray([.5]), jnp.asarray([1.])
        def objective(v, old, G, sigma, mag):
            return self.loss(v, old, G, sigma=sigma, magnitude=mag)["per_item_loss"].sum()
        grads = jax.jit(jax.grad(objective, argnums=(0,1,2,3,4)))(v, old, G, sigma, mag)
        self.close(grads[0], [[.05, -.1]], rtol=.005)
        for frozen in grads[1:]:
            self.close(frozen, 0.)
        terms = self.loss(v, old, G, sigma=sigma)
        self.assertEqual(terms["per_item_loss"].dtype, jnp.float32)
        self.assertEqual(terms["target"].dtype, jnp.float32)

    def test_self_small_reward_signal_survives_large_old_value(self):
        v = jnp.asarray([[1e6]], jnp.float32)
        G = jnp.asarray([[1e-4]], jnp.float32)
        terms = self.loss(v, v, G, anchor_mode="self")
        self.close(terms["per_item_loss"], [1e-8], atol=1e-12)
        grad = jax.jit(jax.grad(lambda current:
            self.loss(current, v, G, anchor_mode="self")["per_item_loss"].sum()))(v)
        self.close(grad, [[2e-4]], atol=1e-9)

    def test_valid_count_keeps_inactive_masked_rows_and_microbatch_equivalence(self):
        v = jnp.ones((5, 1))
        old = jnp.asarray([[1.], [1.], [0.], [1.], [1.]])
        G = jnp.asarray([[.1], [0.], [-2.], [.5], [.2]])
        valid = jnp.asarray([True, True, True, False, True])
        def compute(v, old, G, valid):
            return self.loss(v, old, G, valid_mask=valid, threshold=.5)
        whole = compute(v, old, G, valid)
        self.assertEqual(int(whole["valid_count"]), 4)
        self.assertEqual(int(whole["kept_count"]), 2)
        np.testing.assert_array_equal(whole["masked"], [False, False, True, False, False])
        self.close(whole["per_item_loss"].sum()/whole["valid_count"], .0125)
        objective = lambda values: compute(values, old, G, valid)["per_item_loss"].sum()/4
        whole_grad = jax.grad(objective)(v)
        for size in (1, 2, 5):
            chunks, num, den = [], 0., 0
            for start in range(0, len(v), size):
                sl = slice(start, start+size)
                out = compute(v[sl], old[sl], G[sl], valid[sl])
                num += out["per_item_loss"].sum()
                den += int(out["valid_count"])
                chunks.append(jax.grad(lambda values:
                    compute(values, old[sl], G[sl], valid[sl])["per_item_loss"].sum())(v[sl]))
            self.assertEqual(den, 4)
            self.close(num/den, .0125)
            self.close(jnp.concatenate(chunks)/den, whole_grad)

    def test_all_empty_zero_grad_counts_and_restoration_count(self):
        v = jnp.ones((2, 1))
        for G, old, valid in (
            (jnp.zeros_like(v), v, [True, True]),
            (jnp.full_like(v, -2.), jnp.zeros_like(v), [True, True]),
            (jnp.ones_like(v), v, [False, False]),
        ):
            terms = self.loss(v, old, G, valid_mask=valid, threshold=.5)
            self.assertEqual(int(terms["kept_count"]), 0)
            self.close(terms["per_item_loss"], 0.)
            grad = jax.grad(lambda current: self.loss(current, old, G,
                valid_mask=valid, threshold=.5)["per_item_loss"].sum())(v)
            self.close(grad, 0.)
        restored = self.loss(v, v, jnp.zeros_like(v), restore_anchor_mask=[True, True])
        self.assertEqual(int(restored["kept_count"]), 2)
        self.close(restored["per_item_loss"], 0.)
        # Zero net gradient with retained restoration rows still counts as an
        # optimizer update. The controller, not this loss kernel, owns stepping.

    def test_reference_regularizer_prevents_false_empty_window(self):
        v = jnp.ones((2, 1))
        terms = self.loss(v, v, jnp.zeros_like(v), valid_mask=[True, False],
                          reference_prediction=jnp.zeros_like(v), reference_mse_beta=.1)
        self.assertEqual(int(terms["policy_kept_count"]), 0)
        self.assertEqual(int(terms["kept_count"]), 1)
        self.close(terms["per_item_loss"], [.1, 0.])

    def test_pairwise_tie_restoration_unchanged(self):
        x = np.arange(16, dtype=np.float32).reshape(8, 2)
        cov = covariance(x, [1.,1.,1.,1.,0.,1.,2.,3.], np.repeat([0,1], 4),
                         mode="pairwise", seed=19)
        v = jnp.ones((8, 2))
        terms = art_loss(v, jnp.zeros_like(v), jnp.ones(8),
                         cov["row_directions"], cov["row_guidance_magnitudes"],
                         velocity_mask_metric="fixed_mse", anchor_mode="calibrated",
                         restore_anchor_mask=cov["row_restore_anchor"])
        np.testing.assert_array_equal(terms["keep_mask"][:4], [True]*4)
        self.close(terms["per_item_loss"][:4], 1.)

    def test_no_mismatch_matches_raw_and_x0_self_gradient(self):
        old = jnp.asarray([[.2, .1], [1., -.5]])
        v = old + .01
        G = jnp.asarray([[.1, -.2], [.3, -.1]])
        for space in ("velocity_linear_sigma", "velocity_uniform", "x0"):
            opts = dict(sigma=[.8, .4], loss_space=space, state=jnp.ones_like(v))
            raw = self.loss(v, old, G, anchor_mode="cached", **opts)
            calibrated = self.loss(v, old, G, **opts)
            for key in ("per_item_loss", "movement_score", "outward_alignment", "masked", "keep_mask"):
                self.close(raw[key], calibrated[key])
            raw_grad = jax.grad(lambda current: self.loss(current, old, G,
                anchor_mode="cached", **opts)["per_item_loss"].sum())(v)
            cal_grad = jax.grad(lambda current: self.loss(current, old, G,
                **opts)["per_item_loss"].sum())(v)
            self.close(raw_grad, cal_grad)
        x0_self = jax.grad(lambda current: self.loss(current, old, G,
            anchor_mode="self", sigma=[.8, .4], loss_space="x0",
            state=jnp.ones_like(v))["per_item_loss"].sum())(old)
        self.close(x0_self, np.asarray(G)*np.array([.8, .4])[:,None])

    def test_invalid_anchor_mode_and_valid_shape_rejected(self):
        v = jnp.ones((2, 1))
        with self.assertRaisesRegex(ValueError, "anchor_mode"):
            self.loss(v, v, v, anchor_mode="latest")
        with self.assertRaisesRegex(ValueError, "valid_mask"):
            self.loss(v, v, v, valid_mask=[True])


if __name__ == "__main__":
    unittest.main()
