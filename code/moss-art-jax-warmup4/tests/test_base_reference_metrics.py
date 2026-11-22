"""Numerical diagnostics against autodiff; no model/GPU required."""
import math
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from moss_art.art import art_loss
from moss_art.base_reference_metrics import reference_strength_metrics
from moss_art.trainer import aggregate_gradients


class CotangentTests(unittest.TestCase):
    def test_exact_decomposition_matches_autodiff_across_spaces_and_anchors(self):
        current=jnp.asarray([[[.5,.4],[.3,.2]],[[.15,-.2],[.1,.3]],[[.7,.8],[.9,1.]]])
        old=jnp.full_like(current,.1)
        reference=jnp.zeros_like(current)
        beta=.2
        for space in ('velocity_linear_sigma','velocity_uniform','x0'):
            for anchor in ('cached','self','calibrated'):
                def objective(v):
                    return art_loss(v,old,jnp.asarray([.7,.4,.3]),jnp.ones_like(v),jnp.asarray([.2,.1,.3]),
                        state=jnp.ones_like(v),loss_space=space,anchor_mode=anchor,
                        velocity_mask_metric='fixed_mse',corrective_mask_enabled=False,
                        reference_prediction=reference,reference_mse_beta=beta,
                        valid_mask=jnp.asarray([True,True,False]))
                terms=objective(current)
                policy=jax.grad(lambda v:objective(v)['per_item_policy_loss'].sum())(current)
                base=jax.grad(lambda v:beta*objective(v)['per_item_reference_mse'].sum())(current)
                total=jax.grad(lambda v:objective(v)['per_item_loss'].sum())(current)
                with self.subTest(space=space,anchor=anchor):
                    np.testing.assert_allclose(total,policy+base,rtol=1e-5,atol=1e-7)
                    np.testing.assert_allclose(terms['per_item_policy_output_grad_sq'],jnp.sum(policy**2,axis=(1,2)),rtol=1e-5,atol=1e-7)
                    np.testing.assert_allclose(terms['per_item_reference_output_grad_sq'],jnp.sum(base**2,axis=(1,2)),rtol=1e-5,atol=1e-7)
                    np.testing.assert_allclose(terms['per_item_output_grad_dot'],jnp.sum(policy*base,axis=(1,2)),rtol=1e-5,atol=1e-7)
                    self.assertEqual(float(terms['per_item_reference_output_grad_sq'][2]),0.)

    def test_art_mask_does_not_mask_base_regularizer(self):
        v=jnp.full((1,2,2),-.1)
        terms=art_loss(v,jnp.zeros_like(v),jnp.asarray([.5]),jnp.ones_like(v),jnp.asarray([.2]),
                       velocity_mask_metric='fixed_mse',movement_mask_threshold=1e-5,
                       reference_prediction=jnp.zeros_like(v),reference_mse_beta=.01)
        self.assertFalse(bool(terms['keep_mask'][0]))
        self.assertEqual(float(terms['per_item_policy_output_grad_sq'][0]),0.)
        self.assertGreater(float(terms['per_item_reference_output_grad_sq'][0]),0.)

    def test_zero_lora_prediction_has_zero_base_cotangent(self):
        v=jnp.full((1,2,2),.1)
        terms=art_loss(v,v,jnp.asarray([.5]),jnp.ones_like(v),jnp.asarray([.2]),
                       velocity_mask_metric='fixed_mse',reference_prediction=v,reference_mse_beta=.01)
        self.assertEqual(float(terms['per_item_reference_mse'][0]),0.)
        self.assertEqual(float(terms['per_item_reference_output_grad_sq'][0]),0.)


class MetricTests(unittest.TestCase):
    def test_loss_ratio_and_output_competition_are_distinct(self):
        metrics=reference_strength_metrics(2.,5.,.1,4,
            dict(policy_sq=4.,reference_sq=16.,dot=-8.,policy_zero_count=1,reference_zero_count=0))
        self.assertEqual(metrics['base_reference_loss_to_art_ratio'],.25)
        self.assertEqual(metrics['base_reference_loss_fraction'],.2)
        self.assertEqual(metrics['base_reference_output_grad_to_art_ratio'],2.)
        self.assertEqual(metrics['base_reference_output_grad_cosine'],-1.)
        self.assertEqual(metrics['art_zero_output_grad_fraction'],.25)
        self.assertAlmostEqual(metrics['base_reference_output_grad_cancellation_fraction'],2/3)

    def test_zero_art_emits_defined_flags_instead_of_inf_or_fake_zero_ratio(self):
        metrics=reference_strength_metrics(0.,3.,.01,2,
            dict(policy_sq=0.,reference_sq=1.,dot=0.,policy_zero_count=2,reference_zero_count=0))
        self.assertEqual(metrics['base_reference_loss_ratio_defined'],0.)
        self.assertEqual(metrics['base_reference_output_grad_ratio_defined'],0.)
        self.assertEqual(metrics['base_reference_only_output_grad'],1.)
        self.assertNotIn('base_reference_loss_to_art_ratio',metrics)
        self.assertNotIn('base_reference_output_grad_to_art_ratio',metrics)
        self.assertNotIn('base_reference_output_grad_cosine',metrics)
        self.assertTrue(all(math.isfinite(x) for x in metrics.values()))

    def test_global_denominator_and_partition_invariance(self):
        def result(n,p,b,d,pz,bz):
            return dict(gradients={'a':np.asarray([n],np.float32)},denominator=n,
                        loss_sum=3*n,masked=0,active=n,policy_loss_sum=2*n,
                        reference_mse_sum=10*n,reference_beta=.1,reference_seconds=.2,
                        policy_output_grad_sq_sum=p,reference_output_grad_sq_sum=b,
                        output_grad_dot_sum=d,policy_output_grad_zero_count=pz,reference_output_grad_zero_count=bz)
        rows=[result(2,4.,1.,-1.,1,0),result(3,9.,4.,-2.,1,1)]
        _,split=aggregate_gradients(rows)
        _,merged=aggregate_gradients([result(5,13.,5.,-3.,2,1)])
        for key in ('art_loss','base_reference_loss','base_reference_loss_to_art_ratio',
                    'art_output_grad_norm','base_reference_output_grad_norm','base_reference_output_grad_cosine',
                    'art_zero_output_grad_fraction','base_reference_zero_output_grad_fraction'):
            with self.subTest(key=key):self.assertEqual(split[key],merged[key])


if __name__=='__main__':unittest.main()
