import importlib.util
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE_ROOT = Path(__file__).parent
if not (MODULE_ROOT/'optimizers.py').is_file():
    MODULE_ROOT = MODULE_ROOT.parent/'moss_art'
NEW = module_from_file('warmup_optimizer_under_test', MODULE_ROOT/'optimizers.py')
LEGACY_PATH = Path(os.environ.get('MOSS_ORIGINAL_OPTIMIZER',
    '$CODE_ROOT/moss-art-jax/moss_art/optimizers.py'))
CHECKPOINT_PATH = LEGACY_PATH.with_name('checkpoint.py')


def options(name='adamw', warmup=4):
    return dict(name=name, momentum=.95, nesterov=True, ns_steps=5, rms_match=.2,
                learning_rate=1e-4, warmup_steps=warmup, betas=[.9, .999],
                epsilon=1e-8, weight_decay=0., max_grad_norm=1.)


def assert_tree_equal(test, a, b):
    test.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class WarmupTests(unittest.TestCase):
    def test_python_rate_first_four_then_constant(self):
        o = options()
        np.testing.assert_allclose([NEW.learning_rate_at_step(o, n) for n in range(6)],
                                  [2.5e-5, 5e-5, 7.5e-5, 1e-4, 1e-4, 1e-4], atol=1e-14)
        self.assertEqual(NEW.learning_rate_at_step(options(warmup=1), 0), 1e-4)
        self.assertEqual(NEW.learning_rate_at_step(options(warmup=0), 0), 1e-4)

    def test_actual_adam_updates_match_rate(self):
        o = options(); tx = NEW.build_optimizer(o)
        params = {'bias': jnp.array([0.], jnp.float32)}
        state = tx.init(params)
        update = jax.jit(tx.update)
        for count in range(6):
            updates, state = update({'bias': jnp.ones_like(params['bias'])}, state, params)
            # Constant gradients isolate the rate; Adam bias correction yields ~1.
            np.testing.assert_allclose(-np.asarray(updates['bias']),
                [NEW.learning_rate_at_step(o, count)], rtol=2e-5, atol=1e-10)
            params = optax.apply_updates(params, updates)

    def test_muon_and_adam_partition_share_warmup(self):
        o = options(name='muon'); tx = NEW.build_optimizer(o)
        constant = NEW.build_optimizer(options(name='muon', warmup=0))
        params = {'matrix': jnp.array([[1.,2.],[3.,4.]], jnp.float32),
                  'bias': jnp.array([1.,2.], jnp.float32)}
        gradients = jax.tree.map(jnp.ones_like, params)
        state, base_state = tx.init(params), constant.init(params)
        for count in range(6):
            updates, state = tx.update(gradients, state, params)
            reference, base_state = constant.update(gradients, base_state, params)
            fraction = min((count+1)/4, 1.)
            for actual, base in zip(jax.tree.leaves(updates), jax.tree.leaves(reference)):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(base)*fraction,
                                           rtol=1e-6, atol=1e-10)

    @unittest.skipUnless(LEGACY_PATH.is_file(), 'Needs read-only original optimizer source')
    def test_disabled_warmup_preserves_exact_legacy_state_and_updates(self):
        legacy = module_from_file('legacy_optimizer_reference', LEGACY_PATH)
        for name in ('adamw', 'muon'):
            with self.subTest(name=name):
                old = options(name=name, warmup=0); old.pop('warmup_steps')
                new = dict(old, warmup_steps=0)
                a, b = NEW.build_optimizer(new), legacy.build_optimizer(old)
                params = {'matrix': jnp.array([[1.,2.],[3.,4.]], jnp.float32),
                          'bias': jnp.array([1.,2.], jnp.float32)}
                sa, sb = a.init(params), b.init(params)
                assert_tree_equal(self, sa, sb)
                for count in range(4):
                    gradient = jax.tree.map(lambda p: jnp.ones_like(p)*(count+1)/10, params)
                    ua, sa = a.update(gradient, sa, params)
                    ub, sb = b.update(gradient, sb, params)
                    assert_tree_equal(self, ua, ub)
                    assert_tree_equal(self, sa, sb)
                # The compatibility default also works for older dict callers.
                c = NEW.build_optimizer(old)
                assert_tree_equal(self, c.init(params), a.init(params))

    @unittest.skipUnless(CHECKPOINT_PATH.is_file(), 'Needs original checkpoint implementation')
    def test_schedule_state_roundtrip_matches_continuous_training(self):
        checkpoint = module_from_file('original_checkpoint_roundtrip', CHECKPOINT_PATH)
        for name in ('adamw', 'muon'):
            with self.subTest(name=name):
                o = options(name=name); tx = NEW.build_optimizer(o)
                params = {'matrix': jnp.array([[1.,2.],[3.,4.]], jnp.float32),
                          'bias': jnp.array([1.,2.], jnp.float32)}
                state = tx.init(params)
                for count in range(2):
                    grad = jax.tree.map(lambda p: jnp.ones_like(p)*(count+1)/10, params)
                    update, state = tx.update(grad, state, params)
                    params = optax.apply_updates(params, update)
                # Exercise the production pickle-free checkpoint serializer,
                # including its optimizer-state namedtuple allowlist.
                with tempfile.TemporaryDirectory() as folder:
                    metadata = dict(update=2, optimizer_updates=2)
                    file = Path(folder)/'step2'
                    checkpoint.save_checkpoint(file, params, state, metadata)
                    restored_params, restored_state, recovered = checkpoint.load_checkpoint(
                        file, expected_metadata=metadata)
                    self.assertEqual(recovered['optimizer_updates'], 2)
                resumed = NEW.build_optimizer(o)
                for count in range(2, 6):
                    grad = jax.tree.map(lambda p: jnp.ones_like(p)*(count+1)/10, params)
                    update, state = tx.update(grad, state, params)
                    other, restored_state = resumed.update(grad, restored_state, restored_params)
                    params = optax.apply_updates(params, update)
                    restored_params = optax.apply_updates(restored_params, other)
                    assert_tree_equal(self, params, restored_params)
                    assert_tree_equal(self, state, restored_state)

    def test_microbatch_computation_does_not_advance_schedule(self):
        o = options(); tx = NEW.build_optimizer(o)
        params = {'bias': jnp.array([0.], jnp.float32)}
        state = tx.init(params)
        snapshot = jax.tree.map(lambda value: np.array(value, copy=True), state)
        accumulated = jnp.zeros_like(params['bias'])
        for _ in range(8):
            accumulated += jnp.ones_like(params['bias'])/8
            self.assertEqual(NEW.learning_rate_at_step(o, 0), 2.5e-5)
        assert_tree_equal(self, state, snapshot)
        updates, _ = tx.update({'bias':accumulated}, state, params)
        np.testing.assert_allclose(-np.asarray(updates['bias']), [2.5e-5], rtol=2e-5)

    def test_warmup_validation(self):
        for value in (-1, True, 4., '4', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NEW.validate_optimizer(options(warmup=value))
        for value in (-1, True, 1., '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NEW.learning_rate_at_step(options(), value)

    @unittest.skipUnless(LEGACY_PATH.parent.is_dir(), 'Needs read-only config dependencies')
    def test_config_schema_accepts_warmup_and_default_is_zero(self):
        sys.path.insert(0, str(LEGACY_PATH.parent.parent))
        cfg = module_from_file('moss_art.config_warmup_under_test',
                               MODULE_ROOT/'config.py')
        self.assertEqual(cfg.DEFAULTS['optimizer']['warmup_steps'], 0)
        merged = cfg._merge(copy.deepcopy(cfg.DEFAULTS), {'optimizer':{'warmup_steps':4}})
        cfg._validate_types(merged, cfg.DEFAULTS)
        NEW.validate_optimizer(merged['optimizer'])
        for value in (True, 4., '4'):
            actual = cfg._merge(copy.deepcopy(cfg.DEFAULTS), {'optimizer':{'warmup_steps':value}})
            with self.subTest(value=value), self.assertRaises(ValueError):
                cfg._validate_types(actual, cfg.DEFAULTS)


if __name__ == '__main__':
    unittest.main()
