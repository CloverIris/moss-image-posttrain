"""Checkpoint continuation is bitwise-equivalent on CPU; corruption is refused."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
import optax
from moss_art.checkpoint import (CheckpointError, config_fingerprint,
                                 save_checkpoint, load_checkpoint)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="moss-art-checkpoint-test-")
        self.root = Path(self.temporary.name)
        self.params = {"block.0": {"A": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)*.01,
                                   "B": jnp.zeros((3, 2), jnp.float32)}}
        self.optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adamw(.001))
        self.state = self.optimizer.init(self.params)
        self.config = {"rank": 2, "alpha": 2, "model": "test", "learning_rate": .001}
        self.metadata = {"config_fingerprint": config_fingerprint(self.config), "step": 2,
                         "round_index": 1, "prompt_cursor": 4, "rng": [1, 55],
                         "lora": {"rank": 2}, "data_sha256": "fixture"}

    def tearDown(self):
        self.temporary.cleanup()

    def assertTreesEqual(self, first, second):
        leaves_a, tree_a = jax.tree.flatten(first)
        leaves_b, tree_b = jax.tree.flatten(second)
        self.assertEqual(tree_a, tree_b)
        self.assertEqual(len(leaves_a), len(leaves_b))
        for a, b in zip(leaves_a, leaves_b):
            np.testing.assert_array_equal(a, b)

    def update(self, params, state, step):
        gradient = jax.grad(lambda p: sum(jnp.mean((leaf-(step+1)*.1)**2)
                                         for leaf in jax.tree.leaves(p)))(params)
        updates, state = self.optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state

    def test_save_resume_exact_optimizer_trajectory(self):
        params, state = self.params, self.state
        for step in range(2):
            params, state = self.update(params, state, step)
        path = save_checkpoint(self.root/"step_000002", params, state, self.metadata)
        loaded, loaded_state, metadata = load_checkpoint(path, expected_metadata={
            "config_fingerprint": config_fingerprint(self.config), "lora": {"rank": 2}})
        self.assertTreesEqual(params, loaded)
        self.assertTreesEqual(state, loaded_state)
        self.assertEqual(metadata, self.metadata)
        for step in range(2, 5):
            params, state = self.update(params, state, step)
            loaded, loaded_state = self.update(loaded, loaded_state, step)
        self.assertTreesEqual(params, loaded)
        self.assertTreesEqual(state, loaded_state)
        self.assertEqual(sorted(p.name for p in path.iterdir()), ["COMMITTED", "arrays.npz", "manifest.json"])
        with np.load(path/"arrays.npz", allow_pickle=False) as payload:
            self.assertTrue(all(not payload[key].dtype.hasobject for key in payload.files))

    def test_metadata_fingerprint_and_incompatible_resume(self):
        self.assertEqual(config_fingerprint({"a": 1, "b": 2}), config_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(config_fingerprint({"rank": 2}), config_fingerprint({"rank": 4}))
        path = save_checkpoint(self.root/"step_000002", self.params, self.state, self.metadata)
        with self.assertRaisesRegex(CheckpointError, "incompatible resume"):
            load_checkpoint(path, expected_metadata={"lora": {"rank": 4}})
        with self.assertRaisesRegex(CheckpointError, "missing resume"):
            load_checkpoint(path, expected_metadata={"model_sha256": "required"})
        with self.assertRaises(FileExistsError):
            save_checkpoint(path, self.params, self.state, self.metadata)

    def test_payload_corruption_and_incomplete_directory_rejected(self):
        path = save_checkpoint(self.root/"step_000002", self.params, self.state, self.metadata)
        payload = path/"arrays.npz"
        data = bytearray(payload.read_bytes())
        data[len(data)//2] ^= 1
        payload.write_bytes(data)
        with self.assertRaisesRegex(CheckpointError, "payload checksum"):
            load_checkpoint(path)
        incomplete = self.root/"incomplete"
        incomplete.mkdir()
        (incomplete/"manifest.json").write_text("{}")
        with self.assertRaises(CheckpointError):
            load_checkpoint(incomplete)

    def test_manifest_corruption_rejected(self):
        path = save_checkpoint(self.root/"step_000002", self.params, self.state, self.metadata)
        manifest = path/"manifest.json"
        manifest.write_bytes(manifest.read_bytes()+b" ")
        with self.assertRaisesRegex(CheckpointError, "manifest checksum"):
            load_checkpoint(path)

    def test_failed_atomic_publish_cleans_temporary(self):
        with mock.patch("moss_art.checkpoint.os.rename", side_effect=OSError("simulated power loss")):
            with self.assertRaisesRegex(OSError, "simulated power loss"):
                save_checkpoint(self.root/"step_000002", self.params, self.state, self.metadata)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bfloat16_rng_and_nested_types_roundtrip(self):
        params = {"bf16": jnp.ones((2, 3), jnp.bfloat16), "rng": jax.random.key(18),
                  "nested": [None, True, 5, (jnp.array(7, jnp.int32),)]}
        path = save_checkpoint(self.root/"step_000002", params, self.state, {"rng_array": np.array([1, 2], np.uint32)})
        loaded, state, metadata = load_checkpoint(path)
        self.assertEqual(str(loaded["bf16"].dtype), "bfloat16")
        np.testing.assert_array_equal(loaded["bf16"], params["bf16"])
        np.testing.assert_array_equal(jax.random.key_data(loaded["rng"]), jax.random.key_data(params["rng"]))
        self.assertEqual(metadata["rng_array"], [1, 2])
        self.assertEqual(loaded["nested"][:3], [None, True, 5])
        self.assertIsInstance(loaded["nested"][3], tuple)
        self.assertTreesEqual(state, self.state)

    def test_unknown_namedtuple_descriptor_rejected(self):
        path = save_checkpoint(self.root/"step_000002", self.params, self.state, self.metadata)
        manifest = json.loads((path/"manifest.json").read_bytes())
        def poison(spec):
            if spec["kind"] == "namedtuple":
                spec["class"] = "os.system"
                return True
            for child in spec.get("items", []):
                if isinstance(child, dict) and poison(child):
                    return True
            return False
        self.assertTrue(poison(manifest["optimizer_state"]))
        content = json.dumps(manifest).encode()
        (path/"manifest.json").write_bytes(content)
        (path/"COMMITTED").write_text(hashlib.sha256(content).hexdigest()+"\n")
        with self.assertRaisesRegex(CheckpointError, "unknown optimizer"):
            load_checkpoint(path)

    def test_object_arrays_nonfinite_arrays_and_metadata_rejected(self):
        with self.assertRaises(TypeError):
            save_checkpoint(self.root/"object", {"bad": np.array([object()], object)}, (), {})
        with self.assertRaises(ValueError):
            save_checkpoint(self.root/"nan", {"bad": np.array([np.nan], np.float32)}, (), {})
        with self.assertRaises(ValueError):
            save_checkpoint(self.root/"nan_metadata", {}, (), {"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
