import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from moss_art.assets import component_identity, experiment_identity


class AssetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model'
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        (self.cache / 'manifest.json').write_text('{"revision":1}')
        for component in ('transformer', 'vae'):
            directory = self.model / component
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text('{"layers":1}')
            header = json.dumps({'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            (directory / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', 1.0))
        self.config = {'model': {'kind': 'moss', 'path': str(self.model)}, 'data': {'cache': str(self.cache)}}

    def tearDown(self):
        self.temporary.cleanup()

    def test_stable_identity_and_scope(self):
        first = experiment_identity(self.config)
        self.assertEqual(first, experiment_identity(self.config))
        self.assertIn('not full tensor payload', first['weight_hash_scope'])
        self.assertEqual(set(first['components']), {'transformer', 'vae'})
        self.assertIn('text_cache_manifest', first)

    def test_weight_replacement_and_configuration_change(self):
        before = experiment_identity(self.config)
        weight = self.model / 'transformer/model.safetensors'
        stat = weight.stat()
        payload = weight.read_bytes()
        weight.write_bytes(payload[:-4] + struct.pack('<f', 2.0))
        # Guarantee a distinct filesystem modification timestamp without sleep.
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(before['fingerprint'], experiment_identity(self.config)['fingerprint'])
        before = experiment_identity(self.config)
        (self.model / 'vae/config.json').write_text('{"layers":2}')
        self.assertNotEqual(before['fingerprint'], experiment_identity(self.config)['fingerprint'])

    def test_rebuilt_cache_invalidates_identity(self):
        before = experiment_identity(self.config)
        (self.cache / 'manifest.json').write_text('{"revision":2,"max_sequence_length":32}')
        self.assertNotEqual(before['fingerprint'], experiment_identity(self.config)['fingerprint'])

    def test_missing_and_corrupt_weights_fail_early(self):
        weight = self.model / 'vae/model.safetensors'
        weight.write_bytes(struct.pack('<Q', 1000000))
        with self.assertRaisesRegex(ValueError, 'header extent'):
            experiment_identity(self.config)
        weight.unlink()
        with self.assertRaises(FileNotFoundError):
            component_identity(self.model, 'vae')

    def test_siglip_assets_and_tokenizer_bound(self):
        reward = self.model / 'transformer'
        (reward / 'tokenizer.json').write_text('{"revision":1}')
        self.config['reward'] = {'kind': 'siglip', 'model_path': str(reward)}
        before = experiment_identity(self.config)
        self.assertIn('reward_assets', before)
        (reward / 'tokenizer.json').write_text('{"revision":2}')
        self.assertNotEqual(before['fingerprint'], experiment_identity(self.config)['fingerprint'])

    def test_tiny_requires_no_model_or_cache(self):
        tiny = {'model': {'kind': 'tiny', 'path': '/does/not/exist'}, 'data': {'cache': '/also/missing'}}
        identity = experiment_identity(tiny)
        self.assertEqual(identity['model_kind'], 'tiny')
        self.assertNotIn('components', identity)
        self.assertNotIn('text_cache_manifest', identity)
        self.assertIn('source_files', identity)


if __name__ == '__main__':
    unittest.main()
