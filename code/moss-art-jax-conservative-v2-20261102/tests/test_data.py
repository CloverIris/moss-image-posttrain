import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from moss_art.data import (CACHE_VERSION, PromptCache, PromptRecord, atomic_json,
                           load_prompts, model_identity, sha256_file, text_key)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_jsonl_validation_and_metadata(self):
        file = self.root / "prompts.jsonl"
        file.write_text('\n' + json.dumps({"id": "a", "prompt": "a cat", "source": "manual"}) + '\n')
        records = load_prompts(file, default_negative_prompt="blur")
        self.assertEqual(records[0].negative_prompt, "blur")
        self.assertEqual(records[0].metadata, {"source": "manual"})
        file.write_text('{"id":"a","prompt":"cat"}\n{"id":"a","prompt":"dog"}\n')
        with self.assertRaisesRegex(ValueError, "unique"):
            load_prompts(file)
        for invalid in ('[]', '{"prompt":42}', '{"prompt":" "}', '{"prompt":"cat","negative_prompt":null}', '{"prompt":"cat","metadata":[]}'):
            file.write_text(invalid)
            with self.assertRaises(ValueError):
                load_prompts(file)

    def test_sample_id_alias(self):
        file = self.root / "prompts.jsonl"
        file.write_text('{"sample_id":"stable-id","prompt":"cat"}\n')
        self.assertEqual(load_prompts(file)[0].id, 'stable-id')
        file.write_text('{"id":"a","sample_id":"b","prompt":"cat"}\n')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            load_prompts(file)

    def _model(self):
        model = self.root / "model"
        (model / "text_encoder").mkdir(parents=True)
        (model / "tokenizer").mkdir()
        (model / "text_encoder/config.json").write_text('{}')
        (model / "tokenizer/tokenizer.json").write_text('{}')
        header = b'{}'
        (model / "text_encoder/model.safetensors").write_bytes(struct.pack('<Q', len(header)) + header)
        return model

    def _cache(self, model):
        cache = self.root / "cache"
        cache.mkdir()
        manifest = {"schema_version": CACHE_VERSION, "encoder": "moss-native-jax-qwen3.5",
                    "model_identity": model_identity(model), "entries": {}}
        # Synthetic small features exist only inside this unit test.
        for text, length in (("cat", 3), ("", 1)):
            file = cache / (text_key(text) + '.npz')
            embedding = np.arange(length * 4, dtype=np.float32).reshape(1, length, 4)
            np.savez(file, embedding=embedding, attention_mask=np.ones((1, length), bool))
            manifest['entries'][text_key(text)] = {"file": file.name, "shape": list(embedding.shape), "sha256": sha256_file(file)}
        atomic_json(cache / 'manifest.json', manifest)
        return cache

    def test_variable_length_and_negative_sharing(self):
        model = self._model()
        cache = self._cache(model)
        reader = PromptCache(cache, model)
        value = reader.load(PromptRecord('a', 'cat'))
        self.assertEqual(value['positive'].shape, (1, 3, 4))
        self.assertEqual(value['negative'].shape, (1, 1, 4))
        self.assertEqual(text_key(' cat '), text_key('cat'))
        with self.assertRaises(KeyError):
            reader.load_text('uncached')

    def test_identity_change_and_corruption_rejected(self):
        model = self._model()
        cache = self._cache(model)
        (model / 'tokenizer/tokenizer.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            PromptCache(cache, model)
        file = cache / (text_key('cat') + '.npz')
        file.write_bytes(file.read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'Corrupted'):
            PromptCache(cache).load_text('cat')

    def test_cache_cannot_escape_directory(self):
        model = self._model()
        cache = self._cache(model)
        manifest = json.loads((cache / 'manifest.json').read_text())
        manifest['entries'][text_key('cat')]['file'] = '../outside.npz'
        atomic_json(cache / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            PromptCache(cache).load_text('cat')


if __name__ == '__main__':
    unittest.main()
