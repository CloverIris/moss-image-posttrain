import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from moss_art.vllm_contract import format_prompt
from moss_art.vllm_qwen import make_model_view, unpack_features


class AdapterTests(unittest.TestCase):
    def test_native_token_embed_all_tokens_and_guards(self):
        import torch
        from types import SimpleNamespace
        from moss_art.vllm_qwen import unpack_token_embeddings
        hidden = torch.arange(3 * 4096, dtype=torch.float32).reshape(3, 4096).to(torch.bfloat16)
        def output(value=hidden, ids=None):
            return SimpleNamespace(prompt_token_ids=[1, 2, 3] if ids is None else ids,
                                   outputs=SimpleNamespace(data=value))
        actual = unpack_token_embeddings(output(), [1, 2, 3])
        np.testing.assert_array_equal(actual[0], hidden.float().numpy())
        self.assertEqual(actual.shape, (1, 3, 4096))
        for invalid in (output(ids=[1, 3, 2]), output(hidden[:1]), output(hidden.double()),
                        output(torch.zeros_like(hidden)), output(torch.full_like(hidden, float('nan')))):
            with self.assertRaises(ValueError):
                unpack_token_embeddings(invalid, [1, 2, 3])

    def test_template_preserves_moss_body(self):
        body = 'prompt:\nA white cat.\nnegative prompt:\ntext'
        self.assertEqual(format_prompt(body), '<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n<think>\n')

    def test_model_view_is_symlinked_and_source_config_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, processor, view = root/'model', root/'processor', root/'view'
            (model/'text_encoder').mkdir(parents=True)
            (model/'tokenizer').mkdir()
            processor.mkdir()
            config = {'model_type': 'qwen3_5', 'architectures': ['Qwen3_5Model'],
                      'tie_word_embeddings': False, 'text_config': {'hidden_size':4096, 'num_hidden_layers':32, 'tie_word_embeddings':False}}
            (model/'text_encoder/config.json').write_text(json.dumps(config))
            for path in [model/'text_encoder/a.safetensors', model/'tokenizer/tokenizer.json',
                         processor/'preprocessor_config.json', processor/'video_preprocessor_config.json']:
                path.write_text('{}')
            with patch('moss_art.vllm_qwen.model_identity', return_value={'fingerprint':'one'}):
                make_model_view(model, view, processor)
                make_model_view(model, view, processor)
            self.assertTrue((view/'a.safetensors').is_symlink())
            self.assertEqual(json.loads((model/'text_encoder/config.json').read_text()), config)
            saved = json.loads((view/'config.json').read_text())
            self.assertTrue(saved['text_config']['tie_word_embeddings'])
            with patch('moss_art.vllm_qwen.model_identity', return_value={'fingerprint':'two'}):
                with self.assertRaises(ValueError):
                    make_model_view(model, view, processor)

    def test_feature_layout_token_alignment_dtype_and_finiteness(self):
        import torch
        payload = {'hidden_states': torch.ones((3,1,4096),dtype=torch.bfloat16),
                   'token_ids': torch.tensor([1,2,3])}
        result = unpack_features(payload, [1,2,3])
        self.assertEqual(result.shape, (1,3,4096))
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaises(ValueError):
            unpack_features(payload, [1,3,2])
        with self.assertRaises(ValueError):
            unpack_features({**payload,'hidden_states':payload['hidden_states'].float()}, [1,2,3])
        with self.assertRaisesRegex(ValueError, 'All-zero'):
            unpack_features({**payload,'hidden_states':torch.zeros_like(payload['hidden_states'])},[1,2,3])
        payload['hidden_states'][0,0,0] = float('nan')
        with self.assertRaises(ValueError):
            unpack_features(payload,[1,2,3])


if __name__ == '__main__':
    unittest.main()
