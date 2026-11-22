import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from moss_art.config import DEFAULTS
from moss_art.vllm_prepare import prepare_cache


class CacheWriterTests(unittest.TestCase):
    def test_batch_dedup_drop_and_resume_without_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            prompts=root/'prompts.jsonl'
            prompts.write_text('\n'.join(json.dumps(x) for x in [
                {'id':'a','prompt':'white cat'}, {'id':'b','prompt':'white cat'},
                {'id':'long','prompt':'word '*100},
            ]))
            c=copy.deepcopy(DEFAULTS)
            c['preparation']['backend']='vllm'
            c['data'].update(prompts=str(prompts),cache=str(root/'cache'),context_length=32,short_context_length=32)
            identity={'fingerprint':'same','weight_fingerprint_method':'header+size+mtime_ns+path'}
            class FakeEncoder:
                metadata={'version':'fixed'}
                def __init__(self,*args,**kwargs):pass
                def encode_token_lists(self,lists):
                    return [np.ones((1,len(ids),4096),np.float32) for ids in lists]
            patches=[
                patch('moss_art.vllm_prepare.model_identity',return_value=identity),
                patch('moss_art.data.model_identity',return_value=identity),
                patch('moss_art.vllm_qwen.tokenizer_for',return_value=object()),
                patch('moss_art.vllm_qwen.text_tokens',side_effect=lambda tok,text:np.arange(13+len(text.split()),dtype=np.int32)),
                patch('moss_art.vllm_prepare.select_devices'),
            ]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches:stack.enter_context(p)
                with patch('moss_art.vllm_qwen.VllmPrefill',FakeEncoder):
                    result=prepare_cache(c)
                self.assertEqual(result['records'],2)
                self.assertEqual(result['dropped'],1)
                self.assertEqual(result['unique_texts'],2)
                self.assertEqual(result['encoded_now'],2)
                with patch('moss_art.vllm_qwen.VllmPrefill',side_effect=AssertionError('must not load model')):
                    result=prepare_cache(c)
                self.assertEqual(result['encoded_now'],0)
                from moss_art.data import PromptCache,load_prompts
                cache=PromptCache(c['data']['cache'])
                self.assertEqual(cache.load(load_prompts(prompts)[0])['positive'].shape,(1,15,4096))
                for entry in cache.manifest['entries'].values():
                    with np.load(Path(c['data']['cache'])/entry['file']) as saved:
                        self.assertEqual(saved['input_ids'].dtype,np.int32)
                        self.assertTrue(saved['attention_mask'].all())


if __name__=='__main__':unittest.main()
