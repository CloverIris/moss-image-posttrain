import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import score_kawai as worker


class KawaiValidationTests(unittest.TestCase):
    def test_pinned_class_contract_rejects_reordered_or_missing_labels(self):
        labels = {str(i):v for i,v in enumerate(worker.CLASS_LABELS)}
        self.assertEqual(worker.validate_class_contract(labels, worker.CLASS_LABELS, worker.CLASS_WEIGHTS), labels)
        for bad in ({**labels, '0':'C_tier'}, {k:v for k,v in labels.items() if k != '4'}, {'LABEL_0':'SS_tier'}):
            with self.assertRaises(ValueError):
                worker.validate_class_contract(bad, worker.CLASS_LABELS, worker.CLASS_WEIGHTS)
        with self.assertRaises(ValueError):
            worker.validate_class_contract(labels, worker.CLASS_LABELS[::-1], worker.CLASS_WEIGHTS)
        with self.assertRaises(ValueError):
            worker.validate_class_contract(labels, worker.CLASS_LABELS, [1,1,.5,-.5,-1])

    def test_items_keep_identity_and_require_image_prompt_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'a.png').write_bytes(b'not decoded during validation')
            source = root/'input.json'
            row = dict(id='p0_r0', prompt_id=0, image='a.png')
            source.write_text(json.dumps(dict(items=[row])))
            result = worker.load_items(source)
            self.assertEqual(result[0]['id'], row['id'])
            self.assertEqual(result[0]['image'], str((root/'a.png').resolve()))
            for rows in ([row,row], [{**row, 'image':'missing.png'}], [{**row,'prompt_id':None}], [{**row,'id':1}]):
                source.write_text(json.dumps(dict(items=rows)))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    worker.load_items(source)

    def test_settings_reject_reserved_gpu_dtype_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'model'; model.mkdir()
            (model/'model.safetensors').write_bytes(b'fixture')
            (model/'config.json').write_text(json.dumps(dict(id2label={str(i):v for i,v in enumerate(worker.CLASS_LABELS)})))
            cfg = dict(input='input.json',output='output.json',model_path='model',device=0)
            settings = worker.validate_settings(cfg, root)
            self.assertEqual(settings['batch_size'], 2)
            self.assertEqual(settings['dtype'], 'float32')
            for change in (dict(device=4),dict(device=True),dict(dtype='float16'),dict(batch_size=0),dict(output='input.json')):
                with self.assertRaises(ValueError):
                    worker.validate_settings({**cfg,**change}, root)

    def test_import_does_not_load_torch_or_transformers(self):
        code = "import score_kawai,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        p = subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__ == '__main__':
    unittest.main()
