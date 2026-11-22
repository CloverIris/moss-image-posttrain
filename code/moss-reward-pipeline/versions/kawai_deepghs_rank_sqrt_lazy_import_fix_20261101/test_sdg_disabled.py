"""SDG-disabled rewards must omit defect claims and never launch SDG."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reward_fusion import fuse
import run_reward_pipeline as pipeline


def calibration():
    return dict(weights=dict(kawai=.4, deepghs=.4, hps=.2), output_offset=.5,
                output_gain=.08, parameters={k:dict(mean=1., scale=2.)
                for k in ('kawai', 'deepghs', 'hps')})


class DisabledTests(unittest.TestCase):
    def test_disabled_only_uses_three_models_and_keeps_scales(self):
        result=fuse(dict(kawai=3., deepghs=1., hps=-1.), calibration(),
                    0., sdg_enabled=False)
        self.assertAlmostEqual(result['reward'], .516)
        self.assertEqual(result['reward'], result['aesthetic_reward'])
        self.assertEqual(result['fusion_weights'], dict(kawai=.4, deepghs=.4, hps=.2))
        self.assertEqual(set(result['contributions']), {'kawai','deepghs','hps'})
        self.assertNotIn('artifact_penalty', result)

    def test_no_silent_missing_or_fabricated_components(self):
        raw=dict(kawai=3., deepghs=1., hps=-1.)
        with self.assertRaises(ValueError): fuse(raw, calibration(), 0.)
        with self.assertRaises(ValueError): fuse(raw, calibration(), 1., sdg_enabled=False)
        with self.assertRaises(ValueError):
            fuse(dict(raw, sdg_artifact=0.), calibration(), 0., sdg_enabled=False)
        for model in raw:
            with self.assertRaises(ValueError):
                fuse({k:v for k,v in raw.items() if k!=model}, calibration(),
                     0., sdg_enabled=False)

    @unittest.skipIf(os.name=='nt', 'fcntl archive lock is Linux-specific')
    def test_full_disabled_path_does_not_read_or_launch_sdg(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'one.png'; image.write_bytes(b'image fixture')
            pipeline.save(root/'calibration.json', calibration())
            pipeline.save(root/'batch.json', dict(stage_order=['deepghs','hps']))
            cfg=dict(sdg_enabled=False, artifact_weight=0., calibration=str(root/'calibration.json'),
                     batch_config=str(root/'batch.json'), allowed_archive_root=str(root),
                     archive_root=str(root/'archives'), controller_python='not launched',
                     batch_root=str(root), stage_timeout=1)
            request=dict(version=1, items=[dict(id='a',sample_id='p',image=str(image),prompt='original')])
            def batch(argv, log, timeout):
                pipeline.save(Path(argv[argv.index('--output')+1]),
                    dict(status='complete',items=[dict(id='a', deepghs=dict(status='ok',percentile=1.),
                                                       hps=dict(status='ok',mu=-1.))]))
            with patch.object(pipeline,'provenance',return_value={}), \
                 patch.object(pipeline,'run_program',side_effect=batch), \
                 patch.object(pipeline,'run_kawai_parallel',return_value=dict(items=[dict(id='a',kawai=3.)])), \
                 patch.object(pipeline,'run_sdg_parallel',side_effect=AssertionError('SDG launched')):
                result=pipeline.run(cfg,request,root/'result.json')
            detail=result['details'][0]
            self.assertAlmostEqual(result['scores'][0],.516)
            self.assertEqual(set(detail['raw_scores']),{'kawai','deepghs','hps'})
            self.assertNotIn('sdg',detail['model_details'])
            self.assertNotIn('artifact_count',detail)
            archive=Path(result['archive'])
            self.assertFalse((archive/'sdg.config.json').exists())
            self.assertFalse((archive/'sdg.json').exists())
            self.assertNotIn('artifact_image_fraction',json.loads((archive/'summary.json').read_text()))
            self.assertEqual(json.loads((archive/'manifest.json').read_text())['stage_order'],
                             ['deepghs','hps','kawai'])


if __name__=='__main__': unittest.main()
