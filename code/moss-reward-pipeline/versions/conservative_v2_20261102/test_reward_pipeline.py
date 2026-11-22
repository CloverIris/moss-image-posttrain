import copy
import tempfile
from pathlib import Path
import unittest

from reward_fusion import fuse,index_items
from run_reward_pipeline import canonical_items


class RewardPipelineTests(unittest.TestCase):
    def calibration(self):
        return dict(weights=dict(kawai=.4,deepghs=.4,hps=.2),output_offset=.5,output_gain=.08,
                    parameters={k:dict(mean=1.,scale=2.) for k in ('kawai','deepghs','hps')})

    def test_fusion_contributions_reconstruct_exact_reward(self):
        result=fuse(dict(kawai=3.,deepghs=1.,hps=-1.,sdg_artifact=-.05),self.calibration())
        self.assertAlmostEqual(result['aesthetic_reward'],.516)
        self.assertAlmostEqual(result['reward'],.466)
        self.assertAlmostEqual(result['output_offset']+sum(result['contributions'].values()),result['reward'])

    def test_offline_reweight_keeps_raw_and_fixed_scales(self):
        raw=dict(kawai=3.,deepghs=1.,hps=-1.,sdg_artifact=-.05)
        original=copy.deepcopy(raw)
        score=fuse(raw,self.calibration(),0.,weights=dict(kawai=0.,deepghs=1.,hps=0.))
        self.assertEqual(score['reward'],.5)
        self.assertEqual(raw,original)

    def test_invalid_component_is_never_zero_filled(self):
        for bad in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                fuse(dict(kawai=bad,deepghs=1,hps=1,sdg_artifact=0),self.calibration())

    def test_model_results_are_joined_by_id(self):
        rows=[dict(id='b',value=2),dict(id='a',value=1)]
        result=index_items(rows,['a','b'],'test')
        self.assertEqual(result['a']['value'],1)
        with self.assertRaises(ValueError):index_items(rows,['a'],'test')
        with self.assertRaises(ValueError):index_items(rows+rows,['a','b'],'test')

    def test_raw_prompt_preserved_and_image_content_in_identity(self):
        with tempfile.TemporaryDirectory() as root:
            image=Path(root)/'image.png';image.write_bytes(b'one')
            text='prompt:\n{a: b}\nNegative Prompt:\nfoo'
            request=dict(version=1,items=[dict(prompt=text,image=str(image),prompt_index=0,rollout_id=0)])
            first=canonical_items(request)[0]
            self.assertEqual(first['prompt'],text)
            self.assertEqual(first['prompt_id'],'0')
            image.write_bytes(b'two')
            second=canonical_items(request)[0]
            self.assertNotEqual(first['image_sha256'],second['image_sha256'])


if __name__=='__main__':unittest.main()
