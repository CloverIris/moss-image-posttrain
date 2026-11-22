"""CPU-only optional-head fusion and durable-pipeline integration tests."""
import builtins
import copy
import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reward_fusion as fusion
import run_reward_pipeline as pipeline


def calibration():
    return dict(weights=dict(kawai=.4, deepghs=.4, hps=.2), output_offset=.5,
                output_gain=.08, parameters={name: dict(mean=1., scale=2.)
                                             for name in ('kawai', 'deepghs', 'hps')})


class NoKawaiReads(dict):
    def __getitem__(self, key):
        if key == 'kawai':
            raise AssertionError('Disabled Kawai calibration was accessed')
        return super().__getitem__(key)


@contextmanager
def archive_lock_test_support():
    # Production uses Linux fcntl. Windows unit tests replace only that OS lock,
    # without changing production locking or importing an accelerator library.
    if os.name == 'nt':
        fake = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
        with patch.dict(sys.modules, {'fcntl': fake}):
            yield
    else:
        yield


class OptionalFusionTests(unittest.TestCase):
    def test_deepghs_hps_reconstruct_without_kawai_or_sdg(self):
        raw = dict(deepghs=3., hps=-1.)
        cfg = calibration()
        original = copy.deepcopy((raw, cfg))
        result = fusion.fuse(raw, cfg, 0., sdg_enabled=False, kawai_enabled=False,
                             weights=dict(deepghs=2/3, hps=1/3))
        self.assertEqual(result['enabled_models'], ['deepghs', 'hps'])
        self.assertFalse(result['kawai_enabled'])
        self.assertFalse(result['sdg_enabled'])
        self.assertAlmostEqual(result['reward'], .5 + .08 / 3)
        self.assertEqual(result['reward'], result['aesthetic_reward'])
        self.assertEqual(result['reward'], .5 + math.fsum(result['contributions'].values()))
        for field in ('normalized_scores', 'contributions', 'fusion_weights'):
            self.assertEqual(set(result[field]), {'deepghs', 'hps'})
        self.assertNotIn('artifact_penalty', result)
        self.assertEqual((raw, cfg), original)

    def test_disabled_default_preserves_relative_frozen_weights(self):
        cfg = calibration()
        cfg['weights'] = NoKawaiReads(cfg['weights'])
        cfg['parameters'] = NoKawaiReads(cfg['parameters'])
        result = fusion.fuse(dict(deepghs=3., hps=-1.), cfg, 0.,
                             kawai_enabled=False, sdg_enabled=False)
        self.assertAlmostEqual(result['fusion_weights']['deepghs'], 2/3)
        self.assertAlmostEqual(result['fusion_weights']['hps'], 1/3)
        self.assertAlmostEqual(result['reward'], .5 + .08 / 3)

    def test_explicit_two_head_weights_need_no_kawai_calibration(self):
        cfg = calibration()
        cfg['weights'].pop('kawai')
        cfg['parameters'].pop('kawai')
        result = fusion.fuse(dict(deepghs=3., hps=-1.), cfg, 0.,
                             kawai_enabled=False, sdg_enabled=False,
                             weights=dict(deepghs=2/3, hps=1/3))
        self.assertAlmostEqual(result['reward'], .5 + .08 / 3)

    def test_kawai_and_sdg_switches_are_independent(self):
        raw = dict(deepghs=3., hps=-1., sdg_artifact=-.05)
        result = fusion.fuse(raw, calibration(), kawai_enabled=False,
                             weights=dict(deepghs=2/3, hps=1/3))
        self.assertAlmostEqual(result['reward'], .5 + .08 / 3 - .05)
        self.assertEqual(set(result['contributions']), {'deepghs', 'hps', 'sdg_artifact'})
        self.assertEqual(result['enabled_models'], ['deepghs', 'hps'])
        legacy = fusion.fuse(dict(kawai=3., deepghs=1., hps=-1., sdg_artifact=-.05), calibration())
        self.assertAlmostEqual(legacy['reward'], .466)
        self.assertEqual(legacy['enabled_models'], ['kawai', 'deepghs', 'hps'])

    def test_missing_enabled_scores_never_become_zero(self):
        for model in ('deepghs', 'hps'):
            for invalid in (None, float('nan'), float('inf'), float('-inf')):
                raw = dict(deepghs=3., hps=-1.)
                raw[model] = invalid
                with self.subTest(model=model, invalid=invalid), self.assertRaises(ValueError):
                    fusion.fuse(raw, calibration(), 0., kawai_enabled=False, sdg_enabled=False)
            raw = dict(deepghs=3., hps=-1.)
            raw.pop(model)
            with self.assertRaises(ValueError):
                fusion.fuse(raw, calibration(), 0., kawai_enabled=False, sdg_enabled=False)
        with self.assertRaises(ValueError):
            fusion.fuse(dict(kawai=0., deepghs=3., hps=-1.), calibration(), 0.,
                        kawai_enabled=False, sdg_enabled=False)

    def test_enabled_sets_and_flag_types_are_strict(self):
        for value in (0, 1, 'false', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fusion.fuse(dict(deepghs=3., hps=-1.), calibration(), 0.,
                            kawai_enabled=value, sdg_enabled=False)
        for weights in (dict(kawai=0., deepghs=2/3, hps=1/3),
                        dict(deepghs=1.), dict(deepghs=1., hps=1.),
                        dict(deepghs=float('nan'), hps=0.),
                        dict(deepghs=2., hps=-1.)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fusion.fuse(dict(deepghs=3., hps=-1.), calibration(), 0.,
                            kawai_enabled=False, sdg_enabled=False, weights=weights)


class OptionalPipelineTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('run.py', 'worker.py', 'hpsv3pp_torch/engine.py'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# CPU-only provenance fixture\n', encoding='utf-8')
        pipeline.save(root / 'native.json', {})
        pipeline.save(root / 'batch.json', dict(stage_order=['deepghs', 'hps'],
            hps=dict(project=str(root), native_config=str(root / 'native.json'))))
        pipeline.save(root / 'calibration.json', calibration())
        image = root / 'sample.png'
        image.write_bytes(b'not decoded; valid file for hashing fixture')
        cfg = dict(kawai_enabled=False, sdg_enabled=False, artifact_weight=0.,
                   fusion_weights=dict(deepghs=2/3, hps=1/3),
                   calibration=str(root / 'calibration.json'), batch_config=str(root / 'batch.json'),
                   batch_root=str(root), allowed_archive_root=str(root), archive_root=str(root / 'archives'),
                   controller_python='not executed', stage_timeout=1)
        request = dict(version=1, items=[dict(id=key, sample_id='prompt-'+key,
            prompt='raw prompt is unchanged', image=str(image)) for key in ('a', 'b')])
        def score_batch(argv, log, timeout):
            output = Path(argv[argv.index('--output')+1])
            pipeline.save(output, dict(status='complete', items=[
                dict(id='b', deepghs=dict(status='ok', percentile=1.), hps=dict(status='ok', mu=1.)),
                dict(id='a', deepghs=dict(status='ok', percentile=3.), hps=dict(status='ok', mu=-1.))]))
        return cfg, request, score_batch

    def test_disabled_pipeline_never_imports_reads_or_starts_optional_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, batch = self.fixture(root)
            original_bytes = Path.read_bytes
            original_import = builtins.__import__
            def guarded_bytes(path):
                if path.name in ('score_kawai.py', 'sdg_parallel.py', 'run_sdg_reward.py'):
                    raise AssertionError('Disabled optional source/model was read')
                return original_bytes(path)
            def guarded_import(name, *args, **kwargs):
                if name in ('score_kawai', 'sdg_parallel'):
                    raise AssertionError('Disabled optional worker was imported')
                return original_import(name, *args, **kwargs)
            with archive_lock_test_support(), \
                 patch.object(pipeline, 'run_program', side_effect=batch), \
                 patch.object(pipeline, 'run_kawai_parallel', side_effect=AssertionError('Kawai started')), \
                 patch.object(pipeline, 'run_sdg_parallel', side_effect=AssertionError('SDG started')), \
                 patch.object(Path, 'read_bytes', guarded_bytes), \
                 patch('builtins.__import__', guarded_import):
                result = pipeline.run(cfg, request, root / 'output.json')
            self.assertEqual(result['enabled_models'], ['deepghs', 'hps'])
            self.assertFalse(result['kawai_enabled'])
            self.assertAlmostEqual(result['scores'][0], .5 + .08 / 3)
            self.assertEqual(result['scores'][1], .5)
            for detail in result['details']:
                for field in ('raw_scores', 'normalized_scores', 'contributions', 'model_details'):
                    self.assertEqual(set(detail[field]), {'deepghs', 'hps'})
                self.assertEqual(detail['enabled_models'], ['deepghs', 'hps'])
            archive = Path(result['archive'])
            manifest = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))
            summary = json.loads((archive / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['stage_order'], ['deepghs', 'hps'])
            self.assertEqual(summary['enabled_models'], ['deepghs', 'hps'])
            self.assertEqual(set(summary['aesthetic_weights']), {'deepghs', 'hps'})
            self.assertTrue(all(Path(p).name not in ('score_kawai.py', 'sdg_parallel.py')
                                for p in manifest['code_hashes']))
            self.assertFalse((archive / 'kawai.json').exists())
            self.assertFalse((archive / 'kawai_dp').exists())
            self.assertFalse((archive / 'sdg.json').exists())
            with (archive / 'scores.csv').open(encoding='utf-8-sig', newline='') as stream:
                fields = csv.DictReader(stream).fieldnames
            self.assertEqual(fields, ['id', 'prompt_id', 'image', 'deepghs', 'hps', 'aesthetic_reward', 'reward'])

    def test_legacy_enabled_kawai_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, batch = self.fixture(root)
            cfg.pop('kawai_enabled')
            cfg.pop('fusion_weights')
            with archive_lock_test_support(), patch.object(pipeline, 'run_program', side_effect=batch), \
                 patch.object(pipeline, 'run_kawai_parallel', return_value=dict(items=[
                     dict(id='b', kawai=1.), dict(id='a', kawai=3.)])), \
                 patch.object(pipeline, 'run_sdg_parallel', side_effect=AssertionError('SDG started')):
                result = pipeline.run(cfg, request, root / 'output.json')
            self.assertEqual(result['enabled_models'], ['kawai', 'deepghs', 'hps'])
            expected = fusion.fuse(dict(kawai=3., deepghs=3., hps=-1.), calibration(), 0., sdg_enabled=False)
            self.assertEqual(result['scores'][0], expected['reward'])
            self.assertEqual(set(result['details'][0]['raw_scores']), {'kawai', 'deepghs', 'hps'})

    def test_disabled_kawai_keeps_optional_sdg_pipeline_working(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, batch = self.fixture(root)
            pipeline.save(root / 'sdg.json', {})
            cfg.update(sdg_enabled=True, artifact_weight=1., sdg_config=str(root / 'sdg.json'),
                       sdg_python='not executed', sdg_root=str(root / 'sdg'))
            sdg = dict(scores=[-.05, -.05], details=[
                dict(id=key, artifact_penalty=-.05, artifact_count=1, ignored_misalignment_count=0)
                for key in ('b', 'a')])
            with archive_lock_test_support(), patch.object(pipeline, 'provenance', return_value={}), \
                 patch.object(pipeline, 'run_program', side_effect=batch), \
                 patch.object(pipeline, 'run_kawai_parallel', side_effect=AssertionError('Kawai started')), \
                 patch.object(pipeline, 'run_sdg_parallel', return_value=sdg):
                result = pipeline.run(cfg, request, root / 'output.json')
            self.assertAlmostEqual(result['scores'][0], .5 + .08 / 3 - .05)
            self.assertEqual(set(result['details'][0]['raw_scores']), {'deepghs', 'hps', 'sdg_artifact'})
            self.assertEqual(set(result['details'][0]['model_details']), {'deepghs', 'hps', 'sdg'})
            self.assertEqual(result['enabled_models'], ['deepghs', 'hps'])
            manifest = json.loads((Path(result['archive']) / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['stage_order'], ['deepghs', 'hps', 'sdg'])

    def test_invalid_new_weight_set_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, _ = self.fixture(root)
            cfg['fusion_weights'] = dict(kawai=0., deepghs=2/3, hps=1/3)
            with patch.object(pipeline, 'run_program', side_effect=AssertionError('Model stage started')):
                with self.assertRaises(ValueError):
                    pipeline.run(cfg, request, root / 'output.json')


if __name__ == '__main__':
    unittest.main()
