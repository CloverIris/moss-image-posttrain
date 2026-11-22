"""Rank fusion at the durable train / fixed-validation scoring boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_reward_pipeline as pipeline
from test_kawai_disabled import OptionalPipelineTests, archive_lock_test_support


class RankPipelineTests(unittest.TestCase):
    def fixture(self, root, *, evaluation=True, validation_only=False):
        cfg, request, _ = OptionalPipelineTests().fixture(root)
        cfg.update(fusion_mode='rank_power_rank', rank_power_gamma=.5,
                   rank_training_group_size=8, rank_eval_group_size=4,
                   checkpoint_eval_reward_mode='frozen_calibration')
        image = request['items'][0]['image']
        items = []
        if not validation_only:
            items += [dict(id=f't{i}', sample_id='same-prompt', image=image,
                           prompt='synthetic test', index=i) for i in range(8)]
        if evaluation or validation_only:
            # Deliberately use the same prompt ID: domain isolation must still
            # work without depending on a caller's prefix convention.
            items += [dict(id=f'e{i}', sample_id='same-prompt', image=image,
                           prompt='synthetic test', index=i,
                           checkpoint_eval=dict(job_id='test-checkpoint')) for i in range(4)]
        request['items'] = items

        def score_batch(argv, log, timeout):
            incoming = json.loads(Path(argv[argv.index('--input') + 1]).read_text())['items']
            result = []
            for row in incoming:
                i = int(row['id'][1:])
                # D/H agree in training and disagree in validation.
                deep, hps = (float(i), float(i)) if row['id'][0] == 't' else (float(i), float(3-i))
                result.append(dict(id=row['id'], deepghs=dict(status='ok', percentile=deep),
                                   hps=dict(status='ok', mu=hps)))
            pipeline.save(argv[argv.index('--output') + 1], dict(status='complete', items=result[::-1]))
        return cfg, request, score_batch

    def test_mixed_groups_rank_train_and_keep_validation_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, request, scorer = self.fixture(root)
            original = copy.deepcopy(request)
            with archive_lock_test_support(), patch.object(pipeline, 'run_program', side_effect=scorer):
                result = pipeline.run(cfg, request, root/'output.json')
            self.assertEqual(request, original)
            train, evaluation = result['details'][:8], result['details'][8:]
            self.assertAlmostEqual(sum(r['reward'] for r in train)/8, .5)
            for i, row in enumerate(train):
                self.assertAlmostEqual(row['reward'], i/7)
                self.assertEqual(row['reward'], row['rank_reward'])
                self.assertEqual(row['reward_semantics'], 'rank_power_rank_training')
                self.assertEqual(row['rank_group_size'], 8)
                self.assertEqual(row['components']['frozen_reward'], row['frozen_reward'])
            for row in evaluation:
                self.assertEqual(row['reward'], row['frozen_reward'])
                self.assertEqual(row['reward_semantics'], 'frozen_calibration_validation')
                self.assertEqual(row['rank_group_size'], 4)
            self.assertAlmostEqual(sum(r['rank_reward'] for r in evaluation)/4, .5)
            self.assertNotAlmostEqual(sum(r['reward'] for r in evaluation)/4, .5)
            summary = json.loads((Path(result['archive'])/'summary.json').read_text())
            self.assertEqual(summary['mean_semantics'], 'mixed_domains_not_quality')
            self.assertEqual(set(summary['domains']), {'training', 'checkpoint_eval'})
            self.assertEqual(summary['domains']['training']['images'], 8)
            self.assertEqual(summary['domains']['checkpoint_eval']['images'], 4)
            for row in result['details']:
                self.assertEqual(set(row['raw_scores']), {'deepghs', 'hps'})
                self.assertEqual(row['contributions_space'], 'frozen_calibration')
                self.assertEqual(row['frozen_reward'], .5 + sum(row['contributions'].values()))

    def test_validation_only_flush_keeps_absolute_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, request, scorer = self.fixture(root, validation_only=True)
            with archive_lock_test_support(), patch.object(pipeline, 'run_program', side_effect=scorer):
                result = pipeline.run(cfg, request, root/'output.json')
            self.assertEqual(len(result['scores']), 4)
            self.assertTrue(all(r['reward'] == r['frozen_reward'] for r in result['details']))
            self.assertTrue(all(r['rank_group_size'] == 4 for r in result['details']))

    def test_incomplete_groups_and_invalid_config_fail_before_model_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, request, scorer = self.fixture(root, evaluation=False)
            bad_cases = []
            short = copy.deepcopy(request); short['items'].pop()
            bad_cases.append((cfg, short))
            for gamma in (0, -1, 1.1, float('nan'), float('inf'), True):
                bad_cases.append((dict(cfg, rank_power_gamma=gamma), request))
            for extra in (dict(fusion_mode='typo'), dict(rank_training_group_size=1),
                          dict(checkpoint_eval_reward_mode='rank'), dict(sdg_enabled=True)):
                bad_cases.append((dict(cfg, **extra), request))
            malformed = copy.deepcopy(request); malformed['items'][0]['checkpoint_eval'] = {}
            bad_cases.append((cfg, malformed))
            with patch.object(pipeline, 'run_program', side_effect=AssertionError('Model stage started')):
                for bad_cfg, bad_request in bad_cases:
                    with self.subTest(config=bad_cfg), self.assertRaises(ValueError):
                        pipeline.run(bad_cfg, bad_request, root/'output.json')


if __name__ == '__main__':
    unittest.main()
