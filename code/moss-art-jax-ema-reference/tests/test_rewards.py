import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from moss_art.rewards import as_image_batch, score_images


class RewardTests(unittest.TestCase):
    def test_toy_is_brightness_only(self):
        images = np.stack([np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 255, np.uint8)])
        np.testing.assert_array_equal(score_images(images, ['black', 'white'], {'kind': 'toy'}), [0, 1])
        with self.assertRaises(ValueError):
            score_images(images, ['one prompt'], {'kind': 'toy'})
        with self.assertRaises(ValueError):
            as_image_batch(np.full((2, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            as_image_batch(np.full((2, 2, 3), 2.0))

    def test_command_protocol_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / 'scorer.py'
            script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\nr=json.loads(Path(a.request).read_text());assert r['version']==1\nassert all(Path(i['image']).exists() for i in r['items'])\nPath(a.output).write_text(json.dumps({'scores':[len(i['prompt']) for i in r['items']]}))\n")
            images = np.zeros((2, 4, 4, 3), np.uint8)
            value = score_images(images, ['cat', 'elephant'], {'kind': 'command', 'command': [sys.executable, str(script)]})
            np.testing.assert_array_equal(value, [3, 8])
            value = score_images(images, ['a', 'b'], {'kind': 'command', 'command': [sys.executable, str(script), '--request', '{request}', '--output', '{output}']})
            np.testing.assert_array_equal(value, [1, 1])

    def test_bad_reward_fails_instead_of_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / 'scorer.py'
            script.write_text("import sys\nfrom pathlib import Path\nPath(sys.argv[-1]).write_text('{\"scores\":[NaN]}')\n")
            with self.assertRaisesRegex(ValueError, 'finite'):
                score_images(np.zeros((1, 4, 4, 3), np.uint8), ['cat'], {'kind': 'command', 'command': [sys.executable, str(script)]})
            with self.assertRaisesRegex(RuntimeError, 'subprocess failed'):
                score_images(np.zeros((1, 4, 4, 3), np.uint8), ['cat'], {'kind': 'command', 'command': [sys.executable, '-c', 'raise RuntimeError("bad reward")']})


if __name__ == '__main__':
    unittest.main()
