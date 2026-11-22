import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.writer.record_writer import RecordWriter

from curate_tensorboard import CuratedExporter, DirectoryLock, read_points, scalar_value


def write_events(path, entries):
    with path.open('wb') as stream:
        writer = RecordWriter(stream)
        writer.write(event_pb2.Event(file_version='brain.Event:2').SerializeToString())
        for tag, step, wall, number, tensor in entries:
            value = (summary_pb2.Summary.Value(tag=tag, tensor=tensor_pb2.TensorProto(
                dtype=types_pb2.DT_FLOAT, float_val=[number])) if tensor else
                summary_pb2.Summary.Value(tag=tag, simple_value=number))
            writer.write(event_pb2.Event(step=step, wall_time=wall,
                summary=summary_pb2.Summary(value=[value])).SerializeToString())
        writer.flush()


class CurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'raw'; self.source.mkdir()
        self.dest = self.root/'clean'
        self.config = dict(source_log_dir=str(self.source), log_dir=str(self.dest),
            mapping={'raw/loss':'train/loss', 'raw/reward':'reward/final'},
            categories=[dict(title='Overview', charts=[dict(title='Reward', tags=['reward/final'])])])

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowlist_formats_timestamp_idempotent_restart_and_source_hash(self):
        path = self.source/'events.out.tfevents.test'
        write_events(path, [('raw/loss',2,1000.25,.5,False),
                            ('raw/reward',0,999.75,.75,True),
                            ('raw/ignored',99,9000.,999.,False)])
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with CuratedExporter(self.config) as exporter:
            self.assertEqual(exporter.sync()['added_points'], 2)
            self.assertEqual(exporter.sync()['added_points'], 0)
        with CuratedExporter(self.config) as exporter:
            self.assertEqual(exporter.sync()['added_points'], 0)
        found = sorted(read_points(self.dest))
        self.assertEqual(found, [('reward/final',0,999.75,.75), ('train/loss',2,1000.25,.5)])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_new_source_file_adds_only_new_point(self):
        write_events(self.source/'events.out.tfevents.one', [('raw/loss',1,1.,.5,False)])
        with CuratedExporter(self.config) as exporter:
            exporter.sync()
            write_events(self.source/'events.out.tfevents.two', [('raw/loss',1,2.,.5,True),('raw/loss',2,3.,.4,True)])
            self.assertEqual(exporter.sync()['added_points'], 1)
        self.assertEqual(len(list(read_points(self.dest))), 2)

    def test_changed_scalar_fails_without_partial_new_points(self):
        write_events(self.source/'events.out.tfevents.one', [('raw/loss',1,1.,.5,False)])
        with CuratedExporter(self.config) as exporter:
            exporter.sync()
            write_events(self.source/'events.out.tfevents.two', [('raw/reward',2,3.,.9,False), ('raw/loss',1,4.,.7,False)])
            with self.assertRaisesRegex(ValueError, 'Historical scalar changed'):
                exporter.sync()
        self.assertEqual(len(list(read_points(self.dest))), 1)

    def test_non_scalar_tensor_and_histogram_are_ignored(self):
        self.assertIsNone(scalar_value(summary_pb2.Summary.Value(tag='array', tensor=tensor_pb2.TensorProto(
            dtype=types_pb2.DT_FLOAT, float_val=[1.], tensor_shape={'dim':[{'size':1}]}))))
        self.assertIsNone(scalar_value(summary_pb2.Summary.Value(tag='text', tensor=tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING, string_val=[b'text']))))
        self.assertIsNone(scalar_value(summary_pb2.Summary.Value(tag='hist', histo={})))

    def test_nonfinite_selected_value_fails(self):
        write_events(self.source/'events.out.tfevents.one', [('raw/loss',1,1.,float('nan'),False)])
        with CuratedExporter(self.config) as exporter:
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                exporter.sync()

    def test_layout_has_exact_allowlist_tags(self):
        with CuratedExporter(self.config) as exporter:
            layout = layout_pb2.Layout.FromString(exporter.layout.value[0].tensor.string_val[0])
            patterns = list(layout.category[0].chart[0].multiline.tag)
            self.assertFalse(layout.category[0].closed)
            self.assertEqual(patterns, ['^reward/final$'])
            self.assertIsNotNone(re.match(patterns[0], 'reward/final'))
            self.assertIsNone(re.match(patterns[0], 'reward/final_768'))
        bad = {**self.config, 'log_dir':str(self.root/'bad'), 'categories':[
            dict(title='Bad', charts=[dict(title='Unknown', tags=['secret/raw'])])]}
        with self.assertRaisesRegex(ValueError, 'outside allowlist'):
            CuratedExporter(bad)

    def test_source_and_destination_must_be_separate(self):
        for dest in (self.source, self.source/'nested'):
            with self.assertRaisesRegex(ValueError, 'separate'):
                CuratedExporter({**self.config, 'log_dir':str(dest)})

    def test_mapping_identity_changes_need_new_directory(self):
        with CuratedExporter(self.config):
            pass
        changed = {**self.config, 'mapping':{'raw/loss':'new/loss'}}
        changed['categories'] = []
        with self.assertRaisesRegex(ValueError, 'new destination'):
            CuratedExporter(changed)

    def test_evaluation_timing_is_real_record_and_idempotent(self):
        run = self.root/'run'; step=run/'training/steps/step_00000002'; step.mkdir(parents=True)
        path=step/'reward_evaluation.json';path.write_text(json.dumps(dict(update=2, seconds=17.25)))
        config={**self.config, 'evaluation':dict(run_dir=str(run),tag='perf/reward_seconds')}
        with CuratedExporter(config) as exporter:
            self.assertEqual(exporter.sync()['added_points'],1)
            self.assertEqual(exporter.sync()['added_points'],0)
        self.assertEqual(list(read_points(self.dest)), [('perf/reward_seconds',2,path.stat().st_mtime,17.25)])

    def test_directory_lock_blocks_second_owner(self):
        with DirectoryLock(self.dest):
            with self.assertRaises(RuntimeError):
                with DirectoryLock(self.dest):
                    pass

    def test_layout_category_closed_setting_is_preserved(self):
        config = {**self.config, 'categories':[
            dict(title='Folded', closed=True, charts=[dict(title='Loss', tags=['train/loss'])])]}
        with CuratedExporter(config) as exporter:
            layout = layout_pb2.Layout.FromString(exporter.layout.value[0].tensor.string_val[0])
            self.assertTrue(layout.category[0].closed)

    def test_sync_never_imports_tensorflow_runtime(self):
        write_events(self.source/'events.out.tfevents.one', [('raw/loss',1,1.,.5,False)])
        with CuratedExporter(self.config) as exporter:
            exporter.sync()
        self.assertNotIn('tensorflow', sys.modules)


if __name__ == '__main__':
    unittest.main()
