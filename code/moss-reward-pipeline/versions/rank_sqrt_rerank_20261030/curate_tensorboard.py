#!/usr/bin/env python3
"""Mirror an exact scalar allowlist into a separate, compact TensorBoard run.

Source events remain read-only. A point retains its source step, wall time and
float32 scalar value. The destination can be rebuilt from source at any time.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import time
import types
import uuid

# Use TensorBoard's supported no-TensorFlow sentinel in this standalone process.
# EventFileLoader then uses the bundled Python TFRecord reader and cannot load
# TensorFlow's runtime or initialize a CUDA context while training is active.
from tensorboard import compat as _tb_compat
_tb_compat.notf = types.ModuleType('tensorboard.compat.notf')

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
from tensorboard.plugins.custom_scalar import layout_pb2, metadata
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.util import tensor_util


def float32(value):
    value = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    if not math.isfinite(value):
        raise ValueError('A selected scalar is nonfinite')
    return value


def scalar_value(value):
    if value.HasField('simple_value'):
        return float32(value.simple_value)
    if value.HasField('tensor'):
        tensor = value.tensor
        numeric = {types_pb2.DT_FLOAT, types_pb2.DT_DOUBLE, types_pb2.DT_INT32,
                   types_pb2.DT_INT64, types_pb2.DT_UINT32, types_pb2.DT_UINT64,
                   types_pb2.DT_INT16, types_pb2.DT_UINT16, types_pb2.DT_INT8,
                   types_pb2.DT_UINT8, types_pb2.DT_HALF, types_pb2.DT_BFLOAT16}
        if tensor.dtype not in numeric or tensor.tensor_shape.dim:
            return None
        array = tensor_util.make_ndarray(tensor)
        return float32(array.item()) if array.ndim == 0 else None
    return None


def read_points(directory, mapping=None):
    for path in sorted(Path(directory).glob('events.out.tfevents.*')):
        if not path.is_file():
            continue
        for event in EventFileLoader(str(path)).Load():
            if not event.HasField('summary'):
                continue
            for value in event.summary.value:
                if mapping is not None and value.tag not in mapping:
                    continue
                number = scalar_value(value)
                if number is not None:
                    yield (mapping[value.tag] if mapping is not None else value.tag,
                           int(event.step), float(event.wall_time), number)


def layout_summary(categories, allowed):
    layout = layout_pb2.Layout()
    for item in categories:
        category = layout.category.add(title=item['title'], closed=item.get('closed', False))
        for chart in item['charts']:
            unknown = set(chart['tags']) - allowed
            if unknown:
                raise ValueError('Layout references tags outside allowlist: '+str(sorted(unknown)))
            if not chart['tags']:
                raise ValueError('Layout chart has no tags')
            output = category.chart.add(title=chart['title'])
            # TensorBoard interprets these strings as regexes, not literal tags.
            output.multiline.tag.extend('^'+re.escape(tag)+'$' for tag in chart['tags'])
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=metadata.CONFIG_SUMMARY_TAG, metadata=metadata.create_summary_metadata(),
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,
                                      string_val=[layout.SerializeToString()]))])


class DirectoryLock:
    def __init__(self, directory):
        self.path = Path(directory)/'.curation.lock'
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if os.name == 'nt':
            import msvcrt
            if self.path.stat().st_size == 0:
                self.stream.write(b'0'); self.stream.flush()
            self.stream.seek(0)
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self.stream.close(); self.stream = None
                raise RuntimeError('Curated run already has an active exporter') from exc
        else:
            import fcntl
            try:
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.stream.close(); self.stream = None
                raise RuntimeError('Curated run already has an active exporter') from exc
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close(); self.stream = None


def insert_point(index, point):
    tag, step, wall_time, value = point
    key = (tag, step)
    if key in index:
        previous = index[key]
        if previous[3] != value:
            raise ValueError(f'Historical scalar changed: {tag} at step {step}: {previous[3]} -> {value}')
        if wall_time < previous[2]:
            index[key] = point
        return False
    index[key] = point
    return True


class CuratedExporter:
    def __init__(self, config):
        self.config = config
        self.source = Path(config['source_log_dir']).resolve()
        self.destination = Path(config['log_dir']).resolve()
        if self.destination == self.source or self.destination.is_relative_to(self.source):
            raise ValueError('Curated destination must be separate from source logs')
        if not self.source.is_dir():
            raise FileNotFoundError(self.source)
        self.mapping = dict(config['mapping'])
        if not self.mapping or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                                   for k, v in self.mapping.items()):
            raise ValueError('mapping needs nonempty source and destination tag strings')
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError('Multiple source tags cannot map to the same destination tag')
        self.evaluation = config.get('evaluation')
        allowed = set(self.mapping.values())
        if self.evaluation:
            if self.evaluation['tag'] in allowed:
                raise ValueError('evaluation.tag must not collide with mapped source tags')
            allowed.add(self.evaluation['tag'])
        self.layout = layout_summary(config.get('categories', []), allowed)
        self.destination.mkdir(parents=True, exist_ok=True)
        identity = dict(source_log_dir=str(self.source), mapping=self.mapping,
                        evaluation=self.evaluation)
        self.identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity_path = self.destination/'curation_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text())['sha256'] != self.identity_hash:
                raise ValueError('Curated mapping/source changed; use a new destination directory')
        else:
            identity_path.write_text(json.dumps(dict(sha256=self.identity_hash, **identity), indent=2)+'\n')
        self.index = {}
        for point in read_points(self.destination):
            insert_point(self.index, point)
        stamp = f'{time.time_ns()}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}'
        self.file = (self.destination/('events.out.tfevents.'+stamp+'.curated')).open('xb')
        self.writer = RecordWriter(self.file)
        self.writer.write(event_pb2.Event(wall_time=time.time(), file_version='brain.Event:2').SerializeToString())
        self.writer.write(event_pb2.Event(wall_time=time.time(), step=0, summary=self.layout).SerializeToString())
        self.writer.flush()

    def evaluation_points(self):
        if not self.evaluation:
            return
        root = Path(self.evaluation['run_dir'])/'training'/'steps'
        for path in sorted(root.glob('step_*/reward_evaluation.json')):
            row = json.loads(path.read_text())
            step, seconds = row['update'], row['seconds']
            if type(step) is not int or step < 0 or type(seconds) not in (int, float) or seconds < 0:
                raise ValueError('Invalid committed reward evaluation timing: '+str(path))
            yield (self.evaluation['tag'], step, path.stat().st_mtime, float32(seconds))

    def sync(self):
        # Validate the complete pending set before writing any scalar events.
        pending = {}
        for point in list(read_points(self.source, self.mapping)) + list(self.evaluation_points() or []):
            key = point[:2]
            if key in self.index:
                if self.index[key][3] != point[3]:
                    raise ValueError(f'Historical scalar changed: {key}')
            else:
                insert_point(pending, point)
        for point in sorted(pending.values(), key=lambda p:(p[2], p[1], p[0])):
            tag, step, wall_time, value = point
            summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)])
            self.writer.write(event_pb2.Event(wall_time=wall_time, step=step, summary=summary).SerializeToString())
        self.writer.flush()
        self.index.update(pending)
        state = dict(time=time.time(), source=str(self.source), log_dir=str(self.destination),
                     points=len(self.index), tags=len({key[0] for key in self.index}),
                     mapping_sha256=self.identity_hash, added_points=len(pending))
        path = self.destination/'curation_state.json'
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(path)
        return state

    def close(self):
        if self.writer is not None:
            self.writer.flush(); self.file.close(); self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    interval = config.get('interval_seconds', 10)
    if type(interval) not in (int, float) or not 0 < interval <= 60:
        raise ValueError('interval_seconds must be in (0, 60]')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with DirectoryLock(config['log_dir']):
        with CuratedExporter(config) as exporter:
            while not stopped:
                state = exporter.sync()
                if state['added_points']:
                    print(json.dumps(dict(event='curated_scalars_updated', **state)), flush=True)
                if config.get('once', False):
                    break
                time.sleep(interval)


if __name__ == '__main__':
    main()
