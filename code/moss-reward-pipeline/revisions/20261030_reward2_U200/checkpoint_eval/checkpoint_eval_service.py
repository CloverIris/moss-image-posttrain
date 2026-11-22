#!/usr/bin/env python3
"""CPU-only checkpoint generation service with durable scoring outbox/acks.

Each committed checkpoint is pinned before a worker starts. GPU4 only generates
the fixed validation images. A separate training reward wrapper scores them and
writes an acknowledgement; this service never loads a reward model.
"""
import argparse
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import tempfile
import time
import uuid


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_bytes(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    # Windows scanners can briefly open the old state without FILE_SHARE_DELETE.
    # Retry that sharing violation while keeping the old complete file intact.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if os.name != 'nt' or attempt == 9: raise
            time.sleep(.02*(attempt+1))
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value)+b'\n')


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable artifact differs: '+str(path))
    else:
        atomic_json(path, value)


def committed_manifest(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Checkpoint directory must not be a symlink')
    content = (path/'manifest.json').read_bytes()
    checksum = (path/'COMMITTED').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}', checksum) or digest_bytes(content) != checksum:
        raise ValueError('Checkpoint COMMITTED/manifest checksum mismatch')
    value = json.loads(content)
    return checksum, value


class DirectoryLock:
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.stream.write(b'0'); self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close(); self.stream = None
            raise RuntimeError('Another service holds lock: '+str(self.path)) from exc
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close(); self.stream = None


def process_identity(pid):
    """Linux birth identity prevents signalling a reused PID on recovery."""
    try:
        text = Path('/proc/'+str(pid)+'/stat').read_text()
        fields = text[text.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        return dict(start_ticks=fields[19],
                    argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')[:-1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def json_process_identity(pid):
    value = process_identity(pid)
    if value:
        return dict(start_ticks=value['start_ticks'],
                    argv=[x.decode(errors='replace') for x in value['argv']])
    return None


def stop_child(process, grace=10):
    """Only a Popen child launched by this process, in its own process group."""
    if process.poll() is not None:
        if os.name == 'posix':
            # A failed group leader can exit before its native worker children.
            # This is still the fresh, known-owned group created by Popen.
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: return
            deadline = time.monotonic()+grace
            while time.monotonic() < deadline:
                try: os.killpg(process.pid, 0)
                except ProcessLookupError: return
                time.sleep(.05)
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, check=False)
    else:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if os.name == 'nt': process.kill()
        else:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        process.wait(timeout=grace)


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values),
                std=statistics.pstdev(values), min=min(values), max=max(values))


def validate_config(config):
    for key in ('source_checkpoint_dir', 'service_root', 'source_config', 'fixed_prompts'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError('Missing path configuration: '+key)
    if config.get('device', 4) != 4:
        raise ValueError('This service is reserved for physical GPU4')
    count = config.get('images_per_prompt', 4)
    if type(count) is not int or count < 1:
        raise ValueError('images_per_prompt must be positive')
    seeds = config.get('fixed_seeds')
    if not isinstance(seeds, list) or len(seeds) != count or any(
        type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != count:
        raise ValueError('fixed_seeds must contain one unique uint32 seed per image')
    for key in ('height', 'width'):
        value = config.get(key, 768)
        if type(value) is not int or value <= 0 or value % 16:
            raise ValueError(key+' must be a positive multiple of16')
    if type(config.get('expected_prompt_count', 4)) is not int or config.get('expected_prompt_count', 4) < 1:
        raise ValueError('expected_prompt_count must be positive')
    interval = config.get('poll_interval_seconds', 10)
    if type(interval) not in (int, float) or not 0 < interval <= 60:
        raise ValueError('poll_interval_seconds must be in(0,60]')
    retries = config.get('max_retries', 1)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError('max_retries must be an integer from0 to3')
    for key in ('export_command', 'infer_command', 'summary_command'):
        command = config.get(key)
        if key == 'summary_command' and command is None: continue
        if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
            raise ValueError(key+' must be a nonempty argv list')
    return config


class CheckpointEvalService:
    def __init__(self, config):
        self.config = validate_config(dict(config))
        self.root = Path(config['service_root']).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.source = Path(config['source_checkpoint_dir']).resolve()
        self.outbox = Path(config.get('outbox_dir', str(self.root/'outbox'))).resolve()
        self.acks = Path(config.get('ack_dir', str(self.outbox))).resolve()
        self.outbox.mkdir(parents=True, exist_ok=True); self.acks.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root/'state.json'
        self.stopped, self.child, self.consuming_ack = False, None, False
        fixed = self.root/'fixed'; fixed.mkdir(exist_ok=True)
        self.prompt_snapshot = self._freeze_file(config['fixed_prompts'], fixed/('prompts'+Path(config['fixed_prompts']).suffix))
        self.source_config_snapshot = self._freeze_file(config['source_config'], fixed/'source_config.json')
        identity_config = {k:v for k,v in config.items() if k not in
            ('once', 'poll_interval_seconds', 'occupied_gpu_limit_mib', 'require_gpu_idle',
             'stage_timeout_seconds', 'kill_grace_seconds')}
        identity = dict(config=identity_config, fixed_prompts_sha256=digest_file(self.prompt_snapshot),
                        source_config_sha256=digest_file(self.source_config_snapshot))
        self.identity = digest_bytes(canonical(identity))
        immutable_json(self.root/'service_identity.json', dict(sha256=self.identity, **identity))
        self.state = read_json(self.state_path) if self.state_path.exists() else dict(
            version=1, identity_sha256=self.identity, tasks={}, discovery_errors={})
        if self.state.get('identity_sha256') != self.identity:
            raise ValueError('Service identity changed; use a new service_root')
        self._recover()
        self.save()

    def _freeze_file(self, source, destination):
        source, destination = Path(source).resolve(), Path(destination)
        value = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != value:
                raise ValueError('Frozen service input changed: '+str(source))
        else:
            atomic_bytes(destination, value)
        return destination.resolve()

    def save(self):
        self.state['updated_at'] = time.time()
        atomic_json(self.state_path, self.state)

    def _recover(self):
        for task in self.state['tasks'].values():
            active = task.get('active_process')
            if active:
                actual = json_process_identity(active['pid'])
                if actual and actual == active.get('identity'):
                    if os.name != 'posix':
                        raise RuntimeError('An orphan worker needs attention: '+str(active['pid']))
                    try: os.killpg(active['pid'], signal.SIGTERM)
                    except ProcessLookupError: pass
                    deadline = time.monotonic()+self.config.get('kill_grace_seconds', 10)
                    while json_process_identity(active['pid']) and time.monotonic() < deadline:
                        time.sleep(.1)
                    if json_process_identity(active['pid']):
                        try: os.killpg(active['pid'], signal.SIGKILL)
                        except ProcessLookupError: pass
                task['active_process'] = None
            status = task['status']
            if status == 'done':
                completion = read_json(task['completion'])
                if digest_file(completion['ack']) != completion['ack_sha256'] or \
                        digest_file(completion['summary']) != completion['summary_sha256']:
                    raise ValueError('Completed evaluation artifact changed: '+task['checkpoint_id'])
                self._validate_ready(read_json(task['ready']))
                if completion.get('evaluation_result') and digest_file(completion['evaluation_result']) != completion['evaluation_result_sha256']:
                    raise ValueError('Completed raw scores changed: '+task['checkpoint_id'])
            elif status == 'ready_for_scoring':
                self._validate_ready(read_json(task['ready']))
            elif status == 'running':
                attempt = Path(task['last_attempt'])
                resolved = read_json(attempt/'task.json')
                ready_path = Path(resolved['output_dir'])/'ready.json'
                if ready_path.is_file():
                    ready = read_json(ready_path); self._validate_ready(ready)
                    immutable_json(self.outbox/(task['checkpoint_id']+'.ready.json'), ready)
                    task.update(status='ready_for_scoring', ready=str(ready_path), active_process=None)
                elif (attempt/'infer.complete.json').is_file():
                    complete = read_json(attempt/'infer.complete.json')
                    if digest_file(complete['output']) != complete['sha256']:
                        raise ValueError('Completed inference request changed before publication')
                    self._publish_ready(task, resolved)
                else:
                    task.update(status='pending', active_process=None,
                                last_error='Previous controller interrupted; original attempt retained')

    def _snapshot(self, source, checkpoint_id, checksum):
        destination = self.root/'checkpoints'/checkpoint_id
        if destination.exists():
            if committed_manifest(destination)[0] != checksum:
                raise ValueError('Pinned checkpoint changed')
            return destination
        destination.parent.mkdir(exist_ok=True)
        temporary = destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.pinning')
        temporary.mkdir()
        # Failed partial pins remain available for diagnosis; no user data deleted.
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Checkpoint snapshot cannot contain symlinks')
            target = temporary/path.relative_to(source)
            if path.is_dir(): target.mkdir(exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
        if committed_manifest(source)[0] != checksum or committed_manifest(temporary)[0] != checksum:
            raise ValueError('Checkpoint changed while pinning')
        temporary.replace(destination)
        return destination

    def discover(self):
        added = 0
        for path in sorted(self.source.glob('step_*')):
            match = re.fullmatch(r'step_(\d+)', path.name)
            if not match or not path.is_dir(): continue
            update = int(match[1])
            if update < self.config.get('min_update', 0): continue
            if self.config.get('max_update') is not None and update > self.config['max_update']: continue
            if not (path/'COMMITTED').is_file(): continue
            try:
                checksum, _ = committed_manifest(path)
                checkpoint_id = path.name+'_'+checksum[:16]
                if checkpoint_id in self.state['tasks']: continue
                snapshot = self._snapshot(path, checkpoint_id, checksum)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.state['discovery_errors'][str(path)] = dict(time=time.time(), error=str(exc))
                continue
            self.state['tasks'][checkpoint_id] = dict(checkpoint_id=checkpoint_id, update=update,
                checkpoint_manifest_sha256=checksum, source_checkpoint=str(path),
                checkpoint=str(snapshot), status='pending', attempts=0, failures=0,
                created_at=time.time(), active_process=None)
            added += 1
        if added or self.state['discovery_errors']: self.save()
        return added

    def gpu_idle(self):
        if not self.config.get('require_gpu_idle', True): return True
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        memory = {int(line.split(',')[0]):int(line.split(',')[1]) for line in result.splitlines()}
        if 4 not in memory: raise RuntimeError('Physical GPU4 is unavailable')
        return memory[4] <= self.config.get('occupied_gpu_limit_mib', 1000)

    def _resolved(self, task, attempt):
        result = dict(self.config)
        result.update(version=1, job_id=task['checkpoint_id'], checkpoint_id=task['checkpoint_id'],
            source_step=task['update'], update=task['update'],
            checkpoint=task['checkpoint'], source_checkpoint=task['source_checkpoint'],
            checkpoint_manifest_sha256=task['checkpoint_manifest_sha256'],
            source_config=str(Path(self.config['source_config']).resolve()),
            source_config_snapshot=str(self.source_config_snapshot),
            source_config_sha256=digest_file(self.source_config_snapshot),
            fixed_prompts=str(self.prompt_snapshot), prompts=str(self.prompt_snapshot),
            fixed_prompts_sha256=digest_file(self.prompt_snapshot),
            expected_prompt_sha256=digest_file(self.prompt_snapshot),
            fixed_seeds=self.config['fixed_seeds'], seeds=self.config['fixed_seeds'],
            seed=self.config.get('seed', self.config['fixed_seeds'][0]),
            images_per_prompt=self.config.get('images_per_prompt', 4),
            expected_prompt_count=self.config.get('expected_prompt_count', 4),
            height=self.config.get('height', 768), width=self.config.get('width', 768), device=4,
            audit_dir=str(attempt), output_dir=str(attempt/'generated'), images_dir=str(attempt/'generated'/'images'),
            adapter=str(attempt/'adapter.safetensors'), request=str(attempt/'generated'/'eval_request.json'),
            scores=str(attempt/'scoring'/'result.json'), summary=str(attempt/'summary.json'))
        result['config'] = str(attempt/'task.json')
        return result

    def _run_stage(self, name, task, resolved):
        if digest_file(resolved['source_config']) != resolved['source_config_sha256']:
            raise ValueError('Original source config changed after service plan was frozen')
        command = self.config[name+'_command']
        argv = []
        for value in command:
            for key in ('checkpoint', 'adapter', 'config', 'output_dir', 'request', 'scores',
                        'summary', 'checkpoint_id', 'source_config', 'fixed_prompts'):
                value = value.replace('{'+key+'}', str(resolved[key]))
            argv.append(value)
        folder = Path(resolved['audit_dir'])
        atomic_json(folder/(name+'.command.json'), dict(argv=argv, stage=name,
            environment=dict(CUDA_VISIBLE_DEVICES='4' if name=='infer' else '')))
        env = dict(os.environ)
        env.update({str(k):str(v) for k,v in self.config.get('worker_environment', {}).items()})
        env['CUDA_VISIBLE_DEVICES'] = '4' if name=='infer' else ''
        if name != 'infer': env['JAX_PLATFORMS'] = 'cpu'
        elif env.get('JAX_PLATFORMS') == 'cpu': env.pop('JAX_PLATFORMS')
        timeout = self.config.get('stage_timeout_seconds', {}).get(name, 14400 if name=='infer' else 300)
        started, last_scan = time.monotonic(), 0.
        with (folder/(name+'.log')).open('x', encoding='utf-8') as log:
            options = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           cwd=self.config.get('worker_cwd'), env=env)
            if os.name == 'nt': options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else: options['start_new_session'] = True
            process = subprocess.Popen(argv, **options); self.child = process
            task['active_process'] = dict(pid=process.pid, identity=json_process_identity(process.pid), argv=argv)
            self.save()
            try:
                while process.poll() is None:
                    if self.stopped: raise InterruptedError('Service stopping')
                    if time.monotonic()-started > timeout: raise TimeoutError(name+' timed out')
                    if time.monotonic()-last_scan >= self.config.get('poll_interval_seconds', 10):
                        self.discover()
                        if not self.consuming_ack: self.consume_acks()
                        last_scan = time.monotonic()
                    time.sleep(min(.5, self.config.get('poll_interval_seconds', 10)))
                if process.returncode:
                    raise RuntimeError(f'{name} exited{process.returncode}; see {folder/(name+".log")}')
            finally:
                stop_child(process, self.config.get('kill_grace_seconds', 10))
                self.child = None; task['active_process'] = None; self.save()
        expected = Path(resolved['adapter'] if name=='export' else resolved['summary'] if name=='summary' else resolved['request'])
        if digest_file(resolved['source_config']) != resolved['source_config_sha256']:
            raise ValueError('Original source config changed during worker execution')
        if not expected.is_file() or expected.stat().st_size == 0:
            raise ValueError(name+' exited without its required output')
        atomic_json(folder/(name+'.complete.json'), dict(time=time.time(), seconds=time.monotonic()-started,
            output=str(expected), sha256=digest_file(expected)))

    def _validate_request(self, resolved):
        request = read_json(resolved['request']); items = request.get('items')
        if request.get('version') != 1 or not isinstance(items, list):
            raise ValueError('Inference must produce a version1 request with items')
        per_prompt = self.config.get('images_per_prompt', 4)
        if len(items) != per_prompt*self.config.get('expected_prompt_count', 4):
            raise ValueError('Incorrect fixed evaluation image count')
        groups, plan, images = {}, [], []
        folder = Path(resolved['output_dir']).resolve()
        for row in items:
            image = Path(row['image']).resolve()
            if not image.is_relative_to(folder) or not image.is_file() or not image.stat().st_size:
                raise ValueError('Generated image missing or outside its immutable attempt')
            if not isinstance(row.get('prompt'), str) or type(row.get('seed')) is not int:
                raise ValueError('Evaluation request needs original prompt and actual integer seed metadata')
            key = next((k for k in ('prompt_index', 'prompt_id', 'sample_id') if k in row), None)
            if 'rollout_id' not in row and 'repeat' in row: row['rollout_id'] = row['repeat']
            if key is None or 'rollout_id' not in row:
                raise ValueError('Evaluation request needs prompt group and rollout_id')
            group = canonical({key:row[key]}).decode()
            groups.setdefault(group, []).append(row['rollout_id'])
            conditions = {k:row[k] for k in ('prompt_index', 'prompt_id', 'sample_id', 'rollout_id',
                                            'seed', 'sampling_method') if k in row}
            conditions.update(prompt_sha256=digest_bytes(row['prompt'].encode()),
                negative_prompt_sha256=digest_bytes(row.get('negative_prompt', '').encode()),
                height=resolved['height'], width=resolved['width'])
            plan.append(conditions)
            images.append(dict(path=str(image), sha256=digest_file(image)))
        if any(not isinstance(row.get('id'), (str, int)) for row in items) or \
                len({str(row['id']) for row in items}) != len(items):
            raise ValueError('Each request image needs a unique id')
        if len({x['path'] for x in images}) != len(images):
            raise ValueError('Duplicate generated image paths')
        if len(groups) != self.config.get('expected_prompt_count', 4) or any(
            len(v) != per_prompt or len(set(v)) != per_prompt for v in groups.values()):
            raise ValueError('Each fixed prompt must have the requested unique rollout count')
        immutable_json(self.root/'fixed_request_plan.json', plan)
        worker_plan = request.get('context', {}).get('fixed_plan_sha256')
        if worker_plan is not None:
            if not isinstance(worker_plan, str) or not re.fullmatch('[0-9a-f]{64}', worker_plan):
                raise ValueError('Invalid native worker fixed plan hash')
            immutable_json(self.root/'fixed_native_plan.json', dict(sha256=worker_plan))
        return request, images, digest_bytes(canonical(plan))

    def _publish_ready(self, task, resolved):
        request, images, plan_hash = self._validate_request(resolved)
        ready = dict(version=1, kind='checkpoint_eval', status='ready_for_scoring',
            job_id=task['checkpoint_id'], checkpoint_id=task['checkpoint_id'],
            source_step=task['update'], update=task['update'],
            checkpoint=task['checkpoint'], source_checkpoint=task['source_checkpoint'],
            checkpoint_manifest_sha256=task['checkpoint_manifest_sha256'],
            request=resolved['request'], request_path=resolved['request'],
            request_sha256=digest_file(resolved['request']),
            result_dir=str(Path(resolved['audit_dir'])/'scoring'),
            request_plan_sha256=plan_hash, output_dir=resolved['output_dir'],
            fixed_plan_sha256=request.get('context', {}).get('fixed_plan_sha256', plan_hash),
            fixed_prompts_sha256=resolved['fixed_prompts_sha256'],
            fixed_seeds=resolved['fixed_seeds'], height=resolved['height'], width=resolved['width'],
            images_per_prompt=resolved['images_per_prompt'], prompt_count=resolved['expected_prompt_count'],
            image_count=len(request['items']), images=images,
            namespace='checkpoint_eval:'+task['checkpoint_id'],
            ack=str(self.acks/(task['checkpoint_id']+'.ack.json')))
        path = Path(resolved['output_dir'])/'ready.json'
        immutable_json(path, ready); immutable_json(self.outbox/(task['checkpoint_id']+'.ready.json'), ready)
        task.update(status='ready_for_scoring', ready=str(path), active_process=None,
                    generation_completed_at=time.time())
        self.save()
        print(json.dumps(dict(event='checkpoint_eval_ready', checkpoint_id=task['checkpoint_id'],
                              images=ready['image_count'], outbox=str(self.outbox/(task['checkpoint_id']+'.ready.json')))), flush=True)

    def _validate_ready(self, ready):
        if digest_file(ready['request']) != ready['request_sha256']:
            raise ValueError('Ready request changed')
        for image in ready['images']:
            if digest_file(image['path']) != image['sha256']:
                raise ValueError('Ready image changed: '+image['path'])

    def consume_acks(self):
        if self.consuming_ack: return 0
        self.consuming_ack = True
        try:
            return self._consume_acks()
        finally:
            self.consuming_ack = False

    def _consume_acks(self):
        completed = 0
        for task in self.state['tasks'].values():
            if self.stopped: break
            if task['status'] != 'ready_for_scoring': continue
            ready = read_json(task['ready']); ack_path = Path(ready['ack'])
            if not ack_path.is_file(): continue
            try:
                ack = read_json(ack_path)
                if ack.get('status') not in ('scored', 'complete', 'done') or any(ack.get(k) != ready[k] for k in
                    ('checkpoint_id', 'checkpoint_manifest_sha256', 'request_sha256')):
                    raise ValueError('Scoring acknowledgement identity/status mismatch')
                if ack.get('job_id', ready['job_id']) != ready['job_id']:
                    raise ValueError('Scoring acknowledgement job identity mismatch')
                result_file = ack.get('evaluation_result', ack.get('result_path'))
                payload = read_json(result_file) if isinstance(result_file, str) else ack
                scores = payload.get('scores')
                if not isinstance(scores, list) or len(scores) != ready['image_count'] or any(
                    type(v) not in (int,float) or not math.isfinite(v) for v in scores):
                    raise ValueError('Acknowledgement needs exactly one finite score per image')
                items = read_json(ready['request'])['items']; details = payload.get('details')
                if details is not None:
                    if not isinstance(details, list) or len(details) != len(items):
                        raise ValueError('Acknowledgement detail count differs')
                    for item, detail in zip(items, details):
                        if isinstance(detail, dict) and detail.get('image') and Path(detail['image']).resolve() != Path(item['image']).resolve():
                            raise ValueError('Acknowledgement image order differs')
                self._validate_ready(ready)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                task['ack_error'] = dict(time=time.time(), path=str(ack_path), error=str(exc))
                self.save(); continue
            summary = dict(version=1, checkpoint_id=task['checkpoint_id'], update=task['update'],
                checkpoint_manifest_sha256=task['checkpoint_manifest_sha256'],
                request=ready['request'], request_sha256=ready['request_sha256'],
                fixed_prompts_sha256=ready['fixed_prompts_sha256'],
                request_plan_sha256=ready['request_plan_sha256'], fixed_seeds=ready['fixed_seeds'],
                height=ready['height'], width=ready['width'], image_count=len(scores),
                reward=stats(scores), worker_summary=payload.get('summary'),
                ack=str(ack_path), ack_sha256=digest_file(ack_path))
            summary_path = Path(task['last_attempt'])/'summary.json'; immutable_json(summary_path, summary)
            if self.config.get('summary_command'):
                if task.get('report_failures', 0) > self.config.get('summary_max_retries', 1): continue
                task['report_attempts'] = task.get('report_attempts', 0)+1
                report_folder = Path(task['last_attempt'])/'reporting'/('attempt_%04d'%task['report_attempts'])
                report_folder.mkdir(parents=True, exist_ok=False)
                resolved = read_json(Path(task['last_attempt'])/'task.json')
                resolved.update(ack=str(ack_path), evaluation_result=result_file or str(ack_path),
                    scores=result_file or str(ack_path), summary=str(summary_path),
                    config=str(report_folder/'task.json'), audit_dir=str(report_folder))
                atomic_json(resolved['config'], resolved); self.save()
                try:
                    self._run_stage('summary', task, resolved)
                except InterruptedError as exc:
                    task['report_error'] = dict(error=str(exc), time=time.time())
                    self.save(); continue
                except Exception as exc:
                    task['report_failures'] = task.get('report_failures', 0)+1
                    task['report_error'] = dict(error=str(exc), time=time.time())
                    self.save(); continue
            completion = dict(checkpoint_id=task['checkpoint_id'], ack=str(ack_path),
                ack_sha256=summary['ack_sha256'], summary=str(summary_path), summary_sha256=digest_file(summary_path))
            if result_file:
                completion.update(evaluation_result=result_file, evaluation_result_sha256=digest_file(result_file))
            completion_path = Path(ready['output_dir'])/'completion.json'; immutable_json(completion_path, completion)
            task.update(status='done', completion=str(completion_path), completed_at=time.time(), ack_error=None)
            completed += 1
            print(json.dumps(dict(event='checkpoint_eval_done', checkpoint_id=task['checkpoint_id'],
                                  images=len(scores), mean_reward=summary['reward']['mean'])), flush=True)
        if completed: self.save()
        return completed

    def next_task(self):
        eligible = [task for task in self.state['tasks'].values() if task['status']=='pending' or
                    task['status']=='failed' and task.get('failures',0) <= self.config.get('max_retries',1)]
        return min(eligible, key=lambda x:(x['update'],x['created_at'])) if eligible else None

    def process_task(self, task):
        if not self.gpu_idle(): return False
        task['attempts'] += 1
        attempt = self.root/'evaluations'/task['checkpoint_id']/('attempt_%04d'%task['attempts'])
        attempt.mkdir(parents=True, exist_ok=False)
        resolved = self._resolved(task, attempt); atomic_json(attempt/'task.json', resolved)
        task.update(status='running', owner_pid=os.getpid(), last_attempt=str(attempt), started_at=time.time())
        self.save()
        try:
            self._run_stage('export', task, resolved)
            if self.stopped: raise InterruptedError('Service stopping')
            if not self.gpu_idle(): raise InterruptedError('GPU4 became occupied before inference')
            self._run_stage('infer', task, resolved)
            self._publish_ready(task, resolved)
        except InterruptedError as exc:
            task.update(status='pending', active_process=None, last_error=str(exc))
            atomic_json(attempt/'status.json', dict(status='interrupted', error=str(exc), time=time.time()))
            self.save()
        except Exception as exc:
            task.update(status='failed', active_process=None, failures=task.get('failures',0)+1,
                        last_error=str(exc))
            atomic_json(attempt/'status.json', dict(status='failed', error=str(exc), time=time.time()))
            self.save()
            print(json.dumps(dict(event='checkpoint_eval_failed',checkpoint_id=task['checkpoint_id'],
                                  attempt=task['attempts'],error=str(exc))),flush=True)
        return True

    def run(self):
        while not self.stopped:
            self.discover(); self.consume_acks()
            task = self.next_task()
            if task is not None:
                if self.process_task(task): continue
            if self.config.get('once', False): break
            time.sleep(self.config.get('poll_interval_seconds', 10))
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(); config = validate_config(read_json(args.config))
    root = Path(config['service_root']).resolve(); root.mkdir(parents=True, exist_ok=True)
    gpu_lock = config.get('gpu_lock_path', str(Path(tempfile.gettempdir())/
        ('moss-checkpoint-eval-'+digest_bytes(getpass.getuser().encode())[:16]+'-gpu4.lock')))
    with DirectoryLock(root/'.service.lock'), DirectoryLock(gpu_lock):
        service = CheckpointEvalService(config)
        def stop(*_): service.stopped = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        atomic_json(root/'service_process.json',dict(pid=os.getpid(),host=socket.gethostname(),
                                                   started_at=time.time(),gpu=4))
        service.run()


if __name__ == '__main__':
    main()
