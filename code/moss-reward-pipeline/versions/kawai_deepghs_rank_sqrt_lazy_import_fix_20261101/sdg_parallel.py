"""CPU-only DP launcher: durable shard files and strict ID-based merging.

No Torch, JAX or vLLM import occurs here. The existing SDG worker is the only
process that loads a model. GPU4 is never selected by run_sdg_parallel.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def _identities(items):
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Each image must be an object')
        identity = item.get('id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Every image needs a unique nonempty string id')
        seen.add(identity)
    return seen


def balanced_shards(items, replicas=4):
    """Keep each prompt intact, balancing image counts; preserve group order."""
    if type(replicas) is not int or replicas < 1:
        raise ValueError('replicas must be a positive integer')
    if not isinstance(items, list) or not items:
        raise ValueError('items must be a nonempty list')
    _identities(items)
    groups = OrderedDict()
    for item in items:
        key = item.get('prompt_id')
        if type(key) not in (str, int) or key == '':
            raise ValueError('Each image needs an explicit string/integer prompt_id')
        # An integer 1 and a string "1" are different source prompt identities.
        groups.setdefault((type(key).__name__, key), []).append(item)
    shards = [[] for _ in range(replicas)]
    for group in sorted(groups.values(), key=lambda rows: -len(rows)):
        rank = min(range(replicas), key=lambda i: len(shards[i]))
        shards[rank].extend(group)
    return shards


def _snapshot():
    if os.name != 'posix':
        return []
    result = subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True, check=True)
    return [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]


def _descendants(roots, snapshot):
    owned = set(roots)
    while True:
        found = {pid for pid, ppid in snapshot if ppid in owned}
        if found <= owned:
            return owned - set(roots)
        owned.update(found)


def _stop_owned(processes, known_children):
    """Terminate only launched roots/groups and observed descendants."""
    roots = [p.pid for p in processes if p.poll() is None]
    try:
        children = set(known_children) | _descendants(roots, _snapshot())
    except (OSError, subprocess.SubprocessError):
        children = set(known_children)
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    for pid in children:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(.05)
    for process in processes:
        if process.poll() is None:
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
    if os.name == 'posix':
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            pass


def run_jobs(jobs, timeout=900, *, poll_interval=.2):
    """Launch all supplied pending jobs concurrently and fail as one stage.

    Each job: {argv:[...], log:PATH, output:PATH, cwd?:PATH, env?:dict,
               name?:str}. Existing output paths are rejected: the caller must
    validate/cache completed work before passing only pending jobs here.
    Returns process timing records in job order. Does not parse output schemas.
    """
    if not isinstance(jobs, list):
        raise ValueError('jobs must be a list')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    for job in jobs:
        if not isinstance(job.get('argv'), list) or not job['argv'] or any(not isinstance(x, str) for x in job['argv']):
            raise ValueError('Every job needs a nonempty argv list of strings')
        if Path(job['output']).exists():
            raise FileExistsError('run_jobs only accepts pending jobs without an existing output')
    processes, logs, children, records = [], [], set(), []
    started = time.monotonic()
    try:
        for rank, job in enumerate(jobs):
            log_path = Path(job['log']); log_path.parent.mkdir(parents=True, exist_ok=True)
            Path(job['output']).parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open('a', encoding='utf-8'); logs.append(stream)
            env = dict(os.environ); env.update(job.get('env') or {})
            process = subprocess.Popen(job['argv'], cwd=job.get('cwd'), env=env,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            records.append(dict(name=job.get('name', str(rank)), pid=process.pid,
                                output=str(job['output']), log=str(log_path),
                                started_at=time.time()))
            print(json.dumps(dict(event='reward_replica_started', **records[-1])), flush=True)
        pending = set(range(len(processes)))
        while pending:
            children.update(_descendants([p.pid for p in processes if p.poll() is None], _snapshot()))
            for rank in list(pending):
                code = processes[rank].poll()
                if code is None:
                    continue
                records[rank].update(exit_code=code, finished_at=time.time(),
                                     wall_seconds=time.monotonic()-started)
                if code != 0:
                    raise RuntimeError(f"Reward replica {records[rank]['name']} exited {code}; see {records[rank]['log']}")
                if not Path(jobs[rank]['output']).is_file():
                    raise RuntimeError(f"Reward replica {records[rank]['name']} exited without an output; see {records[rank]['log']}")
                pending.remove(rank)
            if pending:
                if time.monotonic()-started > timeout:
                    raise TimeoutError(f'Reward stage timed out after {timeout} seconds')
                time.sleep(poll_interval)
        return records
    finally:
        _stop_owned(processes, children)
        for stream in logs:
            stream.close()


def merge_shard_responses(items, shards, responses):
    """Match each score/detail pair by its explicit image id, never shard order."""
    expected = _identities(items)
    if len(shards) != len(responses):
        raise ValueError('Expected exactly one response per shard')
    table = {}
    for shard, response in zip(shards, responses):
        shard_ids = _identities(shard)
        if not isinstance(response, dict):
            raise ValueError('Shard response must be an object')
        scores, details = response.get('scores'), response.get('details')
        if not isinstance(scores, list) or not isinstance(details, list) or len(scores) != len(shard) or len(details) != len(shard):
            raise ValueError('Incomplete SDG shard scores/details')
        found = set()
        for score, detail in zip(scores, details):
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('SDG failure/nonfinite score cannot become zero penalty')
            if not isinstance(detail, dict) or detail.get('mode') != 'artifact_only':
                raise ValueError('Expected artifact_only SDG detail')
            if detail.get('alpha_misalignment') != 0.:
                raise ValueError('misalignment must not contribute to reward')
            identity = detail.get('id')
            if identity not in shard_ids or identity in found or identity in table:
                raise ValueError('SDG shard returned missing, duplicate or foreign image id')
            if detail.get('reward') != score:
                raise ValueError('SDG scores/details disagree')
            found.add(identity); table[identity] = (float(score), detail)
        if found != shard_ids:
            raise ValueError('Incomplete SDG shard identity set')
    if set(table) != expected:
        raise ValueError('Merged SDG output lost or added image IDs')
    return dict(scores=[table[row['id']][0] for row in items],
                details=[table[row['id']][1] for row in items], mode='artifact_only')


def run_sdg_parallel(items, sdg_config, sdg_python, sdg_root, work_dir, timeout=900):
    """Run four independent SDG BF16 replicas on physical GPUs 0,1,2,3.

    The outer pipeline owns run/config fingerprints. Completed per-shard
    response files are reused only after validating complete matching IDs.
    """
    if sdg_config.get('mode') != 'artifact_only' or sdg_config.get('alpha_misalignment', 0.) != 0.:
        raise ValueError('SDG must use artifact_only, alpha_misalignment=0')
    devices = sdg_config.get('replica_devices', [[0],[1],[2],[3]])
    if not isinstance(devices, list) or len(devices) != 4 or any(not isinstance(v,list) or len(v) != 1 for v in devices):
        raise ValueError('SDG requires four single-GPU DP replicas')
    flat = [v[0] for v in devices]
    if any(type(v) is not int for v in flat) or set(flat) != {0,1,2,3}:
        raise ValueError('Use the four GB203(SM120) GPUs 0..3; GPU4 is reserved')
    shards = balanced_shards(items, 4)
    root, work = Path(sdg_root), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    jobs, response_paths, reused = [], [], []
    for rank, (shard, physical) in enumerate(zip(shards, flat)):
        directory = work / ('dp%02d'%rank); directory.mkdir(exist_ok=True)
        config_path, request_path, response_path = directory/'config.json', directory/'request.json', directory/'response.json'
        response_paths.append(response_path)
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding='utf-8'))
            merge_shard_responses(shard, [shard], [response])
            reused.append(rank)
            continue
        settings = dict(sdg_config, device='cuda:'+str(physical))
        request = dict(version=1, items=shard, config={k:settings[k] for k in
                       ('model_path','device','batch_size','metric','threads') if k in settings})
        config_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if not shard:
            response_path.write_text(json.dumps(dict(scores=[],details=[],mode='artifact_only'))+'\n')
            continue
        jobs.append(dict(name='sdg_dp%02d'%rank,
                         argv=[str(sdg_python),str(root/'run_sdg_reward.py'),'--config',str(config_path),
                               '--request',str(request_path),'--output',str(response_path)],
                         cwd=str(root), log=str(directory/'worker.log'), output=str(response_path)))
    timing = run_jobs(jobs, timeout=timeout)
    responses = [json.loads(path.read_text(encoding='utf-8')) for path in response_paths]
    merged = merge_shard_responses(items, shards, responses)
    merged['parallel'] = dict(mode='DP4', devices=flat, shard_images=[len(s) for s in shards],
                              reused_shards=reused, launched_jobs=timing,
                              shard_responses=[str(p) for p in response_paths])
    (work/'response.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return merged
