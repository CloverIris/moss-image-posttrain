#!/usr/bin/env python3
"""Four sequential batch stages, four compute GPUs; file input and output."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from defect_prompt import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent
STAGES = ['deepghs', 'tagger', 'hps', 'gemma']
DEFECT_STAGES = ('gemma', 'qwen')
SUPPORTED_STAGES = STAGES + ['qwen']


def stage_order(config, requested=None):
    stages = requested if requested is not None else config.get('stage_order', STAGES)
    if not isinstance(stages, list) or not stages or len(stages) != len(set(stages)):
        raise ValueError('stage_order must be a non-empty list without duplicates')
    if any(stage not in SUPPORTED_STAGES or stage not in config for stage in stages):
        raise ValueError('Every selected stage must be supported and configured')
    return stages


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_items(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.jsonl':
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            if 'items' in data:
                data = data['items']
            elif 'prompts' in data:
                data = [dict(id=str(group['prompt_id'])+'_'+str(run['roll_id']),
                             prompt_id=group['prompt_id'], roll_id=run['roll_id'],
                             image=run.get('rollout_image') or run['image'],
                             prompt=group.get('natural_language_prompt', ''),
                             tags=group.get('original_tags', []))
                        for group in data['prompts'] for run in group['rollouts']]
            else:
                raise ValueError('JSON object must contain items or prompts')
    if not isinstance(data, list) or not data:
        raise ValueError('Input must contain a non-empty image list')
    result, identities = [], set()
    for index, item in enumerate(data):
        image = Path(item['image']).expanduser()
        image = (image if image.is_absolute() else path.parent/image).resolve()
        if not image.is_file(): raise ValueError(f'Image missing: {image}')
        identity = str(item.get('id') or '_'.join(str(item[k]) for k in ('prompt_id', 'roll_id') if k in item) or index)
        if identity in identities: raise ValueError(f'Duplicate id: {identity}')
        identities.add(identity)
        prompt = item.get('prompt', '')
        if not isinstance(prompt, str): raise ValueError('prompt must be a string')
        tags = item.get('tags', [])
        if isinstance(tags, str): tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError('tags must be a string list or comma-separated string')
        result.append(dict(item, id=identity, image=str(image), prompt=prompt, tags=tags))
    return result


def inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): dict(uuid=parts[1].strip(), used_mib=int(parts[2]))
            for line in text.splitlines() if (parts := line.split(','))}


def split_groups(items, replicas):
    """Keep each prompt intact and balance the number of images per DP replica."""
    grouped = defaultdict(list)
    for item in items:
        key = item.get('prompt_id') or item['prompt'] or item['id']
        grouped[key].append(item)
    shards = [[] for _ in range(replicas)]
    for group in sorted(grouped.values(), key=lambda rows: -len(rows)):
        index = min(range(replicas), key=lambda i: len(shards[i]))
        shards[index].extend(group)
    return shards


def descendants(roots):
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    edges = [tuple(map(int, line.split())) for line in rows.splitlines()]
    owned = set(roots)
    while True:
        found = {pid for pid, parent in edges if parent in owned}
        if found <= owned: return owned
        owned.update(found)


def stop_owned(processes, known):
    # vLLM may create child process groups; also terminate descendants captured
    # while their owning driver was alive. Never select processes by model name.
    owned = descendants([p.pid for p in processes if p.poll() is None]) | known
    for pid in sorted(owned, reverse=True):
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    deadline = time.monotonic()+20
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(.2)
    for pid in sorted(owned, reverse=True):
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass


def stage_metrics(stage, replicas, wall):
    metrics = [r['metrics'] for r in replicas]
    start = min(m['inference_started_at'] for m in metrics)
    end = max(m['inference_finished_at'] for m in metrics)
    new_images = sum(m['images'] for m in metrics)
    return dict(stage=stage, dp_replicas=len(metrics),
                tp_size=metrics[0].get('tp_size', len(replicas[0]['gpus'])),
                pp_size=metrics[0].get('pp_size', 1),
                images=sum(m['total_images'] for m in metrics), newly_scored_images=new_images,
                stage_wall_seconds=wall,
                stage_images_per_second=new_images/wall if new_images else None,
                inference_span_seconds=end-start,
                inference_images_per_second=new_images/(end-start) if new_images and end>start else None,
                generated_tokens=sum(m.get('generated_tokens', 0) for m in metrics),
                errors=sum(m.get('errors', 0) for m in metrics), replicas=replicas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config.json'))
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stages', nargs='+', choices=SUPPORTED_STAGES,
                        help='Run only these stages, in the specified order')
    args = parser.parse_args()
    def terminate(signum, frame): raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    stages = stage_order(config, args.stages)
    for stage in stages:
        if stage in DEFECT_STAGES and config[stage].get('prompt_version') != PROMPT_VERSION:
            raise ValueError(stage+' requires prompt_version='+PROMPT_VERSION+'; use a new output path')
    items = load_items(args.input)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(output.stem+'.work')
    identity = dict(config=config, items=items)
    if args.stages is not None: identity['selected_stages'] = stages
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    gpus = inventory()
    reserved = set(config.get('reserved_devices', [4]))
    plans = {}
    compute = None
    for stage in stages:
        devices = config[stage]['replica_devices']
        flat = [i for replica in devices for i in replica]
        if stage in DEFECT_STAGES:
            llm = config[stage]['llm']
            world = int(llm.get('tensor_parallel_size', 1)) * int(llm.get('pipeline_parallel_size', 1))
        else: world = 2 if stage == 'hps' else 1
        if world not in (1, 2, 4): raise ValueError('Per-replica GPU count must divide four')
        expected = [world] * (4 // world)
        if list(map(len, devices)) != expected or len(set(flat)) != 4 or any(i not in gpus for i in flat):
            raise ValueError(f'{stage} requires four distinct GPUs grouped as {expected}')
        if reserved & set(flat): raise ValueError(f'{stage} selects a reserved GPU')
        if compute is not None and compute != set(flat): raise ValueError('All stages must use the same four compute GPUs')
        compute = set(flat)
        plans[stage] = split_groups(items, len(devices))
    plan = dict(images=len(items), stage_order=stages, reserved_devices=sorted(reserved),
                stages={stage: [dict(dp_rank=i, gpus=devices, images=len(plans[stage][i]))
                                for i, devices in enumerate(config[stage]['replica_devices'])] for stage in stages},
                output=str(output))
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    import fcntl
    lock = output.with_name(output.name+'.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error: raise RuntimeError('Another job is writing this output') from error
    if output.exists() and not args.resume: raise FileExistsError('Output exists; choose a new file or --resume')
    if work.exists():
        if not args.resume: raise FileExistsError('Work exists; choose a new file or --resume')
        if json.loads((work/'run.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Resume input/config differs from the original job')
    else:
        work.mkdir()
        save(work/'run.json', dict(fingerprint=fingerprint, input=str(Path(args.input).resolve()), config=config, plan=plan))
        save(work/'input.json', dict(items=items))
    if args.resume and output.exists():
        prior=json.loads(output.read_text())
        if prior.get('status')=='complete' and prior.get('image_count')==len(items):
            print(json.dumps(dict(event='already_complete', output=str(output))),flush=True); return

    start=time.monotonic()
    all_results={}
    all_metrics={}
    for stage in stages:
        stage_dir=work/stage
        stage_dir.mkdir(exist_ok=True)
        jobs=[]
        active=[]
        for rank, devices in enumerate(config[stage]['replica_devices']):
            shard=plans[stage][rank]
            if not shard: continue
            directory=stage_dir/('dp%02d'%rank)
            directory.mkdir(exist_ok=True)
            job=dict(rank=rank, devices=devices, directory=directory, items=shard)
            jobs.append(job)
            if not (directory/(stage+'.metrics.json')).exists(): active.append(job)
        barrier=stage_dir/'barriers'/str(time.time_ns())
        if active: barrier.mkdir(parents=True)
        current=inventory()
        if any(current[gpu]['used_mib']>config.get('busy_memory_mib',1500) for job in active for gpu in job['devices']):
            raise RuntimeError(f'{stage}: selected compute GPU is busy; no unrelated job will be stopped')
        processes=[]
        known=set()
        running={}
        stage_start=time.monotonic()
        try:
            for job in active:
                rank, devices, directory=job['rank'],job['devices'],job['directory']
                local=json.loads(json.dumps(config))
                local[stage]['devices']=devices
                local['_batch']=dict(dp_rank=rank, participants=[j['rank'] for j in active],
                                     barrier_dir=str(barrier), barrier_timeout_seconds=900)
                save(directory/'config.json',local)
                save(directory/'input.json',dict(items=job['items']))
                settings=config[stage]
                python=settings['python']
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus[i]['uuid'] for i in devices),
                         CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                         HF_HOME=config['hf_home'],PYTHONUNBUFFERED='1',
                         OMP_NUM_THREADS=str(settings.get('cpu_threads',2)),VLLM_WORKER_MULTIPROC_METHOD='spawn')
                env['PATH']=str(Path(python).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
                if stage in DEFECT_STAGES: env.pop('VLLM_MARLIN_INPUT_DTYPE',None)
                if stage=='deepghs':
                    env['ONNX_MODE']='cuda'
                    env['LD_LIBRARY_PATH']=':'.join(settings.get('cuda_library_paths',[])+[env.get('LD_LIBRARY_PATH','')])
                command=[python]
                if stage=='hps':command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
                command += [str(ROOT/'worker.py'),'--stage',stage,'--work',str(directory)]
                with (directory/(stage+'.log')).open('a') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(process)
                running[rank]=process
                job['process']=process
                print(json.dumps(dict(event='started',stage=stage,dp_rank=rank,pid=process.pid,
                                      gpus=devices,images=len(job['items']))),flush=True)
            last_sample=0
            while running:
                known.update(descendants([p.pid for p in processes if p.poll() is None]))
                for rank,process in list(running.items()):
                    code=process.poll()
                    if code is None:continue
                    if code!=0:raise RuntimeError(f'{stage} DP{rank} exited {code}; see {stage_dir/("dp%02d"%rank)/(stage+".log")}')
                    if not (stage_dir/('dp%02d'%rank)/(stage+'.metrics.json')).exists():
                        raise RuntimeError(f'{stage} DP{rank} exited without final metrics')
                    del running[rank]
                if time.monotonic()-last_sample>=5:
                    sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                                    '--format=csv,noheader,nounits'],text=True)
                    with (work/'gpu_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(stage=stage,elapsed_seconds=time.monotonic()-start,gpus=sample.splitlines()))+'\n')
                    last_sample=time.monotonic()
                if running:time.sleep(1)
        finally:
            stop_owned(processes, known)

        # CUDA teardown can finish just after process exit. Wait before handing
        # the same GPUs to the next model, without touching the reserved GPU.
        release_deadline=time.monotonic()+30
        while active and any(inventory()[gpu]['used_mib']>config.get('busy_memory_mib',1500) for gpu in compute):
            if time.monotonic()>release_deadline:raise RuntimeError(f'{stage}: compute GPUs were not released after exit')
            time.sleep(.5)

        table={}
        replica_metrics=[]
        for job in jobs:
            directory=job['directory']
            rows=[json.loads(line) for line in (directory/(stage+'.jsonl')).read_text().splitlines() if line.strip()]
            shard_ids={item['id'] for item in job['items']}
            if len(rows)!=len(shard_ids) or {row['id'] for row in rows}!=shard_ids:
                raise ValueError(f'{stage} DP{job["rank"]}: missing or duplicate image results')
            table.update({row['id']:row['result'] for row in rows})
            replica_metrics.append(dict(dp_rank=job['rank'],gpus=job['devices'],
                                        metrics=json.loads((directory/(stage+'.metrics.json')).read_text())))
        if len(table)!=len(items):raise ValueError(f'{stage}: incomplete merged result')
        all_results[stage]=table
        existing=stage_dir/'stage.metrics.json'
        if active or not existing.exists():
            metrics=stage_metrics(stage,replica_metrics,time.monotonic()-stage_start)
            save(existing,metrics)
        else:metrics=json.loads(existing.read_text())
        all_metrics[stage]=metrics
        save(stage_dir/'results.json',dict(items=[dict(id=item['id'],result=table[item['id']]) for item in items]))
        print(json.dumps(dict(event='stage_complete',stage=stage,stage_wall_seconds=metrics['stage_wall_seconds'],
                              inference_images_per_second=metrics['inference_images_per_second'])),flush=True)

    elapsed=time.monotonic()-start
    combined=[dict(item,**{stage:all_results[stage][item['id']] for stage in stages}) for item in items]
    save(output,dict(schema_version=2,status='complete',image_count=len(items),config=config,metrics=all_metrics,
                     orchestration=dict(mode='sequential_batch_stages',stage_order=stages,reserved_devices=sorted(reserved),
                                        wall_seconds=elapsed,images_per_second=len(items)/elapsed if not args.resume else None,
                                        resumed=args.resume),items=combined))
    print(json.dumps(dict(event='complete',images=len(items),wall_seconds=elapsed,
                          images_per_second=len(items)/elapsed if not args.resume else None,output=str(output))),flush=True)


if __name__=='__main__':main()
