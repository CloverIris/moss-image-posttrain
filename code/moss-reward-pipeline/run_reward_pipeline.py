#!/usr/bin/env python3
"""Durable reward command: DeepGHS/HPS plus optional Kawai/SDG, fixed fusion."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from reward_fusion import aesthetic_models, fuse, index_items, resolve_weights


def run_sdg_parallel(*args, **kwargs):
    # Disabled stages must not import their worker implementation.
    from sdg_parallel import run_sdg_parallel as implementation
    return implementation(*args, **kwargs)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def canonical_items(request):
    if request.get('version') != 1 or not request.get('items'):
        raise ValueError('Expected version=1 and a nonempty items list')
    items, identities = [], set()
    for i, row in enumerate(request['items']):
        image = Path(row['image']).resolve()
        if not image.is_file() or not isinstance(row.get('prompt'), str):
            raise ValueError('Each input needs an existing image and unchanged prompt')
        prompt_id = str(row.get('sample_id', row.get('prompt_index', row.get('prompt_id', ''))))
        if not prompt_id:
            raise ValueError('Each input needs sample_id, prompt_index or prompt_id')
        key = str(row.get('id') or f"p{prompt_id}_r{row.get('rollout_id', row.get('roll_id', i))}_i{row.get('index', i)}")
        if key in identities:
            raise ValueError('Duplicate input image ID')
        identities.add(key)
        items.append(dict(row, id=key, prompt_id=prompt_id,
                          image=str(image), prompt=row['prompt'],
                          image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    return items


def run_program(argv, log, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nATTEMPT '+str(time.time())+'\n');stream.flush()
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise
    if code:
        tail = log.read_text(errors='replace')[-6000:]
        raise RuntimeError(f'Stage exited {code}; full log: {log}\n{tail}')


def provenance(cfg):
    files = [Path(__file__),Path(__file__).with_name('reward_fusion.py'),
             Path(cfg['batch_root'])/'run.py',Path(cfg['batch_root'])/'worker.py']
    if cfg.get('kawai_enabled', True) or cfg.get('sdg_enabled', True):
        files.append(Path(__file__).with_name('sdg_parallel.py'))
    if cfg.get('kawai_enabled', True):
        files.append(Path(__file__).with_name('score_kawai.py'))
    if cfg.get('sdg_enabled', True):
        files.extend([Path(cfg['sdg_root'])/'run_sdg_reward.py',
                      Path(cfg['sdg_root'])/'sdg_artifact_reward.py',
                      Path(cfg['sdg_root'])/'sdg_output.py',Path(cfg['sdg_config'])])
        sdg_settings = json.loads(Path(cfg['sdg_config']).read_text())
        sdg_repo = Path(sdg_settings['repo'])
        files.extend([Path(cfg['sdg_root'])/'run_sdg_pilot.py',
                  sdg_repo/'boxflow_grpo/src/flow_factory/rewards/dense_grpo_rewards.py',
                  sdg_repo/'boxflow_grpo/src/flow_factory/utils/pixel_reward_map.py'])
    batch=json.loads(Path(cfg['batch_config']).read_text())
    files.extend([Path(batch['hps']['native_config']),
                  Path(batch['hps']['project'])/'hpsv3pp_torch/engine.py'])
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run_kawai_parallel(items, cfg, archive):
    from sdg_parallel import balanced_shards, run_jobs
    devices=cfg.get('kawai_devices',[0,1,2,3])
    if len(devices)!=4 or set(devices)!={0,1,2,3}:
        raise ValueError('Kawai DP4 requires physical GPUs 0..3; GPU4 is reserved')
    shards=balanced_shards(items,4)
    jobs,pending,paths=[],[],[]
    for rank,(physical,shard) in enumerate(zip(devices,shards)):
        directory=archive/'kawai_dp'/('dp%02d'%rank);directory.mkdir(parents=True,exist_ok=True)
        output=directory/'response.json';paths.append(output)
        if output.exists():
            index_items(json.loads(output.read_text())['items'],[x['id'] for x in shard],'kawai cached shard')
            continue
        if not shard:
            save(output,dict(items=[],metadata=dict(device=physical,empty=True)))
            continue
        attempt=directory/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        save(attempt/'input.json',dict(items=shard))
        local=dict(cfg['kawai'],device=physical,input=str(attempt/'input.json'),output=str(attempt/'result.json'))
        save(attempt/'config.json',local)
        jobs.append(dict(name='kawai_dp%02d'%rank,
                         argv=[cfg['kawai_python'],str(Path(__file__).with_name('score_kawai.py')),'--config',str(attempt/'config.json')],
                         log=str(attempt/'worker.log'),output=str(attempt/'result.json')))
        pending.append((attempt/'result.json',output,shard))
    timing=run_jobs(jobs,timeout=cfg['stage_timeout'])
    for source,output,shard in pending:
        result=json.loads(source.read_text());index_items(result['items'],[x['id'] for x in shard],'kawai new shard');save(output,result)
    responses=[json.loads(p.read_text()) for p in paths]
    table=index_items([row for result in responses for row in result['items']],[r['id'] for r in items],'kawai DP4')
    return dict(items=[table[row['id']] for row in items],metadata=dict(parallel='DP4',devices=devices,
                shard_images=[len(s) for s in shards],replicas=[r['metadata'] for r in responses],launched_jobs=timing))


def run(cfg, request, output):
    kawai_enabled = cfg.get('kawai_enabled', True)
    enabled_models = list(aesthetic_models(kawai_enabled))
    sdg_enabled = cfg.get('sdg_enabled', True)
    if type(sdg_enabled) is not bool:
        raise ValueError('sdg_enabled must be boolean')
    if not sdg_enabled and cfg['artifact_weight'] != 0:
        raise ValueError('Disabled SDG requires artifact_weight=0')
    items = canonical_items(request)
    calibration = json.loads(Path(cfg['calibration']).read_text())
    fusion_weights = resolve_weights(calibration, weights=cfg.get('fusion_weights'),
                                     kawai_enabled=kawai_enabled)
    batch_cfg=json.loads(Path(cfg['batch_config']).read_text())
    sdg_cfg=json.loads(Path(cfg['sdg_config']).read_text()) if sdg_enabled else None
    identity = dict(items=items, config=cfg, calibration=calibration,
                    batch_config=batch_cfg,sdg_config=sdg_cfg,code=provenance(cfg))
    fingerprint = digest(identity)
    context = request.get('context', {})
    root = Path(context.get('step_dir', cfg['archive_root'])).resolve()
    if not root.is_relative_to(Path(cfg['allowed_archive_root']).resolve()):
        raise ValueError('Reward archives must be on the configured durable NVMe volume')
    archive = root/'reward_models'/fingerprint[:24]
    archive.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (archive/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = archive/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing reward archive identity mismatch')
        save(manifest,dict(fingerprint=fingerprint,context=context,config=cfg,calibration=calibration,
                           code_hashes=identity['code'],image_count=len(items),
                           stage_order=['deepghs','hps']+(['kawai'] if kawai_enabled else [])+(['sdg'] if sdg_enabled else []),
                           tagger_enabled=False,sdg_enabled=sdg_enabled,
                           kawai_enabled=kawai_enabled,enabled_models=enabled_models,
                           fusion_weights=fusion_weights))
        save(archive/'request.json',request);save(archive/'input.json',dict(items=items))
        save(archive/'calibration.json',calibration)
        save(archive/'batch.config.json',batch_cfg)
        if sdg_enabled:
            save(archive/'sdg.config.json',sdg_cfg)
        snapshot=archive/'code'
        snapshot.mkdir(exist_ok=True)
        for index,path in enumerate(identity['code']):
            source=Path(path)
            (snapshot/(str(index)+'_'+source.name)).write_bytes(source.read_bytes())
        if (archive/'response.json').exists():
            response=json.loads((archive/'response.json').read_text())
            save(output,response);return response
        # The reused controller only contains the two requested model stages.
        if batch_cfg['stage_order'] != ['deepghs','hps'] or 'tagger' in batch_cfg:
            raise ValueError('Batch configuration must contain DeepGHS/HPS and no tagger')
        batch_out=archive/'deepghs_hps.json'
        if not batch_out.exists() or json.loads(batch_out.read_text()).get('status')!='complete':
            run_program([cfg['controller_python'],str(Path(cfg['batch_root'])/'run.py'),
                         '--config',str(archive/'batch.config.json'),'--input',str(archive/'input.json'),
                         '--output',str(batch_out),'--stages','deepghs','hps','--resume'],
                         archive/'deepghs_hps.log',cfg['stage_timeout'])
        batch=json.loads(batch_out.read_text())
        aligned=index_items(batch['items'],[r['id'] for r in items],'deepghs_hps')
        k={}
        if kawai_enabled:
            kawai_out=archive/'kawai.json'
            if not kawai_out.exists():
                save(kawai_out,run_kawai_parallel(items,cfg,archive))
            kawai=json.loads(kawai_out.read_text())
            k=index_items(kawai['items'],[r['id'] for r in items],'kawai')
        s={}
        if sdg_enabled:
            sdg_out=archive/'sdg.json'
            if not sdg_out.exists():
                settings=dict(sdg_cfg,replica_devices=[[g] for g in cfg.get('sdg_devices',[0,1,2,3])])
                save(sdg_out,run_sdg_parallel(items,settings,cfg['sdg_python'],cfg['sdg_root'],
                                            archive/'sdg_dp',timeout=cfg['stage_timeout']))
            sdg=json.loads(sdg_out.read_text())
            if len(sdg['scores'])!=len(items) or len(sdg['details'])!=len(items):
                raise ValueError('SDG response count mismatch')
            s=index_items(sdg['details'],[r['id'] for r in items],'sdg')
        details=[]
        for row in items:
            key=row['id'];d,h=aligned[key]['deepghs'],aligned[key]['hps']
            if d.get('status')!='ok' or h.get('status')!='ok':
                raise ValueError('Invalid component score')
            raw=dict(deepghs=d['percentile'],hps=h['mu'])
            if kawai_enabled:
                raw['kawai']=k[key]['kawai']
            if sdg_enabled:
                raw['sdg_artifact']=s[key]['artifact_penalty']
            fused=fuse(raw,calibration,cfg['artifact_weight'],weights=fusion_weights,
                       sdg_enabled=sdg_enabled,kawai_enabled=kawai_enabled)
            detail=dict(id=key,global_id=fingerprint[:24]+':'+key,collection_id=fingerprint[:24],
                        index=row.get('index'),prompt_id=row['prompt_id'],
                        image=row['image'],image_sha256=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest(),
                        raw_scores=raw,**fused,
                        model_details=dict(deepghs=d,hps=h),
                        archive=str(archive),config_fingerprint=fingerprint)
            if kawai_enabled:
                detail['model_details']['kawai']=k[key]
            if sdg_enabled:
                detail.update(artifact_count=s[key]['artifact_count'],
                              ignored_misalignment_count=s[key]['ignored_misalignment_count'])
                detail['model_details']['sdg']=s[key]
            details.append(detail)
        with (archive/'per_image.jsonl').open('w',encoding='utf-8') as stream:
            for row in details:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        columns=['id','prompt_id','image']+enabled_models+['aesthetic_reward','reward']
        if sdg_enabled:
            columns+=['sdg_artifact','artifact_count']
        with (archive/'scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in details:
                writer.writerow({**{k:row[k] for k in columns if k in row},**row['raw_scores']})
        scores=[row['reward'] for row in details]
        response=dict(scores=scores,details=details,archive=str(archive),fingerprint=fingerprint,
                      sdg_enabled=sdg_enabled,kawai_enabled=kawai_enabled,
                      enabled_models=enabled_models)
        summary=dict(images=len(items),mean=statistics.mean(scores),
             std=statistics.pstdev(scores),min=min(scores),max=max(scores),
             sdg_enabled=sdg_enabled,kawai_enabled=kawai_enabled,enabled_models=enabled_models,
             artifact_weight=cfg['artifact_weight'],aesthetic_weights=fusion_weights)
        if sdg_enabled:
            summary['artifact_image_fraction']=sum(r['artifact_count']>0 for r in details)/len(details)
        save(archive/'summary.json',summary)
        save(archive/'response.json',response)
        save(output,response)
        return response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    request=json.loads(args.request.read_text(encoding='utf-8-sig'))
    response=run(cfg,request,args.output)
    print(json.dumps(dict(event='composite_reward_complete',images=len(response['scores']),
                         archive=response['archive']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
