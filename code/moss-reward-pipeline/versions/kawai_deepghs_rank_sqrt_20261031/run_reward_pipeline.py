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

from reward_fusion import aesthetic_models, fuse, fuse_group, index_items, resolve_weights


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


def active_models(cfg):
    """One active-model contract drives scoring, fusion, provenance and reports."""
    models = aesthetic_models(cfg.get('kawai_enabled', True), cfg.get('enabled_models'))
    if ('enabled_models' in cfg and 'kawai_enabled' in cfg and
            cfg['kawai_enabled'] != ('kawai' in models)):
        raise ValueError('kawai_enabled contradicts enabled_models')
    return models


def rank_groups(items, cfg):
    """Validate complete groups before loading any reward model.

    Checkpoint validation shares scoring hardware, never a ranking population.
    Its frozen absolute reward remains authoritative for quality and retention.
    """
    mode = cfg.get('fusion_mode', 'frozen_calibration')
    if mode == 'frozen_calibration':
        return []
    if mode != 'rank_power_rank':
        raise ValueError('Unknown fusion_mode')
    gamma = cfg.get('rank_power_gamma', .5)
    if type(gamma) not in (float, int) or not 0 < gamma <= 1:
        raise ValueError('rank_power_gamma must be finite and in (0, 1]')
    if cfg.get('sdg_enabled', True) or cfg['artifact_weight'] != 0:
        raise ValueError('rank_power_rank requires SDG disabled and artifact_weight=0')
    if cfg.get('checkpoint_eval_reward_mode', 'frozen_calibration') != 'frozen_calibration':
        raise ValueError('Fixed validation must retain frozen_calibration quality reward')
    sizes = dict(training=cfg.get('rank_training_group_size', 8),
                 checkpoint_eval=cfg.get('rank_eval_group_size', 4))
    if any(type(k) is not int or k < 2 for k in sizes.values()):
        raise ValueError('Rank group sizes must be integers >= 2')
    grouped = {}
    for index, row in enumerate(items):
        checkpoint = row.get('checkpoint_eval')
        if checkpoint is not None:
            if not isinstance(checkpoint, dict) or not checkpoint.get('job_id'):
                raise ValueError('Checkpoint rank group requires an explicit job_id')
            domain, job = 'checkpoint_eval', str(checkpoint['job_id'])
        else:
            domain, job = 'training', ''
        key = (domain, job, row['prompt_id'])
        grouped.setdefault(key, []).append(index)
    for (domain, job, prompt), indices in grouped.items():
        if len(indices) != sizes[domain]:
            raise ValueError(f'Incomplete {domain} rank group: expected {sizes[domain]}, got {len(indices)}')
    return [(key[0], indices) for key, indices in grouped.items()]


def apply_rank_fusion(details, items, calibration, cfg, weights):
    groups = rank_groups(items, cfg)
    if not groups:
        return details
    output = [None] * len(details)
    for domain, indices in groups:
        rows = [dict(details[i], evaluation_domain=domain) for i in indices]
        ranked = fuse_group(rows, calibration, gamma=cfg.get('rank_power_gamma', .5),
                            weights=weights, kawai_enabled='kawai' in active_models(cfg),
                            enabled_models=list(active_models(cfg)), expected_group_size=len(indices))
        tied = len({r['rank_reward'] for r in ranked}) < len(ranked)
        for index, row in zip(indices, ranked):
            if domain == 'checkpoint_eval':
                row['reward'] = row['frozen_reward']
                row['aesthetic_reward'] = row['frozen_aesthetic_reward']
                row['reward_semantics'] = 'frozen_calibration_validation'
            else:
                row['reward_semantics'] = 'rank_power_rank_training'
            # The native training auditor already flattens this namespace. No
            # change to the frozen DiT/training source is needed for new charts.
            row['components'] = dict(frozen_reward=row['frozen_reward'],
                rank_reward=row['rank_reward'], rank_fused_score=row['rank_fused_score'],
                final_rank_has_ties=float(tied),
                model_rank_percentiles=row['model_rank_percentiles'],
                rank_power_scores=row['rank_component_scores'],
                rank_contributions=row['weighted_rank_contributions'])
            output[index] = row
    return output


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
    models = active_models(cfg)
    if 'kawai' in models or cfg.get('sdg_enabled', True):
        files.append(Path(__file__).with_name('sdg_parallel.py'))
    if 'kawai' in models:
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
    if 'hps' in models:
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
    enabled_models = list(active_models(cfg))
    kawai_enabled = 'kawai' in enabled_models
    batch_stages = [name for name in ('deepghs', 'hps') if name in enabled_models]
    sdg_enabled = cfg.get('sdg_enabled', True)
    if type(sdg_enabled) is not bool:
        raise ValueError('sdg_enabled must be boolean')
    if not sdg_enabled and cfg['artifact_weight'] != 0:
        raise ValueError('Disabled SDG requires artifact_weight=0')
    items = canonical_items(request)
    rank_groups(items, cfg)
    calibration = json.loads(Path(cfg['calibration']).read_text())
    fusion_weights = resolve_weights(calibration, weights=cfg.get('fusion_weights'),
                                     kawai_enabled=kawai_enabled, enabled_models=enabled_models)
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
                           stage_order=batch_stages+(['kawai'] if kawai_enabled else [])+(['sdg'] if sdg_enabled else []),
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
        # Only selected heads may construct workers or read model-specific files.
        if batch_cfg['stage_order'] != batch_stages or 'tagger' in batch_cfg:
            raise ValueError('Batch stage_order must exactly match active DeepGHS/HPS heads; no tagger')
        aligned = {}
        if batch_stages:
            batch_name = '_'.join(batch_stages)
            batch_out=archive/(batch_name+'.json')
            if not batch_out.exists() or json.loads(batch_out.read_text()).get('status')!='complete':
                run_program([cfg['controller_python'],str(Path(cfg['batch_root'])/'run.py'),
                             '--config',str(archive/'batch.config.json'),'--input',str(archive/'input.json'),
                             '--output',str(batch_out),'--stages',*batch_stages,'--resume'],
                             archive/(batch_name+'.log'),cfg['stage_timeout'])
            batch=json.loads(batch_out.read_text())
            aligned=index_items(batch['items'],[r['id'] for r in items],batch_name)
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
            key=row['id']
            raw, model_details = {}, {}
            for name in batch_stages:
                component = aligned[key][name]
                if component.get('status') != 'ok':
                    raise ValueError('Invalid component score: '+name)
                raw[name] = component['percentile' if name == 'deepghs' else 'mu']
                model_details[name] = component
            if kawai_enabled:
                raw['kawai']=k[key]['kawai']
                model_details['kawai']=k[key]
            if sdg_enabled:
                raw['sdg_artifact']=s[key]['artifact_penalty']
            fused=fuse(raw,calibration,cfg['artifact_weight'],weights=fusion_weights,
                       sdg_enabled=sdg_enabled,kawai_enabled=kawai_enabled,enabled_models=enabled_models)
            detail=dict(id=key,global_id=fingerprint[:24]+':'+key,collection_id=fingerprint[:24],
                        index=row.get('index'),prompt_id=row['prompt_id'],
                        image=row['image'],image_sha256=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest(),
                        raw_scores=raw,**fused,
                        model_details=model_details,
                        archive=str(archive),config_fingerprint=fingerprint)
            if sdg_enabled:
                detail.update(artifact_count=s[key]['artifact_count'],
                              ignored_misalignment_count=s[key]['ignored_misalignment_count'])
                detail['model_details']['sdg']=s[key]
            details.append(detail)
        details = apply_rank_fusion(details, items, calibration, cfg, fusion_weights)
        with (archive/'per_image.jsonl').open('w',encoding='utf-8') as stream:
            for row in details:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        columns=['id','prompt_id','image']+enabled_models+['aesthetic_reward','reward']
        if cfg.get('fusion_mode') == 'rank_power_rank':
            columns += ['frozen_reward', 'rank_reward', 'rank_fused_score', 'reward_semantics']
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
        if cfg.get('fusion_mode') == 'rank_power_rank':
            response.update(fusion_mode='rank_power_rank', rank_power_gamma=cfg.get('rank_power_gamma', .5),
                            contribution_semantics='frozen_calibration_diagnostic')
        summary=dict(images=len(items),mean=statistics.mean(scores),
             std=statistics.pstdev(scores),min=min(scores),max=max(scores),
             sdg_enabled=sdg_enabled,kawai_enabled=kawai_enabled,enabled_models=enabled_models,
             artifact_weight=cfg['artifact_weight'],aesthetic_weights=fusion_weights)
        if cfg.get('fusion_mode') == 'rank_power_rank':
            domains = {}
            for domain in ('training', 'checkpoint_eval'):
                rows = [r for r in details if r['evaluation_domain'] == domain]
                if rows:
                    values = [r['reward'] for r in rows]
                    domains[domain] = dict(images=len(rows), reward_semantics=rows[0]['reward_semantics'],
                        mean=statistics.mean(values), std=statistics.pstdev(values),
                        frozen_reward_mean=statistics.mean(r['frozen_reward'] for r in rows),
                        rank_reward_mean=statistics.mean(r['rank_reward'] for r in rows))
            summary.update(fusion_mode='rank_power_rank', rank_power_gamma=cfg.get('rank_power_gamma', .5),
                contribution_semantics='frozen_calibration_diagnostic', domains=domains,
                mean_semantics='mixed_domains_not_quality' if len(domains)>1 else next(iter(domains.values()))['reward_semantics'])
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
