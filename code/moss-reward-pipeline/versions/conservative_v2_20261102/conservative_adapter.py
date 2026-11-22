"""Strict ID joins and prompt-conditioned guards around the conservative kernel."""
import hashlib,json,math
from dataclasses import asdict,replace
from pathlib import Path
import importlib.util,sys
_kernel_path=Path(__file__).with_name('conservative_reward.py')
_kernel_name='_moss_conservative_kernel_'+hashlib.sha256(str(_kernel_path.resolve()).encode()).hexdigest()[:20]
_kernel=sys.modules.get(_kernel_name)
if _kernel is None:
    _spec=importlib.util.spec_from_file_location(_kernel_name,_kernel_path);_kernel=importlib.util.module_from_spec(_spec);sys.modules[_kernel_name]=_kernel
    try:_spec.loader.exec_module(_kernel)
    except BaseException:sys.modules.pop(_kernel_name,None);raise
Candidate,HeadCalibration,RewardPolicy,TagRule=(_kernel.Candidate,_kernel.HeadCalibration,_kernel.RewardPolicy,_kernel.TagRule)
catalog_from_native_config,guard_from_native_tags,fuse_group,validate_training_contract=(_kernel.catalog_from_native_config,_kernel.guard_from_native_tags,_kernel.fuse_group,_kernel.validate_training_contract)

def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha_text(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def checked(reference,label):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError(label+' requires pinned path and SHA256')
    path=Path(reference['path']);expected=reference['sha256']
    if not path.is_absolute() or not isinstance(expected,str) or len(expected)!=64 or sha_file(path)!=expected:raise ValueError(label+' checksum/path mismatch')
    return json.loads(path.read_text(encoding='utf-8-sig'))

def groups(items,cfg):
    if cfg.get('fusion_mode')!='conservative_v2':raise ValueError('Expected conservative_v2 mode')
    if cfg.get('sdg_enabled',True) or cfg.get('artifact_weight')!=0 or set(cfg.get('enabled_models',[]))!={'kawai','deepghs'}:raise ValueError('Conservative v2 requires exactly Kawai/DeepGHS, no SDG/HPS')
    if cfg.get('checkpoint_eval_reward_mode','frozen_calibration')!='frozen_calibration':raise ValueError('Fixed heldout must keep frozen absolute reward')
    if any(k in cfg for k in ('rank_power_gamma','rank_training_group_size','rank_eval_group_size')):raise ValueError('Remove both rank stages/options for conservative mode')
    size=cfg.get('conservative_group_size')
    if type(size) is not int or size<2:raise ValueError('Explicit conservative_group_size required')
    table={}
    for i,row in enumerate(items):
        ev=row.get('checkpoint_eval');domain='checkpoint_eval' if ev is not None else 'training'
        if ev is not None and (not isinstance(ev,dict) or not ev.get('job_id')):raise ValueError('Checkpoint job identity missing')
        # Collection-local groups stay separate when the same dataset record repeats.
        key=(domain,str(ev['job_id']) if ev else '',str(row.get('prompt_index',row['prompt_id'])))
        table.setdefault(key,[]).append(i)
    for key,indices in table.items():
        expected=cfg.get('conservative_eval_group_size',4) if key[0]=='checkpoint_eval' else size
        if len(indices)!=expected:raise ValueError('Incomplete '+key[0]+' conservative group')
        if len({(items[i]['prompt_id'],items[i]['prompt'],items[i].get('negative_prompt','')) for i in indices})!=1:raise ValueError('Mixed original prompt/CFG texts in group')
    return [(k[0],v) for k,v in table.items()]

def load_policy(cfg):
    opts=cfg.get('conservative',{});raw=checked(opts.get('policy'),'conservative policy')
    if raw.get('schema')!=1 or raw.get('approved') is not True:raise ValueError('Conservative calibration is not approved; no production defaults')
    for field in ('guidance_gain','reference_mse_beta'):
        value=raw['learner'][field]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive calibrated learner '+field+' required')
    evidence=checked(raw['artifacts']['evidence'],'calibration evidence')
    if evidence.get('calibration_id')!=raw['calibration_id'] or type(evidence.get('human_labeled_images')) is not int or evidence['human_labeled_images']<=0:raise ValueError('Missing matching human calibration evidence')
    catalog_json=checked(raw['artifacts']['catalog'],'native catalog')
    catalog=catalog_from_native_config(catalog_json['tags'],catalog_json['tags_split'])
    targets=checked(raw['artifacts']['targets'],'target sidecar')
    if targets.get('schema')!=1 or targets.get('calibration_id')!=raw['calibration_id']:raise ValueError('Target rules/calibration mismatch')
    indexed={}
    for row in targets['targets']:
        key=row['prompt_id']
        if not isinstance(key,str) or not key or key in indexed:raise ValueError('Missing or duplicate sidecar prompt ID')
        for field in ('prompt_sha256','negative_prompt_sha256'):
            if not isinstance(row.get(field),str) or len(row[field])!=64:raise ValueError('Original prompt and negative text hashes required')
        rules=[TagRule(**r) for r in row['rules']]
        # Validate duplicate/contradictory rules even when no image is scored yet.
        guard_from_native_tags({},rules,catalog,min_reliability=raw['min_guard_reliability'])
        coverage=row['aesthetic_coverage']
        if type(coverage) not in (int,float) or not math.isfinite(coverage) or not 0<=coverage<=1:raise ValueError('Uncalibrated coverage')
        indexed[key]=(row,rules)
    head=raw['heads']
    policy=RewardPolicy(raw['calibration_id'],HeadCalibration(**head['kawai']),HeadCalibration(**head['deepghs']),raw['aesthetic_coverage'],raw['min_guard_reliability'],raw['aesthetic_cap'],cfg['conservative_group_size'])
    return dict(raw=raw,policy=policy,catalog=catalog,targets=indexed,policy_sha256=opts['policy']['sha256'],catalog_sha256=raw['artifacts']['catalog']['sha256'])

def validate_context(context,bundle):
    c=context.get('reward_contract') if isinstance(context,dict) else None
    if not c or c.get('name')!='conservative_v2' or c.get('reference_mode')!='base':raise ValueError('Live learner did not declare conservative frozen-Base contract')
    if c.get('calibration_id')!=bundle['policy'].calibration_id or c.get('policy_sha256')!=bundle['policy_sha256']:raise ValueError('Reward and learner calibration identities differ')
    validate_training_contract(c['covariance_options'],reference_mse_beta=c['reference_mse_beta'])
    wanted=bundle['raw']['learner']
    if c['covariance_options']['guidance_gain']!=wanted['guidance_gain'] or c['reference_mse_beta']!=wanted['reference_mse_beta']:raise ValueError('Learner gain/Base beta differ from calibrated policy')

def validate_predictions(response,items,bundle):
    if response.get('schema')!=1 or response.get('status')!='complete' or response.get('unfiltered') is not True or response.get('catalog_sha256')!=bundle['catalog_sha256']:raise ValueError('Guard requires full unfiltered probabilities from the pinned catalog')
    identity=bundle['raw']['tagger_identity']
    if response.get('tagger_identity')!=identity:raise ValueError('Tagger model/dtype/implementation identity differs from calibration')
    table={}
    for row in response.get('items',[]):
        key=row['id']
        if key in table:raise ValueError('Duplicate tagger image ID')
        table[key]=row
    if set(table)!={x['id'] for x in items}:raise ValueError('Tagger IDs differ from training image IDs')
    for item in items:
        row=table[item['id']]
        if row.get('image_sha256')!=item['image_sha256']:raise ValueError('Tagger/image SHA mismatch')
        if row.get('status') not in ('ok','error'):raise ValueError('Explicit tagger result status required')
        if row['status']=='ok':
            probs=row.get('probabilities')
            if not isinstance(probs,dict) or set(probs)!=set(bundle['catalog']):raise ValueError('Tagger categories incomplete')
            for category,names in bundle['catalog'].items():
                if set(probs[category])!=names:raise ValueError('Filtered or incomplete native probabilities')
                if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in probs[category].values()):raise ValueError('Invalid tag probability')
        elif not row.get('error'):raise ValueError('Failed tagger calls require an archived error')
    return table

def apply(details,items,cfg,bundle,response):
    grouped=groups(items,cfg);train=[items[i] for domain,indices in grouped if domain=='training' for i in indices]
    predicted=validate_predictions(response,train,bundle) if train else {}
    result=[dict(row,fusion_mode='conservative_v2',calibration_id=bundle['policy'].calibration_id,contributions_space='frozen_calibration') for row in details];diagnostics=[]
    for domain,indices in grouped:
        if domain=='checkpoint_eval':
            for i in indices:result[i].update(evaluation_domain=domain,reward_semantics='frozen_calibration_validation',frozen_reward=result[i]['reward'])
            continue
        candidates=[];guards=[]
        pid=items[indices[0]]['prompt_id'];target=bundle['targets'].get(pid)
        for i in indices:
            item=items[i];raw=details[i]['raw_scores'];score=predicted[item['id']]
            if target:
                spec,rules=target
                if sha_text(item['prompt'])!=spec['prompt_sha256'] or sha_text(item.get('negative_prompt',''))!=spec['negative_prompt_sha256']:raise ValueError('Sidecar is bound to different original prompt text')
            else:rules=[]
            probabilities=score.get('probabilities',{}) if score['status']=='ok' else {}
            guard=guard_from_native_tags(probabilities,rules,bundle['catalog'],min_reliability=bundle['policy'].min_guard_reliability)
            if score['status']=='error':guard=replace(guard,status='unknown',reliability=0.0,reason='tagger call failed: '+score['error'])
            guards.append(asdict(guard));candidates.append(Candidate(item['id'],pid,raw['kawai'],raw['deepghs'],guard))
        policy=replace(bundle['policy'],aesthetic_coverage=target[0]['aesthetic_coverage'] if target else 0.0)
        fused=fuse_group(candidates,policy);group_id=len(diagnostics)
        rowdiag=asdict(fused);rowdiag.update(prompt_id=pid,group_id=group_id,guards=guards,coverage=policy.aesthetic_coverage,known_pass=sum(g['status']=='pass' for g in guards),known_fail=sum(g['status']=='fail' for g in guards),unknown=sum(g['status']=='unknown' for g in guards));diagnostics.append(rowdiag)
        pairs=len(indices)*(len(indices)-1)//2
        for position,i in enumerate(indices):
            original=result[i]['reward'];result[i].update(frozen_reward=original,frozen_aesthetic_reward=result[i]['aesthetic_reward'],reward=fused.rewards[position],evaluation_domain=domain,reward_semantics='conservative_v2_training',calibration_id=policy.calibration_id,conservative_group_id=group_id,guard=guards[position])
            result[i]['components']=dict(frozen_absolute_reward=original,conservative_reward=fused.rewards[position],reward_mad=fused.reward_mad,signal_active=float(fused.signal_active),aesthetic_pair_fraction=fused.aesthetic_pairs/pairs,constraint_pair_fraction=fused.constraint_pairs/pairs,abstained_pair_fraction=fused.abstained_pairs/pairs,guard_pass=float(guards[position]['status']=='pass'),guard_fail=float(guards[position]['status']=='fail'),guard_unknown=float(guards[position]['status']=='unknown'),coverage=policy.aesthetic_coverage)
    return result,diagnostics

def collect_predictions(items,cfg,bundle,archive,run_program):
    options=cfg['conservative']['tagger'];response=archive/'pixai_probabilities.json'
    if response.exists():return json.loads(response.read_text(encoding='utf-8-sig'))
    if options['mode']=='cached':
        # Offline fixtures/replay only; online training must score its new images.
        if cfg['conservative'].get('offline_replay') is not True:raise ValueError('Cached tagger is allowed only in explicit offline replay')
        data=checked(options['response'],'cached tagger response')
        response.write_text(json.dumps(data,allow_nan=False));return data
    if options['mode']!='jax':raise ValueError('Unsupported conservative tagger provider')
    request=archive/'pixai_request.json';settings=archive/'pixai_config.json'
    request.write_text(json.dumps({'items':items},ensure_ascii=False,allow_nan=False))
    settings.write_text(json.dumps(dict(options,tagger_identity=bundle['raw']['tagger_identity'],catalog=bundle['raw']['artifacts']['catalog'])))
    run_program([options['python'],str(Path(__file__).with_name('score_pixai_guard.py')),'--config',str(settings),'--request',str(request),'--output',str(response)],archive/'pixai.log',cfg['stage_timeout'])
    return json.loads(response.read_text(encoding='utf-8-sig'))
