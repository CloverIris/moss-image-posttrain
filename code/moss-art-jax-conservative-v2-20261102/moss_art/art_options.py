"""One validated covariance contract shared by staged, direct and worker paths."""
import math,hashlib,json
from pathlib import Path

LEGACY_DEFAULTS=dict(scale_mode='unit_normalized_covariance',guidance_gain=1.0,zero_signal_mode='exact',zero_signal_atol=0.0,reward_contract='legacy',reference_mode='base',calibration_id='',policy_file='',policy_sha256='')

def covariance_options(art):
    options=dict(mode=art['feedback'],**{k:art.get(k,LEGACY_DEFAULTS[k]) for k in ('scale_mode','guidance_gain','zero_signal_mode','zero_signal_atol')})
    if options['scale_mode'] not in ('unit_normalized_covariance','reward_mad','raw_covariance','reward_std_normalized_covariance'):raise ValueError('Unknown ART scale_mode')
    if options['zero_signal_mode'] not in ('exact','historical','threshold'):raise ValueError('Unknown ART zero_signal_mode')
    for k in ('guidance_gain','zero_signal_atol'):
        x=options[k]
        if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('Invalid ART '+k)
    if options['zero_signal_mode']=='threshold' and options['zero_signal_atol']<=0 or options['zero_signal_mode']!='threshold' and options['zero_signal_atol']!=0:raise ValueError('Invalid zero signal tolerance')
    contract=art.get('reward_contract','legacy')
    if contract not in ('legacy','conservative_v2'):raise ValueError('Unknown reward contract')
    if art.get('reference_mode','base')!='base':raise ValueError('This trainer implements a frozen Base reference only')
    if contract=='conservative_v2':
        if options['mode']!='scalar' or options['scale_mode']!='reward_mad' or options['zero_signal_mode']!='exact' or options['zero_signal_atol']!=0 or options['guidance_gain']<=0:raise ValueError('Conservative v2 needs scalar/reward_mad/exact/positive gain')
        beta=art.get('reference_mse_beta',0)
        if type(beta) not in (int,float) or not math.isfinite(beta) or beta<=0:raise ValueError('Conservative v2 requires a calibrated positive frozen-Base beta')
        if not isinstance(art.get('calibration_id'),str) or not art['calibration_id'].strip():raise ValueError('Conservative v2 requires an explicit calibration_id')
    return options

def validate_policy_reference(art):
    if art.get('reward_contract','legacy')!='conservative_v2':return
    path=Path(art.get('policy_file',''))
    if not path.is_absolute() or not path.is_file():raise ValueError('Conservative approved policy_file is required before inference')
    value=path.read_bytes()
    if hashlib.sha256(value).hexdigest()!=art.get('policy_sha256'):raise ValueError('Conservative policy SHA mismatch')
    policy=json.loads(value)
    if policy.get('schema')!=1 or policy.get('approved') is not True or policy.get('calibration_id')!=art['calibration_id']:raise ValueError('Conservative calibration not approved or mismatched')
    for field in ('guidance_gain','reference_mse_beta'):
        x=policy['learner'][field]
        if type(x) not in (int,float) or not math.isfinite(x) or x<=0:raise ValueError('Uncalibrated learner '+field)
    if policy['learner']['guidance_gain']!=art['guidance_gain'] or policy['learner']['reference_mse_beta']!=art['reference_mse_beta']:raise ValueError('ART gain/Base beta do not match calibrated policy')
    for key in ('catalog','targets','evidence'):
        reference=policy['artifacts'][key];source=Path(reference['path'])
        if not source.is_absolute() or hashlib.sha256(source.read_bytes()).hexdigest()!=reference['sha256']:raise ValueError('Changed conservative artifact: '+key)

def declared_contract(art):
    return dict(name=art.get('reward_contract','legacy'),calibration_id=art.get('calibration_id',''),policy_sha256=art.get('policy_sha256',''),covariance_options=covariance_options(art),reference_mode=art.get('reference_mode','base'),reference_mse_beta=art.get('reference_mse_beta',0))

def validate_details(art,details,rewards):
    if art.get('reward_contract','legacy')!='conservative_v2':return
    if not isinstance(details,list) or len(details)!=len(rewards):raise ValueError('Conservative v2 requires ordered reward details')
    identities=set()
    for i,(row,value) in enumerate(zip(details,rewards)):
        if row.get('index')!=i or row.get('reward_semantics')!='conservative_v2_training' or row.get('calibration_id')!=art['calibration_id']:raise ValueError('Reward calibration/semantics/order do not match learner')
        key=row.get('id')
        if not isinstance(key,str) or not key or key in identities:raise ValueError('Invalid conservative image identity')
        identities.add(key)
        if not math.isfinite(float(value)) or not -1.000001<=float(value)<=1.000001 or not math.isclose(float(value),float(row['reward']),abs_tol=1e-6):raise ValueError('Conservative rewards differ from ordered details')

def validate_stats(art,stats):
    if art.get('reward_contract','legacy')!='conservative_v2':return
    import numpy as np
    gain=covariance_options(art)['guidance_gain']
    expected=np.where(stats['group_active'],np.asarray(stats['group_reward_abs_means'])*gain,0.)
    if not np.allclose(stats['group_guidance_magnitudes'],expected,rtol=1e-5,atol=0):raise ValueError('Prepared statistics erased conservative reward amplitude')
