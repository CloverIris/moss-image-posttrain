"""CPU-only synthetic endpoint/gradient checks; never construct a MOSS model."""
import copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import jax,jax.numpy as jnp
from moss_art.art import covariance,art_loss
from moss_art.art_options import covariance_options,declared_contract,validate_stats,validate_details,validate_policy_reference
from moss_art.config import DEFAULTS,fingerprint

def art():return dict(feedback='scalar',reward_contract='conservative_v2',reference_mode='base',reference_mse_beta=.1,calibration_id='SYNTHETIC-TEST-ONLY',scale_mode='reward_mad',guidance_gain=.2,zero_signal_mode='exact',zero_signal_atol=0.,eps=1e-5)
def rows(r):return [dict(id=str(i),index=i,reward=float(x),reward_semantics='conservative_v2_training',calibration_id='SYNTHETIC-TEST-ONLY') for i,x in enumerate(r)]

def test_true_covariance_mad_scale_and_exact_zero():
    x=np.arange(24,dtype=np.float32).reshape(8,3);r=np.linspace(-.2,.2,8,dtype=np.float32);g=np.zeros(8,np.int32);a=art()
    strong=covariance(x,r,g,**covariance_options(a));weak=covariance(x,r*.1,g,**covariance_options(a));zero=covariance(x,r*0,g,**covariance_options(a))
    validate_stats(a,strong);validate_stats(a,weak);validate_stats(a,zero)
    np.testing.assert_allclose(weak['group_guidance_magnitudes'],strong['group_guidance_magnitudes']*.1,rtol=1e-5)
    np.testing.assert_allclose(weak['group_directions'],strong['group_directions'],rtol=1e-5)
    assert not np.asarray(zero['group_active']).any();assert np.count_nonzero(zero['group_guidance_magnitudes'])==0

def test_output_space_gradient_reflects_confidence_and_retains_base_anchor():
    a=art();x=np.arange(24,dtype=np.float32).reshape(8,3);r=np.linspace(-.2,.2,8,dtype=np.float32);g=np.zeros(8,np.int32)
    def gradient(rewards,current,old,reference):
        s=covariance(x,rewards,g,**covariance_options(a));direction=jnp.repeat(s['group_directions'],8,axis=0);magnitude=jnp.repeat(s['group_guidance_magnitudes'],8)
        return jax.grad(lambda value:art_loss(value,old,jnp.full(8,.5),direction,magnitude,dt=jnp.full(8,-.02),reference_prediction=reference,reference_mse_beta=.1)['per_item_loss'].sum())(current)
    z=jnp.zeros((8,3));strong=gradient(r,z,z,z);weak=gradient(r*.1,z,z,z)
    np.testing.assert_allclose(weak,strong*.1,rtol=1e-5,atol=1e-7)
    assert np.count_nonzero(gradient(r*0,z,z,z))==0
    ones=jnp.full((8,3),.1);anchor=gradient(r*0,ones,ones,z);assert np.linalg.norm(anchor)>0

def test_worker_fallback_and_precomputed_stats_use_same_contract():
    from moss_art.worker import NativeWorker
    a=art();r=np.linspace(-.2,.2,8,dtype=np.float32);x=np.arange(24,dtype=np.float32).reshape(8,3);g=np.zeros(8,np.int32)
    fake=SimpleNamespace(c={'art':a,'sampling':{'group_size':8}},cache={'endpoints':x,'groups':g,'selected':[0]})
    NativeWorker.prepare_update(fake,rewards=r);validate_stats(a,fake.stats)
    bad=dict(fake.stats,group_guidance_magnitudes=np.ones(1))
    with pytest.raises(ValueError):NativeWorker.prepare_update(fake,rewards=r,stats=bad)

def test_staged_prepare_training_really_passes_gain_and_semantics(tmp_path):
    from moss_art.stage_pipeline import prepare_training
    c=copy.deepcopy(DEFAULTS);c['art'].update(art());c['runtime']['output_dir']=str(tmp_path/'output');c['training']['updates_per_rollout']=2;c['sampling'].update(group_size=8,train_timesteps=[0]);c['reward']['kind']='command'
    r=np.linspace(-.2,.2,8,dtype=np.float32);x=np.arange(24,dtype=np.float32).reshape(8,3);g=np.zeros(8,np.int32)
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'COMMITTED').write_text('synthetic')
    (bundle/'evaluated.json').write_text(json.dumps(dict(source_commit='synthetic',scores=r.tolist(),details=rows(r))))
    payload=dict(config=c,shards=[{'endpoints':x}],groups=g,rollout_start=0,resolution_label=None,outputs=[dict(sample_ids=['p']*8,seeds=list(range(8)))])
    result=prepare_training(c,bundle,payload,{'update':0})
    s=result['prepared'][0]['stats'];validate_stats(c['art'],s)
    assert result['covariance_options']['scale_mode']=='reward_mad'
    assert np.all(np.asarray(s['group_guidance_magnitudes'])<1)
    bad=rows(r);bad[0]['reward_semantics']='rank_power_rank_training'
    (bundle/'evaluated.json').write_text(json.dumps(dict(source_commit='synthetic',scores=r.tolist(),details=bad)))
    with pytest.raises(ValueError):prepare_training(c,bundle,payload,{'update':0})

def test_bad_modes_gain_beta_and_reference_fail_closed():
    for changes in ({'feedback':'pairwise'},{'scale_mode':'unit_normalized_covariance'},{'guidance_gain':0},{'reference_mse_beta':0},{'reference_mode':'ema'},{'zero_signal_mode':'historical'}):
        with pytest.raises(ValueError):covariance_options(dict(art(),**changes))
    a=art();assert declared_contract(a)['name']=='conservative_v2'
    with pytest.raises(ValueError):validate_details(a,[dict(index=0,reward=0)],np.zeros(1))
    with pytest.raises(ValueError):validate_policy_reference(a)

def test_legacy_defaults_leave_fingerprint_unchanged():
    from moss_art.art_options import LEGACY_DEFAULTS
    a=copy.deepcopy(DEFAULTS);b=copy.deepcopy(DEFAULTS)
    for k in LEGACY_DEFAULTS:b['art'].pop(k,None)
    assert fingerprint(a)==fingerprint(b)
    assert covariance_options(a['art'])['scale_mode']=='unit_normalized_covariance'

def test_pipeline_signed_scores_reach_staged_covariance_without_reranking(tmp_path,monkeypatch):
    """Synthetic head fixtures through the actual reward and staged ART functions."""
    reward_root=os.environ.get('MOSS_CONSERVATIVE_REWARD_ROOT')
    if not reward_root:pytest.skip('isolated reward root not configured')
    monkeypatch.syspath_prepend(reward_root)
    from test_conservative_pipeline import fixture,archive_lock_test_support
    import run_reward_pipeline as pipeline
    from unittest.mock import patch
    from moss_art.stage_pipeline import prepare_training
    cfg,request,scorer,kawai=fixture(tmp_path)
    with archive_lock_test_support(),patch.object(pipeline,'run_program',side_effect=scorer),patch.object(pipeline,'run_kawai_parallel',side_effect=kawai),patch.object(pipeline,'apply_rank_fusion',side_effect=AssertionError('unexpected rank')):
        response=pipeline.run(cfg,request,tmp_path/'response.json')
    c=copy.deepcopy(DEFAULTS);c['art'].update(art());c['runtime']['output_dir']=str(tmp_path/'output');c['training']['updates_per_rollout']=2;c['sampling'].update(group_size=2,train_timesteps=[0]);c['reward']['kind']='command'
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'COMMITTED').write_text('synthetic-e2e')
    (bundle/'evaluated.json').write_text(json.dumps(dict(source_commit='synthetic-e2e',scores=response['scores'],details=response['details'])))
    payload=dict(config=c,shards=[{'endpoints':np.array([[0.,1.,2.],[3.,4.,5.]],np.float32)}],groups=np.zeros(2,np.int32),rollout_start=0,resolution_label=None,outputs=[dict(sample_ids=['p']*2,seeds=[0,1])])
    result=prepare_training(c,bundle,payload,{'update':0});stats=result['prepared'][0]['stats']
    validate_stats(c['art'],stats)
    np.testing.assert_allclose(stats['group_guidance_magnitudes'],np.mean(np.abs(response['scores']))*.2,rtol=1e-5)
    assert 0<float(np.asarray(stats['group_guidance_magnitudes'])[0])<.2
