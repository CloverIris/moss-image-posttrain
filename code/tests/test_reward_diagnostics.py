import json,os,subprocess,sys
import numpy as np
import pytest
from moss_art.reward_diagnostics import normalize_records,record_available


def samples(ids):return [dict(index=i,sample_id=label,seed=100+i) for i,label in enumerate(ids)]


def test_normalization_uses_complete_interleaved_prompt_groups():
    result=normalize_records(samples(["a","b","a","b"]),[.1,.6,.3,1.],2)
    np.testing.assert_allclose([r["mad_normalized_reward"] for r in result["rows"]],[-1,-1,1,1],atol=1e-6)
    assert abs(result["summary"]["mad_normalized_mean"])<1e-6
    with pytest.raises(ValueError,match="Incomplete"):normalize_records(samples(["a","a","b"]),[.1,.2,.3],2)


def test_constant_reward_group_is_exact_zero():
    result=normalize_records(samples(["a"]*8),[.6789]*8,8)
    assert [r["mad_normalized_reward"] for r in result["rows"]]==[0.]*8
    assert result["summary"]["constant_group_fraction"]==1.
    assert result["groups"][0]["reward_std"]==0.


def test_logged_weights_reproduce_art_mad_scaled_covariance():
    from moss_art.art import covariance
    rng=np.random.default_rng(18);ids=np.tile(np.arange(3),8);r=rng.uniform(.2,.9,len(ids)).astype(np.float32);x=rng.normal(size=(len(ids),4)).astype(np.float32)
    result=normalize_records(samples([str(i) for i in ids]),r,8)
    normalized=np.array([row["mad_normalized_reward"] for row in result["rows"]],np.float32)
    expected=np.stack([np.sum(normalized[ids==g,None]*x[ids==g],axis=0)/7 for g in range(3)])
    actual=covariance(x,r,ids,group_size=8)
    np.testing.assert_allclose(expected,np.asarray(actual["group_covariance"])/np.asarray(actual["group_reward_abs_means"])[:,None],rtol=2e-5,atol=2e-6)


def test_imports_no_accelerator_framework():
    command="import sys;import moss_art.reward_diagnostics;assert 'jax' not in sys.modules;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,"-c",command],check=True)


def test_saved_records_are_idempotent_and_keep_raw_scores(tmp_path):
    output=tmp_path/"training";images=output/"rollout_images/rollout_000000";images.mkdir(parents=True);rewards=output/"rewards";rewards.mkdir()
    (tmp_path/"config.json").write_text(json.dumps(dict(runtime=dict(output_dir=str(output)),training=dict(rollouts_per_prompt=2))))
    (images/"samples.json").write_text(json.dumps(samples(["a","a","b","b"])))
    raw=dict(rollout=0,update=0,details=[dict(reward=r) for r in [.1,.3,.5,.9]])
    source=rewards/"rollout_000000.json";source.write_text(json.dumps(raw));before=source.read_bytes()
    (output/"rollouts.jsonl").write_text(json.dumps(dict(rollout=0,height=768,width=768))+"\n")
    class Logger:
        writer=None
        def write(self,*args):pass
    done={};assert record_available(tmp_path,Logger(),done)==1
    assert record_available(tmp_path,Logger(),done)==0
    assert source.read_bytes()==before
    saved=json.loads((output/"reward_norm/rollout_000000.json").read_text())
    assert len(saved["rows"])==4 and saved["height"]==768
    assert len((output/"reward_norm/summary.jsonl").read_text().splitlines())==1
