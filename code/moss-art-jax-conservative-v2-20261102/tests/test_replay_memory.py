import copy,pickle
import jax,jax.numpy as jnp
import numpy as np
import pytest
from moss_art.replay import host_snapshot,canonical_contexts,shard_replay
from moss_art.config import DEFAULTS
from moss_art.trainer import paired_replay_payloads,aggregate_gradients
from moss_art.art import covariance,prompt_windows
from test_paired_replay import worker,Pool


def test_snapshot_copies_shared_device_buffers_once_and_pickle_preserves_aliases():
    array=jnp.ones((1,4,2),jnp.float32)
    snapshot=host_snapshot({"pairs":[(array,array)]*8})
    result=pickle.loads(pickle.dumps(snapshot))
    assert all(pair[0] is result["pairs"][0][0] and pair[1] is pair[0] for pair in result["pairs"])
    assert isinstance(result["pairs"][0][0],np.ndarray)


@pytest.mark.parametrize("feedback",["scalar","pairwise"])
@pytest.mark.parametrize("slots",[3,4])
def test_sharded_replay_preserves_gradients_and_reduces_rows(feedback,slots):
    c=copy.deepcopy(DEFAULTS)
    c["sampling"].update(height=32,width=32,steps=4,train_timesteps=list(range(slots)),group_size=8,
        prompts_per_rollout=1,split_group_across_workers=True,sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[])
    c["art"]["feedback"]=feedback
    record={"sample_id":"synthetic","prompt":"synthetic fixture","negative_prompt":""}
    ws=[worker(c) for _ in range(4)];pool=Pool(ws)
    outputs=[w.rollout([record],[100+rank*2,101+rank*2],samples_per_prompt=2) for rank,w in enumerate(ws)]
    endpoints=np.concatenate([o["endpoints"] for o in outputs]);groups=np.zeros(8,np.int32)
    stats={k:np.asarray(v) for k,v in covariance(endpoints,np.linspace(.1,.9,8,dtype=np.float32),groups,mode=feedback,group_size=8).items()}
    windows=prompt_windows(groups,slots,seed=42)
    reference=worker(c);reference.rollout([record],list(range(100,108)));reference.prepare_update(stats=stats,windows=windows)
    payloads=paired_replay_payloads(pool,groups,stats,windows);pool.call("prepare_update",payloads)
    assert all(len(w.cache["endpoints"])<8 for w in ws)
    if slots==4:assert all(w.cache["states"].shape[1]==2 for w in ws)
    for w in ws:
        assert all(pair is w.cache["contexts"][0] for pair in w.cache["contexts"])
        exported=w.export_replay()
        assert all(pair is exported["contexts"][0] for pair in exported["contexts"])
    for window in range(2):
        actual,metrics=aggregate_gradients(pool.call("gradient",[{"window":window}]*4))
        expected,ref_metrics=aggregate_gradients([reference.gradient(window)])
        np.testing.assert_allclose(actual["theta"],expected["theta"],rtol=4e-6,atol=4e-7)
        assert metrics["items"]==ref_metrics["items"]==8*slots//2
    for w in ws:
        w.clear_replay()
        assert w.cache is None and w.stats=={} and w.windows==[]
