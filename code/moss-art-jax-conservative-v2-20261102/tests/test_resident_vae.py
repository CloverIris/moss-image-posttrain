"""Explicit VAE ownership and phase boundaries without external model assets."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.mesh_runtime import MeshPool
from moss_art.evaluate import LatentDecoder, MeshLatentDecoder


class FakeDecoder:
    built=[]

    def __init__(self,*args,**kwargs):
        self.params={"weight":np.zeros((3,4),np.float32)}
        self.closed=0;self.calls=[]
        self.built.append(self)

    def __call__(self,latents):
        assert not self.closed
        self.calls.append(latents.copy())
        return np.repeat(latents[:,:1,:1,:1].astype(np.uint8).transpose(0,2,3,1),3,axis=-1)

    def close(self):
        self.closed+=1;self.params=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def pool_fixture(monkeypatch,*,resident=True,precision="mxfp8"):
    monkeypatch.setattr("moss_art.evaluate.MeshLatentDecoder",FakeDecoder)
    FakeDecoder.built=[]
    pool=MeshPool.__new__(MeshPool)
    pool.config={"sampling":{"vae_resident_rollout":resident},"model":{
        "path":"unused","vae_precision":"float16","vae_low_precision_stages":["input"],
        "vae_conv_backend":"xla","vae_conv_workspace_mib":256}}
    pool.worker=SimpleNamespace(model=SimpleNamespace(base_precision=precision,release_base=lambda:None),
        cache={},buckets={},text_cache={},updates_seen=0,lora={})
    pool._vae_decoder=None;pool._vae_build_count=0
    pool.size=4;pool.devices=[SimpleNamespace(memory_stats=lambda:{}) for _ in range(4)]
    pool.parallel=object();pool.workers=list(pool.devices)
    pool.info=[];pool.shards=None;pool.prepared=None
    pool.memory_events=[];pool.memory_event=pool.memory_events.append
    return pool


def endpoints(offset=0):
    return [np.full((2,128,1,1),offset+i,np.float32) for i in range(4)]


def test_resident_reuses_one_decoder_and_preserves_output_order(monkeypatch):
    pool=pool_fixture(monkeypatch)
    decoder=pool.ensure_rollout_vae()
    assert pool.ensure_rollout_vae() is decoder
    first=pool.decode(endpoints());second=pool.decode(endpoints(10))
    assert len(FakeDecoder.built)==1 and decoder.closed==0
    assert len(decoder.calls)==2 and pool._vae_build_count==1
    for rank in range(4):
        np.testing.assert_array_equal(first[rank]["images"],rank)
        np.testing.assert_array_equal(second[rank]["images"],10+rank)
    pool.release_vae(reason="test")
    pool.release_vae(reason="test_again")
    assert decoder.closed==1 and pool._vae_decoder is None
    assert pool.memory_events.count("after_rollout_vae_load")==1
    assert pool.memory_events.count("after_rollout_vae_release")==1
    replacement=pool.ensure_rollout_vae()
    assert replacement is not decoder and pool._vae_build_count==2
    pool.close()
    assert replacement.closed==1


@pytest.mark.parametrize("resident",[False,True])
def test_paused_decode_keeps_temporary_decoder_path(monkeypatch,resident):
    pool=pool_fixture(monkeypatch,resident=resident)
    pool.worker=None
    pool.decode(endpoints());pool.decode(endpoints())
    assert len(FakeDecoder.built)==2
    assert all(d.closed==1 for d in FakeDecoder.built)
    assert pool._vae_decoder is None


def test_disabled_residency_does_not_allocate_during_rollout(monkeypatch):
    pool=pool_fixture(monkeypatch,resident=False,precision="bf16")
    assert pool.ensure_rollout_vae() is None and FakeDecoder.built==[]


def test_bf16_cannot_retain_rollout_vae_and_release_precedes_reload(monkeypatch):
    pool=pool_fixture(monkeypatch)
    decoder=pool.ensure_rollout_vae()
    events=[]
    def pause():
        assert decoder.closed==1
        events.append("pause");pool.worker=None
    def resume(*,base_precision):
        assert pool._vae_decoder is None
        events.append("resume")
        pool.worker=SimpleNamespace(model=SimpleNamespace(base_precision=base_precision,precision_contract=lambda:{}))
    pool.pause=pause;pool.resume=resume;pool.parameter_version=3
    pool._ensure_precision("bf16")
    assert events==["pause","resume"] and decoder.closed==1
    with pytest.raises(RuntimeError,match="active MXFP8"):
        pool.ensure_rollout_vae()
    # The same-precision early return must also enforce release.
    pool._vae_decoder=FakeDecoder()
    stale=pool._vae_decoder
    pool._ensure_precision("bf16")
    assert stale.closed==1 and pool._vae_decoder is None


def test_pause_releases_decoder_before_dit_and_is_idempotent(monkeypatch):
    pool=pool_fixture(monkeypatch)
    decoder=pool.ensure_rollout_vae()
    events=[]
    def release_base():
        assert decoder.closed==1
        events.append("dit_release")
    pool.worker.model.release_base=release_base
    pool.pause();pool.pause()
    assert events==["dit_release"] and pool.worker is None
    assert decoder.closed==1
    pool.close()


def test_prepare_update_drops_vae_even_if_bad_replay_is_rejected(monkeypatch):
    pool=pool_fixture(monkeypatch)
    decoder=pool.ensure_rollout_vae()
    pool.shards=[{}]*4
    with pytest.raises(ValueError,match="version"):
        pool.call("prepare_update",[{}]*4)
    assert decoder.closed==1 and pool._vae_decoder is None


def test_reward_drops_decoder_before_loading_scorer(monkeypatch):
    from moss_art import silva_jax
    pool=pool_fixture(monkeypatch)
    decoder=pool.ensure_rollout_vae()
    pool.config["reward"]={"kind":"silva","model_path":"unused","backbone_path":"unused",
        "dtype":"float32","attention_backend":"native","batch_size":4}
    class Scorer:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            assert decoder.closed==1 and pool._vae_decoder is None
            return cls()
        def score(self,images,*,batch_size):return [1.]*len(images)
        def close(self):pass
    monkeypatch.setattr(silva_jax,"SilvaScorerJax",Scorer)
    assert pool.score(np.zeros((4,2,2,3),np.uint8))==[1.]*4


def test_decoder_close_deletes_parameters_and_only_own_jit_cache():
    decoder=LatentDecoder.__new__(LatentDecoder)
    parameter=jnp.ones((3,4),jnp.float32)
    cleared=[]
    decoder.params={"weight":parameter}
    decoder.decode=SimpleNamespace(clear_cache=lambda:cleared.append(True))
    decoder.close();decoder.close()
    assert parameter.is_deleted() and cleared==[True]
    assert decoder.params is None and decoder.decode is None


def test_mesh_decode_transfers_pixels_to_host_and_releases_transient_arrays():
    decoder=MeshLatentDecoder.__new__(MeshLatentDecoder)
    inputs=[];outputs=[]
    def put(chunk):
        tensor=jax.device_put(chunk);inputs.append(tensor);return tensor
    def decode(params,tensor):
        output=jnp.repeat(tensor[:,:1,:1,:1],3,axis=1)
        outputs.append(output);return output
    decoder.parallel=SimpleNamespace(size=4,put=put)
    decoder.params={};decoder.decode=decode
    result=decoder(np.zeros((5,128,1,1),np.float32))
    assert result.shape==(5,1,1,3) and result.dtype==np.uint8
    assert len(inputs)==len(outputs)==2
    assert all(x.is_deleted() for x in inputs+outputs)
    np.testing.assert_array_equal(result,128)
