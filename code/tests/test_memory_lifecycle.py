import copy,json
import jax,jax.numpy as jnp,numpy as np
import pytest
from moss_art.checkpoint import _encode,save_checkpoint,load_checkpoint
from moss_art.mesh_runtime import MeshPool


def test_checkpoint_encoder_keeps_fp32_device_arrays_until_member_write(tmp_path):
    x=jnp.arange(128,dtype=jnp.float32).reshape(8,16)
    arrays={};_encode({"weight":x},arrays)
    assert arrays["array_000000"] is x
    path=save_checkpoint(tmp_path/"checkpoint",{"weight":x},{"moment":x*2},{"step":1})
    lora,opt,meta=load_checkpoint(path)
    # Saving must not attach a persistent host cache to the original master.
    assert x._npy_value is None
    np.testing.assert_array_equal(lora["weight"],x)
    np.testing.assert_array_equal(opt["moment"],x*2)


def test_mesh_replay_split_uses_views():
    pool=MeshPool.__new__(MeshPool);pool.size=4
    data=np.arange(8*4*2,dtype=np.float32).reshape(8,4,2)
    pair=(np.ones((1,3,2),np.float32),np.zeros((1,3,2),np.float32))
    cache=dict(states=data,old=data+1,endpoints=data[:,0],groups=np.zeros(8,np.int32),contexts=[pair]*8)
    shards=pool._split_cache(cache)
    for shard in shards:
        for k in ("states","old","endpoints","groups"):assert np.shares_memory(shard[k],cache[k])
        assert all(p is pair for p in shard["contexts"])


@pytest.mark.skipif(len(jax.local_devices())!=4,reason="requires four logical devices")
def test_mesh_reload_reuses_lora_without_random_initialization(monkeypatch):
    from test_mesh_dp import configuration
    from moss_art.model import MossModel
    original=MossModel.init_lora;calls=[]
    def init(self,*args,**kwargs):calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(MossModel,"init_lora",init)
    with MeshPool(configuration()) as pool:
        assert len(calls)==1
        before=pool.worker.lora
        pool.pause();pool.resume()
        assert len(calls)==1
        for k in before:assert pool.worker.lora[k] is before[k]


@pytest.mark.skipif(len(jax.local_devices())!=4,reason="requires four logical devices")
def test_mesh_update_keeps_lora_on_devices_and_clears_replay_before_save(tmp_path,monkeypatch):
    import moss_art.trainer as trainer
    from test_mesh_dp import configuration
    c=configuration();c["training"].update(prompts_per_update=2,rollouts_per_prompt=4,training_points_per_trajectory=3,max_updates=2,evaluate_every=1)
    c["runtime"]["output_dir"]=str(tmp_path/"training")
    data=tmp_path/"data.jsonl";data.write_text(''.join(json.dumps(dict(id=str(i),prompt="synthetic fixture "+str(i)))+'\n' for i in range(5)))
    c["data"]["prompts"]=str(data)
    from moss_art.config import validate
    validate(c);events=[]
    call=MeshPool.call;save=trainer._save
    def capture_call(self,op,*args,**kwargs):
        events.append(op)
        return call(self,op,*args,**kwargs)
    def capture_save(config,output,lora,opt,meta):
        assert events[-1]=="clear_replay"
        assert all(isinstance(v,jax.Array) and v.is_fully_replicated for v in lora.values())
        return save(config,output,lora,opt,meta)
    monkeypatch.setattr(MeshPool,"call",capture_call);monkeypatch.setattr(trainer,"_save",capture_save)
    out=trainer.train(c)
    rows=[json.loads(l) for l in (out/"metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows]==[1,2]
