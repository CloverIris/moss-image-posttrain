import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import jax,jax.numpy as jnp
from moss_art.config import DEFAULTS,validate
from moss_art.silva_jax import resize_uint8,preprocess,aesthetic_head


def test_uint8_preprocessing_matches_torch_oracle():
    fixture=json.loads(Path(__file__).with_name("silva_resize_oracle.json").read_text())
    for case in fixture["cases"]:
        h,w=case["shape"]
        x=np.random.default_rng(h*10000+w).integers(0,256,(h,w,3),dtype=np.uint8)
        assert hashlib.sha256(resize_uint8(x).tobytes()).hexdigest()==case["uint8_sha256"]
        assert hashlib.sha256(preprocess([x])[0].tobytes()).hexdigest()==case["pixels_sha256"]


def test_ordinal_head_keeps_original_calibration_and_saturates():
    p={"norm.weight":jnp.ones(2),"norm.bias":jnp.zeros(2),"head.latent.weight":jnp.array([[1.,0.]]),
       "head.latent.bias":jnp.zeros(1),"head.base_threshold":jnp.array(0.),"head.raw_deltas":jnp.zeros(4),
       "cal_lat_knots":jnp.array([-.5,0.,.5]),"cal_score_knots":jnp.array([.1,.3,.9])}
    x=jnp.array([[-1.,1.],[0.,0.],[1.,-1.]])
    actual=aesthetic_head(p,x,hidden_dims=(),calibrated=True)
    np.testing.assert_allclose(actual["reward"],[.1,.3,.9],atol=2e-7)
    latent=np.asarray(actual["latent"]);thresholds=np.cumsum(np.full(4,np.log(2)))
    expected=(1/(1+np.exp(-(latent[:,None]-thresholds)))).mean(-1)
    np.testing.assert_allclose(actual["raw_score"],expected,atol=2e-7)
    assert not np.allclose(actual["raw_score"],actual["reward"])


def test_silva_config_requires_calibrated_metric_and_backbone():
    c=copy.deepcopy(DEFAULTS);c["reward"].update(kind="silva",metric="calibrated",backbone_path="/fixture/backbone",dtype="float32",attention_backend="native")
    validate(c)
    for values in ({"metric":"raw"},{"backbone_path":""},{"attention_backend":"cudnn"}):
        bad=copy.deepcopy(c);bad["reward"].update(values)
        with pytest.raises(ValueError):validate(bad)


def test_weights_share_one_mapping_per_shard(tmp_path):
    from safetensors.numpy import save_file
    from vendor.moss_jax.weights import load_host_params
    values={f"x{i}":np.arange(i+1,dtype=np.float32) for i in range(20)}
    save_file(values,str(tmp_path/"model.safetensors"))
    loaded=load_host_params(tmp_path,"")
    assert len({id(v.base) for v in loaded.values()})==1
    for key,value in values.items():
        np.testing.assert_array_equal(loaded[key],value)
        assert not loaded[key].flags.writeable


def test_mesh_silva_reward_preserves_image_order(monkeypatch):
    from moss_art.mesh_runtime import MeshPool
    from moss_art import silva_jax
    calls=[]
    class FakeScorer:
        @classmethod
        def from_pretrained(cls,*args,**kw):calls.append(kw);return cls()
        def score(self,images,*,batch_size):return [dict(reward=float(x[0,0,0])/255,raw_score=.5,latent=0.) for x in images]
        def close(self):pass
    monkeypatch.setattr(silva_jax,"SilvaScorerJax",FakeScorer)
    pool=MeshPool.__new__(MeshPool);pool.config=copy.deepcopy(DEFAULTS);pool.config["reward"]["kind"]="silva"
    pool.devices=[jax.local_devices()[0]]*4;pool.size=4;pool.memory_event=lambda _:None
    x=np.arange(9,dtype=np.uint8)[:,None,None,None]*np.ones((9,2,2,3),np.uint8)
    result=pool.score(x)
    np.testing.assert_allclose([r["reward"] for r in result],np.arange(9)/255)
    assert len(calls)==1 and calls[0]["data_parallel_size"]==4


@pytest.mark.skipif(len(jax.local_devices())!=4,reason="requires four logical devices")
def test_silva_features_and_scores_are_batch_sharded():
    from moss_art.silva_jax import SilvaScorerJax
    from moss_art.data_parallel import get_data_parallel
    rng=np.random.default_rng(19);params={};d=8
    def linear(name,out_dim,in_dim):
        params[name+".weight"]=rng.normal(0,.1,(out_dim,in_dim)).astype(np.float32)
        params[name+".bias"]=np.zeros(out_dim,np.float32)
    def norm(name):params.update({name+".weight":np.ones(d,np.float32),name+".bias":np.zeros(d,np.float32)})
    params["vision_model.embeddings.patch_embedding.weight"]=rng.normal(0,.1,(d,3,2,2)).astype(np.float32)
    params["vision_model.embeddings.patch_embedding.bias"]=np.zeros(d,np.float32)
    params["vision_model.embeddings.position_embedding.weight"]=np.zeros((4,d),np.float32)
    prefix="vision_model.encoder.layers.0."
    norm(prefix+"layer_norm1");norm(prefix+"layer_norm2")
    for name in ("q","k","v","out"):linear(prefix+"self_attn."+name+"_proj",d,d)
    linear(prefix+"mlp.fc1",16,d);linear(prefix+"mlp.fc2",d,16)
    norm("vision_model.post_layernorm");norm("vision_model.head.layernorm")
    params["vision_model.head.probe"]=rng.normal(0,.1,(1,1,d)).astype(np.float32)
    params["vision_model.head.attention.in_proj_weight"]=rng.normal(0,.1,(3*d,d)).astype(np.float32)
    params["vision_model.head.attention.in_proj_bias"]=np.zeros(3*d,np.float32)
    linear("vision_model.head.attention.out_proj",d,d);linear("vision_model.head.mlp.fc1",16,d);linear("vision_model.head.mlp.fc2",d,16)
    head={"norm.weight":np.ones(d,np.float32),"norm.bias":np.zeros(d,np.float32),"head.latent.weight":np.ones((1,d),np.float32),
          "head.latent.bias":np.zeros(1,np.float32),"head.base_threshold":np.float32(0),"head.raw_deltas":np.zeros(4,np.float32),
          "cal_lat_knots":np.array([-1.,0.,1.],np.float32),"cal_score_knots":np.array([.1,.5,.9],np.float32),"cal_fitted":np.array(True)}
    vc=dict(num_hidden_layers=1,num_attention_heads=2,layer_norm_eps=1e-6,patch_size=2,image_size=4);hc=dict(hidden_dims=[])
    dp=get_data_parallel(4,0)
    distributed=SilvaScorerJax(jax.device_put(params,dp.replicated),jax.device_put(head,dp.replicated),vc,hc,device=jax.local_devices()[0],dtype=jnp.float32,backend="native",parallel=dp)
    serial=SilvaScorerJax(jax.device_put(params,jax.local_devices()[0]),jax.device_put(head,jax.local_devices()[0]),vc,hc,device=jax.local_devices()[0],dtype=jnp.float32,backend="native")
    pixels=rng.normal(size=(8,3,4,4)).astype(np.float32)
    actual=distributed.embed_pixels(pixels);expected=serial.embed_pixels(pixels)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    assert actual.sharding.is_equivalent_to(dp.batch(2),2)
    scores=distributed.score_features(actual)
    assert all(v.sharding.is_equivalent_to(dp.batch(1),1) for v in scores.values())
    distributed.close();serial.close()
