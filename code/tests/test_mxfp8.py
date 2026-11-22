import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from moss_art import mxfp8 as mx


def test_mx_quantization_zero_boundary_and_scale_layout():
    maxima=np.array([0,1e-30,1,448,np.nextafter(np.float32(448),np.float32(np.inf)),896,1e20],np.float32)
    x=np.repeat(maxima[:,None],32,axis=1)
    q,s=mx.quantize(jnp.asarray(x))
    assert q.dtype==jnp.float8_e4m3fn and s.dtype==jnp.float8_e8m0fnu
    assert s.shape==(len(maxima),1)
    scales=np.asarray(s.astype(jnp.float32))[:,0]
    np.testing.assert_array_equal(scales[3:6],np.array([1,2,2],np.float32))
    assert np.all(scales*448>=maxima)
    assert np.isfinite(np.asarray(mx.dequantize(q,s,jnp.float32))).all()
    np.testing.assert_array_equal(np.asarray(q[0].astype(jnp.float32)),0)


@pytest.mark.parametrize('dgrad',['bf16','mxfp8'])
def test_frozen_linear_padded_forward_and_surrogate_dgrad(dgrad):
    rng=np.random.default_rng(12)
    x=jnp.asarray(rng.normal(size=(2,7,128)),jnp.bfloat16)
    w=jnp.asarray(rng.normal(size=(256,128)),jnp.bfloat16)
    q,s=mx.quantize(w)
    qt,st=mx.quantize(w.T) if dgrad=='mxfp8' else (None,None)
    y,pb=jax.vjp(lambda value:mx.frozen_matmul(value,q,s,qt,st),x)
    xq,xs=mx.quantize(x.reshape(-1,128))
    ref=(mx.dequantize(xq,xs,jnp.float32)@mx.dequantize(q,s,jnp.float32).T).reshape(2,7,256)
    np.testing.assert_allclose(y,ref,rtol=2e-5,atol=2e-4)
    dy=jnp.asarray(rng.normal(size=y.shape),jnp.bfloat16).astype(jnp.float32)
    dx=pb(dy)[0]
    if dgrad=='bf16':
        expected=(dy.astype(jnp.bfloat16).astype(jnp.float32)@mx.dequantize(q,s,jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    else:
        gq,gs=mx.quantize(dy.astype(jnp.bfloat16).reshape(-1,256))
        expected=(mx.dequantize(gq,gs,jnp.float32)@mx.dequantize(qt,st,jnp.float32).T).reshape(x.shape).astype(jnp.bfloat16)
    np.testing.assert_allclose(dx.astype(jnp.float32),expected.astype(jnp.float32),rtol=1e-2,atol=.125)


def test_lora_factors_receive_nonzero_gradients_and_frozen_base_is_compressed():
    from moss_art.model import MossModel
    model=MossModel.tiny(dtype=jnp.bfloat16,base_precision='mxfp8',targets='all',
        config=dict(dim=128,ffn_dim=256,num_heads=4,text_dim=128,num_layers=1),offload_activations=False)
    assert model.mxfp8_modules==6
    assert all(model.base[k].dtype==jnp.float8_e4m3fn for k in model.base if mx.eligible(k,model.base[k]))
    lora=model.init_lora(5)
    x=jnp.ones((1,8,2,2));context=jnp.ones((1,3,128))*.2
    y,pb=model.predict_with_pullback(lora,x,.5,context)
    gradients=pb(jnp.ones_like(y)/y.size)
    assert all(np.isfinite(np.asarray(v)).all() for v in gradients.values())
    assert sum(np.linalg.norm(np.asarray(v)) for k,v in gradients.items() if 'lora_up' in k)>0
    assert all(v.dtype==jnp.float32 for v in lora.values())
    # No original BF16 transformer projection weights are retained.
    assert model.base['layers.0.proj.q.weight'].nbytes==128*128


def test_mxfp8_sft_config_and_tp_guard(tmp_path):
    from moss_art.sft import load_config
    p=tmp_path/'config.json';p.write_text(json.dumps({'model':{'base_precision':'mxfp8'}}))
    assert load_config(p)['model']['mxfp8_dgrad']=='bf16'
    p.write_text(json.dumps({'model':{'base_precision':'mxfp8'},'runtime':{'tensor_parallel_size':2,'devices':[0,1]}}))
    with pytest.raises(ValueError,match='tensor_parallel_size'):load_config(p)
    p.write_text(json.dumps({'model':{'base_precision':'fp8'}}))
    with pytest.raises(ValueError,match='base_precision'):load_config(p)

def test_mxfp8_checkpoint_optimizer_and_adapter_roundtrip(tmp_path):
    import optax
    from moss_art.model import MossModel
    from moss_art.checkpoint import save_checkpoint, load_checkpoint
    from moss_art.lora import export_safetensors, import_safetensors
    model=MossModel.tiny(dtype=jnp.bfloat16,base_precision='mxfp8',targets='all',
        config=dict(dim=128,ffn_dim=256,num_heads=4,text_dim=128,num_layers=1),offload_activations=False)
    tx=optax.adam(1e-4)
    a=model.init_lora(4);state=tx.init(a)
    x=jnp.ones((1,8,2,2));context=jnp.ones((1,3,128))*.1
    def step(a,state):
        y,pb=model.predict_with_pullback(a,x,.4,context)
        grad=pb(2*y/y.size)
        update,state=tx.update(grad,state,a)
        return optax.apply_updates(a,update),state
    a1,s1=step(a,state)
    save_checkpoint(tmp_path/'step1',a1,s1,{'precision':'mxfp8','update':1})
    expected,expected_state=step(a1,s1)
    restored,restored_state,metadata=load_checkpoint(tmp_path/'step1',expected_metadata={'precision':'mxfp8'})
    actual,actual_state=step(restored,restored_state)
    for left,right in zip(jax.tree.leaves((expected,expected_state)),jax.tree.leaves((actual,actual_state))):
        np.testing.assert_array_equal(left,right)
    export_safetensors(tmp_path/'adapter.safetensors',actual,alpha=model.alpha)
    loaded=import_safetensors(tmp_path/'adapter.safetensors',a,alpha=model.alpha)
    for key in actual:np.testing.assert_array_equal(loaded[key],actual[key])