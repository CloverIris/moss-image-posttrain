"""Checkpoint location and segment replay must preserve full LoRA gradients."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moss_art.model import MossModel
from moss_art.conditioning import pad_context

@pytest.mark.parametrize("interval",[1,2,3,8])
@pytest.mark.parametrize("offload",[False,True])
def test_segmented_vjp_matches_autodiff(interval,offload):
    m=MossModel.tiny(seed=82,config={"num_layers":5},offload_activations=offload,checkpoint_interval=interval)
    rng=np.random.default_rng(7)
    a={k:v+jnp.asarray(rng.normal(0,.01,v.shape),jnp.float32) for k,v in m.init_lora().items()}
    x=rng.normal(size=(1,8,2,2)).astype(np.float32)
    ctx,length=pad_context(rng.normal(size=(1,3,24)).astype(np.float32),8)
    kwargs=dict(context_lengths=length,skip_layers=(1,3))
    out,tape=m._forward(a,x,.4,ctx,record=True,**kwargs)
    assert len(tape["checkpoints"])==(3+interval-1)//interval
    assert all(isinstance(v,np.ndarray if offload else jax.Array) for v in tape["checkpoints"].values())
    cot=jnp.asarray(rng.normal(size=x.shape),jnp.float32)
    actual=m._backward(a,tape,cot)
    expected=jax.grad(lambda aa:jnp.sum(m.predict(aa,x,.4,ctx,**kwargs)*cot))(a)
    for k in expected:
        np.testing.assert_allclose(actual[k],expected[k],rtol=5e-5,atol=2e-6,err_msg=k)
    # Pullbacks remain reusable; reconstructing segments must not consume tape.
    again=m._backward(a,tape,cot)
    for k in actual: np.testing.assert_array_equal(again[k],actual[k])

@pytest.mark.parametrize("interval",[0,-1,True,1.5])
def test_invalid_checkpoint_interval(interval):
    with pytest.raises(ValueError,match="checkpoint_interval"):
        MossModel.tiny(checkpoint_interval=interval)
