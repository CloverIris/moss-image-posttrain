"""Fixed prompt shape preserves full-attention semantics and compilation reuse."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from moss_art.conditioning import pad_context,padded_rope,condition_block
from moss_art.model import MossModel
from moss_art import model as model_module
from moss_art.attention import attention
from vendor.moss_jax import dit

def inputs():
    rng=np.random.default_rng(281)
    model=MossModel.tiny(seed=92,targets="all")
    adapters=model.init_lora()
    adapters={k:v+jnp.asarray(rng.normal(0,.01,v.shape),jnp.float32) for k,v in adapters.items()}
    state=rng.normal(size=(1,8,2,2)).astype(np.float32)
    context=rng.normal(size=(1,7,24)).astype(np.float32)
    cotangent=rng.normal(size=state.shape).astype(np.float32)
    return model,adapters,state,context,cotangent

@pytest.mark.parametrize("length",[1,3,7])
@pytest.mark.parametrize("capacity",[16,17])
def test_padded_output_and_lora_gradient_match_unpadded(length,capacity):
    model,adapters,state,context,cotangent=inputs()
    context=context[:,:length]
    original,back=model.predict_with_pullback(adapters,state,.4,context)
    grad=back(cotangent)
    padded,valid=pad_context(context,capacity)
    # Nonzero junk in the padding tests actual masking, not just zero values.
    padded[:,length:]=np.random.default_rng(90).normal(0,20,padded[:,length:].shape)
    actual,back=model.predict_with_pullback(adapters,state,.4,padded,context_lengths=valid)
    actual_grad=back(cotangent)
    np.testing.assert_allclose(actual,original,rtol=2e-5,atol=2e-6)
    for key in grad:
        np.testing.assert_allclose(actual_grad[key],grad[key],rtol=4e-4,atol=2e-6)

def test_dynamic_length_reuses_forward_backward_condition_and_rope_jits():
    model,adapters,state,context,cotangent=inputs()
    kernels=(model._block_forward,model._block_vjp,padded_rope,condition_block,
             dit._compiled_linear,model_module._final_vjp)
    for kernel in kernels:
        kernel.clear_cache()
    counts=[]
    for length in (3,7,1):
        padded,valid=pad_context(context[:,:length],16)
        value,back=model.predict_with_pullback(adapters,state,.3,padded,context_lengths=valid)
        jax.block_until_ready(back(cotangent))
        counts.append([kernel._cache_size() for kernel in kernels])
    assert counts[0]==counts[1]==counts[2]
    assert all(count>0 for count in counts[0])
    padded,valid=pad_context(context,16)
    larger=np.zeros((1,8,3,2),np.float32)
    value,back=model.predict_with_pullback(adapters,larger,.3,padded,context_lengths=valid)
    jax.block_until_ready(back(jnp.ones_like(value)))
    assert model._block_forward._cache_size()>counts[-1][0]

def test_real_length_rope_matches_original_valid_tokens():
    for length in (1,7):
        padded=padded_rope(2,3,16,128,10000.,jnp.asarray([length],jnp.int32),dtype=jnp.float32)
        original=dit.rope_embeddings(2,3,length,128,dtype=jnp.float32)
        np.testing.assert_allclose(padded[:,:6+length],original,rtol=2e-7,atol=1e-7)

def test_cfg_masks_positive_and_negative_independently():
    model,adapters,state,context,cotangent=inputs()
    positive,pl=pad_context(context,16)
    negative,nl=pad_context(context[:,:2],16)
    expected,back=model.predict_with_pullback(adapters,state,.4,context,cfg_scale=3.,
        negative_context=context[:,:2],negative_skip_layers=(1,))
    grad=back(cotangent)
    actual,back=model.predict_with_pullback(adapters,state,.4,positive,cfg_scale=3.,
        negative_context=negative,negative_skip_layers=(1,),context_lengths=pl,negative_context_lengths=nl)
    actual_grad=back(cotangent)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    for key in grad:
        np.testing.assert_allclose(actual_grad[key],grad[key],rtol=4e-4,atol=2e-6)

def test_cudnn_gets_dynamic_prefix_lengths_without_dense_mask(monkeypatch):
    captured=[]
    def kernel(q,k,v,**kwargs):
        captured.append(kwargs)
        assert "mask" not in kwargs and "bias" not in kwargs
        assert kwargs["implementation"]=="cudnn" and kwargs["is_causal"] is False
        np.testing.assert_array_equal(kwargs["query_seq_lengths"],[5,9])
        np.testing.assert_array_equal(kwargs["key_value_seq_lengths"],[5,9])
        return jnp.ones_like(q)
    monkeypatch.setattr(jax.nn,"dot_product_attention",kernel)
    q=jnp.ones((2,17,256),jnp.bfloat16)
    rope=dit.rope_embeddings(1,1,16,128,dtype=jnp.bfloat16)
    result=attention(q,q,q,rope,2,backend="cudnn",sequence_lengths=jnp.asarray([5,9],jnp.int32))
    assert len(captured)==1
    np.testing.assert_array_equal(result[0,5:],0)
    np.testing.assert_array_equal(result[1,9:],0)

def test_padding_rejects_truncation_and_invalid_valid_lengths():
    model,adapters,state,context,_=inputs()
    with pytest.raises(ValueError,match="no truncation"):
        pad_context(context,6)
    padded,valid=pad_context(context,16)
    for lengths in ([0],[17],[1,2],[1.5]):
        with pytest.raises(ValueError,match="context_lengths"):
            model.predict(adapters,state,.4,padded,context_lengths=np.asarray(lengths))


def test_native_chunked_mask_matches_individual_unpadded_rows():
    rng=np.random.default_rng(19)
    q,k,v=[jnp.asarray(rng.normal(size=(2,145,16)),jnp.float32) for _ in range(3)]
    rope=dit.rope_embeddings(1,1,144,8,dtype=jnp.float32)
    lengths=jnp.asarray([17,139],jnp.int32)
    actual=attention(q,k,v,rope,2,sequence_lengths=lengths)
    for row,length in enumerate((17,139)):
        expected=attention(q[row:row+1,:length],k[row:row+1,:length],
                           v[row:row+1,:length],rope[:,:length],2)
        np.testing.assert_allclose(actual[row:row+1,:length],expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_array_equal(actual[row,length:],0)
    grad=jax.grad(lambda value:attention(q,k,value,rope,2,sequence_lengths=lengths).sum())(v)
    np.testing.assert_array_equal(grad[0,17:],0)
    np.testing.assert_array_equal(grad[1,139:],0)
