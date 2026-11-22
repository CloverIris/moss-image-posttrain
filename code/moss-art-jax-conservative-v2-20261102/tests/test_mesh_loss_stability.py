"""ART loss AD and padding semantics across the per-item Mesh boundary."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from moss_art.art import art_loss
from moss_art.mesh_runtime import _loss_vjp,_loss_vjp_items

OPTIONS=dict(noise=.8,kl=1e-5,eps=1e-5)

def fixture():
    rng=np.random.default_rng(43)
    direction=rng.normal(size=(4,2,3,2)).astype(np.float32)
    normalized=direction/np.sqrt(np.mean(direction**2,axis=(1,2,3),keepdims=True))
    current=np.zeros_like(direction);current[0]=-.5*normalized[0];current[2]=.25;current[3]=100.
    return [current,np.zeros_like(current),np.full(4,.5,np.float32),direction,
            np.array([1.,0.,0.,1.],np.float32),np.full(4,-.05,np.float32),
            np.array([False,False,True,False]),np.array([True,True,True,False])]

def reference(args,mode):
    current,old,sigma,direction,magnitude,coef,anchor,valid=args
    def objective(value):
        terms=art_loss(value,old,sigma,direction,magnitude,ode_velocity_coefficient=coef,
            restore_anchor_mask=anchor,valid_mask=valid,q2_reference_sde_noise_level=.8,
            q2_reference_kl_threshold=1e-5,eps=1e-5,anchor_mode=mode)
        return jnp.sum(terms['per_item_loss']),(jnp.sum(terms['masked']),jnp.sum(terms['keep_mask']))
    return jax.value_and_grad(objective,has_aux=True)(current)

@pytest.mark.parametrize('mode',['cached','self','calibrated'])
def test_matches_scalar_reference_for_mask_padding_anchor_and_inactive_rows(mode):
    args=[jnp.asarray(v) for v in fixture()]
    (expected,counts),gradient=reference(args,mode)
    (actual,actual_counts),actual_gradient=_loss_vjp(*args,**OPTIONS,anchor_mode=mode)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-7)
    assert actual_counts==tuple(int(v) for v in counts)
    np.testing.assert_allclose(actual_gradient,gradient,rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(np.asarray(actual_gradient)[[1,3]],0.)
    if mode=='cached':assert actual_counts==(1,1)

@pytest.mark.parametrize('mode',['cached','self','calibrated'])
def test_gradient_numerators_add_before_original_window_normalization(mode):
    args=fixture();args[0]=np.full_like(args[0],.03)
    args[4]=np.ones(4,np.float32);args[6]=np.zeros(4,bool);args[7]=np.array([True,False,True,True])
    (expected,_),grad=reference([jnp.asarray(v) for v in args],mode)
    pieces=[];losses=[]
    for start in (0,2):
        (loss,_),piece=_loss_vjp(*[jnp.asarray(v[start:start+2]) for v in args],**OPTIONS,anchor_mode=mode)
        losses.append(loss);pieces.append(piece)
    # Padding is excluded, but genuine masked/inactive rows stay in the divisor.
    denominator=int(args[-1].sum())
    np.testing.assert_allclose(sum(losses)/denominator,float(expected)/denominator,rtol=2e-6)
    np.testing.assert_allclose(np.concatenate(pieces)/denominator,np.asarray(grad)/denominator,rtol=2e-6,atol=1e-7)

def test_no_signal_is_exact_zero_loss_and_cotangent():
    args=fixture();args[3].fill(0);args[4].fill(0);args[6].fill(False)
    (loss,counts),gradient=_loss_vjp(*[jnp.asarray(v) for v in args],**OPTIONS)
    assert loss==0 and counts==(0,0)
    np.testing.assert_array_equal(gradient,0.)

def test_sharded_loss_vectors_have_no_cross_device_scalar_collective():
    if len(jax.devices())<4:pytest.skip('requires four logical devices')
    mesh=Mesh(np.asarray(jax.devices()[:4]),('dp',))
    args=[jax.device_put(v,NamedSharding(mesh,P('dp',*(None for _ in range(v.ndim-1))))) for v in fixture()]
    (items,aux),gradient=_loss_vjp_items(*args,**OPTIONS)
    assert items.shape==(4,) and all(a.shape==(4,) for a in aux)
    assert items.sharding.is_equivalent_to(NamedSharding(mesh,P('dp')),1)
    assert gradient.sharding.is_equivalent_to(args[0].sharding,args[0].ndim)
    hlo=_loss_vjp_items.lower(*args,**OPTIONS).compile().as_text().lower()
    assert 'all-reduce' not in hlo and 'all-gather' not in hlo
    (value,counts),actual_gradient=_loss_vjp(*args,**OPTIONS)
    (expected,expected_counts),expected_gradient=reference([jnp.asarray(v) for v in fixture()],'cached')
    np.testing.assert_allclose(value,expected,rtol=2e-6,atol=1e-7)
    assert counts==tuple(int(v) for v in expected_counts)
    np.testing.assert_allclose(actual_gradient,expected_gradient,rtol=2e-6,atol=1e-7)
