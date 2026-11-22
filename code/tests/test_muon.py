import copy
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from moss_art.config import DEFAULTS
from moss_art.optimizers import build_optimizer, parameter_labels


def options(**kw):
    return dict(DEFAULTS["optimizer"], name="muon", **kw)


def numpy_ns(g, steps=5):
    x = g.astype(np.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (np.linalg.norm(x) + np.float32(1e-7))
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445*x + (-4.775*a + 2.0315*(a@a)) @ x
    return x.T if transposed else x


@pytest.mark.parametrize("shape", [(4,32),(32,4),(4,4)])
@pytest.mark.parametrize("nesterov", [True,False])
def test_muon_matches_independent_reference(shape,nesterov):
    o = options(learning_rate=1e-3,weight_decay=.02,max_grad_norm=.4,nesterov=nesterov)
    rng = np.random.default_rng(74)
    p = {"matrix":rng.normal(size=shape).astype(np.float32)}
    expected=p["matrix"].copy(); momentum=np.zeros_like(expected)
    tx=build_optimizer(o); state=tx.init(p)
    for _ in range(3):
        g=rng.normal(size=shape).astype(np.float32)
        clipped=g*min(1.,o["max_grad_norm"]/np.linalg.norm(g))
        momentum=o["momentum"]*momentum+clipped
        direction=clipped+o["momentum"]*momentum if nesterov else momentum
        expected=(1-o["learning_rate"]*o["weight_decay"])*expected-o["learning_rate"]*o["rms_match"]*np.sqrt(max(shape))*numpy_ns(direction)
        update,state=tx.update({"matrix":g},state,p);p=optax.apply_updates(p,update)
        np.testing.assert_allclose(p["matrix"],expected,rtol=2e-6,atol=5e-7)
    assert all(x.dtype==jnp.float32 for x in jax.tree.leaves(state) if x.ndim)


def test_muon_fallback_grouping_zero_gradient_and_adam_equivalence():
    o=options(weight_decay=.1,max_grad_norm=100.)
    p={"down":jnp.ones((4,16)),"up":jnp.zeros((16,4)),"rank1":jnp.ones((16,1)),
       "bias":jnp.ones(16),"dora_scale":jnp.ones((2,8))}
    labels=parameter_labels(p)
    assert labels==dict(down="muon",up="muon",rank1="adamw",bias="adamw",dora_scale="adamw")
    tx=build_optimizer(o);state=tx.init(p)
    adam=build_optimizer(dict(o,name="adamw"));astate=adam.init(p)
    for i in range(3):
        g=jax.tree.map(lambda x:jnp.zeros_like(x) if i==0 else jnp.ones_like(x)*.1,p)
        u,state=tx.update(g,state,p);a,astate=adam.update(g,astate,p)
        for k in ("rank1","bias","dora_scale"):np.testing.assert_array_equal(u[k],a[k])
        if i==0:
            for k in ("up","down"):np.testing.assert_allclose(u[k],-o["learning_rate"]*o["weight_decay"]*p[k])
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(u))
        p=optax.apply_updates(p,u)


def test_muon_checkpoint_exact_resume_and_reject_optimizer_change(tmp_path):
    from moss_art.checkpoint import save_checkpoint,load_checkpoint,CheckpointError
    p={"matrix":jnp.arange(48,dtype=jnp.float32).reshape(4,12)*.01,"rank1":jnp.ones((1,12))}
    tx=build_optimizer(options());state=tx.init(p)
    def step(p,state):
        g=jax.tree.map(lambda x:jnp.cos(x),p)
        u,s=tx.update(g,state,p)
        return optax.apply_updates(p,u),s
    p,state=step(p,state)
    save_checkpoint(tmp_path/"step1",p,state,{"optimizer":"muon"})
    expected=step(p,state)
    restored,s,_=load_checkpoint(tmp_path/"step1",expected_metadata={"optimizer":"muon"})
    actual=step(restored,s)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
    with pytest.raises(CheckpointError):load_checkpoint(tmp_path/"step1",expected_metadata={"optimizer":"adamw"})


@pytest.mark.parametrize("bad",[{"name":"sgd"},{"momentum":1.},{"momentum":-.1},{"ns_steps":0},
    {"ns_steps":1.5},{"rms_match":0.},{"nesterov":"true"},{"rms_match":True}])
def test_muon_config_validation_both_entrypoints(tmp_path,bad):
    from moss_art.config import load_config
    from moss_art.sft import load_config as sft_config
    f=tmp_path/"config.json";f.write_text(json.dumps({"optimizer":bad}))
    for loader in (load_config,sft_config):
        with pytest.raises(ValueError):loader(f)


def test_adam_default_behavior_unchanged():
    o=copy.deepcopy(DEFAULTS["optimizer"])
    assert o["name"]=="adamw"
    p={"x":jnp.ones((4,16))};g={"x":jnp.arange(64,dtype=jnp.float32).reshape(4,16)}
    tx=build_optimizer(o)
    old=optax.chain(optax.clip_by_global_norm(o["max_grad_norm"]),optax.adamw(o["learning_rate"],
        b1=o["betas"][0],b2=o["betas"][1],eps=o["epsilon"],weight_decay=o["weight_decay"]))
    for a,b in zip(jax.tree.leaves(tx.update(g,tx.init(p),p)),jax.tree.leaves(old.update(g,old.init(p),p))):np.testing.assert_array_equal(a,b)
