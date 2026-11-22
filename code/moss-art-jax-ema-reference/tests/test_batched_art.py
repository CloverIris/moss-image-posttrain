"""Real sample batching preserves seed identity and fixed ART denominators."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moss_art.art import covariance,prompt_windows
from moss_art.trainer import aggregate_gradients
from test_paired_replay import worker

class BatchedPolicy:
    config={"latent_channels":2}
    def predict(self,a,x,sigma,context,**kw):
        time=jnp.asarray(sigma).reshape((-1,)+(1,)*(x.ndim-1))
        ctx=jnp.mean(context,axis=(1,2)).reshape((-1,)+(1,)*(x.ndim-1))
        return .15*x+a["theta"]*jnp.tanh(x+ctx)+.03*time**2
    def predict_with_pullback(self,a,x,sigma,context,**kw):
        value,pb=jax.vjp(lambda aa:self.predict(aa,x,sigma,context,**kw),a)
        return value,lambda cot:pb(cot)[0]

@pytest.mark.parametrize("batch_size",[2,4])
def test_rollout_and_gradient_batch_equal_serial(batch_size):
    from moss_art.config import DEFAULTS
    c=copy.deepcopy(DEFAULTS)
    c["sampling"].update(height=32,width=32,steps=4,train_timesteps=[0,1,2,3],group_size=4,
        sampler="euler",scheduler="moss",guidance_scale=1.,skip_layers=[])
    serial=worker(c);serial.model=BatchedPolicy()
    c=copy.deepcopy(c);c["sampling"]["rollout_batch_size"]=batch_size;c["training"]["microbatch_size"]=batch_size
    batched=worker(c);batched.model=BatchedPolicy()
    record={"sample_id":"synthetic","prompt":"synthetic fixture","negative_prompt":""}
    outputs=[w.rollout([record],[1,2,3,4]) for w in (serial,batched)]
    for key in ("endpoints",):np.testing.assert_allclose(outputs[0][key],outputs[1][key],rtol=2e-6,atol=2e-7)
    for key in ("states","old"):np.testing.assert_allclose(serial.cache[key],batched.cache[key],rtol=2e-6,atol=2e-7)
    groups=np.zeros(4,np.int32);stats={k:np.asarray(v) for k,v in covariance(outputs[0]["endpoints"],[.1,.3,.7,.9],groups).items()}
    windows=prompt_windows(groups,4,seed=42)
    for w in (serial,batched):w.prepare_update(stats=stats,windows=windows)
    for i in range(2):
        grads=[]
        for w in (serial,batched):
            g,metrics=aggregate_gradients([w.gradient(i)]);grads.append(g)
            assert metrics["items"]==8
        np.testing.assert_allclose(grads[0]["theta"],grads[1]["theta"],rtol=3e-6,atol=3e-7)
        value={"theta":serial.lora["theta"]-.001*grads[0]["theta"]}
        for w in (serial,batched):w.set_lora(value)
