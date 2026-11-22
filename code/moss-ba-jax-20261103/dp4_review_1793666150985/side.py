"""Text-conditioned BA SideNet on MOSS packed latents (NCHW)."""
import jax
import jax.numpy as jnp
from flax import linen as nn

def time_features(value, batch):
    frequencies = jnp.exp(-jnp.log(10000.) * jnp.arange(32, dtype=jnp.float32)/32)
    angle = jnp.broadcast_to(jnp.float32(value), (batch,))[:,None] * 1000 * frequencies
    return jnp.concatenate((jnp.cos(angle),jnp.sin(angle)),axis=-1)

class SideNet(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, x, v, t, h, condition):
        w = self.width
        dtype = jnp.bfloat16
        pos,neg,lp,ln = condition
        context = nn.Dense(w,dtype=dtype,name="qwen_linear")(jnp.concatenate((pos,neg),axis=1))
        branch = self.param("branch_embedding",nn.initializers.normal(.02),(2,w))
        role = jnp.concatenate((jnp.zeros(pos.shape[1],jnp.int32),jnp.ones(neg.shape[1],jnp.int32)))
        context = (context + branch[role]).astype(dtype)
        mask = jnp.concatenate((jnp.arange(pos.shape[1])<lp[:,None],jnp.arange(neg.shape[1])<ln[:,None]),axis=1)
        pooled = jnp.sum(jnp.where(mask[...,None],context,0).astype(jnp.float32),axis=1)/jnp.maximum(mask.sum(1)[:,None],1)
        emb = jnp.concatenate((time_features(t,len(x)), time_features(h,len(x)), pooled),axis=-1)
        emb = nn.silu(nn.Dense(w,dtype=dtype,name="condition")(emb))
        features = jnp.concatenate((x,v),axis=1).transpose(0,2,3,1).astype(dtype)
        features = nn.Conv(w,(3,3),dtype=dtype,name="input")(features)
        for i in range(3):
            y = nn.RMSNorm(dtype=dtype,name=f"norm_{i}")(features)
            scale,bias = jnp.split(nn.Dense(2*w,dtype=dtype,name=f"film_{i}")(emb),2,axis=-1)
            y = y*(1+scale[:,None,None,:])+bias[:,None,None,:]
            y = nn.Conv(w,(3,3),feature_group_count=w,dtype=dtype,name=f"depthwise_{i}")(y)
            features = features + nn.Dense(w,dtype=dtype,name=f"pointwise_{i}")(nn.silu(y))
            if i == 1:
                low = nn.avg_pool(features,(2,2),strides=(2,2))
                b,rows,cols,_ = low.shape
                tokens = nn.RMSNorm(dtype=dtype,name="attention_norm")(low.reshape(b,rows*cols,w))
                q,k,val = jnp.split(nn.Dense(3*w,dtype=dtype,name="spatial_qkv")(tokens),3,axis=-1)
                heads = lambda a: a.reshape(b,-1,4,w//4)
                spatial = jax.nn.dot_product_attention(heads(q),heads(k),heads(val)).reshape(tokens.shape)
                ck,cv = jnp.split(nn.Dense(2*w,dtype=dtype,name="text_kv")(context),2,axis=-1)
                text = jax.nn.dot_product_attention(heads(q),heads(ck),heads(cv),mask=mask[:,None,None,:]).reshape(tokens.shape)
                mixed = nn.Dense(w,dtype=dtype,name="att_out")(spatial+text).reshape(low.shape)
                features = features+jnp.repeat(jnp.repeat(mixed,2,axis=1),2,axis=2)
        y = nn.silu(nn.RMSNorm(dtype=dtype,name="output_norm")(features))
        residual = nn.Conv(x.shape[1],(3,3),kernel_init=nn.initializers.zeros,dtype=dtype,name="output")(y)
        return jnp.float32(h)*residual.transpose(0,3,1,2).astype(jnp.float32)

