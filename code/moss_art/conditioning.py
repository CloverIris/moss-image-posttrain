"""Fixed-shape prompt buffers with dynamic valid lengths, without dense GPU masks."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from vendor.moss_jax import dit
from .attention import attention

def pad_context(context, capacity):
    """Pad on host before device_put: varying source lengths never enter a JIT."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Prompt capacity must be a positive integer")
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3 or context.shape[0] != 1 or not 0 < context.shape[1] <= capacity:
        raise ValueError(f"Context must be [1,1..{capacity},hidden]; no truncation is allowed")
    if not np.isfinite(context).all():
        raise ValueError("Context must be finite")
    padded = np.zeros((1,capacity,context.shape[2]),np.float32)
    padded[:,:context.shape[1]] = context
    return padded, np.asarray([context.shape[1]],np.int32)

@partial(jax.jit, static_argnums=(0,1,2,3,4), static_argnames=("dtype",))
def padded_rope(height, width, capacity, head_dim, base, valid_lengths, *, dtype, bands=None):
    """Original MOSS coordinates, but image f-coordinate uses the REAL length."""
    num_bands = head_dim//8
    if bands is None or bands.dtype != jnp.float32:
        bands = jnp.float32(1.) / jnp.power(jnp.float32(base),
            jnp.arange(num_bands,dtype=jnp.float32)/jnp.float32(num_bands))
    else:
        bands = bands.astype(jnp.float32)
    row,column = jnp.meshgrid(jnp.arange(height,dtype=jnp.float32),
                              jnp.arange(width,dtype=jnp.float32),indexing="ij")
    def per_sample(length):
        image = jnp.stack((jnp.zeros_like(row),
            jnp.broadcast_to(length.astype(jnp.float32),row.shape),
            row-jnp.float32((height-1)/2), column-jnp.float32((width-1)/2)),axis=-1).reshape(height*width,4)
        context = jnp.zeros((capacity,4),jnp.float32).at[:,1].set(jnp.arange(capacity,dtype=jnp.float32))
        phase = jnp.concatenate((image,context),axis=0)[...,None]*bands
        rope = jnp.concatenate((jnp.sin(phase),jnp.cos(phase)),axis=-2)
        return dit._cast(rope.reshape(height*width+capacity,head_dim),dtype)
    return jax.vmap(per_sample)(valid_lengths)

@partial(jax.jit, static_argnums=(3,5))
def condition_block(params,x,rope,heads,valid_lengths,backend):
    """The two frozen condition blocks need the same length masking as the DiT."""
    q,k,v = jnp.split(dit.linear(params,"block.qkv",
        dit.rms_norm(x,params["block.norm1.weight"])),3,axis=-1)
    shape=(*q.shape[:2],heads,q.shape[-1]//heads)
    q=dit.rms_norm(q.reshape(shape),params["block.q_norm.weight"]).reshape(q.shape)
    k=dit.rms_norm(k.reshape(shape),params["block.k_norm.weight"]).reshape(k.shape)
    x=dit._add(x,dit.linear(params,"block.proj",
        attention(q,k,v,rope,heads,backend=backend,sequence_lengths=valid_lengths)))
    y=dit.rms_norm(x,params["block.norm2.weight"])
    hidden=dit.linear(params,"block.ffn_up",y)
    gate=dit.linear(params,"block.ffn_gate",y)
    return dit._add(x,dit.linear(params,"block.ffn_down",dit.situ(hidden,gate)))
