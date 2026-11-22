"""Native JAX Qwen3.5 prefill, including reference vision and interleaved MRoPE.

Host arrays and layer streaming bound VRAM on the dedicated 16 GiB encoder GPU.
No autoregressive decode, LM head or KV cache is allocated. Image pixels and
chat token IDs are supplied by the existing CPU processor, without a Torch model.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from vendor.moss_jax import text_encoder as te
from vendor.moss_jax.weights import load_host_params,load_config


def multimodal_positions(ids, grids, image_token_id, merge=2):
    ids=np.asarray(ids)
    if ids.shape[0]!=1:
        raise ValueError("Prefill accepts one independent condition at a time")
    positions=np.zeros((3,1,ids.shape[1]),np.int32)
    offset=0; cursor=0
    for t,h,w in grids:
        if t!=1:
            raise ValueError("Only static reference images are supported")
        found=np.flatnonzero(ids[0,cursor:]==image_token_id)
        if not len(found): raise ValueError("Missing image placeholders")
        start=cursor+int(found[0]); n=(h//merge)*(w//merge)
        if not np.all(ids[0,start:start+n]==image_token_id):
            raise ValueError("Image grid/placeholder mismatch")
        text=start-cursor
        positions[:,0,cursor:start]=np.arange(text)[None]+offset
        offset+=text
        hh,ww=np.meshgrid(np.arange(h//merge),np.arange(w//merge),indexing="ij")
        positions[:,0,start:start+n]=np.stack((np.zeros(n),hh.ravel(),ww.ravel())).astype(np.int32)+offset
        offset+=max(h,w)//merge; cursor=start+n
    if np.any(ids[0,cursor:]==image_token_id): raise ValueError("Unexpected image placeholders")
    positions[:,0,cursor:]=np.arange(ids.shape[1]-cursor)[None]+offset
    return positions


def vision_coordinates(h,w,merge=2):
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing="ij")
    def order(a): return a.reshape(h//merge,merge,w//merge,merge).transpose(0,2,1,3).reshape(-1)
    return np.stack((order(yy),order(xx)),axis=-1)


def interpolation(h,w,side,merge=2):
    # Bilinear, align_corners=True on the learned position embedding grid.
    y=np.linspace(0,side-1,h,dtype=np.float32)
    x=np.linspace(0,side-1,w,dtype=np.float32)
    xy=vision_coordinates(h,w,merge)
    yy,xx=y[xy[:,0]],x[xy[:,1]]
    lo_y=np.floor(yy).astype(int);lo_x=np.floor(xx).astype(int)
    hi_y=np.minimum(lo_y+1,side-1);hi_x=np.minimum(lo_x+1,side-1)
    dy=yy-lo_y;dx=xx-lo_x
    return (np.stack((lo_y*side+lo_x,lo_y*side+hi_x,hi_y*side+lo_x,hi_y*side+hi_x),axis=1),
            np.stack(((1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx),axis=1).astype(np.float32))


def linear(p,name,x):
    # Torch Linear/Conv bias is fused into the FP32 GEMM accumulator. Do not
    # round the dot to BF16 before adding bias: vision outliers amplify it.
    value=jnp.matmul(x,p[name+".weight"].T,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    if name+".bias" in p:value=value+p[name+".bias"].astype(jnp.float32)
    return lax.optimization_barrier(value.astype(x.dtype))


def layer_norm(p,name,x):
    z=x.astype(jnp.float32)
    z=(z-jnp.mean(z,axis=-1,keepdims=True))*lax.rsqrt(jnp.var(z,axis=-1,keepdims=True)+1e-6)
    return lax.optimization_barrier((z*p[name+".weight"].astype(jnp.float32)+p[name+".bias"].astype(jnp.float32)).astype(x.dtype))


@partial(jax.jit,static_argnames=("heads","backend"))
def vision_block(p,x,cos,sin,length,*,heads,backend):
    h=layer_norm(p,"norm1",x)
    q,k,v=jnp.moveaxis(linear(p,"attn.qkv",h).reshape(x.shape[0],3,heads,-1),1,0)
    def rotate(z):
        z32=z.astype(jnp.float32);a,b=jnp.split(z32,2,axis=-1)
        return (z32*cos[:,None,:]+jnp.concatenate((-b,a),axis=-1)*sin[:,None,:]).astype(z.dtype)
    q,k=rotate(q),rotate(k)
    if backend=="cudnn":
        att=jax.nn.dot_product_attention(q[None],k[None],v[None],
            query_seq_lengths=length,key_value_seq_lengths=length,implementation="cudnn")[0]
    else:
        logits=jnp.einsum("qhd,khd->hqk",q.astype(jnp.float32),k.astype(jnp.float32))*(q.shape[-1]**-.5)
        logits=jnp.where(jnp.arange(x.shape[0])[None,None,:]<length[0],logits,-jnp.inf)
        att=jnp.einsum("hqk,khd->qhd",jax.nn.softmax(logits,axis=-1),v.astype(jnp.float32)).astype(x.dtype)
    x=lax.optimization_barrier(x+linear(p,"attn.proj",att.reshape(x.shape)))
    h=linear(p,"mlp.linear_fc1",layer_norm(p,"norm2",x))
    h=jax.nn.gelu(h.astype(jnp.float32),approximate=True).astype(x.dtype)
    return lax.optimization_barrier(x+linear(p,"mlp.linear_fc2",h))


@jax.jit
def merge_patches(p,x):
    x=layer_norm(p,"norm",x).reshape(-1,p["linear_fc1.weight"].shape[1])
    x=linear(p,"linear_fc1",x)
    x=jax.nn.gelu(x.astype(jnp.float32),approximate=False).astype(x.dtype)
    return linear(p,"linear_fc2",x)


@jax.jit
def patch_embed(weight,bias,pixels,position):
    # Conv3d has one output per supplied flattened C,T,H,W patch.
    value=jnp.matmul(pixels,weight.reshape(weight.shape[0],-1).T,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    value=lax.optimization_barrier((value+bias.astype(jnp.float32)).astype(pixels.dtype))
    return lax.optimization_barrier(value+position)


class JaxPrefill:
    def __init__(self,model,*,backend="cudnn",vision_capacity=1024,vision_precision="float32",gpu_layers=0,gpu_embedding=False,gpu_vision=False):
        self.params=load_host_params(model,"text_encoder")
        self.config=load_config(model,"text_encoder")
        self.config["text_config"]["jax_attention_backend"]=backend
        if vision_precision not in ("float32","bfloat16"):raise ValueError("Unsupported vision_precision")
        self.vision_dtype=getattr(jnp,vision_precision)
        self.backend=backend
        self.vision_capacity=vision_capacity
        self.device=jax.devices()[0]
        self.calls=0
        layers=self.config["text_config"]["num_hidden_layers"]
        if type(gpu_layers) is not int or not 0<=gpu_layers<=layers:
            raise ValueError("gpu_layers must lie between zero and model depth")
        self.gpu_layers=gpu_layers
        self.gpu_embedding=gpu_embedding
        self.gpu_vision=gpu_vision
        # Spread resident layers uniformly: 16/32 gives even-numbered layers (1-based).
        self.gpu_layer_indices=[((i+1)*layers//gpu_layers)-1 for i in range(gpu_layers)]
        resident=[]
        for key in list(self.params):
            is_layer=key.startswith("language_model.layers.")
            index=int(key.split(".")[2]) if is_layer else -1
            if ((is_layer and index in self.gpu_layer_indices) or
                (gpu_embedding and key=="language_model.embed_tokens.weight") or
                (gpu_vision and key.startswith("visual."))):
                self.params[key]=jax.device_put(self.params[key],self.device)
                resident.append(self.params[key])
        jax.block_until_ready(resident)
        self.resident_bytes=sum(x.nbytes for x in resident)

    def _block(self,prefix):
        return {k[len(prefix):]:v for k,v in self.params.items() if k.startswith(prefix)}

    @staticmethod
    def _release_temporary(value,source):
        if not isinstance(source,jax.Array) or value.dtype!=source.dtype:
            value.delete()

    def vision(self,pixels,grid):
        t,h,w=map(int,grid)
        if t!=1: raise ValueError("Only images are supported")
        cfg=self.config["vision_config"];count=h*w
        capacity=self.vision_capacity
        if count>capacity: raise ValueError("Reference exceeds vision patch capacity")
        coords=vision_coordinates(h,w,cfg["spatial_merge_size"])
        dim=cfg["hidden_size"]//cfg["num_heads"]
        theta=cfg.get("rope_parameters",{}).get("rope_theta",10000.)
        inv=1/(theta**(np.arange(0,dim//2,2,dtype=np.float32)/(dim//2)))
        angles=(coords[...,None].astype(np.float32)*inv).reshape(count,-1)
        angles=np.concatenate((angles,angles),axis=-1)
        cos=jnp.asarray(np.pad(np.cos(angles),((0,capacity-count),(0,0))))
        sin=jnp.asarray(np.pad(np.sin(angles),((0,capacity-count),(0,0))))
        indices,weights=interpolation(h,w,int(cfg["num_position_embeddings"]**.5),cfg["spatial_merge_size"])
        table=self.params["visual.pos_embed.weight"]
        pos=np.sum(np.asarray(table[indices],np.float32)*weights[:,:,None],axis=1)
        pos=jnp.asarray(np.pad(pos,((0,capacity-count),(0,0))),dtype=self.vision_dtype)
        pix=jnp.asarray(np.pad(pixels,((0,capacity-count),(0,0))),dtype=self.vision_dtype)
        wgt=jnp.asarray(self.params["visual.patch_embed.proj.weight"],dtype=self.vision_dtype)
        bias=jnp.asarray(self.params["visual.patch_embed.proj.bias"],dtype=self.vision_dtype)
        x=patch_embed(wgt,bias,pix,pos);x.block_until_ready()
        self._release_temporary(wgt,self.params["visual.patch_embed.proj.weight"])
        self._release_temporary(bias,self.params["visual.patch_embed.proj.bias"])
        for i in range(cfg["depth"]):
            source=self._block(f"visual.blocks.{i}.")
            block={k:jnp.asarray(v,dtype=self.vision_dtype) for k,v in source.items()}
            x=vision_block(block,x,cos,sin,jnp.asarray([count],jnp.int32),heads=cfg["num_heads"],backend=self.backend if self.vision_dtype!=jnp.float32 else "sdpa")
            x.block_until_ready()
            for k,a in block.items(): self._release_temporary(a,source[k])
        source=self._block("visual.merger.")
        block={k:jnp.asarray(v,dtype=self.vision_dtype) for k,v in source.items()}
        merged=merge_patches(block,x)
        result=np.asarray(merged)[:count//(cfg["spatial_merge_size"]**2)].copy()
        merged.delete()
        for k,a in block.items(): self._release_temporary(a,source[k])
        return result

    def encode(self,ids,capacity,pixels=None,grids=()):
        ids=np.asarray(ids,np.int32)
        n=ids.shape[1]
        if ids.shape[0]!=1 or not 0<n<=capacity: raise ValueError("Invalid prefill length/capacity")
        x=np.asarray(self.params["language_model.embed_tokens.weight"][ids]).copy()
        image_id=self.config["image_token_id"]
        if len(grids):
            visual=[];offset=0
            for grid in grids:
                count=int(np.prod(grid))
                visual.append(self.vision(pixels[offset:offset+count],grid));offset+=count
            visual=np.concatenate(visual)
            if visual.shape[0]!=int(np.sum(ids==image_id)): raise ValueError("Visual token mismatch")
            x[ids==image_id]=visual
        positions=multimodal_positions(ids,grids,image_id,self.config["vision_config"]["spatial_merge_size"])
        padded_ids=np.pad(ids,((0,0),(0,capacity-n)))
        mask=np.arange(capacity)[None]<n
        embeds=np.pad(x,((0,0),(0,capacity-n),(0,0)))
        positions=np.pad(positions,((0,0),(0,0),(0,capacity-n)))
        out=te.text_encode(self.params,padded_ids,mask,self.config,streaming=True,
                           inputs_embeds=embeds,position_ids=positions)
        result=np.asarray(out,dtype=np.float32)[:,:n].copy();out.delete();self.calls+=1
        if not np.isfinite(result).all(): raise ValueError("Nonfinite JAX Qwen prefill")
        return result
