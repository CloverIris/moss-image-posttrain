"""Offline native-JAX SigLIP2 patch14/384 + pinned SILVA calibrated aesthetic head.

Raw MAP pooled embeddings (never L2-normalized), exact GELU in the SILVA head,
SigLIP tanh GELU and original calibration LUT. No Torch dependency at runtime.
"""
from functools import partial, lru_cache
from pathlib import Path
import json
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from jax import lax
from vendor.moss_jax.weights import load_host_params


@lru_cache(maxsize=64)
def _resize_weights(source,target):
    # PyTorch 2.11 CPU uint8 bilinear-AA: per-axis int16 coefficients and
    # round/saturate after each separable pass (not PIL's 22-bit weights).
    scale=source/target;support=max(scale,1.);rows=[];maximum=0.
    for index in range(target):
        center=scale*(index+.5)
        start=max(int(center-support+.5),0);end=min(int(center+support+.5),source)
        weights=[max(0.,1.-abs((j-center+.5)/max(scale,1.))) for j in range(start,end)]
        total=sum(weights);weights=np.asarray([w/total for w in weights],np.float64)
        maximum=max(maximum,float(weights.max()));rows.append((start,weights))
    precision=0
    while precision<22 and int(.5+maximum*(1<<(precision+1)))<(1<<15):precision+=1
    return tuple((start,np.floor(weights*(1<<precision)+.5).astype(np.int32)) for start,weights in rows),precision


def resize_uint8(image,size=384):
    if image.shape[1]!=size:
        rows,precision=_resize_weights(image.shape[1],size)
        out=np.empty((image.shape[0],size,3),np.uint8)
        for index,(start,weights) in enumerate(rows):
            value=(image[:,start:start+len(weights)].astype(np.int32)*weights[None,:,None]).sum(axis=1,dtype=np.int32)
            out[:,index]=np.clip((value+(1<<(precision-1)))>>precision,0,255)
        image=out
    if image.shape[0]!=size:
        rows,precision=_resize_weights(image.shape[0],size)
        out=np.empty((size,image.shape[1],3),np.uint8)
        for index,(start,weights) in enumerate(rows):
            value=(image[start:start+len(weights)].astype(np.int32)*weights[:,None,None]).sum(axis=0,dtype=np.int32)
            out[index]=np.clip((value+(1<<(precision-1)))>>precision,0,255)
        image=out
    return image


def preprocess(images,size=384):
    rows=[]
    for image in images:
        if isinstance(image,(str,Path)):
            with Image.open(image) as im:pixels=np.asarray(im.convert("RGB"))
        elif isinstance(image,Image.Image):pixels=np.asarray(image.convert("RGB"))
        else:pixels=np.asarray(image)
        if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:raise ValueError("SILVA expects RGB uint8 images")
        x=resize_uint8(pixels,size).astype(np.float32)
        rows.append(((x-np.float32(127.5))/np.float32(127.5)).transpose(2,0,1))
    if not rows:raise ValueError("Empty SILVA image batch")
    return np.stack(rows)


def _linear(p,name,x):
    y=jnp.matmul(x,p[name+".weight"].T,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    return (y+p[name+".bias"].astype(jnp.float32)).astype(x.dtype)


def _norm(p,name,x,eps):
    y=x.astype(jnp.float32);mean=jnp.mean(y,axis=-1,keepdims=True)
    var=jnp.mean(jnp.square(y-mean),axis=-1,keepdims=True)
    y=(y-mean)*lax.rsqrt(var+jnp.float32(eps))
    return (y*p[name+".weight"].astype(jnp.float32)+p[name+".bias"].astype(jnp.float32)).astype(x.dtype)


def _attention(q,k,v,heads,backend):
    q=q.reshape(*q.shape[:2],heads,-1);k=k.reshape(*k.shape[:2],heads,-1);v=v.reshape(*v.shape[:2],heads,-1)
    if backend=="cudnn":out=jax.nn.dot_product_attention(q,k,v,implementation="cudnn")
    else:
        logits=jnp.einsum("bqhd,bkhd->bhqk",q.astype(jnp.float32),k.astype(jnp.float32),precision=lax.Precision.HIGHEST)*jnp.float32(q.shape[-1]**-.5)
        out=jnp.einsum("bhqk,bkhd->bqhd",jax.nn.softmax(logits,axis=-1),v.astype(jnp.float32),precision=lax.Precision.HIGHEST).astype(v.dtype)
    return out.reshape(out.shape[0],out.shape[1],-1)


@partial(jax.jit,static_argnames=("patch","size"))
def _embed(p,pixels,*,patch,size):
    x=pixels.transpose(0,2,3,1);grid=size//patch
    x=x[:,:grid*patch,:grid*patch].reshape(x.shape[0],grid,patch,grid,patch,3).transpose(0,1,3,5,2,4).reshape(x.shape[0],grid*grid,-1)
    w=p["patch_embedding.weight"].reshape(p["patch_embedding.weight"].shape[0],-1)
    y=jnp.matmul(x,w.T,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    y=(y+p["patch_embedding.bias"].astype(jnp.float32)).astype(pixels.dtype)
    return (y+p["position_embedding.weight"]).astype(pixels.dtype)


@partial(jax.jit,static_argnames=("heads","eps","backend"))
def _block(p,x,*,heads,eps,backend):
    y=_norm(p,"layer_norm1",x,eps)
    q,k,v=(_linear(p,"self_attn."+name+"_proj",y) for name in ("q","k","v"))
    x=x+_linear(p,"self_attn.out_proj",_attention(q,k,v,heads,backend))
    y=_norm(p,"layer_norm2",x,eps)
    y=_linear(p,"mlp.fc2",jax.nn.gelu(_linear(p,"mlp.fc1",y),approximate=True))
    return x+y


@partial(jax.jit,static_argnames=("heads","eps","backend"))
def _pool(p,x,*,heads,eps,backend):
    x=_norm(p,"post_layernorm",x,eps)
    probe=jnp.broadcast_to(p["head.probe"],(x.shape[0],1,x.shape[-1]))
    weight=p["head.attention.in_proj_weight"];bias=p["head.attention.in_proj_bias"]
    d=x.shape[-1];values=[]
    for i,value in enumerate((probe,x,x)):
        y=jnp.matmul(value,weight[i*d:(i+1)*d].T,precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
        values.append((y+bias[i*d:(i+1)*d].astype(jnp.float32)).astype(x.dtype))
    x=_linear(p,"head.attention.out_proj",_attention(*values,heads,backend))
    y=_norm(p,"head.layernorm",x,eps)
    y=_linear(p,"head.mlp.fc2",jax.nn.gelu(_linear(p,"head.mlp.fc1",y),approximate=True))
    return (x+y)[:,0].astype(jnp.float32)


@partial(jax.jit,static_argnames=("hidden_dims","calibrated"))
def aesthetic_head(p,embedding,*,hidden_dims,calibrated):
    x=_norm(p,"norm",embedding.astype(jnp.float32),1e-5)
    for index,_ in enumerate(hidden_dims):x=jax.nn.gelu(_linear(p,f"trunk.{index*3}",x),approximate=False)
    latent=_linear(p,"head.latent",x)[:,0]
    thresholds=p["head.base_threshold"]+jnp.cumsum(jax.nn.softplus(p["head.raw_deltas"]))
    raw=jax.nn.sigmoid(latent[:,None]-thresholds[None]).mean(axis=-1)
    if calibrated:
        xp,fp=p["cal_lat_knots"],p["cal_score_knots"]
        index=jnp.clip(jnp.searchsorted(xp,latent,side="left"),1,len(xp)-1)
        x0,x1=xp[index-1],xp[index];y0,y1=fp[index-1],fp[index]
        t=jnp.clip((latent-x0)/jnp.maximum(x1-x0,1e-12),0,1)
        score=y0+t*(y1-y0)
    else:score=raw
    return dict(reward=score,raw_score=raw,latent=latent)


class SilvaScorerJax:
    def __init__(self,backbone,head,vision_config,head_config,*,device,dtype,backend,parallel=None):
        self.backbone,self.head=backbone,head
        self.vision_config,self.head_config=vision_config,head_config
        self.device,self.dtype,self.backend=device,dtype,backend
        self.parallel=parallel
        self.calibrated=bool(np.asarray(head["cal_fitted"]))
        self.blocks=[self._view(backbone,f"vision_model.encoder.layers.{i}.") for i in range(vision_config["num_hidden_layers"])]
        self.embeddings=self._view(backbone,"vision_model.embeddings.")
        self.pool_params={k.removeprefix("vision_model."):v for k,v in backbone.items() if k.startswith(("vision_model.head.","vision_model.post_layernorm."))}

    @staticmethod
    def _view(params,prefix):return {k[len(prefix):]:v for k,v in params.items() if k.startswith(prefix)}

    @classmethod
    def from_pretrained(cls,model_dir,backbone_dir,*,device=None,dtype="float32",attention_backend="native",data_parallel_size=1):
        device=device or jax.devices()[0]
        parallel=None
        if data_parallel_size>1:
            from .data_parallel import get_data_parallel
            parallel=get_data_parallel(data_parallel_size,4)
        placement=parallel.replicated if parallel else device
        if dtype not in ("float32","bfloat16"):raise ValueError("SILVA dtype must be float32 or bfloat16")
        if attention_backend not in ("native","cudnn"):raise ValueError("Unsupported SILVA attention backend")
        if attention_backend=="cudnn" and (device.platform!="gpu" or dtype!="bfloat16"):raise ValueError("SILVA cuDNN requires CUDA BF16")
        model_dir,backbone_dir=Path(model_dir),Path(backbone_dir)
        vc=json.loads((backbone_dir/"config.json").read_text())["vision_config"]
        vc={"layer_norm_eps":1e-6,"hidden_act":"gelu_pytorch_tanh",**vc}
        hc=json.loads((model_dir/"config.json").read_text())
        if vc["image_size"]!=384 or vc["patch_size"]!=14 or vc["hidden_size"]!=hc["embedding_dim"] or vc["hidden_act"]!="gelu_pytorch_tanh":raise ValueError("SILVA requires the pinned SigLIP2 patch14-384 backbone")
        if hc.get("n_residual_blocks",0)!=0:raise ValueError("This pinned SILVA release has no residual trunk blocks")
        pp=json.loads((backbone_dir/"preprocessor_config.json").read_text())
        if pp["resample"]!=2 or pp["image_mean"]!=[.5]*3 or pp["image_std"]!=[.5]*3 or pp["size"]!={"height":384,"width":384}:raise ValueError("Unsupported SILVA preprocessing")
        def upload(values,target_dtype):
            result={}
            for k,v in values.items():
                array=np.asarray(v,dtype=np.dtype(target_dtype)) if np.asarray(v).dtype.kind!="b" else np.asarray(v)
                result[k]=jax.device_put(array,placement);result[k].block_until_ready()
            return result
        backbone=upload(load_host_params(backbone_dir,"",prefix="vision_model."),getattr(jnp,dtype))
        head=upload(load_host_params(model_dir,""),jnp.float32)
        if not bool(np.asarray(head["cal_fitted"])):raise ValueError("SILVA calibrated reward requires a fitted calibration LUT")
        return cls(backbone,head,vc,hc,device=device,dtype=getattr(jnp,dtype),backend=attention_backend,parallel=parallel)

    def embed_pixels(self,pixels):
        c=self.vision_config
        x=jax.device_put(np.asarray(pixels),self.parallel.batch(4) if self.parallel else self.device).astype(self.dtype)
        x=_embed(self.embeddings,x,patch=c["patch_size"],size=c["image_size"])
        for block in self.blocks:
            x=_block(block,x,heads=c["num_attention_heads"],eps=c["layer_norm_eps"],backend=self.backend)
            x.block_until_ready()
        return _pool(self.pool_params,x,heads=c["num_attention_heads"],eps=c["layer_norm_eps"],backend=self.backend)

    def score_features(self,features):
        return aesthetic_head(self.head,jax.device_put(features,self.parallel.batch(2) if self.parallel else self.device),hidden_dims=tuple(self.head_config["hidden_dims"]),calibrated=self.calibrated)

    def score(self,images,*,batch_size=4):
        if batch_size<1:raise ValueError("SILVA batch_size must be positive")
        batch_size*=self.parallel.size if self.parallel else 1
        results=[]
        for begin in range(0,len(images),batch_size):
            chunk=list(images[begin:begin+batch_size]);count=len(chunk);chunk+=chunk[-1:]*(batch_size-count)
            features=self.embed_pixels(preprocess(chunk))
            values={k:np.asarray(v)[:count] for k,v in self.score_features(features).items()}
            if not np.isfinite(values["reward"]).all():raise FloatingPointError("Nonfinite SILVA reward")
            results.extend({k:float(v[i]) for k,v in values.items()} for i in range(count))
        return results

    def close(self):
        for value in jax.tree.leaves((self.backbone,self.head)):
            if not value.is_deleted():value.delete()
        self.backbone={};self.head={};self.blocks=[];self.embeddings={};self.pool_params={}
