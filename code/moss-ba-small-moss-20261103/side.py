"""Trainable MOSS-shaped BA SideNet with FP32 masters and BF16 compute."""
from dataclasses import dataclass
import sys
import numpy as np
import jax
import jax.numpy as jnp
from common import NATIVE

sys.path.insert(0,str(NATIVE))
from vendor.moss_jax import dit
from moss_art.conditioning import padded_rope, condition_block
from moss_art.model import _dit_block


@dataclass(frozen=True)
class SideNet:
    width: int = 768
    ffn_dim: int = 2048
    depth: int = 11
    heads: int = 6
    text_dim: int = 4096
    channels: int = 128
    freq_dim: int = 256
    backend: str = "cudnn"

    def shapes(self):
        d,f,head=self.width,self.ffn_dim,self.width//self.heads
        if d%self.heads or head%8:raise ValueError("Invalid MOSS attention dimensions")
        shapes={}
        def linear(name,inp,out,bias=True):
            shapes[name+".weight"]=(out,inp)
            if bias:shapes[name+".bias"]=(out,)
        def norm(name,n):shapes[name+".weight"]=(n,)
        linear("video_embedder",self.channels*2,d)
        linear("text_embedder.proj",self.text_dim,d)
        shapes["text_roles"]=(2,d)
        for name in ("time_embedding","interval_embedding"):
            linear(name+".0",self.freq_dim,d)
            linear(name+".2",d,d)
        linear("shared_block_modulation",d,6*d)
        linear("video_head_modulation",d,2*d)
        for i in range(2):
            name=f"text_embedder.blocks.{i}"
            norm(name+".norm1",d);norm(name+".norm2",d)
            norm(name+".q_norm",head);norm(name+".k_norm",head)
            linear(name+".qkv",d,3*d);linear(name+".proj",d,d)
            linear(name+".ffn_up",d,f);linear(name+".ffn_gate",d,f)
            linear(name+".ffn_down",f,d)
        for i in range(self.depth):
            name=f"layers.{i}"
            norm(name+".norm1",d);norm(name+".norm2",d)
            norm(name+".q_norm",head);norm(name+".k_norm",head)
            for v in ("q","k","v","o"):linear(name+".proj."+v,d,d,False)
            linear(name+".proj.ffn_up_gate",d,2*f,False)
            linear(name+".proj.ffn_down",f,d,False)
        norm("final_norm_video",d)
        linear("final_linear_video",d,self.channels,False)
        return shapes

    def init(self,key,*_):
        seed=int(np.asarray(jax.random.key_data(key),dtype=np.uint32).astype(np.uint64).sum())
        rng=np.random.default_rng(seed)
        p={}
        for name,shape in self.shapes().items():
            if "norm" in name and name.endswith(".weight"):
                a=np.ones(shape,np.float32)
            elif name.endswith(".bias") or name=="final_linear_video.weight":
                a=np.zeros(shape,np.float32)
            else:
                scale=.02 if name=="text_roles" else 1/np.sqrt(shape[-1])
                a=rng.standard_normal(shape,dtype=np.float32)*np.float32(scale)
            p[name]=jax.device_put(a)
        # Start residual branches with modest open gates and no time modulation.
        p["shared_block_modulation.weight"]=jnp.zeros_like(p["shared_block_modulation.weight"])
        bias=np.zeros((6,self.width),np.float32);bias[[2,5]]=.5
        p["shared_block_modulation.bias"]=jnp.asarray(bias.reshape(-1))
        p["video_head_modulation.weight"]=jnp.zeros_like(p["video_head_modulation.weight"])
        p["interval_embedding.2.weight"]=jnp.zeros_like(p["interval_embedding.2.weight"])
        return {"params":p}

    @staticmethod
    def view(params,prefix):
        return {"block"+k[len(prefix):]:v for k,v in params.items() if k.startswith(prefix+".")}

    def embed_time(self,p,value,prefix,batch):
        half=self.freq_dim//2
        freq=jnp.power(jnp.float64(10000),-jnp.arange(half,dtype=jnp.float64)/half)
        times=jnp.broadcast_to(jnp.float32(value),(batch,))*1000
        phase=times.astype(jnp.float64)[:,None]*freq
        features=jnp.concatenate((jnp.cos(phase),jnp.sin(phase)),axis=-1).astype(jnp.float32)
        y=dit.linear(p,prefix+".0",features)
        y=dit._cast(jax.nn.silu(y.astype(jnp.float32)),jnp.bfloat16)
        return dit.linear(p,prefix+".2",y)

    def apply(self,variables,x,v,t,h,condition):
        p=jax.tree.map(lambda a:a.astype(jnp.bfloat16),variables["params"])
        pos,neg,lp,ln=condition
        batch,_,height,width=x.shape
        capacity=pos.shape[1]+neg.shape[1]
        locations=jnp.arange(capacity)[None,:]
        positive=locations<lp[:,None]
        def gather(data,indices):
            return jnp.take_along_axis(data,jnp.clip(indices,0,data.shape[1]-1)[...,None],axis=1)
        packed=jnp.where(positive[...,None],gather(pos,locations),gather(neg,locations-lp[:,None]))
        valid=locations<(lp+ln)[:,None]
        packed=jnp.where(valid[...,None],packed,0)
        text=dit.linear(p,"text_embedder.proj",packed)
        text=dit._add(text,p["text_roles"][jnp.where(positive,0,1)])
        text=jnp.where(valid[...,None],text,0)
        lengths=(lp+ln).astype(jnp.int32)
        rope=padded_rope(height,width,capacity,self.width//self.heads,10000.,lengths,dtype=jnp.bfloat16)
        def refine(weights,values,positions,lens):
            return condition_block(weights,values,positions,self.heads,lens,self.backend)
        refine=jax.checkpoint(refine,prevent_cse=True)
        for i in range(2):
            text=refine(self.view(p,f"text_embedder.blocks.{i}"),text,rope[:,height*width:],lengths)
        image=jnp.concatenate((x,v),axis=1).reshape(batch,2*self.channels,height*width).transpose(0,2,1)
        image=dit.linear(p,"video_embedder",image)
        time=dit._add(self.embed_time(p,t,"time_embedding",batch),self.embed_time(p,h,"interval_embedding",batch))
        modulation=dit.linear(p,"shared_block_modulation",time).reshape(batch,6,self.width)
        head=dit.linear(p,"video_head_modulation",time).reshape(batch,2,self.width)
        values=jnp.concatenate((image,text),axis=1)
        joint_lengths=lengths+jnp.int32(height*width)
        def block(weights,values,rope,mod,lengths):
            return _dit_block(weights,{},values,rope,mod,self.heads,1.,self.backend,lengths)
        block=jax.checkpoint(block,prevent_cse=True)
        for i in range(self.depth):
            values=block(self.view(p,f"layers.{i}"),values,rope,modulation,joint_lengths)
        image=values[:,:height*width]
        out,_,_=dit._final(p,image,head)
        return jnp.float32(h)*out.transpose(0,2,1).reshape(batch,self.channels,height,width).astype(jnp.float32)
