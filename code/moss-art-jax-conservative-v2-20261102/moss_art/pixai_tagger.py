"""Frozen PixAI Tagger v1.0 in native JAX, with dense soft TAG rewards.

No Torch/Transformers/timm dependency in inference. Original state-dict names
are retained; metadata alone never substitutes for pretrained weights.
Only the released no-CLS, no-relative-position ViTDet architecture is accepted.
"""
from __future__ import annotations
from functools import partial, lru_cache
import json
import math
from pathlib import Path
import struct
import warnings
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from jax import lax

METADATA = Path(__file__).resolve().parents[1] / 'vendor/pixai_tagger/metadata'


def validate_config(c):
    for key in ('img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads', 'window_size', 'num_classes'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['img_size'] % c['patch_size'] or c['pretrain_img_size'] % c['patch_size']:
        raise ValueError('Image sizes must be divisible by patch_size')
    if c['embed_dim'] % c['num_heads'] or (c['embed_dim']//c['num_heads']) % 4:
        raise ValueError('head_dim must be divisible by four')
    if c.get('retain_cls_token') or c.get('rel_pos_blocks') or c.get('init_values'):
        raise ValueError('CLS tokens, relative-position biases and LayerScale are not supported by this release port')
    if c.get('norm_layer') != 'LayerNorm' or c.get('act_layer') != 'GELU':
        raise ValueError('Only the released LayerNorm/GELU architecture is supported')
    ids = c['global_att_blocks']
    if not ids or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < c['depth'] for i in ids):
        raise ValueError('global_att_blocks must be nonempty sorted unique valid block indices')
    if ids[-1] != c['depth']-1:
        raise ValueError('The final block must be a global attention block')
    if len(c.get('tags', [])) != c['num_classes'] or len(set(c['tags'])) != c['num_classes']:
        raise ValueError('Tag vocabulary must match num_classes and contain no duplicates')
    if sum(n for _, n in c['tags_split']) != c['num_classes'] or any(n <= 0 for _, n in c['tags_split']):
        raise ValueError('Category splits must cover the vocabulary')
    if c.get('tile_abs_pos') and not c.get('use_abs_pos'):
        raise ValueError('tile_abs_pos requires use_abs_pos')
    if c.get('use_abs_pos') and not c.get('tile_abs_pos'):
        raise ValueError('The released tiled absolute positions are required; interpolation is not implemented')


def expected_shapes(c):
    validate_config(c)
    d, h, depth = c['embed_dim'], int(c['embed_dim']*c['mlp_ratio']), c['depth']
    shapes = {'patch_embed.proj.weight': (d, c['in_chans'], c['patch_size'], c['patch_size'])}
    if c.get('bias_patch_embed'): shapes['patch_embed.proj.bias'] = (d,)
    if c.get('use_abs_pos'):
        shapes['pos_embed'] = (1, (c['pretrain_img_size']//c['patch_size'])**2 + int(c['pretrain_use_cls_token']), d)
    for norm in ('ln_pre', 'ln_post'):
        if c.get(norm):
            shapes[norm+'.weight']=(d,); shapes[norm+'.bias']=(d,)
    for i in range(depth):
        p=f'blocks.{i}.'
        for norm in ('norm1','norm2'):
            shapes[p+norm+'.weight']=(d,); shapes[p+norm+'.bias']=(d,)
        for name, shape, bias in [('attn.qkv',(3*d,d),c['qkv_bias']), ('attn.proj',(d,d),True), ('mlp.fc1',(h,d),True), ('mlp.fc2',(d,h),True)]:
            shapes[p+name+'.weight']=shape
            if bias: shapes[p+name+'.bias']=(shape[0],)
    shapes.update({'head_pool.q':(d,), 'head_pool.norm.weight':(d,), 'head_pool.norm.bias':(d,),
                   'head_pool.pos_embed':((c['img_size']//c['patch_size'])**2,d),
                   'head_pool.kv.weight':(2*d,d), 'head_pool.kv.bias':(2*d,),
                   'head_pool.proj.weight':(d,d), 'head_pool.proj.bias':(d,),
                   'head.weight':(c['num_classes'],d), 'head.bias':(c['num_classes'],)})
    return shapes


def _round(x, dtype):
    if dtype == jnp.bfloat16: x = lax.reduce_precision(x.astype(jnp.float32),8,7)
    elif dtype == jnp.float16: x = lax.reduce_precision(x.astype(jnp.float32),5,10)
    y=x.astype(dtype)
    return lax.optimization_barrier(y) if dtype in (jnp.bfloat16,jnp.float16) else y


def _add(x,y): return _round(x.astype(jnp.float32)+y.astype(jnp.float32),x.dtype)


def _linear(p,name,x):
    w=p[name+'.weight']; x=x.astype(w.dtype)
    y=lax.dot_general(x,w,(((x.ndim-1,),(1,)),((),())),precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    if name+'.bias' in p: y=y+p[name+'.bias'].astype(jnp.float32)
    return _round(y,w.dtype)


def _norm(p,name,x):
    y=x.astype(jnp.float32);mean=jnp.mean(y,-1,keepdims=True)
    variance=jnp.mean((y-mean)**2,-1,keepdims=True)
    y=(y-mean)*lax.rsqrt(variance+jnp.float32(1e-5))
    return _round(y*p[name+'.weight'].astype(jnp.float32)+p[name+'.bias'].astype(jnp.float32),x.dtype)


@partial(jax.jit,static_argnums=(0,1,2))
def rotary_phase(side,head_dim,source_side):
    bands=jnp.float32(1)/jnp.power(jnp.float32(10000),jnp.arange(0,head_dim,4,dtype=jnp.float32)/jnp.float32(head_dim))
    indices=jnp.arange(side*side,dtype=jnp.float32)
    scale=jnp.float32(source_side/side)
    x=(indices % side)*scale;y=jnp.floor(indices/side)*scale
    return jnp.concatenate((x[:,None]*bands,y[:,None]*bands),-1)


def _rotate(x,phase):
    y=x.astype(jnp.float32).reshape(*x.shape[:-1],x.shape[-1]//2,2)
    real,imag=y[...,0],y[...,1];cos=jnp.cos(phase)[None,:,None,:];sin=jnp.sin(phase)[None,:,None,:]
    y=jnp.stack((real*cos-imag*sin,real*sin+imag*cos),-1)
    return _round(y.reshape(x.shape),x.dtype)


def _sdpa(q,k,v,backend):
    # Layout B,N,H,D. Window padding is intentionally NOT masked: the original
    # ViTDet attends to the normalized/biased padded window tokens as well.
    if backend == 'cudnn':
        if q.dtype not in (jnp.bfloat16,jnp.float16): raise ValueError('cuDNN requires BF16/FP16 QKV')
        nq,nk=q.shape[1],k.shape[1];pq,pk=(-nq)%2,(-nk)%2;lengths={}
        if pq or pk:
            q=jnp.pad(q,((0,0),(0,pq),(0,0),(0,0)))
            k=jnp.pad(k,((0,0),(0,pk),(0,0),(0,0)));v=jnp.pad(v,((0,0),(0,pk),(0,0),(0,0)))
            lengths={'query_seq_lengths':jnp.full((q.shape[0],),nq,jnp.int32),'key_value_seq_lengths':jnp.full((q.shape[0],),nk,jnp.int32)}
        out=jax.nn.dot_product_attention(q,k,v,implementation='cudnn',is_causal=False,**lengths)
        return out[:,:nq]
    if backend != 'native': raise ValueError('attention_backend must be native or cudnn')
    split=jnp.float32(q.shape[-1]**-.25);key=k.astype(jnp.float32)*split;value=v.astype(jnp.float32)
    def chunk(query):
        logits=jnp.einsum('bthd,bshd->bhts',query.astype(jnp.float32)*split,key,precision=lax.Precision.HIGHEST)
        return jnp.einsum('bhts,bshd->bthd',jax.nn.softmax(logits,-1),value,precision=lax.Precision.HIGHEST)
    n=q.shape[1];size=min(n,128);count=(n+size-1)//size
    padded=jnp.pad(q,((0,0),(0,count*size-n),(0,0),(0,0)))
    chunks=padded.reshape(q.shape[0],count,size,q.shape[2],q.shape[3]).transpose(1,0,2,3,4)
    result=lax.map(chunk,chunks).transpose(1,0,2,3,4).reshape(q.shape[0],count*size,q.shape[2],q.shape[3])[:,:n]
    return _round(result,v.dtype)


def partition(x,window):
    b,h,w,d=x.shape;ph=(-h)%window;pw=(-w)%window
    x=jnp.pad(x,((0,0),(0,ph),(0,pw),(0,0)))
    return x.reshape(b,(h+ph)//window,window,(w+pw)//window,window,d).transpose(0,1,3,2,4,5).reshape(-1,window,window,d)


def unpartition(x,window,shape):
    b,h,w,d=shape;hp=h+(-h)%window;wp=w+(-w)%window
    return x.reshape(b,hp//window,wp//window,window,window,d).transpose(0,1,3,2,4,5).reshape(b,hp,wp,d)[:,:h,:w]


def _block(p,x,phase,heads,window,backend):
    shape=x.shape;y=_norm(p,'norm1',x)
    if window: y=partition(y,window)
    b,h,w,d=y.shape
    qkv=_linear(p,'attn.qkv',y).reshape(b,h*w,3,heads,d//heads)
    q,k,v=qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
    if phase is not None:q=_rotate(q,phase);k=_rotate(k,phase)
    y=_sdpa(q,k,v,backend).reshape(b,h,w,d);y=_linear(p,'attn.proj',y)
    if window:y=unpartition(y,window,shape)
    x=_add(x,y);y=_linear(p,'mlp.fc1',_norm(p,'norm2',x))
    y=_round(jax.nn.gelu(y.astype(jnp.float32),approximate=False),y.dtype)
    return _add(x,_linear(p,'mlp.fc2',y))


_block_jit=jax.jit(_block,static_argnums=(3,4,5))


def _stem(p,images,c):
    dtype=p['patch_embed.proj.weight'].dtype;images=images.astype(dtype)
    w=p['patch_embed.proj.weight']
    x=lax.conv_general_dilated(images,w,(c['patch_size'],c['patch_size']),'VALID',dimension_numbers=('NCHW','OIHW','NHWC'),precision=lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
    if 'patch_embed.proj.bias' in p:x=x+p['patch_embed.proj.bias'].astype(jnp.float32)
    x=_round(x,dtype)
    if c.get('use_abs_pos'):
        pos=p['pos_embed'][:,int(c['pretrain_use_cls_token']):];side=c['pretrain_img_size']//c['patch_size'];pos=pos.reshape(1,side,side,-1)
        pos=jnp.tile(pos,(1,x.shape[1]//side+1,x.shape[2]//side+1,1))[:,:x.shape[1],:x.shape[2]]
        x=_add(x,pos)
    if c.get('ln_pre'):x=_norm(p,'ln_pre',x)
    return x


def _pool_head(p,x,heads):
    b,h,w,d=x.shape;x=x.reshape(b,h*w,d);x=_norm(p,'head_pool.norm',x)
    x=_add(x,p['head_pool.pos_embed'][None]);q=jnp.broadcast_to(p['head_pool.q'],(b,1,d)).reshape(b,1,heads,d//heads)
    k,v=jnp.split(_linear(p,'head_pool.kv',x),2,-1)
    k=k.reshape(b,h*w,heads,d//heads);v=v.reshape(b,h*w,heads,d//heads)
    # A single-query pool is cheap in reference math and keeps its softmax FP32.
    y=_sdpa(q,k,v,'native').reshape(b,1,d)[:,0]
    return _linear(p,'head',_linear(p,'head_pool.proj',y)).astype(jnp.float32)


_pool_head_jit=jax.jit(_pool_head,static_argnums=(2,))

@lru_cache(maxsize=8)
def _stem_kernel(config_json):
    config=json.loads(config_json)
    return jax.jit(lambda p,x:_stem(p,x,config))

def _local(params,prefix):return {k[len(prefix):]:v for k,v in params.items() if k.startswith(prefix)}


def forward(params,images,config,*,attention_backend='native',capture=False):
    """Released ViTDet inference; images are normalized NCHW, logits are FP32."""
    validate_config(config)
    if images.ndim != 4 or tuple(images.shape[1:]) != (config['in_chans'],config['img_size'],config['img_size']):
        raise ValueError('Expected normalized NCHW images at configured resolution')
    stem_params={k:v for k,v in params.items() if k.startswith(('patch_embed.','ln_pre.')) or k=='pos_embed'}
    stem_settings={k:config[k] for k in ('patch_size','use_abs_pos','pretrain_img_size','pretrain_use_cls_token','ln_pre')}
    x=_stem_kernel(json.dumps(stem_settings,sort_keys=True))(stem_params,images);trace={'stem':x} if capture else None
    side=config['img_size']//config['patch_size'];head_dim=config['embed_dim']//config['num_heads']
    phases={}
    if config.get('use_rope'):
        for n in {side,config['window_size']}:
            source=config.get('rope_pt_size') or config['window_size']
            phases[n]=rotary_phase(n,head_dim,source if config.get('use_interp_rope') else n)
    for i in range(config['depth']):
        window=0 if i in config['global_att_blocks'] else config['window_size'];n=window or side
        phase=phases.get(n)
        x=_block_jit(_local(params,f'blocks.{i}.'),x,phase,config['num_heads'],window,attention_backend)
        if capture:trace[f'blocks.{i}']=x
    if config.get('ln_post'):x=_norm(params,'ln_post',x)
    head_params={k:v for k,v in params.items() if k.startswith(('head_pool.','head.'))}
    logits=_pool_head_jit(head_params,x,config['num_heads'])
    if capture:trace['logits']=logits;return logits,trace
    return logits


def _resize_axis(image,output,axis):
    """Torchvision float bilinear resize, half-pixel centers, antialias=True."""
    length=image.shape[axis]
    if length==output:return image
    scale=length/output;support=max(1.,scale)
    centers=(np.arange(output,dtype=np.float32)+np.float32(.5))*np.float32(scale)-np.float32(.5)
    start=np.floor(centers-support).astype(np.int32);indices=start[:,None]+np.arange(math.ceil(2*support)+2,dtype=np.int32)[None]
    weights=np.maximum(np.float32(0),np.float32(1)-np.abs(indices.astype(np.float32)-centers[:,None])/np.float32(support))
    weights*=((indices>=0)&(indices<length));weights/=weights.sum(-1,keepdims=True)
    values=np.take(image,np.clip(indices,0,length-1),axis=axis)
    shape=[1]*values.ndim;shape[axis]=output;shape[axis+1]=weights.shape[1]
    return np.sum(values*weights.reshape(shape),axis=axis+1,dtype=np.float32)


def preprocess(images,size=1008):
    """PIL/path or HWC uint8 RGB inputs -> deterministic FP32 normalized NCHW."""
    if isinstance(images,(str,Path,Image.Image)) or isinstance(images,np.ndarray) and images.ndim==3:images=[images]
    rows=[]
    for image in images:
        if isinstance(image,(str,Path)):
            with Image.open(image) as source:image=source.copy()
        if isinstance(image,Image.Image):
            if image.mode!='RGB':
                rgba=image.convert('RGBA');canvas=Image.new('RGBA',rgba.size,(255,255,255));canvas.alpha_composite(rgba);image=canvas.convert('RGB')
            image=np.asarray(image)
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Expected PIL images or HWC uint8 RGB arrays')
        h,w=image.shape[:2]
        if min(h,w)<1:raise ValueError('Empty image')
        x=image.astype(np.float32)/np.float32(255);ratio=min(size/h,size/w);nh,nw=int(h*ratio),int(w*ratio)
        if min(nh,nw)<1:raise ValueError('Aspect ratio produces an empty resized axis')
        # ATen bilinear AA computes width before height.
        x=_resize_axis(_resize_axis(x,nw,1),nh,0)
        top,left=(size-nh)//2,(size-nw)//2
        padded=np.zeros((size,size,3),np.float32);padded[top:top+nh,left:left+nw]=x
        rows.append(((padded-np.float32(.5))/np.float32(.5)).transpose(2,0,1))
    if not rows:raise ValueError('Image batch is empty')
    return np.stack(rows)


def load_parameters(path,config,*,dtype=jnp.float32,device=None):
    """Strict original safetensors loader; maps CPU pages and uploads by tensor."""
    import ml_dtypes
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'PixAI weights are required: {path}. Metadata was copied without weights.')
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>16*1024*1024:raise ValueError('Invalid safetensors header size')
        header=json.loads(f.read(n))
    shapes=expected_shapes(config);found={k:v for k,v in header.items() if k!='__metadata__'}
    if set(found)!=set(shapes):raise ValueError(f'Weight keys differ: missing={sorted(set(shapes)-set(found))}, unexpected={sorted(set(found)-set(shapes))}')
    types={'F32':np.float32,'BF16':ml_dtypes.bfloat16,'F16':np.float16};result={};device=device or jax.devices()[0]
    for key,shape in shapes.items():
        item=found[key]
        if tuple(item['shape'])!=shape or item['dtype'] not in types:raise ValueError(f'Invalid weight shape/dtype: {key}')
        source=np.dtype(types[item['dtype']]);start,end=item['data_offsets']
        if end-start!=math.prod(shape)*source.itemsize or 8+n+end>path.stat().st_size:raise ValueError(f'Invalid weight extent: {key}')
        host=np.memmap(path,mode='r',offset=8+n+start,dtype=source,shape=shape)
        array=jax.device_put(np.asarray(host,dtype=np.dtype(dtype)),device);array.block_until_ready();result[key]=array
    return result


class PixAITagger:
    def __init__(self,params,config,*,attention_backend='native',parallel=None):
        validate_config(config)
        if attention_backend not in ('native','cudnn'):raise ValueError('Unknown attention backend')
        self.params=params;self.config=config;self.attention_backend=attention_backend
        self.parallel=parallel
        self.device=jax.local_devices()[0] if parallel else next(iter(params.values())).device
        if attention_backend=='cudnn' and (self.device.platform!='gpu' or params['head.weight'].dtype not in (jnp.bfloat16,jnp.float16)):
            raise ValueError('cuDNN requires a GPU and BF16/FP16 parameters')
        self.tag_ids={tag:i for i,tag in enumerate(config['tags'])};self.categories={};start=0
        for name,count in config['tags_split']:
            for tag in config['tags'][start:start+count]:self.categories[tag]=name
            start+=count

    @classmethod
    def from_pretrained(cls,weights,*,metadata_dir=METADATA,dtype='float32',attention_backend='native',device=None,data_parallel_size=1):
        parallel=None
        if data_parallel_size>1:
            from .data_parallel import get_data_parallel
            parallel=get_data_parallel(data_parallel_size,4)
        if dtype not in ('float32','bfloat16','float16'):raise ValueError('Unsupported dtype')
        config=json.loads((Path(metadata_dir)/'config.json').read_text())
        preprocessing=json.loads((Path(metadata_dir)/'preprocessor_config.json').read_text())
        if preprocessing['size']!=config['img_size']:raise ValueError('Preprocessor size differs from model size')
        if attention_backend=='cudnn' and (dtype=='float32' or (device or jax.devices()[0]).platform!='gpu'):
            raise ValueError('cuDNN requires a GPU and BF16/FP16 parameters')
        return cls(load_parameters(weights,config,dtype=getattr(jnp,dtype),device=parallel.replicated if parallel else device),config,attention_backend=attention_backend,parallel=parallel)

    def logits(self,pixels):
        pixels=np.asarray(pixels,np.float32)
        if not np.isfinite(pixels).all():raise ValueError('Nonfinite input pixels')
        return forward(self.params,jax.device_put(pixels,self.parallel.batch(4) if self.parallel else self.device),self.config,attention_backend=self.attention_backend)

    def probabilities(self,images,*,batch_size=4):
        if type(batch_size) is not int or batch_size<1:raise ValueError('batch_size must be positive')
        if isinstance(images,(str,Path,Image.Image)) or isinstance(images,np.ndarray) and images.ndim==3:images=[images]
        batch_size*=self.parallel.size if self.parallel else 1
        images=list(images)
        if not images:raise ValueError('Image batch is empty')
        results=[]
        for i in range(0,len(images),batch_size):
            chunk=images[i:i+batch_size];count=len(chunk);chunk=chunk+[chunk[-1]]*(batch_size-count)
            logits=self.logits(preprocess(chunk,self.config['img_size']))
            probs=np.asarray(jax.nn.sigmoid(logits.astype(jnp.float32)))[:count]
            if not np.isfinite(probs).all():raise FloatingPointError('Nonfinite PixAI probabilities')
            results.append(probs)
        return np.concatenate(results)

    def target_indices(self,tags,*,category='general',strict=True):
        selected=[];unknown=[]
        if category is not None and category not in {name for name,_ in self.config['tags_split']}:raise ValueError('Unknown tag category')
        for name in tags:
            if not isinstance(name,str):raise ValueError('Target tags must be strings')
            tag=name.strip()
            if tag not in self.tag_ids:tag=tag.replace(' ','_')
            if tag not in self.tag_ids or category is not None and self.categories[tag]!=category:unknown.append(name);continue
            if tag not in selected:selected.append(tag)
        if unknown and strict:raise ValueError(f'Unknown or out-of-category target tags: {unknown}')
        if unknown:warnings.warn(f'Excluded unsupported target tags: {unknown}',RuntimeWarning)
        if not selected:raise ValueError('No supported target tags remain')
        return np.asarray([self.tag_ids[t] for t in selected],np.int32),selected,unknown

    def rewards(self,images,target_tags,*,batch_size=4,category='general',strict=True):
        # target_tags is one explicit list per image, never inferred from prose.
        if isinstance(images,(str,Path,Image.Image)) or isinstance(images,np.ndarray) and images.ndim==3:images=[images]
        images=list(images)
        if len(images)!=len(target_tags):raise ValueError('One target-tag list is required per image')
        selections=[self.target_indices(tags,category=category,strict=strict) for tags in target_tags]
        probabilities=self.probabilities(images,batch_size=batch_size)
        if len(selections)!=len(probabilities):raise ValueError('One target-tag list is required per image')
        return [{'reward':float(np.mean(row[ids],dtype=np.float64)), 'target_scores':{tag:float(row[idx]) for idx,tag in zip(ids,tags)},
                 'used_tags':tags,'unsupported_tags':unknown} for row,(ids,tags,unknown) in zip(probabilities,selections)]

    def close(self):
        for value in jax.tree.leaves(self.params):value.delete()
        self.params={}
