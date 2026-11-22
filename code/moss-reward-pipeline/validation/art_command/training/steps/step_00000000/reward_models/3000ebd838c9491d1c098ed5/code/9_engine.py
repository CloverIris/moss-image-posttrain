"""Original HF/HPS operators with native DTensor TP or explicit PP placement.

No Accelerate device dispatch, weight offload or custom attention kernels.
Checkpoint pages are mapped on CPU and only local weight shards reach CUDA.
"""
import json
import os
from pathlib import Path
import sys
import time


class NativeHPS:
    def __init__(self,config):
        import torch
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor,Shard
        from torch.distributed.tensor.parallel import parallelize_module,ColwiseParallel,RowwiseParallel
        from torch.distributed.device_mesh import init_device_mesh
        from transformers import AutoConfig
        self.config=config
        self.runtime=config['torch_native']
        self.mode=self.runtime['parallelism']
        self.rank=int(os.environ.get('RANK',0))
        self.local_rank=int(os.environ.get('LOCAL_RANK',0))
        self.world=int(os.environ.get('WORLD_SIZE',1))
        torch.set_num_threads(self.runtime.get('cpu_threads',8))
        if self.mode=='tp':
            if self.world!=2:raise ValueError('TP2 requires torchrun with two ranks')
            torch.cuda.set_device(self.local_rank)
            self.device=torch.device('cuda',self.local_rank)
            dist.init_process_group('nccl',device_id=self.device)
            self.mesh=init_device_mesh('cuda',(self.world,),mesh_dim_names=('tp',))
        elif self.mode=='pp':
            if self.world!=1 or torch.cuda.device_count()!=2:raise ValueError('PP2 uses one process and two GPUs')
            torch.cuda.set_device(0)
            self.device=torch.device('cuda',0)
            self.mesh=None
        else:raise ValueError('parallelism must be tp or pp')
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,config['upstream_hps_dir'])
        from hpsv3.model.qwen3vl_rm import Qwen3VLRewardModelFiLMHybrid
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding,Qwen3VLVisionRotaryEmbedding
        start=time.monotonic()
        cfg=AutoConfig.from_pretrained(config['backbone_dir'],local_files_only=True)
        cfg.text_config.vocab_size=self.runtime.get('vocab_size',151670)
        cfg.pad_token_id=151643
        cfg.tokenizer_padding_side='right'
        for sub in [cfg,cfg.text_config,cfg.vision_config]:
            sub._attn_implementation='sdpa'
            sub.use_cache=False
        with torch.device('meta'):
            model=Qwen3VLRewardModelFiLMHybrid(cfg,output_dim=2,reward_token='special',
                special_token_ids=[151669],rm_head_type='ranknet',cond_dim=256)
        # The score model never calls the vocabulary projection.
        model.lm_head=torch.nn.Identity()
        if self.mode=='tp':
            plan={}
            for i in range(len(model.model.language_model.layers)):
                for name in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','mlp.gate_proj','mlp.up_proj']:
                    plan[f'layers.{i}.{name}']=ColwiseParallel(use_local_output=True)
                for name in ['self_attn.o_proj','mlp.down_proj']:
                    plan[f'layers.{i}.{name}']=RowwiseParallel(input_layouts=Shard(-1),use_local_output=True)
            parallelize_module(model.model.language_model,self.mesh,plan,src_data_rank=None)
        state=torch.load(config['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        if 'model' in state:state=state['model']
        state={k:v for k,v in state.items() if k!='lm_head.weight'}
        expected=set(model.state_dict())
        if set(state)!=expected:
            raise ValueError('Checkpoint mismatch: missing='+repr(sorted(expected-set(state)))+
                ' unexpected='+repr(sorted(set(state)-expected)))
        self.cut=len(model.model.language_model.layers)//2
        vision_dtype=getattr(torch,self.runtime.get('vision_dtype','bfloat16'))
        self.shards=[]
        for name,param in list(model.named_parameters()):
            source=state[name]
            dtype=torch.float32 if name.startswith(('rm_head.','cond_encoder.','film_gen.')) else torch.bfloat16
            if name.startswith('model.visual.'):dtype=vision_dtype
            device=self.device
            if self.mode=='pp':
                on_last=name.startswith(('rm_head.','cap_encoder.','cond_encoder.','film_gen.','model.language_model.norm.'))
                if name.startswith('model.language_model.layers.'):
                    on_last=int(name.split('.')[3])>=self.cut
                device=torch.device('cuda',int(on_last))
            if isinstance(param,DTensor):
                for placement in param.placements:
                    if isinstance(placement,Shard):
                        axis=placement.dim
                        if source.shape[axis]%self.world:raise ValueError('Uneven parameter shard: '+name)
                        width=source.shape[axis]//self.world
                        source=source.narrow(axis,self.rank*width,width)
                local=source.to(device=device,dtype=dtype,copy=True).contiguous()
                value=DTensor.from_local(local,self.mesh,param.placements,run_check=False,
                    shape=param.shape,stride=param.stride())
                self.shards.append(dict(name=name,global_shape=list(param.shape),local_shape=list(local.shape),
                    placements=[str(p) for p in param.placements]))
            else:value=source.to(device=device,dtype=dtype,copy=True)
            parent,key=name.rsplit('.',1)
            setattr(model.get_submodule(parent),key,torch.nn.Parameter(value,requires_grad=False))
        del state,source
        # Rebuild nonpersistent buffers with the original HF classes. The
        # released factory casts them to BF16 before optional FP32 vision.
        text_rope=Qwen3VLTextRotaryEmbedding(cfg.text_config,device='cpu').to(torch.bfloat16).to(self.device)
        text_rope.original_inv_freq=text_rope.inv_freq
        model.model.language_model.rotary_emb=text_rope
        dim=model.model.visual.rotary_pos_emb.inv_freq.numel()*2
        model.model.visual.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(dim).to(torch.bfloat16).to(vision_dtype).to(self.device)
        if any(t.is_meta for _,t in list(model.named_parameters())+list(model.named_buffers())):
            raise ValueError('Unmaterialized model state')
        self.model=model.eval()
        model.per_image_cap=config['scoring'].get('capability_mode','batch')=='per_image'
        if self.mode=='pp':self._install_pipeline_routing()
        if any(hasattr(m,'_hf_hook') for m in model.modules()):raise ValueError('Accelerate hook found')
        torch.backends.cuda.matmul.allow_tf32=False
        if vision_dtype==torch.float32:torch.backends.cudnn.allow_tf32=False
        self._original_sdpa=torch.nn.functional.scaled_dot_product_attention
        if self.runtime.get('attention','cudnn')=='cudnn':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            original=self._original_sdpa
            def sdpa(q,k,v,*args,**kwargs):
                if q.dtype==torch.bfloat16 and q.shape[-1]==128:
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):return original(q,k,v,*args,**kwargs)
                return original(q,k,v,*args,**kwargs)
            torch.nn.functional.scaled_dot_product_attention=sdpa
        self.processor=None
        self.sync()
        self.load_seconds=time.monotonic()-start

    def _install_pipeline_routing(self):
        """Only route tensors; all decoder/model forward math stays upstream."""
        import torch
        language=self.model.model.language_model
        cache={}
        def stage_value(value):
            if isinstance(value,torch.Tensor):
                key=id(value)
                if key not in cache:cache[key]=(value,value.to('cuda:1'))
                return cache[key][1]
            if isinstance(value,tuple):return tuple(stage_value(x) for x in value)
            if isinstance(value,list):return [stage_value(x) for x in value]
            if isinstance(value,dict):return {k:stage_value(x) for k,x in value.items()}
            return value
        def route(original):
            def forward(hidden_states,*args,**kwargs):
                return original(hidden_states.to('cuda:1'),*stage_value(args),**stage_value(kwargs))
            return forward
        for layer in list(language.layers)[self.cut:]:layer.forward=route(layer.forward)
        original=language.forward
        def forward(*args,**kwargs):
            cache.clear()
            try:return original(*args,**kwargs)
            finally:cache.clear()
        language.forward=forward

    def sync(self):
        import torch
        if self.mode=='tp':torch.cuda.synchronize(self.device)
        else:
            for i in range(2):torch.cuda.synchronize(i)

    def barrier(self):
        import torch.distributed as dist
        if self.mode=='tp':dist.barrier()
        self.sync()

    def maximum(self,value):
        import torch
        import torch.distributed as dist
        if self.mode!='tp':return float(value)
        tensor=torch.tensor(float(value),device=self.device,dtype=torch.float64)
        dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
        return tensor.item()

    def forward(self,batch):
        import torch
        with torch.inference_mode():
            return self.model(return_dict=True,iter_values=torch.full((batch['input_ids'].shape[0],),
                float(self.config['scoring']['iter_step']),device=self.device),**batch)

    def score(self,images,prompts):
        import torch
        if isinstance(prompts,str):prompts=[prompts]*len(images)
        if len(images)!=len(prompts) or not images:raise ValueError('One prompt per image required')
        if self.processor is None:
            from hpsv3.inference import HPSv3RewardInferencer
            from transformers import AutoProcessor
            processor=HPSv3RewardInferencer.__new__(HPSv3RewardInferencer)
            processor.processor=AutoProcessor.from_pretrained(self.config['backbone_dir'],local_files_only=True,padding_side='right')
            processor.processor.tokenizer.add_special_tokens({'additional_special_tokens':['<|Reward|>']})
            processor.use_special_tokens=True;processor.device=self.device
            self.processor=processor
        rows=[]
        size=self.config['runtime']['batch_size']
        for start in range(0,len(images),size):
            batch=self.processor.prepare_batch(images[start:start+size],prompts[start:start+size])
            if batch['input_ids'].shape[1]>self.config['runtime']['max_sequence_length']:raise ValueError('Context limit exceeded')
            output=self.forward(batch)
            for values,condition in zip(output['logits'].float().cpu().tolist(),output['cond_pred'].float().cpu().tolist()):
                rows.append(dict(mu=values[0],log_sigma=values[1],sigma=float(torch.tensor(values[1]).exp()),capability=condition[0]))
        return rows

    def close(self):
        import torch
        import torch.distributed as dist
        torch.nn.functional.scaled_dot_product_attention=self._original_sdpa
        if self.mode=='tp':dist.destroy_process_group()
