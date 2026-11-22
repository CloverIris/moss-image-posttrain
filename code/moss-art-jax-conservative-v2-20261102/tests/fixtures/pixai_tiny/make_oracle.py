import json,sys,importlib.util,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[3];W=Path(__file__).resolve().parent
torch.set_num_threads(4);torch.manual_seed(1729)
source=ROOT/'vendor/pixai_tagger/reference/torch_model.py'
spec=importlib.util.spec_from_file_location('pixai_torch_reference',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
c=json.loads((ROOT/'vendor/pixai_tagger/metadata/config.json').read_text());c.update(img_size=24,patch_size=4,pretrain_img_size=8,embed_dim=64,depth=4,num_heads=4,window_size=4,global_att_blocks=[1,3],num_classes=7,tags=['1girl','blue_hair','hat','coat','smile','flower','sample_character'],tags_split=[['general',6],['character',1]])
model=module.ViTDetCls(module.ViTDetClsConfig(**c)).eval()
x=torch.randn(2,3,24,24)*.3
weights={k:v.contiguous() for k,v in model.state_dict().items()};save_file(weights,str(W/'tiny.safetensors'))
(W/'tiny_config.json').write_text(json.dumps(c,indent=2));(W/'preprocessor_config.json').write_text(json.dumps({'size':24}))
arrays={'input':x.numpy()};active={}
def hook(name):
 def callback(mod,inputs,output):arrays[name]=output.detach().float().numpy()
 return callback
# The first block's input is the exact stem output.
model.blocks[0].register_forward_pre_hook(lambda mod,args:arrays.update(stem=args[0].detach().float().numpy()))
for i,b in enumerate(model.blocks):b.register_forward_hook(hook('blocks.'+str(i)))
with torch.inference_mode():arrays['logits']=model(x).float().numpy()
# Preserve complex64 rotary buffers when casting only trainable parameters.
for parameter in model.parameters():parameter.data=parameter.data.bfloat16()
arrays_bf={'input':x.bfloat16().float().numpy()}
arrays_old=arrays;arrays=arrays_bf
with torch.inference_mode():arrays['logits']=model(x.bfloat16()).float().numpy()
arrays=arrays_old
for k,v in arrays_bf.items():arrays['bf16_'+k]=v
# Independent torchvision float preprocessing oracle, including alpha composition.
from PIL import Image
rng=np.random.default_rng(2026)
processor=module.RescalePadProcessor(size=24)
for i,shape in enumerate([(13,31),(31,13),(24,24),(7,11),(49,63),(1,24),(24,1)]):
 image=rng.integers(0,256,(*shape,3),dtype=np.uint8);arrays['raw_'+str(i)]=image
 arrays['preprocessed_'+str(i)]=processor(Image.fromarray(image))['pixel_values'].numpy()
rgba=rng.integers(0,256,(17,31,4),dtype=np.uint8);arrays['raw_rgba']=rgba;arrays['preprocessed_rgba']=processor(Image.fromarray(rgba))['pixel_values'].numpy()
np.savez_compressed(W/'torch_reference.npz',**arrays)
(W/'oracle_info.json').write_text(json.dumps({'torch':torch.__version__,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seed':1729,'note':'Synthetic small architecture only; no pretrained PixAI weights used or transferred. FP32 and BF16 (parameter-only casting preserves complex rotary buffers).'},indent=2))
print('ORACLE READY',len(arrays),flush=True)
