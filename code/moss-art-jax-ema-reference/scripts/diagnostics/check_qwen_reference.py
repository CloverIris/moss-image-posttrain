import os
os.environ["CUDA_VISIBLE_DEVICES"]="GPU-2fbea160-cfc0-82dc-6002-d5ef4c5bf8f4"
import json,numpy as np,torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel,AutoTokenizer,AutoImageProcessor,AutoVideoProcessor,Qwen3VLProcessor
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
root=Path("$CODE_ROOT/moss-art-jax");model="$MODEL_ROOT/MOSS-Image-SFT"
t=AutoTokenizer.from_pretrained(model+"/tokenizer",local_files_only=True)
d="$CODE_ROOT/moss-image/reference-lab/processor"
p=Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(d,local_files_only=True),video_processor=AutoVideoProcessor.from_pretrained(d,local_files_only=True),tokenizer=t,chat_template=t.chat_template)
m=AutoModel.from_pretrained(model+"/text_encoder",dtype=torch.bfloat16,device_map="auto",max_memory={0:"11GiB","cpu":"24GiB"},local_files_only=True,attn_implementation="sdpa").eval().requires_grad_(False)
m.visual.float()
rows=json.loads((root/"cache/sft_1024_smoke/manifest.json").read_text())["identity"]["rows"]
arrays={}
def hook(name):
 def capture(module,args,out):
  v=out.pooler_output if hasattr(out,"pooler_output") else out
  arrays[name]=v.float().cpu().numpy()
 return capture
handles=[m.visual.register_forward_hook(hook("vision"))]
for i in (0,1,26):handles.append(m.visual.blocks[i].register_forward_hook(hook(f"vision_block_{i}")))
for idx,row in enumerate(rows):
 refs=row["conditioning_images"]
 content=[{"type":"image","image":Image.open(path).convert("RGB")} for path in refs]+[{"type":"text","text":row["prompt"]}]
 inputs=p.apply_chat_template([{"role":"user","content":content}],tokenize=True,add_generation_prompt=True,enable_thinking=True,return_dict=True,return_tensors="pt",processor_kwargs={"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":262144}}})
 for k,v in inputs.items():
  if isinstance(v,torch.Tensor):arrays[k]=v.numpy()
 inputs={k:v.to("cuda",dtype=torch.float32) if v.is_floating_point() else v.to("cuda") for k,v in inputs.items() if isinstance(v,torch.Tensor)}
 with torch.inference_mode():
  pos=m.get_rope_index(inputs["input_ids"],inputs["mm_token_type_ids"],image_grid_thw=inputs.get("image_grid_thw"),attention_mask=inputs.get("attention_mask"))[0]
  arrays["positions"]=pos.cpu().numpy()
  out=m(**inputs,use_cache=False,return_dict=True)
  arrays["context"]=out.last_hidden_state.float().cpu().numpy()
 np.savez(root/f"outputs/qwen_reference_fp32_{idx}.npz",**arrays)
 print(json.dumps(dict(event="torch_reference",index=idx,keys=list(arrays))),flush=True)
 arrays={}
