def main():
 from pathlib import Path
 import json,time,numpy as np
 from moss_art.config import load_config
 from moss_art.vllm_prepare import select_devices
 c=load_config("configs/preparation/vllm_qwen_prefill.json")
 select_devices(c["preparation"]["vllm"]["devices"])
 from moss_art.vllm_qwen import VllmPrefill,text_tokens
 p=dict(c["preparation"]["vllm"],enforce_eager=True,max_num_batched_tokens=8192,max_num_seqs=8)
 root=Path("$MOSS_ROOT/audits/vllm_prefill_20261024/repeat_eager")
 e=VllmPrefill(c["model"]["path"],p,root,max_sequence_length=512)
 long=text_tokens(e.tokenizer,"A quiet landscape with trees, distant hills, and warm sunlight. "*36)
 short=text_tokens(e.tokenizer,"prompt:\nA white cat on a blue chair.\nnegative prompt:\nwatermark")
 empty=text_tokens(e.tokenizer,"")
 cases=[('single1',[long],0),('single2',[long],0),('mixed',[empty,short,long],2),('single3',[long],0),('eight',[long]*8,0),('single4',[long],0)]
 first=None;report=[]
 for name,ids,index in cases:
  f=e.encode_token_lists(ids)[index]
  np.savez(root/(name+'.npz'),embedding=f)
  if first is None:first=f.astype('float64')
  a=f.astype('float64');cos=float(np.sum(a*first)/np.sqrt(np.sum(a*a)*np.sum(first*first)));rel=float(np.linalg.norm(a-first)/np.linalg.norm(first))
  result=dict(name=name,cosine=cos,relative_l2=rel,maximum=float(np.abs(a-first).max()))
  report.append(result);print(json.dumps(result),flush=True)
 (root/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
