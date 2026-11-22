def main():
    from pathlib import Path
    import json,os
    from moss_art.config import load_config
    from moss_art.vllm_prepare import select_devices
    c=load_config("configs/preparation/vllm_qwen_prefill.json")
    select_devices(c["preparation"]["vllm"]["devices"])
    from moss_art.vllm_qwen import VllmPrefill,text_tokens
    p=dict(c["preparation"]["vllm"],enforce_eager=True,max_num_batched_tokens=1024,max_num_seqs=2)
    root=Path("$MOSS_ROOT/audits/vllm_prefill_20261024/debug_eager")
    engine=VllmPrefill(c["model"]["path"],p,root,max_sequence_length=512)
    original=engine.llm.generate
    def show(*args,**kw):
     results=original(*args,**kw)
     print("DEBUG_DECODE",[(r.prompt_token_ids,r.outputs[0].token_ids) for r in results],flush=True)
     return results
    engine.llm.generate=show
    ids=[text_tokens(engine.tokenizer,t) for t in ["A quiet landscape with trees, distant hills, and warm sunlight. "*36]]
    features=engine.encode_token_lists(ids)
    import numpy as np
    np.savez(root/"single_long_vllm.npz",embedding=features[0])
    print("DEBUG_FEATURES",[(x.shape,float(abs(x).max())) for x in features],flush=True)

if __name__=="__main__": main()
