#!/usr/bin/env python3
"""Score explicit target tags with native JAX PixAI; accepts only --config."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args(argv)
    c=json.loads(Path(args.config).read_text());required={'metadata_dir','weights','dtype','attention_backend','device','batch_size','category','strict_tags','samples','output'}
    if set(c)!=required:raise ValueError(f'Config requires exactly {sorted(required)}')
    if not isinstance(c['samples'],list) or not c['samples']:raise ValueError('samples must be nonempty')
    if type(c['strict_tags']) is not bool:raise ValueError('strict_tags must be boolean')
    ids=[]
    for sample in c['samples']:
        if set(sample)!={'id','image','tags'} or not isinstance(sample['id'],str) or not sample['id']:raise ValueError('Each sample requires id, image and tags')
        if not isinstance(sample['tags'],list) or not sample['tags']:raise ValueError('Each sample requires a nonempty tag list')
        ids.append(sample['id'])
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample ids')
    # Match production allocator policy, before any JAX import; never alter XLA_FLAGS.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
    if c['device']=='cpu':os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    elif type(c['device']) is int and c['device']>=0:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
        inventory={int(x.split(',')[0]):x.split(',')[1].strip() for x in lines}
        if c['device'] not in inventory:raise ValueError('Requested physical GPU does not exist')
        os.environ['CUDA_VISIBLE_DEVICES']=inventory[c['device']];os.environ['JAX_PLATFORMS']='cuda'
    else:raise ValueError('device must be cpu or a physical GPU index')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    from moss_art.pixai_tagger import PixAITagger
    start=time.perf_counter();model=PixAITagger.from_pretrained(c['weights'],metadata_dir=c['metadata_dir'],dtype=c['dtype'],attention_backend=c['attention_backend'])
    loaded=time.perf_counter()-start
    try:
        started=time.perf_counter();scores=model.rewards([s['image'] for s in c['samples']],[s['tags'] for s in c['samples']],batch_size=c['batch_size'],category=c['category'],strict=c['strict_tags'])
        report={'schema_version':1,'model':'pixai-labs/pixai-tagger-v1.0','implementation':'native-jax','weights':c['weights'],'dtype':c['dtype'],'attention_backend':c['attention_backend'],'pool_attention':'native FP32 softmax','load_seconds':loaded,'score_seconds_including_first_compile':time.perf_counter()-started,'results':[dict(id=s['id'],image=s['image'],**value) for s,value in zip(c['samples'],scores)]}
        output=Path(c['output']);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
        temporary=output.with_suffix(output.suffix+'.tmp');temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');temporary.replace(output)
        print(json.dumps({'output':str(output.resolve()),'images':len(scores),'load_seconds':loaded,'score_seconds':report['score_seconds_including_first_compile']}))
    finally:model.close()


if __name__=='__main__':main()
