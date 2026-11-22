"""Pinned native PixAI probabilities, with no tag filtering or prompt rewriting."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk:=source.read(8*1024*1024):digest.update(chunk)
    return digest.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def run(cfg,request,output):
    device=cfg['device']
    if device=='cpu':os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    elif type(device) is int and 0<=device<=3:os.environ.update(CUDA_VISIBLE_DEVICES=str(device),JAX_PLATFORMS='cuda')
    else:raise ValueError('Explicit CPU or physical GPU0..3 required; GPU4 remains reserved')
    os.environ['OMP_NUM_THREADS']=str(cfg.get('threads',4))
    import sys
    sys.path.insert(0,cfg['trainer_root'])
    identity=cfg['tagger_identity'];catalog=cfg['catalog'];root=Path(catalog['path']).parent
    native=Path(cfg['trainer_root'])/'moss_art/pixai_tagger.py'
    for path,key in [(cfg['weights'],'weights_sha256'),(native,'implementation_sha256'),(root/'preprocessor_config.json','preprocessor_sha256')]:
        if sha(path)!=identity[key]:raise ValueError('Tagger identity checksum mismatch: '+key)
    if sha(catalog['path'])!=catalog['sha256'] or identity['catalog_sha256']!=catalog['sha256']:raise ValueError('Tagger catalog changed')
    if Path(catalog['path']).name!='config.json':raise ValueError('Native metadata must use pinned config.json')
    import jax
    from PIL import Image,UnidentifiedImageError
    from moss_art.pixai_tagger import PixAITagger
    if identity['dtype'] not in ('float32','bfloat16'):raise ValueError('Uncalibrated tagger dtype')
    model=PixAITagger.from_pretrained(cfg['weights'],metadata_dir=root,dtype=identity['dtype'],attention_backend=identity['attention_backend'],device=jax.local_devices()[0])
    records=[];seen=set();response=dict(schema=1,status='incomplete',unfiltered=True,tagger_identity=identity,catalog_sha256=catalog['sha256'],items=records)
    try:
        for item in request['items']:
            if item['id'] in seen or sha(item['image'])!=item['image_sha256']:raise ValueError('Invalid image identity or SHA')
            seen.add(item['id']);record=dict(id=item['id'],image_sha256=item['image_sha256'])
            try:
                with Image.open(item['image']) as image:pixels=image.convert('RGB').copy()
            except (OSError,UnidentifiedImageError) as error:
                record.update(status='error',error=str(error));records.append(record);save(str(output)+'.partial.json',response);continue
            values=model.probabilities([pixels],batch_size=1)[0];offset=0;probabilities={}
            for category,count in model.config['tags_split']:
                probabilities[category]={name:float(value) for name,value in zip(model.config['tags'][offset:offset+count],values[offset:offset+count])};offset+=count
            record.update(status='ok',probabilities=probabilities);records.append(record);save(str(output)+'.partial.json',response)
        response['status']='complete';save(output,response)
    finally:model.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:run(json.loads(Path(a.config).read_text(encoding='utf-8-sig')),json.loads(Path(a.request).read_text(encoding='utf-8-sig')),Path(a.output))
    except BaseException:
        save(a.output+'.failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
