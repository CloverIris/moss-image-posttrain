#!/usr/bin/env python3
"""Recompute fusion from archived raw scores; no models or GPUs."""
import argparse
import json
from pathlib import Path
import statistics
from reward_fusion import fuse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    source=Path(cfg['input']);out=Path(cfg['output'])
    if out.exists() or source.resolve()==out.resolve():raise FileExistsError('Choose a new analysis output')
    sources=sorted(source.rglob('per_image.jsonl')) if source.is_dir() else [source]
    if not sources:raise ValueError('No archived per_image.jsonl files found')
    rows=[]
    for path in sources:
        rows.extend(json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip())
    calibration=json.loads(Path(cfg['calibration']).read_text())
    new=[]
    for row in rows:
        sdg_enabled=cfg.get('sdg_enabled',row.get('sdg_enabled',True))
        raw_for_fusion={k:v for k,v in row['raw_scores'].items()
                        if sdg_enabled or k!='sdg_artifact'}
        score=fuse(raw_for_fusion,calibration,cfg['artifact_weight'],
                   weights=cfg['weights'],sdg_enabled=sdg_enabled)
        new.append(dict(id=row['id'],global_id=row.get('global_id',row['archive']+':'+row['id']),
                        prompt_id=row['prompt_id'],image=row['image'],
                        raw_scores=row['raw_scores'],old_reward=row['reward'],**score))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(input=str(source),config=cfg,items=new,
        mean_reward=statistics.mean(r['reward'] for r in new),
        notes=['Offline fusion only; original records and applied training updates are unchanged.',
               'Frozen component scales and output gain are retained for comparable weight experiments.']),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(images=len(new),output=str(out))))


if __name__=='__main__':main()
