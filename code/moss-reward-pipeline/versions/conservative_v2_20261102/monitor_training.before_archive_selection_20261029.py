#!/usr/bin/env python3
"""Read-only model/process inspection; writes only monitoring records.

Lives outside the checkpoint-signed training source tree. It never imports
ML frameworks, changes a training configuration, or stops a process.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import urllib.request


def read_json(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def json_lines(text):
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result


def inspect(cfg):
    run=Path(cfg['run_dir']).resolve()
    config=read_json(run/'config.json')
    controller=int((run/'controller.pid').read_text())
    processes={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,stat=,args='],text=True).splitlines():
        pid,ppid,stat,args=line.strip().split(None,3)
        processes[int(pid)]=dict(pid=int(pid),ppid=int(ppid),stat=stat,args=args)
    alive=controller in processes and not processes[controller]['stat'].startswith('Z')
    owned={controller} if alive else set()
    while True:
        children={p for p,v in processes.items() if v['ppid'] in owned}
        if children<=owned:break
        owned|=children
    phase='transition' if alive else 'stopped'
    for pid in sorted(owned):
        for candidate in ('infer','eval','train'):
            if '/scripts/'+candidate+'.py' in processes[pid]['args']:
                phase=candidate
    log=(run/'logs/controller.log').read_text(errors='replace')
    events=json_lines(log)
    # Preserve old failure logs without re-reporting a repaired incident.
    segment=log[log.rfind('{"event": "controller_restarted"'):] if '{"event": "controller_restarted"' in log else log
    markers=[line for line in segment.splitlines() if any(x in line for x in
             ('Traceback (most recent call last)', 'FloatingPointError:', 'CheckpointError:', 'ResourceExhaustedError:'))]
    latest=read_json(run/'training/latest.json',{})
    committed_update=int(latest.get('update',0))
    progress=next((e for e in reversed(events) if e.get('event')=='rollout_collection'),None)
    source_update=max(committed_update,int(progress.get('update',0)) if progress else 0)
    last_rollout_progress=progress
    if progress and int(progress.get('update',0))!=source_update:
        progress=None  # A new phase has not reported its first round yet.
    step=run/'training/steps'/('step_%08d'%source_update)
    metrics_file=run/'training/metrics.jsonl'
    metrics=json_lines(metrics_file.read_text()) if metrics_file.exists() else []
    metrics=[m for m in metrics if m.get('event')=='update']
    finite=all(all(type(m.get(k)) not in (int,float) or math.isfinite(m[k]) for k in
                    ('loss','art_loss','base_reference_mse','base_reference_loss','grad_norm','adapter_delta_norm')) for m in metrics)
    checkpoint_valid=None;source_mismatches=[];meta={}
    if latest.get('checkpoint'):
        directory=Path(latest['checkpoint']);content=(directory/'manifest.json').read_bytes()
        checkpoint_valid=(directory/'COMMITTED').read_text().strip()==hashlib.sha256(content).hexdigest()
        meta=json.loads(content)['metadata']
        for name,record in meta.get('experiment',{}).get('source_files',{}).items():
            path=Path(record['resolved_path'])
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                source_mismatches.append(name)
    raw_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw',
                                     '--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in raw_gpu.splitlines():
        a=[x.strip() for x in line.split(',')]
        gpus.append(dict(index=int(a[0]),memory_mib=int(a[1]),util_percent=int(a[2]),
                         temperature_c=int(a[3]),power_w=float(a[4])))
    mem={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
    disk=shutil.disk_usage(run);ramdisk=shutil.disk_usage('/dev/shm')
    tb_status=None;tb_error=None
    try:
        with urllib.request.urlopen(cfg.get('tensorboard_url','http://127.0.0.1:6006')+'/data/logdir',timeout=5) as q:
            tb_status=q.status
    except Exception as exc:tb_error=str(exc)
    sdg=[]
    for path in step.glob('reward_models/*/sdg_dp/dp*/worker.log'):
        decoded=json_lines(path.read_text(errors='replace'))
        item=next((e for e in reversed(decoded) if e.get('event') in ('sdg_batch_done','sdg_reward_done')),None)
        if item:sdg.append(dict(replica=path.parent.name,**item))
    complete=read_json(run/'training/complete.json')
    status='healthy'
    if not alive:status='complete' if complete and committed_update>=config['training']['max_updates'] else 'needs_attention'
    if not finite or checkpoint_valid is False or source_mismatches:status='needs_attention'
    if disk.free/2**30<config['runtime'].get('min_free_disk_gib',10):status='needs_attention'
    return dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        status=status,phase=phase,controller_pid=controller,controller_alive=alive,
        owned_processes=[processes[p] for p in sorted(owned)],progress=progress,
        last_rollout_progress=last_rollout_progress,source_update=source_update,
        completed_update=committed_update,optimizer_updates=meta.get('optimizer_updates',0),
        max_updates=config['training']['max_updates'],latest_checkpoint=latest.get('checkpoint'),
        checkpoint_manifest_commit_valid=checkpoint_valid,checkpoint_payload_note='Not rehashed by this lightweight monitor; strict load verifies payload on resume.',
        source_mismatches=source_mismatches,metrics_count=len(metrics),metrics_finite=finite,
        latest_train=metrics[-1] if metrics else None,current_segment_error_markers=markers[-8:],
        sdg_shards=sdg,gpus=gpus,mem_available_gib=mem['MemAvailable']/2**20,
        swap_used_gib=(mem['SwapTotal']-mem['SwapFree'])/2**20,
        nvme_free_gib=disk.free/2**30,tmpfs_free_gib=ramdisk.free/2**30,
        tensorboard_http=tb_status,tensorboard_error=tb_error,training_complete=complete,
        action='inspection_only',signed_source_frozen=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read_json(args.config);check=inspect(cfg);run=Path(cfg['run_dir'])
    with (run/'monitoring_checks.jsonl').open('a') as f:f.write(json.dumps(check,allow_nan=False)+'\n')
    state=read_json(run/'monitoring_state.json',{})
    state.update(last_checked_at=check['time'],last_status=check['status'],last_phase=check['phase'],
                 last_observed_progress=check['progress'],last_resources=check,
                 completed_update=check['completed_update'],optimizer_updates=check['optimizer_updates'])
    temp=run/'monitoring_state.json.tmp';temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temp.replace(run/'monitoring_state.json')
    compact={k:check[k] for k in ('time','status','phase','source_update','completed_update','optimizer_updates',
             'progress','metrics_count','metrics_finite','source_mismatches','current_segment_error_markers',
             'sdg_shards','gpus','mem_available_gib','swap_used_gib','nvme_free_gib','tmpfs_free_gib','tensorboard_http')}
    print(json.dumps(compact,allow_nan=False))


if __name__=='__main__':main()
