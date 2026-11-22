#!/usr/bin/env python3
"""Run a configured repository's staged training then drain fixed validation.

Standard-library control only. No Qwen preparation, model import or GPU process
is started except by the chosen repository's public run_stages.sh/flush command.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from checkpoint_eval_service import DirectoryLock, atomic_json, digest_file, read_json


DEFAULT_PYTHON = '$CONDA_PREFIX/envs/jax_env/bin/python'


def settings_for(config_path):
    config = Path(config_path).expanduser().resolve()
    if not config.is_file(): raise FileNotFoundError(config)
    directory = config.parent
    launch_config = directory/'launcher.config.json'
    settings = read_json(launch_config) if launch_config.is_file() else {}
    if not isinstance(settings,dict): raise ValueError('launcher.config.json must be a JSON object')
    repo = os.environ.get('MOSS_REPO') or settings.get('repo')
    if not isinstance(repo,str) or not Path(repo).is_absolute():
        raise ValueError('Set absolute MOSS_REPO or launcher.config.json repo')
    repo = Path(repo).resolve()
    if not (repo/'scripts/run_stages.sh').is_file():
        raise FileNotFoundError(repo/'scripts/run_stages.sh')
    python = Path(os.environ.get('MOSS_PYTHON') or settings.get('python') or DEFAULT_PYTHON).expanduser()
    if not python.is_absolute() or not python.is_file(): raise ValueError('Training Python must be an existing absolute path')
    helper = Path(settings.get('flush_helper',str(Path(__file__).with_name('flush_validation.py')))).expanduser()
    if not helper.is_absolute() or not helper.is_file(): raise ValueError('flush_helper must be an existing absolute path')
    flush_config = directory/'flush_validation.config.json'
    if not flush_config.is_file(): raise FileNotFoundError(flush_config)
    grace = settings.get('termination_grace_seconds',30)
    poll = settings.get('poll_interval_seconds',.25)
    if type(grace) not in (int,float) or not 0 < grace <= 300:
        raise ValueError('termination_grace_seconds must be in (0,300]')
    if type(poll) not in (int,float) or not .01 <= poll <= 1:
        raise ValueError('poll_interval_seconds must be in [.01,1]')
    return dict(config=str(config),config_sha256=digest_file(config),run_dir=str(directory),
        repo=str(repo),repo_source='MOSS_REPO' if os.environ.get('MOSS_REPO') else 'launcher.config.json',
        python=str(python.resolve()),flush_helper=str(helper.resolve()),flush_config=str(flush_config),
        launcher_config=str(launch_config) if launch_config.is_file() else None,
        launcher_config_sha256=digest_file(launch_config) if launch_config.is_file() else None,
        termination_grace_seconds=float(grace),poll_interval_seconds=float(poll))


def process_table():
    """Linux birth identifiers allow cleanup of only this launch's descendants."""
    table={}
    if not Path('/proc').is_dir(): return table
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            value=(path/'stat').read_text(); fields=value[value.rfind(')')+2:].split()
            if fields[0]=='Z': continue
            table[int(path.name)]=dict(pid=int(path.name),ppid=int(fields[1]),
                pgid=int(fields[2]),start_ticks=fields[19])
        except (OSError,ValueError,IndexError): pass
    return table


class TrainingFlushLauncher:
    def __init__(self,settings):
        self.settings=settings; self.stop_signal=None; self.child=None; self.owned={}
        self.metadata_dir=Path(settings['run_dir'])/'run_metadata'
        self.metadata_dir.mkdir(parents=True,exist_ok=True)
        identity=f'{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}'
        self.metadata_path=self.metadata_dir/('launcher_'+identity+'.json')
        self.record=dict(version=1,launch_id=identity,pid=os.getpid(),status='starting',
            started_at=time.time(),settings=settings,stages=[],qwen_preparation='external_not_started_by_launcher')

    def save(self):
        self.record['updated_at']=time.time()
        atomic_json(self.metadata_path,self.record)
        atomic_json(self.metadata_dir/'launcher.latest.json',self.record)

    def stop(self,number,*_):
        self.stop_signal=self.stop_signal or number

    def capture_owned(self):
        table=process_table()
        roots={pid for pid,old in self.owned.items() if pid in table and table[pid]['start_ticks']==old['start_ticks']}
        if self.child is not None and self.child.poll() is None and self.child.pid in table:
            roots.add(self.child.pid)
        owned=set(roots)
        while True:
            children={pid for pid,entry in table.items() if entry['ppid'] in owned}
            if children <= owned: break
            owned |= children
        for pid in owned: self.owned[pid]=table[pid]
        return {pid:table[pid] for pid,old in self.owned.items()
            if pid in table and table[pid]['start_ticks']==old['start_ticks']}

    def cleanup_owned(self):
        active=self.capture_owned()
        leader_running=self.child is not None and self.child.poll() is None
        if leader_running:
            # Let run_stages.sh's own trap forward TERM to its setsid phase first.
            self.child.terminate()
        if not active and (self.child is None or self.child.poll() is not None): return []
        if os.name!='posix':
            if self.child is not None:
                try: self.child.wait(timeout=self.settings['termination_grace_seconds'])
                except subprocess.TimeoutExpired: self.child.kill(); self.child.wait()
            return []
        # Include already-orphaned but birth-verified stage groups on failure.
        groups={entry['pgid'] for entry in active.values()}
        for group in groups if not leader_running else ():
            if group==os.getpgrp(): raise RuntimeError('Refusing to signal the supervisor process group')
            try: os.killpg(group,signal.SIGTERM)
            except ProcessLookupError: pass
        deadline=time.monotonic()+self.settings['termination_grace_seconds']
        while time.monotonic()<deadline:
            if not self.capture_owned(): break
            time.sleep(min(.1,self.settings['poll_interval_seconds']))
        remaining=self.capture_owned()
        killed=sorted({entry['pgid'] for entry in remaining.values()})
        for group in killed:
            if group==os.getpgrp(): raise RuntimeError('Refusing to signal the supervisor process group')
            try: os.killpg(group,signal.SIGKILL)
            except ProcessLookupError: pass
        if self.child is not None:
            try: self.child.wait(timeout=5)
            except subprocess.TimeoutExpired: self.child.kill(); self.child.wait(timeout=5)
        return killed

    def run_stage(self,name,argv,cwd):
        stage=dict(name=name,argv=argv,cwd=str(cwd),status='running',started_at=time.time())
        self.record.update(status='running',phase=name); self.record['stages'].append(stage); self.save()
        environment=dict(os.environ,MOSS_PYTHON=self.settings['python'])
        self.owned={}
        options=dict(cwd=cwd,env=environment,stdin=subprocess.DEVNULL)
        if os.name=='posix': options['start_new_session']=True
        else: options['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
        started=time.monotonic()
        try:
            self.child=subprocess.Popen(argv,**options)
            stage['pid']=self.child.pid; self.save(); self.capture_owned()
            while self.child.poll() is None:
                self.capture_owned()
                if self.stop_signal: break
                time.sleep(self.settings['poll_interval_seconds'])
            stage['cleanup_killed_groups']=self.cleanup_owned()
            native_code=self.child.returncode
            code=128+self.stop_signal if self.stop_signal else 128-native_code if native_code<0 else native_code
            stage.update(status='interrupted' if self.stop_signal else 'complete' if code==0 else 'failed',
                native_exit_code=native_code,exit_code=code,finished_at=time.time(),seconds=time.monotonic()-started)
            self.save(); return code
        except BaseException as exc:
            stage['cleanup_killed_groups']=self.cleanup_owned()
            stage.update(status='failed',error=repr(exc),finished_at=time.time(),seconds=time.monotonic()-started)
            self.save(); raise
        finally:
            self.child=None

    def run(self):
        with DirectoryLock(self.metadata_dir/'.training_flush_launcher.lock'):
            self.save()
            try:
                stages=[('training',['bash',str(Path(self.settings['repo'])/'scripts/run_stages.sh'),'--config',self.settings['config']],self.settings['repo']),
                    ('validation_flush',[self.settings['python'],self.settings['flush_helper'],'--config',self.settings['flush_config']],str(Path(self.settings['flush_helper']).parent))]
                for name,argv,cwd in stages:
                    if self.stop_signal:
                        code=128+self.stop_signal; break
                    code=self.run_stage(name,argv,cwd)
                    if code: break
                self.record.update(status='interrupted' if self.stop_signal else 'complete' if code==0 else 'failed',
                    exit_code=code,stop_signal=self.stop_signal,finished_at=time.time())
                self.save(); return code
            except BaseException as exc:
                self.record.update(status='failed',error=repr(exc),exit_code=1,finished_at=time.time())
                self.save(); raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(argv)
    launcher=TrainingFlushLauncher(settings_for(args.config))
    signal.signal(signal.SIGTERM,launcher.stop); signal.signal(signal.SIGINT,launcher.stop)
    code=launcher.run()
    print(json.dumps(dict(event='training_and_validation_launcher_exit',status=launcher.record['status'],
        exit_code=code,metadata=str(launcher.metadata_path))),flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
