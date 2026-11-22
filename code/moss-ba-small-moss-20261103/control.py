"""Identity-checked test/DP launch, without touching historical jobs or services."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import RUN,REFERENCE,PYTHON,read,save,sha
from pool import lock,identity,matches

CODE=Path(__file__).resolve().parent

def launch(gpu_test=False):
    with lock(RUN/'control/launch.lock',True):
        if (RUN/'cancelled.json').exists():raise RuntimeError('User stopped this run')
        for marker in [RUN/'coordinator.json',RUN/'gpu_test.json',*(RUN.glob('rank_*.json'))]:
            if marker.exists():
                item=read(marker)
                if matches(item):
                    print('Existing live role retained',marker,item['pid']);return
        if (RUN/'complete.json').exists():raise RuntimeError('Run already completed')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except OSError:continue
            if str(CODE/'dp_train.py') in cmd:raise RuntimeError('Unrecorded existing coordinator '+proc.name)
        if shutil.disk_usage(RUN).free<12*1024**3:raise OSError('NVMe reserve')
        plan=read(RUN/'plan.json');assert plan['run_sha']==sha(RUN/'config.json')
        for path,expected in read(RUN/'native_sources.json').items():assert sha(path)==expected,path
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        occupied={row.split(', ')[0]:int(row.split(', ')[1]) for row in raw.strip().splitlines()}
        for uuid in plan['uuids'][:1 if gpu_test else 4]:
            if occupied[uuid]>250:raise RuntimeError(f'GPU already in use {uuid} {occupied[uuid]} MiB')
        sources={str(path):sha(path) for path in sorted(CODE.glob('*.py'))}
        if not gpu_test:
            assert read(RUN/'reports/gpu_acceptance.json')['parameters']==101796864
            assert 'PASS CPU:' in (RUN/'reports/cpu_invariants.txt').read_text()
            assert 'PASS: four CPU replicas' in (RUN/'reports/dp_protocol.txt').read_text()
            assert 'PASS: pinned official' in (RUN/'reports/solver_parity.txt').read_text()
            pin=RUN/'training_sources.json'
            if pin.exists():
                assert read(pin)==sources,'Source changed; preserve and validate repair before repinning'
            else:
                save(pin,sources)
                snapshot=RUN/'source';snapshot.mkdir(exist_ok=True)
                for file in CODE.glob('*.py'):shutil.copy2(file,snapshot/file.name)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['uuids'][0],MOSS_BA_RUN=str(RUN),
                 TMPDIR=str(RUN/'control/tmp'),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 XLA_PYTHON_CLIENT_MEM_FRACTION='.90',PYTHONUNBUFFERED='1')
        command=[PYTHON,str(CODE/('verify_side.py' if gpu_test else 'dp_train.py'))]
        if gpu_test:command.append('--gpu')
        if not gpu_test:
            meta=read(RUN/'run_metadata.json');meta.update(phase='training_preflight',launching_at=time.time());save(RUN/'run_metadata.json',meta)
            parent=REFERENCE/'control/student_restart_100m.json'
            state=read(parent);state.update(phase='training_preflight',planned_run=str(RUN));save(parent,state)
        stamp=time.time_ns();log=RUN/'logs'/f"{'gpu_test' if gpu_test else 'dp_train'}_{stamp}.log"
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        birth,cmd=identity(process.pid)
        role=dict(pid=process.pid,birth=birth,cmd=' '.join(os.fsdecode(v) for v in cmd),command=command,run=str(RUN),code=str(CODE),log=str(log),
                  created_at=time.time(),run_id=plan['run_id'])
        save(RUN/('gpu_test.json' if gpu_test else 'coordinator.json'),role)
        save(RUN/'control'/f'launch_{stamp}.json',role)
        print(role,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-test',action='store_true');args=parser.parse_args()
    launch(args.gpu_test)
