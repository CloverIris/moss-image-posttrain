"""Create exactly one fresh student run and audit reuse of completed references."""
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from common import RUN,REFERENCE,BASE,NATIVE,read,save,sha
from pool import lock


def prepare():
    with lock(REFERENCE/'control/student_restart_100m.lock',True):
        control=read(REFERENCE/'control/student_restart_100m.json')
        if control.get('planned_run') not in (None,str(RUN)):
            raise RuntimeError('A different successor was already selected')
        control.update(phase='implementing',planned_run=str(RUN),code=str(Path(__file__).parent),
                       user_instruction='杀死网页生图服务、实现新学生架构 测试 开始训练 0.1B的学生',
                       implementation_authorized_at=time.time(),operation_id=control.get('operation_id',str(uuid.uuid4())))
        save(REFERENCE/'control/student_restart_100m.json',control)
        if (RUN/'config.json').exists():
            print('Existing unique run retained',RUN);return
        RUN.mkdir(parents=True,exist_ok=False)
        for sub in ('reports','logs','pool','control/tmp'): (RUN/sub).mkdir(parents=True)
        cfg=read(REFERENCE/'config.json')
        validation=read(REFERENCE/'reports/reference_complete_validation.json')
        assert validation['verified'] and validation['train_count']==1971 and validation['eval_count']==4
        assert validation['configuration_sha256']==sha(REFERENCE/'config.json')
        assert cfg['sampling']==dict(rollout_mode='bf16',guidance_scale=3.,skip_layers=[22],shift=5.,resolution=1024)
        assert len(cfg['order'])==1971 and len(cfg['eval_plan'])==4
        checked=0
        for split,key in [('train','training'),('eval','evaluation')]:
            for row in validation[key]:
                path=REFERENCE/'latents'/split/f"{row['index']:05d}.npz"
                assert sha(path)==row['sha256'],path
                checked+=1
        # Verify the native implementation used to create the BF16 reference.
        sources=read(REFERENCE/'training_sources.json')
        native={k:v for k,v in sources.items() if str(k).startswith(str(NATIVE)+'/')}
        for path,expected in native.items():assert sha(path)==expected,path
        policies={str(p):sha(p) for p in (BASE/'training_precision_policy.json',BASE/'datasets/next_training_dataset.json')}
        cfg.update(width=768,student=dict(architecture='MOSS_single_stream_BA_SideNet',ffn_dim=2048,depth=11,
                   heads=6,head_dim=128,text_layers=2,channels=128,freq_dim=256,backend='cudnn',parameters=101796864),
                   optimizer=dict(name='adamw',lr=.0003,warmup_updates=32,weight_decay=.01,clip_norm=1.),
                   training=dict(gpus=[0,1,2,3],dp_batch=4,tail_batch=3,total_updates=1972,fresh_student=True,fresh_optimizer=True),
                   reference_run=str(REFERENCE),reference_validation_sha=sha(REFERENCE/'reports/reference_complete_validation.json'),
                   evaluation_steps=[0,128,512,1024,1972],code=str(Path(__file__).parent))
        save(RUN/'config.json',cfg)
        (RUN/'latents').symlink_to(REFERENCE/'latents',target_is_directory=True)
        shutil.copy2(REFERENCE/'producer_complete.json',RUN/'producer_complete.json')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
        devices=[r.split(', ') for r in raw.strip().splitlines()]
        assert all('GB203(SM120)' in devices[i][2] for i in range(4))
        save(RUN/'plan.json',dict(run_sha=sha(RUN/'config.json'),uuids=[devices[i][1] for i in range(4)],
                                run_id=str(uuid.uuid4()),total_updates=1972,global_batch=4,tail_samples=3))
        save(RUN/'native_sources.json',native)
        save(RUN/'reports/input_reuse_audit.json',dict(verified=True,rechecked_sha_files=checked,
             reference_validation=str(REFERENCE/'reports/reference_complete_validation.json'),
             reference_validation_sha=cfg['reference_validation_sha'],native_sources_verified=len(native),policies=policies,
             train_sha=cfg['train_sha'],eval_sha=cfg['eval_sha'],qwen_cache_manifest_sha=cfg['cache_manifest_sha'],
             original_text_unchanged=True,reference_symlink_read_only_by_training=True))
        save(RUN/'run_metadata.json',dict(phase='testing',created_at=time.time(),code=str(Path(__file__).parent),
             parent_run=str(REFERENCE),run=str(RUN),operation_id=control['operation_id'],fresh_optimizer=True,
             fresh_random_student=True,retired_student_checkpoint_reused=False,web_generation_stopped=True))
        print('Prepared',RUN,'reference SHA files',checked,flush=True)

if __name__=='__main__':prepare()
