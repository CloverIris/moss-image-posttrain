"""CPU-only contract, persistence and recovery tests; no model weights needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import checkpoint_eval_service as service


WORKER = r'''
import hashlib, json, os, sys, time
from pathlib import Path
stage, config_path = sys.argv[1:]
c=json.loads(Path(config_path).read_text())
trace=Path(c['trace'])
with trace.open('a') as f:
    f.write(json.dumps({'stage':stage,'checkpoint_id':c['checkpoint_id'],
        'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'), 'platform':os.environ.get('JAX_PLATFORMS')})+'\n')
if stage=='export':
    Path(c['adapter']).write_bytes(b'fake-adapter')
elif stage=='summary':
    marker=Path(c['service_root'])/'report.ok'
    count=sum(1 for line in trace.read_text().splitlines() if json.loads(line)['stage']=='summary')
    if c.get('report_fail_once') and count==1: sys.exit(6)
    marker.write_text(c['checkpoint_id'])
else:
    out=Path(c['output_dir']); out.mkdir(parents=True)
    Path(out/'partial.png').write_bytes(b'partial-image')
    count=sum(1 for line in trace.read_text().splitlines() if json.loads(line)['stage']=='infer')
    if c.get('fail_always') or (c.get('fail_once') and count==1): sys.exit(5)
    if c.get('hang'): time.sleep(10)
    if c.get('create_checkpoint_during_infer') and c['update']==2:
        cp=Path(c['source_checkpoint_dir'])/'step_00000004'; cp.mkdir(exist_ok=True)
        data=json.dumps({'update':4}).encode(); (cp/'manifest.json').write_bytes(data)
        (cp/'COMMITTED').write_text(hashlib.sha256(data).hexdigest())
        (cp/'array.bin').write_bytes(b'weights4'); time.sleep(.25)
    rows=[]
    for p in range(4):
        for r in range(4):
            image=out/('p%d_r%d.png'%(p,r)); image.write_bytes(b'fake-png')
            rows.append(dict(id='p%d:r%d'%(p,r),image=str(image),prompt='fixed prompt %d'%p,
                prompt_id='prompt%d'%p,prompt_index=p,rollout_id=r,
                seed=100+p*4+r+(1 if c.get('vary_plan') and c['update']==4 else 0),
                negative_prompt='blur'))
    Path(c['request']).write_text(json.dumps({'version':1,'items':rows}))
'''


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='moss-checkpoint-eval-test-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base/'source'; self.source.mkdir()
        self.worker = self.base/'worker.py'; self.worker.write_text(WORKER)
        self.prompts = self.base/'prompts.jsonl'
        self.prompts.write_text('\n'.join(json.dumps(dict(id=i,prompt='fake%d'%i)) for i in range(4)))
        self.source_config = self.base/'source_config.json'; self.source_config.write_text('{}')
        self.config = dict(source_checkpoint_dir=str(self.source), service_root=str(self.base/'service'),
            source_config=str(self.source_config), fixed_prompts=str(self.prompts), fixed_seeds=[100,101,102,103],
            expected_prompt_count=4, images_per_prompt=4, width=768, height=768, device=4,
            poll_interval_seconds=.05, require_gpu_idle=False, max_retries=1, once=True,
            trace=str(self.base/'trace.jsonl'),
            export_command=[sys.executable,str(self.worker),'export','{config}'],
            infer_command=[sys.executable,str(self.worker),'infer','{config}'])
        self.quiet = contextlib.redirect_stdout(io.StringIO()); self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def checkpoint(self, update=2, commit=True):
        cp = self.source/('step_%08d'%update); cp.mkdir()
        data = json.dumps(dict(update=update)).encode()
        (cp/'manifest.json').write_bytes(data); (cp/'array.bin').write_bytes(b'weights')
        if commit: (cp/'COMMITTED').write_text(service.digest_bytes(data))
        return cp

    def new_service(self):
        return service.CheckpointEvalService(self.config)

    def tasks(self, daemon):
        return sorted(daemon.state['tasks'].values(), key=lambda x:x['update'])

    def trace(self):
        path = Path(self.config['trace'])
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def ack(self, task, **patch):
        ready=service.read_json(task['ready'])
        result = dict(status='complete', job_id=ready['job_id'], checkpoint_id=ready['checkpoint_id'],
            checkpoint_manifest_sha256=ready['checkpoint_manifest_sha256'],
            request_sha256=ready['request_sha256'], scores=[i/16 for i in range(16)])
        result.update(patch); service.atomic_json(ready['ack'], result)
        return ready, result

    def test_discovery_requires_commit_and_pins_weights(self):
        incomplete=self.checkpoint(2,False); bad=self.checkpoint(4)
        (bad/'COMMITTED').write_text('0'*64); good=self.checkpoint(6)
        daemon=self.new_service(); self.assertEqual(daemon.discover(),1)
        task=self.tasks(daemon)[0]; pinned=Path(task['checkpoint'])
        self.assertEqual(task['update'],6)
        self.assertEqual(os.stat(good/'array.bin').st_ino,os.stat(pinned/'array.bin').st_ino)
        # Simulate keep2 rotation of this TEST-OWNED checkpoint only.
        for child in good.iterdir(): child.unlink()
        good.rmdir()
        self.assertEqual((pinned/'array.bin').read_bytes(),b'weights')
        self.assertTrue(incomplete.is_dir()); self.assertTrue(daemon.state['discovery_errors'])

    def test_generation_ack_restart_exactly_once(self):
        self.checkpoint(); self.config['score_command']=[sys.executable,'must-never-run.py']
        daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        self.assertEqual(task['status'],'ready_for_scoring')
        ready, _=self.ack(task); self.assertTrue(Path(ready['ack']).parent==daemon.outbox)
        self.assertEqual(daemon.consume_acks(),1); self.assertEqual(task['status'],'done')
        summary=service.read_json(Path(task['last_attempt'])/'summary.json')
        self.assertEqual(summary['image_count'],16); self.assertAlmostEqual(summary['reward']['mean'],.46875)
        self.assertEqual([r['stage'] for r in self.trace()],['export','infer'])
        self.assertEqual(self.trace()[0]['cuda'],''); self.assertEqual(self.trace()[0]['platform'],'cpu')
        self.assertEqual(self.trace()[1]['cuda'],'4')
        again=self.new_service(); again.run(); self.assertEqual(len(self.trace()),2)

    def test_all_checkpoints_generate_without_waiting_for_scores(self):
        self.checkpoint(2); self.checkpoint(4)
        daemon=self.new_service(); daemon.run()
        self.assertEqual([x['status'] for x in self.tasks(daemon)],['ready_for_scoring']*2)
        self.assertEqual(len(list(daemon.outbox.glob('*.ready.json'))),2)
        self.assertEqual(len(self.trace()),4)

    def test_discover_during_inference_pins_next_checkpoint(self):
        self.checkpoint(); self.config['create_checkpoint_during_infer']=True
        daemon=self.new_service(); daemon.run()
        self.assertEqual([x['update'] for x in self.tasks(daemon)],[2,4])
        self.assertTrue(all(x['status']=='ready_for_scoring' for x in self.tasks(daemon)))

    def test_retry_keeps_failed_images_and_never_overwrites(self):
        self.checkpoint(); self.config['fail_once']=True
        daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        self.assertEqual((task['status'],task['attempts'],task['failures']),('ready_for_scoring',2,1))
        folder=Path(task['last_attempt']).parent
        self.assertEqual((folder/'attempt_0001'/'generated'/'partial.png').read_bytes(),b'partial-image')
        self.assertTrue((folder/'attempt_0001'/'status.json').is_file())
        self.assertTrue((folder/'attempt_0002'/'generated'/'eval_request.json').is_file())

    def test_failed_retry_is_bounded(self):
        self.checkpoint(); self.config['fail_always']=True
        daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        self.assertEqual((task['status'],task['attempts'],task['failures']),('failed',2,2))
        daemon.run(); self.assertEqual(len(self.trace()),4)

    def test_bad_ack_does_not_regenerate_or_mark_done(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        self.ack(task,request_sha256='wrong'); self.assertEqual(daemon.consume_acks(),0)
        self.assertIn('identity',task['ack_error']['error'])
        self.ack(task,scores=[0]); daemon.consume_acks()
        self.assertIn('finite',task['ack_error']['error']); daemon.run()
        self.assertEqual(task['status'],'ready_for_scoring'); self.assertEqual(len(self.trace()),2)

    def test_external_result_ack_and_raw_score_integrity(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        ready=service.read_json(task['ready']); result=Path(ready['result_dir'])/'result.json'
        service.atomic_json(result,dict(scores=[.5]*16,summary=dict(aesthetic=.8)))
        self.ack(task,evaluation_result=str(result)); daemon.consume_acks()
        self.assertEqual(task['status'],'done'); service.atomic_json(result,dict(scores=[.8]*16))
        with self.assertRaisesRegex(ValueError,'raw scores'): self.new_service()

    def test_fixed_plan_mismatch_fails_new_generation(self):
        self.checkpoint(2); self.checkpoint(4); self.config.update(vary_plan=True,max_retries=0)
        daemon=self.new_service(); daemon.run(); tasks=self.tasks(daemon)
        self.assertEqual(tasks[0]['status'],'ready_for_scoring'); self.assertEqual(tasks[1]['status'],'failed')
        self.assertIn('Immutable artifact',tasks[1]['last_error'])

    def test_inputs_are_frozen_across_restart(self):
        daemon=self.new_service(); self.prompts.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Frozen'): self.new_service()
        self.assertEqual(daemon.prompt_snapshot.read_text().splitlines()[0],json.dumps(dict(id=0,prompt='fake0')))

    def test_ready_image_tamper_rejected_on_restart(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        ready=service.read_json(task['ready']); Path(ready['images'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Ready image'): self.new_service()

    def test_done_artifacts_verified_on_restart(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        ready,_=self.ack(task); daemon.consume_acks()
        Path(ready['ack']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'Completed evaluation artifact'): self.new_service()

    def test_stale_running_resumes_published_ready(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        task.update(status='running',active_process=dict(pid=99999999,identity=None)); daemon.save()
        again=self.new_service(); again.run()
        self.assertEqual(self.tasks(again)[0]['status'],'ready_for_scoring'); self.assertEqual(len(self.trace()),2)

    def test_stale_running_with_complete_infer_does_not_regenerate(self):
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        ready=Path(task['ready']); ready.unlink()
        (daemon.outbox/(task['checkpoint_id']+'.ready.json')).unlink()
        task.update(status='running'); daemon.save()
        again=self.new_service(); again.run()
        self.assertEqual(self.tasks(again)[0]['status'],'ready_for_scoring'); self.assertEqual(len(self.trace()),2)

    def test_gpu_busy_leaves_pending_without_exporting(self):
        self.checkpoint(); daemon=self.new_service()
        with mock.patch.object(daemon,'gpu_idle',return_value=False): daemon.run()
        task=self.tasks(daemon)[0]; self.assertEqual((task['status'],task['attempts']),('pending',0))
        self.assertFalse(self.trace())

    def test_timeout_stops_owned_child_and_keeps_failed_attempt(self):
        self.checkpoint(); self.config.update(hang=True,max_retries=0,stage_timeout_seconds=dict(infer=.15),kill_grace_seconds=.2)
        daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        self.assertEqual(task['status'],'failed'); self.assertIn('timed out',task['last_error'])
        self.assertIsNone(task['active_process']); self.assertIsNone(daemon.child)
        self.assertTrue((Path(task['last_attempt'])/'generated'/'partial.png').exists())

    def test_reporter_retries_only_reporting_not_generation(self):
        self.checkpoint(); self.config.update(report_fail_once=True,
            summary_command=[sys.executable,str(self.worker),'summary','{config}'])
        daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]; self.ack(task)
        self.assertEqual(daemon.consume_acks(),0); self.assertEqual(task['status'],'ready_for_scoring')
        self.assertEqual(daemon.consume_acks(),1); self.assertEqual(task['status'],'done')
        self.assertEqual([r['stage'] for r in self.trace()],['export','infer','summary','summary'])
        self.assertEqual(self.trace()[-1]['cuda'],''); self.assertTrue((daemon.root/'report.ok').exists())
        again=self.new_service(); again.run(); self.assertEqual(len(self.trace()),4)

    def test_exclusive_file_lock(self):
        path=self.base/'exclusive.lock'
        with service.DirectoryLock(path):
            with self.assertRaises(RuntimeError):
                with service.DirectoryLock(path): pass

    def test_cli_once_only_config(self):
        # A CPU fake service must not contend with a live GPU4 evaluation daemon.
        self.config['gpu_lock_path'] = str(self.base/'cpu-test-gpu4.lock')
        self.checkpoint(); config=self.base/'service.json'; service.atomic_json(config,self.config)
        result=subprocess.run([sys.executable,str(Path(service.__file__)),'--config',str(config)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        state=service.read_json(Path(self.config['service_root'])/'state.json')
        self.assertEqual(next(iter(state['tasks'].values()))['status'],'ready_for_scoring')

    def test_real_wrapper_interoperability_192_plus_16(self):
        from test_run_reward_with_validation import WrapperTests
        import run_reward_with_validation as wrapper
        fixture=WrapperTests(methodName='runTest'); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.config['service_root']=str(fixture.service)
        self.checkpoint(); daemon=self.new_service(); daemon.run(); task=self.tasks(daemon)[0]
        result=wrapper.run(fixture.cfg,fixture.request,fixture.step/'reward.json',
            pipeline=fixture.pipeline,splitter=wrapper.split_groups)
        self.assertEqual(len(result['scores']),192)
        self.assertEqual(len(fixture.pipeline.calls[0]['items']),208)
        self.assertEqual(daemon.consume_acks(),1); self.assertEqual(task['status'],'done')
        summary=service.read_json(Path(task['last_attempt'])/'summary.json')
        self.assertEqual(summary['image_count'],16)
        self.assertAlmostEqual(summary['reward']['mean'],1.995)
        again=self.new_service(); again.run(); self.assertEqual(len(self.trace()),2)

    def test_no_accelerator_framework_imported(self):
        self.assertFalse(set(('jax','torch','tensorflow')) & set(sys.modules))


if __name__=='__main__': unittest.main()
