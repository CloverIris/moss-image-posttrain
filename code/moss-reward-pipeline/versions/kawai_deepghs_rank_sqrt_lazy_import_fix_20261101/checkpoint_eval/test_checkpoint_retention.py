"""Only TEST-OWNED temporary hardlinks/adapter copies are ever unlinked."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import checkpoint_eval_service as s


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='moss-retention-test-');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.source=self.base/'training_checkpoints';self.source.mkdir()
        self.source_config=self.base/'source.json';s.atomic_json(self.source_config,{})
        self.prompts=self.base/'fixed.jsonl';self.prompts.write_text('{}\n')
        self.config=dict(source_checkpoint_dir=str(self.source),service_root=str(self.base/'service'),
            source_config=str(self.source_config),fixed_prompts=str(self.prompts),device=4,
            fixed_seeds=[100,101,102,103],images_per_prompt=4,expected_prompt_count=4,
            export_command=['unused'],infer_command=['unused'],summary_command=['unused'],
            logging_log_dir=str(self.base/'tb'),max_update=200,once=True,require_gpu_idle=False,
            retention=dict(release_done_checkpoint_payload=True,done_adapters=dict(enabled=True,
                keep_best=3,keep_latest=2,milestone_interval_updates=40,final_update=200)))
        self.quiet=contextlib.redirect_stdout(io.StringIO());self.quiet.__enter__();self.addCleanup(self.quiet.__exit__,None,None,None)

    def daemon(self):return s.CheckpointEvalService(self.config)

    def done(self,d,update,score=1.):
        cp=self.source/('step_%08d'%update);cp.mkdir()
        (cp/'arrays.npz').write_bytes(('fake-LoRA-and-optimizer-%d'%update).encode())
        s.atomic_json(cp/'manifest.json',dict(payload_sha256=s.digest_file(cp/'arrays.npz'),metadata=dict(update=update)))
        (cp/'COMMITTED').write_text(s.digest_file(cp/'manifest.json'))
        d.discover();task=next(t for t in d.state['tasks'].values() if t['update']==update)
        attempt=d.root/'evaluations'/task['checkpoint_id']/'attempt_0001';attempt.mkdir(parents=True)
        resolved=d._resolved(task,attempt);s.atomic_json(attempt/'task.json',resolved)
        task.update(last_attempt=str(attempt),attempts=1)
        adapter=Path(resolved['adapter']);adapter.write_bytes(('fake-export-%d'%update).encode())
        contract=dict(checkpoint=task['checkpoint'],checkpoint_manifest_sha256=task['checkpoint_manifest_sha256'],
            checkpoint_payload_sha256=s.digest_file(cp/'arrays.npz'),source_config=resolved['source_config'],
            source_config_sha256=resolved['source_config_sha256'],adapter=str(adapter))
        s.atomic_json(adapter.with_name('adapter.safetensors.export.json'),dict(contract=contract,
            contract_sha256=s.digest_bytes(s.canonical(contract)),adapter_sha256=s.digest_file(adapter)))
        s.atomic_json(attempt/'export.complete.json',dict(output=str(adapter),sha256=s.digest_file(adapter)))
        output=Path(resolved['output_dir']);output.mkdir()
        items=[]
        for p in range(4):
            for r in range(4):
                image=output/f'{p}_{r}.png';image.write_bytes(f'original-image-{update}-{p}-{r}'.encode())
                items.append(dict(id=f'{p}:{r}',image=str(image),prompt=f'fixed {p}',prompt_index=p,
                    prompt_id=str(p),rollout_id=r,seed=100+p*4+r))
        s.atomic_json(resolved['request'],dict(version=1,items=items));d._publish_ready(task,resolved)
        ready=s.read_json(task['ready']);ack=Path(ready['ack'])
        result=attempt/'scoring'/'result.json';s.atomic_json(result,dict(status='complete',scores=[score]*16,raw='preserved'))
        s.atomic_json(ack,dict(status='complete',checkpoint_id=task['checkpoint_id'],request_sha256=ready['request_sha256']))
        summary=attempt/'summary.json';s.atomic_json(summary,dict(checkpoint_id=task['checkpoint_id'],reward=dict(mean=score)))
        completion=output/'completion.json';s.atomic_json(completion,dict(checkpoint_id=task['checkpoint_id'],
            ack=str(ack),ack_sha256=s.digest_file(ack),summary=str(summary),summary_sha256=s.digest_file(summary),
            evaluation_result=str(result),evaluation_result_sha256=s.digest_file(result)))
        report=attempt/'validation_report.json';s.atomic_json(report,dict(checkpoint_id=task['checkpoint_id'],source_step=update))
        event=Path(self.config['logging_log_dir'])/f'event_{update}';event.parent.mkdir(exist_ok=True);event.write_bytes(b'event-provenance')
        s.atomic_json(attempt/'validation_report.complete.json',dict(status='complete',checkpoint_id=task['checkpoint_id'],
            source_step=update,report=str(report),report_sha256=s.digest_file(report),report_digest=s.digest_bytes(s.canonical(s.read_json(report))),
            event=str(event),event_sha256=s.digest_file(event)))
        task.update(status='done',completion=str(completion));d.save()
        return task

    def test_default_off_keeps_all_exported_and_checkpoint_bytes(self):
        self.config['retention']={};d=self.daemon();task=self.done(d,2);d.apply_retention()
        self.assertTrue((Path(task['checkpoint'])/'arrays.npz').exists())
        self.assertTrue((Path(task['last_attempt'])/'adapter.safetensors').exists())

    def test_done_payload_unlinks_only_own_pin_and_discover_does_not_repin(self):
        d=self.daemon();task=self.done(d,2)
        keep=[Path(task['ready']),Path(task['completion']),self.source/'step_00000002'/'arrays.npz']
        keep += [Path(x['path']) for x in s.read_json(task['ready'])['images']]
        hashes={str(path):s.digest_file(path) for path in keep}
        payload=Path(task['checkpoint'])/'arrays.npz';d.apply_retention()
        self.assertFalse(payload.exists());self.assertTrue(payload.with_name('arrays.npz.released.json').exists())
        self.assertEqual(d.discover(),0);self.assertFalse(payload.exists());self.assertEqual(len(d.state['tasks']),1)
        for path,checksum in hashes.items():self.assertEqual(s.digest_file(path),checksum)
        again=self.daemon();again.discover();self.assertFalse(payload.exists())
        self.assertEqual(next(iter(again.state['tasks'].values()))['status'],'done')

    def test_best3_latest2_and_40_update_milestones_are_kept(self):
        d=self.daemon();tasks={u:self.done(d,u,score) for u,score in [(0,0),(2,10),(4,9),(6,8),(8,7),(40,6),(200,5)]}
        d.apply_retention()
        for update,task in tasks.items():
            self.assertFalse((Path(task['checkpoint'])/'arrays.npz').exists())
            self.assertEqual((Path(task['last_attempt'])/'adapter.safetensors').exists(),update!=8)
            self.assertTrue((self.source/('step_%08d'%update)/'arrays.npz').exists())
        proof=Path(tasks[8]['last_attempt'])/'adapter.safetensors.released.json';self.assertTrue(proof.exists())
        again=self.daemon();again.apply_retention();self.assertEqual(len(again.state['tasks']),7)

    def test_ready_or_failed_tasks_are_never_retained_away(self):
        d=self.daemon();task=self.done(d,2);task['status']='ready_for_scoring';d.save();d.apply_retention()
        self.assertTrue((Path(task['checkpoint'])/'arrays.npz').exists())
        task['status']='failed';d.apply_retention();self.assertTrue((Path(task['last_attempt'])/'adapter.safetensors').exists())

    def test_missing_report_or_corrupt_adapter_preserves_payload(self):
        for change in ('report','adapter'):
            with self.subTest(change=change):
                # Separate service roots because immutable identities cannot change.
                self.config['service_root']=str(self.base/('service_'+change))
                d=self.daemon();task=self.done(d,2 if change=='report' else 4)
                if change=='report':(Path(task['last_attempt'])/'validation_report.complete.json').unlink()
                else:(Path(task['last_attempt'])/'adapter.safetensors').write_bytes(b'corrupt')
                d.apply_retention();self.assertIn('retention_error',task)
                self.assertTrue((Path(task['checkpoint'])/'arrays.npz').exists())

    def test_missing_payload_without_proof_rejected_on_recovery(self):
        d=self.daemon();task=self.done(d,2);(Path(task['checkpoint'])/'arrays.npz').unlink()
        with self.assertRaisesRegex(ValueError,'without a valid retention proof'):self.daemon()

    def test_crash_after_unlink_before_proof_finishes_without_regeneration(self):
        d=self.daemon();task=self.done(d,2);payload=Path(task['checkpoint'])/'arrays.npz'
        real=s.immutable_json
        def interrupted(path,value):
            if str(path).endswith('.released.json'):raise RuntimeError('simulated power loss')
            return real(path,value)
        with mock.patch.object(s,'immutable_json',side_effect=interrupted):d.apply_retention()
        self.assertFalse(payload.exists());self.assertEqual(task['retention']['checkpoint_payload']['status'],'prepared')
        again=self.daemon();recovered=next(iter(again.state['tasks'].values()))
        self.assertEqual(recovered['status'],'done');self.assertEqual(recovered['attempts'],1)
        self.assertEqual(recovered['retention']['checkpoint_payload']['status'],'released')

    def test_release_proof_tamper_rejected(self):
        d=self.daemon();task=self.done(d,2);d.apply_retention()
        proof=Path(task['checkpoint'])/'arrays.npz.released.json';value=s.read_json(proof);value['intent_sha256']='0'*64;s.atomic_json(proof,value)
        with self.assertRaisesRegex(ValueError,'release proof'):self.daemon()

    def test_foreign_source_directory_is_never_unlinked(self):
        d=self.daemon();task=self.done(d,2);task['checkpoint']=str(self.source/'step_00000002')
        d.apply_retention();self.assertIn('owned pinned directory',task['retention_error']['error'])
        self.assertTrue((self.source/'step_00000002'/'arrays.npz').exists())

    def test_foreign_old_run_adapter_directory_is_never_unlinked(self):
        d=self.daemon();task=self.done(d,2)
        old=self.base/'old_run'/'evaluations'/task['checkpoint_id']/'attempt_0001';old.mkdir(parents=True)
        adapter=old/'adapter.safetensors';adapter.write_bytes(b'old-run-must-not-change')
        task['last_attempt']=str(old);d.apply_retention()
        self.assertIn('owned evaluation attempt',task['retention_error']['error'])
        self.assertEqual(adapter.read_bytes(),b'old-run-must-not-change')


if __name__=='__main__':unittest.main()
