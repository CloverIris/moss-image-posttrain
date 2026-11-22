import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sdg_parallel import balanced_shards, merge_shard_responses, run_jobs, run_sdg_parallel


def rows(prompts=24, count=16):
    return [dict(id=f'p{p}_r{r}',prompt_id=p,prompt='unchanged',image='test.png')
            for p in range(prompts) for r in range(count)]


def response(items):
    details = [dict(id=row['id'],reward=-i/1000,mode='artifact_only',alpha_misalignment=0.)
               for i,row in enumerate(items)]
    return dict(scores=[x['reward'] for x in details],details=details)


class SDGParallelTests(unittest.TestCase):
    def test_p24_k16_groups_are_not_split_or_lost(self):
        items = rows()
        shards = balanced_shards(items)
        self.assertEqual(list(map(len,shards)),[96]*4)
        self.assertEqual({r['id'] for s in shards for r in s},{r['id'] for r in items})
        assignment = {}
        for rank,shard in enumerate(shards):
            for row in shard:
                self.assertEqual(assignment.setdefault(row['prompt_id'],rank),rank)
        self.assertEqual([len(set(r['prompt_id'] for r in s)) for s in shards],[6]*4)

    def test_merge_reorders_using_ids_and_rejects_invalid(self):
        items = rows(4,2)
        shards = balanced_shards(items)
        responses = [response(list(reversed(s))) for s in shards]
        merged = merge_shard_responses(items,shards,responses)
        self.assertEqual([d['id'] for d in merged['details']],[x['id'] for x in items])
        responses[0]['scores'][0] = None
        with self.assertRaises(ValueError):
            merge_shard_responses(items,shards,responses)
        responses = [response(s) for s in shards]
        responses[0]['details'][0]['id'] = 'foreign'
        with self.assertRaises(ValueError):
            merge_shard_responses(items,shards,responses)

    def test_gpu4_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_sdg_parallel(rows(4,1),dict(mode='artifact_only',replica_devices=[[0],[1],[2],[4]]),
                                 sys.executable,tmp,tmp)

    def test_completed_shards_are_reused_and_verified(self):
        items=rows(4,2);shards=balanced_shards(items)
        with tempfile.TemporaryDirectory() as tmp:
            for rank,shard in enumerate(shards):
                p=Path(tmp)/('dp%02d'%rank);p.mkdir()
                (p/'response.json').write_text(json.dumps(response(shard)))
            merged=run_sdg_parallel(items,dict(mode='artifact_only'),sys.executable,tmp,tmp)
            self.assertEqual(merged['parallel']['reused_shards'],[0,1,2,3])
            self.assertEqual(merged['parallel']['launched_jobs'],[])
            (Path(tmp)/'dp00/response.json').write_text(json.dumps(dict(scores=[],details=[])))
            with self.assertRaises(ValueError):
                run_sdg_parallel(items,dict(mode='artifact_only'),sys.executable,tmp,tmp)

    def test_child_failure_does_not_become_zero_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=dict(argv=[sys.executable,'-c','raise SystemExit(7)'],
                     log=str(Path(tmp)/'worker.log'),output=str(Path(tmp)/'response.json'))
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                run_jobs([job],timeout=10,poll_interval=.01)
            self.assertFalse(Path(job['output']).exists())

    def test_cpu_fake_jobs_run_and_module_imports_no_ml(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=[]
            for i in range(4):
                out=Path(tmp)/str(i)/'response.json'
                code='from pathlib import Path; Path('+repr(str(out))+').write_text("{}")'
                jobs.append(dict(argv=[sys.executable,'-c',code],log=str(out.parent/'worker.log'),output=str(out)))
            self.assertEqual(len(run_jobs(jobs,timeout=10,poll_interval=.01)),4)
        code="import sdg_parallel,sys; assert all(x not in sys.modules for x in ('torch','jax','vllm'))"
        proc=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)

    @unittest.skipUnless(os.name == 'posix', 'Linux process subtree cleanup')
    def test_failure_stops_owned_child_but_leaves_other_process(self):
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);child_file=root/'child.pid'
                spawn=('import subprocess,sys,time; from pathlib import Path; '
                       'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                       'Path('+repr(str(child_file))+').write_text(str(p.pid)); time.sleep(30)')
                fail=('import time; from pathlib import Path; p=Path('+repr(str(child_file))+'); '
                      '\nwhile not p.exists(): time.sleep(.01)\ntime.sleep(.2)\nraise SystemExit(7)')
                jobs=[dict(argv=[sys.executable,'-c',code],log=str(root/f'{i}.log'),
                           output=str(root/f'{i}.json')) for i,code in enumerate((spawn,fail))]
                with self.assertRaisesRegex(RuntimeError,'exited 7'):
                    run_jobs(jobs,timeout=10,poll_interval=.02)
                self.assertIsNone(unrelated.poll())
                child_pid=int(child_file.read_text())
                stat=Path(f'/proc/{child_pid}/stat')
                if stat.exists():
                    state=stat.read_text().rsplit(') ',1)[1].split()[0]
                    self.assertEqual(state,'Z','Owned child must have exited, not remain running')
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)


if __name__=='__main__':
    unittest.main()
