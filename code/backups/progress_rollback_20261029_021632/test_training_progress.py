import unittest
from moss_art.training_progress import window_batch_counts,microbatch_progress,window_finished


class ProgressTests(unittest.TestCase):
    def test_real_bucket_imbalance_counts_without_reordering(self):
        buckets=[{(288,288):list(range(n)),(512,512):list(range(f))}
                 for n,f in ((138,54),(138,54),(118,74),(118,74))]
        before=[{k:list(v) for k,v in b.items()} for b in buckets]
        plan=window_batch_counts(buckets,1)
        self.assertEqual(plan,{(288,288):138,(512,512):74})
        self.assertEqual(sum(plan.values()),212)
        self.assertEqual(buckets,before)
        self.assertEqual(sum(window_batch_counts([{(288,288):[0]*128,(512,512):[0]*64}]*4,1).values()),192)

    def test_microbatch_padding_and_empty_plan(self):
        self.assertEqual(window_batch_counts([{(288,288):[0]*5},{(288,288):[0]*2}],2),{(288,288):3})
        self.assertEqual(window_batch_counts([{},{}],1),{})

    def test_progress_is_explicitly_provisional_and_has_no_update_event(self):
        event=microbatch_progress(0,(288,288),16,212,16,138,80.,20.,64)
        self.assertEqual(event['event'],'training_microbatch_progress')
        self.assertFalse(event['committed']);self.assertTrue(event['provisional'])
        self.assertEqual(event['completed_microbatches'],16)
        self.assertEqual(event['total_microbatches'],212)
        self.assertEqual(event['bucket'],[288,288])

    def test_window_log_does_not_mutate_official_pending_event(self):
        original=dict(event='update',update=1,window=0,loss=.2,base_reference_mse=.1)
        result=window_finished(original)
        self.assertEqual(original['event'],'update')
        self.assertEqual(result['event'],'window_finished')
        self.assertFalse(result['committed']);self.assertTrue(result['provisional'])
        self.assertEqual(result['loss'],.2)

    def test_actual_mesh_generator_preserves_bucket_order_padding_and_locations(self):
        import numpy as np
        from types import SimpleNamespace
        from moss_art.mesh_runtime import MeshPool
        pool=MeshPool.__new__(MeshPool)
        pool.config={'art':{'feedback':'scalar'},'training':{'microbatch_size':1}}
        pool.workers=list(range(4))
        pool.worker=SimpleNamespace(model=SimpleNamespace(text_dim=2))
        pool.shards=[];pool.prepared=[]
        lengths=((3,1),(1,3),(2,2),(3,1))
        for short,full in lengths:
            cache=dict(endpoints=np.zeros((2,1,1),np.float32),
                groups=np.asarray([0,1]),contexts=[(np.zeros((1,n,2)),np.zeros((1,n,2))) for n in (3,5)],
                condition_lengths=[(np.asarray([3]),np.asarray([3])),(np.asarray([5]),np.asarray([5]))],
                selected=[0],sigmas=np.asarray([.5]),ode_velocity_coefficients=np.asarray([-.1]),
                states=np.zeros((2,1,1,1)),old=np.zeros((2,1,1,1)))
            windows=[np.asarray([[0,0]]*short+[[1,0]]*full)]
            pool.shards.append(cache)
            pool.prepared.append(dict(windows=windows,stats=dict(group_directions=np.ones((2,1,1)),group_guidance_magnitudes=np.ones(2))))
        actual=list(pool._window_batches(0,include_inactive=True))
        self.assertEqual(len(actual),6)
        self.assertEqual(pool._window_batch_progress,dict(total=6,buckets={(3,3):3,(5,5):3}))
        self.assertEqual([host['positive'].shape[1] for host,_ in actual],[3,3,3,5,5,5])
        self.assertEqual(sum(int(host['valid'].sum()) for host,_ in actual),16)
        self.assertEqual(actual[0][1],[(0,0,0),(1,0,0),(2,0,0),(3,0,0)])
        self.assertEqual(actual[2][1],[(0,0,0),None,None,(3,0,0)])


if __name__=='__main__':unittest.main()
