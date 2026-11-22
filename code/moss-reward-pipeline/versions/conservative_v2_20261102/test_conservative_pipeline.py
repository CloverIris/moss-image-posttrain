"""Synthetic fixtures only; no production thresholds or real model execution."""
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_reward_pipeline as pipeline
import test_kawai_disabled as fixture_support
archive_lock_test_support=fixture_support.archive_lock_test_support

def pin(path,value):
    pipeline.save(path,value)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def fixture(root):
    cfg,request,_=fixture_support.OptionalPipelineTests().fixture(root)
    cfg.update(kawai_enabled=True,enabled_models=['kawai','deepghs'],fusion_weights={'kawai':.5,'deepghs':.5},fusion_mode='conservative_v2',conservative_group_size=2,conservative_eval_group_size=4)
    pipeline.save(root/'batch.json',dict(stage_order=['deepghs']))
    image=request['items'][0]['image'];text=r'Prompt: literal\nframe';negative='explicit negative'
    request['items']=[dict(id=f't{i}',sample_id='p',prompt_index=0,prompt=text,negative_prompt=negative,index=i,image=image) for i in range(2)]
    catalog=pin(root/'catalog.json',dict(tags=['native flat'],tags_split=[['style',1]]))
    evidence=pin(root/'evidence.json',dict(calibration_id='SYNTHETIC-TEST-ONLY',human_labeled_images=2))
    rule=dict(category='style',tag='native flat',desired_present=True,absent_at=.2,present_at=.8,reliability=.95)
    targets=pin(root/'targets.json',dict(schema=1,calibration_id='SYNTHETIC-TEST-ONLY',targets=[dict(prompt_id='p',prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),negative_prompt_sha256=hashlib.sha256(negative.encode()).hexdigest(),aesthetic_coverage=1.,rules=[rule])]))
    identity=dict(weights_sha256='0'*64,implementation_sha256='1'*64,catalog_sha256=catalog['sha256'],preprocessor_sha256='2'*64,dtype='float32',attention_backend='native')
    policy=dict(schema=1,approved=True,calibration_id='SYNTHETIC-TEST-ONLY',artifacts=dict(catalog=catalog,targets=targets,evidence=evidence),heads=dict(kawai=dict(deadband=.01,temperature=.2),deepghs=dict(deadband=.01,temperature=.1)),aesthetic_coverage=1.,min_guard_reliability=.8,aesthetic_cap=.25,learner=dict(guidance_gain=.2,reference_mse_beta=.1),tagger_identity=identity)
    ref=pin(root/'policy.json',policy)
    raw=dict(schema=1,status='complete',unfiltered=True,catalog_sha256=catalog['sha256'],tagger_identity=identity,items=[dict(id=r['id'],image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest(),status='ok',probabilities={'style':{'native flat':.9}}) for r in request['items']])
    response=pin(root/'tagger.json',raw)
    cfg['conservative']=dict(policy=ref,offline_replay=True,tagger=dict(mode='cached',response=response))
    request['context']=dict(reward_contract=dict(name='conservative_v2',calibration_id=policy['calibration_id'],policy_sha256=ref['sha256'],reference_mode='base',reference_mse_beta=.1,covariance_options=dict(mode='scalar',scale_mode='reward_mad',guidance_gain=.2,zero_signal_mode='exact',zero_signal_atol=0.0)))
    def scorer(argv,log,timeout):
        rows=json.loads(Path(argv[argv.index('--input')+1]).read_text())['items']
        pipeline.save(argv[argv.index('--output')+1],dict(status='complete',items=[dict(id=r['id'],deepghs=dict(status='ok',percentile=.8 if r['id'].endswith('0') else .3)) for r in rows][::-1]))
    def kawai(items,cfg,archive):return dict(items=[dict(id=r['id'],kawai=.7 if r['id'].endswith('0') else .2) for r in items][::-1],metadata={})
    return cfg,request,scorer,kawai

class ConservativeIntegrationTests(unittest.TestCase):
    def test_durable_pipeline_bypasses_both_rank_stages_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,scorer,kawai=fixture(root);before=copy.deepcopy(request)
            with archive_lock_test_support(),patch.object(pipeline,'run_program',side_effect=scorer),patch.object(pipeline,'run_kawai_parallel',side_effect=kawai),patch.object(pipeline,'apply_rank_fusion',side_effect=AssertionError('Rank executed')):
                result=pipeline.run(cfg,request,root/'result.json');again=pipeline.run(cfg,request,root/'again.json')
            self.assertEqual(request,before);self.assertEqual(result,again)
            self.assertGreater(result['scores'][0],0);self.assertLess(result['scores'][1],0);self.assertAlmostEqual(sum(result['scores']),0)
            archive=Path(result['archive']);self.assertTrue((archive/'pixai_probabilities.json').is_file());self.assertTrue((archive/'conservative.targets.json').is_file())
            diag=json.loads((archive/'conservative.groups.json').read_text())[0]
            self.assertEqual(diag['aesthetic_pairs']+diag['constraint_pairs']+diag['abstained_pairs'],1)
            self.assertEqual([r['id'] for r in result['details']],['t0','t1'])
            self.assertTrue(all(r['reward_semantics']=='conservative_v2_training' for r in result['details']))
    def test_fixed_validation_absolute_and_repeated_dataset_ids_do_not_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,scorer,kawai=fixture(root);helpers=pipeline._conservative_helpers();bundle=helpers.load_policy(cfg)
            items=pipeline.canonical_items(request);details=[dict(id=r['id'],raw_scores=dict(kawai=.7-i*.4,deepghs=.8-i*.4),reward=.4+i*.01,aesthetic_reward=.4,index=i) for i,r in enumerate(items)]
            more=[dict(r,id='repeat'+r['id'],prompt_index=1,index=i+2) for i,r in enumerate(items)];items+=more;details+=[dict(d,id=r['id'],index=i+2) for i,(d,r) in enumerate(zip(details[:2],more))]
            for i in range(4):items.append(dict(items[0],id='eval'+str(i),prompt_index=0,index=i+4,checkpoint_eval={'job_id':'ckpt'}));details.append(dict(details[0],id='eval'+str(i),index=i+4,reward=.3+i*.01))
            response=json.loads(Path(cfg['conservative']['tagger']['response']['path']).read_text());response['items'] += [dict(response['items'][i],id=r['id']) for i,r in enumerate(more)]
            fused,groups=helpers.apply(details,items,cfg,bundle,response)
            self.assertEqual(len(groups),2);self.assertEqual(len(fused),8)
            self.assertEqual([r['reward'] for r in fused[4:]],[d['reward'] for d in details[4:]])
    def test_unapproved_and_wrong_learner_reject_before_model_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,scorer,kawai=fixture(root)
            cases=[];bad=copy.deepcopy(request);bad['context']['reward_contract']['covariance_options']['scale_mode']='unit_normalized_covariance';cases.append((cfg,bad))
            bad=copy.deepcopy(request);bad['context']['reward_contract']['reference_mode']='ema';cases.append((cfg,bad))
            badcfg=copy.deepcopy(cfg);badcfg['rank_power_gamma']=.5;cases.append((badcfg,request))
            for c,r in cases:
                with patch.object(pipeline,'run_program',side_effect=AssertionError('Model executed')),patch.object(pipeline,'run_kawai_parallel',side_effect=AssertionError('Model executed')),self.assertRaises(ValueError):pipeline.run(c,r,root/'no.json')
            p=json.loads((root/'policy.json').read_text());p['approved']=False;cfg['conservative']['policy']=pin(root/'policy.json',p)
            with patch.object(pipeline,'run_program',side_effect=AssertionError('Model executed')),self.assertRaises(ValueError):pipeline.run(cfg,request,root/'no.json')
    def test_missing_or_filtered_probabilities_are_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,_,_=fixture(root);h=pipeline._conservative_helpers();bundle=h.load_policy(cfg);items=pipeline.canonical_items(request);response=json.loads((root/'tagger.json').read_text())
            bad=copy.deepcopy(response);bad['items'][0]['probabilities']['style'].clear()
            with self.assertRaises(ValueError):h.validate_predictions(bad,items,bundle)
            bad=copy.deepcopy(response);bad['items'][0]['image_sha256']='f'*64
            with self.assertRaises(ValueError):h.validate_predictions(bad,items,bundle)
            bad=copy.deepcopy(response);bad['items'][0]['status']='error';bad['items'][0]['error']='archived image failure';bad['items'][0].pop('probabilities')
            details=[dict(id=r['id'],raw_scores=dict(kawai=.7,deepghs=.8),reward=.3,aesthetic_reward=.3,index=i) for i,r in enumerate(items)]
            fused,_=h.apply(details,items,cfg,bundle,bad);self.assertTrue(all(r['reward']==0 for r in fused))
    def test_native_prompt_hash_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,_,_=fixture(root);h=pipeline._conservative_helpers();bundle=h.load_policy(cfg);items=pipeline.canonical_items(request);items[0]['prompt']='rewritten'
            response=json.loads((root/'tagger.json').read_text());details=[dict(id=r['id'],raw_scores=dict(kawai=.7,deepghs=.8),reward=.3,aesthetic_reward=.3,index=i) for i,r in enumerate(items)]
            with self.assertRaises(ValueError):h.apply(details,items,cfg,bundle,response)
    def test_zero_cap_pair_is_counted_exactly_once(self):
        from conservative_reward import Candidate,Guard,RewardPolicy,HeadCalibration,fuse_group
        policy=RewardPolicy('test',HeadCalibration(0.,1.),HeadCalibration(0.,1.),1.,.8,0.,2)
        g=Guard('pass',.9,'test');r=fuse_group([Candidate('a','p',.7,.8,g),Candidate('b','p',.1,.2,g)],policy)
        self.assertEqual((r.aesthetic_pairs,r.constraint_pairs,r.abstained_pairs),(0,0,1))
    def test_wrapper_fingerprint_uses_same_conservative_artifacts(self):
        path=Path(__file__).parent/'checkpoint_eval/run_reward_with_validation.py';spec=importlib.util.spec_from_file_location('conservative_wrapper_fixture',path);wrapper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wrapper;spec.loader.exec_module(wrapper)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,request,scorer,kawai=fixture(root)
            expected,_=wrapper.pipeline_identity(pipeline,cfg,request)
            with archive_lock_test_support(),patch.object(pipeline,'run_program',side_effect=scorer),patch.object(pipeline,'run_kawai_parallel',side_effect=kawai):actual=pipeline.run(cfg,request,root/'response.json')
            self.assertEqual(expected,actual['fingerprint'])
            summary=wrapper.summary_for(actual['details'],dict(enabled_models=['kawai','deepghs'],kawai_enabled=True,sdg_enabled=False,domain='training'))
            self.assertEqual(summary['mean_semantics'],'relative_zero_mean_not_quality')
    def test_late_versioned_import_does_not_need_persistent_sys_path(self):
        h=pipeline._conservative_helpers()
        self.assertEqual(Path(h.__file__).parent,Path(pipeline.__file__).parent)
        self.assertEqual(Path(h._kernel.__file__).parent,Path(pipeline.__file__).parent)
if __name__=='__main__':unittest.main()
