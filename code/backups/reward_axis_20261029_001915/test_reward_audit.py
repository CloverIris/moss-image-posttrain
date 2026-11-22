"""Small CPU fixtures for durable reward provenance and group diagnostics."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from moss_art.reward_audit import (analyze_rewards, collection_context, json_sha256,
    log_reward_components, new_command_attempt, persist_reward_evaluation)


def fixture(tmp_path):
    c = dict(runtime=dict(output_dir=str(tmp_path/"run")), sampling=dict(group_size=2), art=dict(feedback="scalar"))
    rows = [dict(index=i,prompt_index=g,sample_id="same_dataset_id",rollout_id=i//2,
                 image=str(tmp_path/(str(i)+".png")),prompt="unchanged prompt")
            for i,g in enumerate((0,1,0,1))]
    request = dict(version=1,update=4,rollout=12,config_fingerprint="algorithm_hash",items=rows)
    details = [dict(reward=value,raw_scores=dict(kawai=i+1,deepghs=.5),
                   normalized_scores=dict(kawai=i/2),contributions=dict(kawai=.04*i),
                   aesthetic_reward=value+.01,artifact_penalty=-.01,artifact_count=int(i%2==0),
                   ignored_misalignment_count=1, raw_output_path="/durable/model/raw.jsonl")
               for i,value in enumerate((.1,.6,.3,1.))]
    result = dict(version=2,scores=[.1,.6,.3,1.],details=details,source_commit="commit")
    return c, request, result


def test_audit_groups_use_collection_prompt_index_not_repeated_dataset_id(tmp_path):
    c,request,result = fixture(tmp_path)
    analysis = analyze_rewards(c,request,result)
    np.testing.assert_allclose([row["mad_normalized_reward"] for row in analysis["rows"]],[-1,-1,1,1],atol=1e-6)
    assert [row["group_id"] for row in analysis["rows"]] == [0,1,0,1]
    assert analysis["summary"]["groups"] == 2
    assert analysis["config_sha256"] == json_sha256(c)
    assert analysis["samples_sha256"] == json_sha256(request["items"])
    assert analysis["evaluation_sha256"] == json_sha256(result)
    assert "row_directions" not in json.dumps(analysis)
    assert [row["sample_id"] for row in analysis["rows"]] == ["same_dataset_id"]*4


def test_components_and_artifact_fraction_are_logged_without_raw_arrays(tmp_path):
    c,request,result = fixture(tmp_path)
    analysis = analyze_rewards(c,request,result)
    assert analysis["component_summary"]["raw_scores/kawai"]["mean"] == 2.5
    assert analysis["component_summary"]["raw_scores/kawai"]["count"] == 4
    assert analysis["rows"][0]["components"]["final_reward"] == analysis["rows"][0]["raw_reward"]
    assert analysis["component_summary"]["aesthetic_reward"]["complete_groups"] == 2
    assert analysis["component_summary"]["raw_scores/kawai"]["group_mad_mean"] == 1.
    assert analysis["component_summary"]["raw_scores/kawai"]["pooled_within_prompt_std"] == 1.
    assert analysis["component_summary"]["raw_scores/deepghs"]["constant_group_fraction"] == 1.
    assert len(analysis["component_group_statistics"]["raw_scores/kawai"]) == 2
    assert analysis["sdg_summary"]["artifact_fraction"] == .5
    records = []
    class Logger:
        def write(self,*args): records.append(args)
    log_reward_components(Logger(),12,analysis)
    assert any(row[0] == "reward_components/raw_scores/kawai" for row in records)
    assert any(row[0] == "reward_sdg" and row[2]["artifact_fraction"] == .5 for row in records)
    with pytest.raises(ValueError,match="details"):
        analyze_rewards(c,request,dict(result,details=[]))


def test_full_evaluation_survives_ram_bundle_retirement(tmp_path):
    c,request,result = fixture(tmp_path)
    bundle = tmp_path/"scratch";bundle.mkdir();(bundle/"COMMITTED").write_text("commit")
    context = collection_context(c,request,bundle)
    assert context["step_dir"].endswith("steps/step_00000004")
    assert context["source_commit"] == "commit"
    step = tmp_path/"run/steps/step_00000004";step.mkdir(parents=True)
    (step/"manifest.json").write_text(json.dumps(dict(files=["samples.json"])))
    analysis = persist_reward_evaluation(c,request,bundle,result)
    shutil.rmtree(bundle)
    assert json.loads((step/"reward_evaluation.json").read_text()) == result
    saved = json.loads((step/"reward_analysis.json").read_text())
    assert saved["rows"][0]["raw_reward"] == pytest.approx(.1)
    assert saved["context"]["run_output_dir"] == str((tmp_path/"run").resolve())
    manifest = json.loads((step/"manifest.json").read_text())
    assert "reward_analysis.json" in manifest["files"]
    assert manifest["reward_evaluation"]["source_sha256"] == analysis["evaluation_sha256"]


def test_new_attempt_preserves_previous_attempts(tmp_path):
    context = dict(step_dir=str(tmp_path))
    a = new_command_attempt(context);(a/"response.json").write_text("original")
    b = new_command_attempt(context)
    assert a != b
    assert (a/"response.json").read_text() == "original"


def test_audit_import_loads_no_accelerator_runtime():
    subprocess.run([sys.executable,"-c","import sys;import moss_art.reward_audit;assert 'jax' not in sys.modules;assert 'torch' not in sys.modules"],check=True)


def test_command_context_and_full_logs_are_durable_across_retries(tmp_path):
    from moss_art.stage_pipeline import _score_request
    c,request,_ = fixture(tmp_path)
    script = tmp_path/"fake.py"
    script.write_text('''import argparse,json
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.load(open(a.request));assert r['context']['update']==4;assert r['context']['rollout']==12
assert 'steps/step_00000004' in r['context']['step_dir']
print('x'*20000)
json.dump({'scores':[.1,.6,.3,1.],'details':[{'artifact_count':0}]*4,'retained_extra':'calibration'},open(a.output,'w'))
''')
    c["reward"] = dict(kind="command",command=[sys.executable,str(script)],timeout=30,
        model_path="test",device="cpu",batch_size=4,metric="test",threads=1)
    bundle=tmp_path/"scratch";bundle.mkdir();(bundle/"COMMITTED").write_text("commit")
    scores,details=_score_request(c,request,bundle)
    np.testing.assert_allclose(scores,[.1,.6,.3,1.])
    attempts=tmp_path/"run/steps/step_00000004/reward_attempts"
    first=attempts/"attempt_0001"
    assert (first/"stdout.log").stat().st_size > 20000
    before=(first/"response.json").read_bytes()
    assert json.loads(before)["retained_extra"] == "calibration"
    assert (bundle/"reward_response.json").read_bytes()==before
    _score_request(c,request,bundle)
    assert (attempts/"attempt_0002/response.json").exists()
    assert (first/"response.json").read_bytes()==before


def test_command_failure_keeps_full_stderr_but_bounds_exception(tmp_path):
    from moss_art.stage_pipeline import _score_request
    c,request,_ = fixture(tmp_path)
    script=tmp_path/"failed.py"
    script.write_text("import sys;sys.stderr.write('e'*20000);raise SystemExit(3)")
    c["reward"] = dict(kind="command",command=[sys.executable,str(script)],timeout=30,
        model_path="test",device="cpu",batch_size=4,metric="test",threads=1)
    bundle=tmp_path/"scratch";bundle.mkdir()
    with pytest.raises(RuntimeError) as error:
        _score_request(c,request,bundle)
    assert len(str(error.value)) < 10000
    attempt=tmp_path/"run/steps/step_00000004/reward_attempts/attempt_0001"
    assert (attempt/"stderr.log").stat().st_size==20000
    assert json.loads((attempt/"status.json").read_text())["returncode"]==3


def test_tensorboard_contains_component_group_metrics_and_distributions(tmp_path):
    from moss_art.tensorboard_log import ScalarLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    c,request,result=fixture(tmp_path)
    c["sampling"].update(height=768,width=768)
    c["logging"]=dict(tensorboard=True,log_dir=str(tmp_path/"tb"))
    analysis=analyze_rewards(c,request,result)
    with ScalarLogger(c,tmp_path/"run") as logger:
        log_reward_components(logger,12,analysis)
    events=EventAccumulator(str(tmp_path/"tb"));events.Reload()
    assert "reward_components/raw_scores/kawai/group_mad_mean" in events.Tags()["scalars"]
    assert "reward_components/768x768/raw_scores/kawai/group_std_mean" in events.Tags()["scalars"]
    assert "reward_components/artifact_penalty/distribution" in events.Tags()["histograms"]
    assert events.Scalars("reward_sdg/artifact_fraction")[0].value==.5
