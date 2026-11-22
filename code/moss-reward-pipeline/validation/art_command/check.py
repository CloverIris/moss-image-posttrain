import sys,json,pathlib,numpy as np
sys.path.insert(0,"$CODE_ROOT/moss-art-jax")
from moss_art.config import load_config,fingerprint
from moss_art.stage_pipeline import _score_request
root=pathlib.Path("$DATA_DISK/moss-reward-pipeline/validation/art_command")
c=load_config("$MOSS_ROOT/runs/reward4_P8_K16_T4_mxfp8_20261029/config.json")
c["runtime"]["output_dir"]=str(root/"training")
bundle=root/"bundle";bundle.mkdir(exist_ok=True);(bundle/"COMMITTED").write_text("artifact-integration-validation")
r=json.loads(pathlib.Path("$DATA_DISK/moss-reward-pipeline/validation/request.json").read_text())
r.update(update=0,rollout=0,config_fingerprint=fingerprint(c),config=c)
scores,details=_score_request(c,r,bundle)
assert scores.shape==(4,) and np.isfinite(scores).all()
assert len(details)==4
(root/"result.json").write_text(json.dumps(dict(scores=scores.tolist(),details=details),ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(status="passed",images=4,artifact_only=True,tagger_enabled=False)))
