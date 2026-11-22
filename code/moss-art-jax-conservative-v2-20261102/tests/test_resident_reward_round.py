import copy
import numpy as np
import pytest
from moss_art.config import DEFAULTS, validate
from moss_art.trainer import reward_round

@pytest.mark.parametrize("resident", [False, True])
def test_reward_decodes_before_offload_only_in_resident_mode(monkeypatch, resident):
    c=copy.deepcopy(DEFAULTS)
    c["sampling"]["vae_resident_rollout"]=resident
    c["reward"]["kind"]="http"
    events=[]
    class Pool:
        def ensure_rollout_vae(self): pass
        def pause(self): events.append("pause")
        def decode(self, endpoints):
            events.append("decode")
            return [dict(images=np.zeros((1,8,8,3),np.uint8))]
        def resume(self): events.append("resume_bf16")
    def score(images,prompts,options):
        assert images.dtype==np.uint8
        events.append("score")
        return [.5]
    monkeypatch.setattr("moss_art.rewards.score_images",score)
    outputs=[dict(endpoints=np.zeros((1,128,2,2),np.float32),prompts=["synthetic"])]
    reward,images=reward_round(c,Pool(),outputs)
    np.testing.assert_array_equal(reward,[.5])
    assert events==(["decode","pause","score","resume_bf16"] if resident else
                    ["pause","decode","score","resume_bf16"])

def test_resident_vae_configuration_rejects_incompatible_modes():
    c=copy.deepcopy(DEFAULTS)
    c["sampling"].update(vae_resident_rollout=True,rollout_mode="mxfp8")
    c["runtime"].update(platform="cuda",execution_backend="mesh",devices=[0,1,2,3])
    c["model"].update(kind="moss",dtype="bfloat16",activation_storage="gpu")
    c["training"].update(prompts_per_update=8,rollouts_per_prompt=16,training_points_per_trajectory=4)
    validate(c)
    for section,key,value in [("sampling","rollout_mode","bf16"),("reward","cpu_decode",True),("model","kind","tiny")]:
        bad=copy.deepcopy(c);bad[section][key]=value
        with pytest.raises(ValueError,match="Resident"):validate(bad)
