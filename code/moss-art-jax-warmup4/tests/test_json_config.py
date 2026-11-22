import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from moss_art.config_io import read_mapping,validate_run,RUN_DEFAULTS
from moss_art.config import load_config,fingerprint

def test_json_strict_and_bom(tmp_path):
    p=tmp_path/"a.json"
    for text in ('[]','{"x":NaN}','{"x":1e999}','{"x":1,"x":2}','{"nested":{"x":1,"x":2}}','{"x":1,}'):
        p.write_text(text)
        with pytest.raises(ValueError): read_mapping(p)
    p.write_text('\ufeff{"runtime":{"devices":[0,3,1,2]}}',encoding="utf-8")
    assert read_mapping(p)["runtime"]["devices"]==[0,3,1,2]

def test_run_settings_do_not_change_art_fingerprint(tmp_path):
    p=tmp_path/"c.json";p.write_text('{}')
    c=load_config(p);before=fingerprint(c)
    c["run"]["resume"]="outputs/previous"
    assert fingerprint(c)==before
    c["sampling"]["sampler"]="euler"
    assert fingerprint(c)!=before

@pytest.mark.parametrize("command,run",[("train",{"adapter":"x"}),
    ("sample",{"resume":"x"}),("evaluate",{}),("train",{"resume":"x","init_adapter":"y"}),
    ("doctor",{"seed":3}),("train",{"decode":True}),("sample",{"sample_index":-1})])
def test_run_rejects_inapplicable_fields(command,run):
    with pytest.raises(ValueError):validate_run({**RUN_DEFAULTS,**run},command)

def test_sft_dispatches_only_config_parameters(tmp_path):
    from moss_art.sft import main
    p=tmp_path/"sft.json"
    config={"runtime":{"devices":[0,3,1,2],"tensor_parallel_size":2,"output_dir":"outputs/test"},
            "training":{"batch_size":6,"microbatch_size":3},
            "run":{"sample_index":2,"seed":13,"adapter":"adapter.safetensors"}}
    p.write_text(json.dumps(config))
    with patch("moss_art.sft.sample") as call:
        main(["sample","--config",str(p)])
        c,index,seed,adapter,checkpoint=call.call_args.args
        assert (index,seed,adapter,checkpoint)==(2,13,"adapter.safetensors",None)
        assert c["runtime"]["devices"]==[0,3,1,2]
    config["run"]={"init_adapter":"adapter.safetensors"}
    p.write_text(json.dumps(config))
    with patch("moss_art.sft.train") as call:
        main(["train","--config",str(p)])
        assert call.call_args.args[1:]==(None,"adapter.safetensors")
    with pytest.raises(SystemExit):main(["train","--config",str(p),"--devices","0"])

def test_flat_preprocessing_config_schema(tmp_path):
    import argparse
    from moss_art.config_io import parse_file_arguments
    schema=argparse.ArgumentParser()
    schema.add_argument("--manifest",required=True)
    schema.add_argument("--width",type=int,default=1024)
    schema.add_argument("--conditioning-only",action="store_true")
    p=tmp_path/"prepare.json"
    p.write_text(json.dumps({"manifest":"-literal.jsonl","width":512,"conditioning_only":True}))
    a=parse_file_arguments(schema,["--config",str(p)])
    assert a.manifest=="-literal.jsonl" and a.width==512 and a.conditioning_only
    for override in ({"width":True},{"typo":1},{"conditioning_only":"true"}):
        p.write_text(json.dumps({"manifest":"x",**override}))
        with pytest.raises(ValueError):parse_file_arguments(schema,["--config",str(p)])

def test_all_json_training_presets_validate():
    from moss_art.sft import load_config as sft_config
    for p in [*Path("configs").glob("*.json"),*Path("configs/examples").glob("*.json")]:
        if p.name in ("decode.json","analyze_prompts.json","analyze_prompt_check.json","qwen_prefill_check.json"):continue
        load_config(p)
    for p in Path("configs/sft").glob("*.json"):
        if p.name=="prepare.json":continue
        sft_config(p)


def test_sft_beta_rejects_ignored_training_shift(tmp_path):
    from moss_art.sft import load_config
    p=tmp_path/"sft.json"
    p.write_text(json.dumps({"sampling":{"scheduler":"beta","training_shift":2.}}))
    with pytest.raises(ValueError,match="training_shift"):
        load_config(p)
    p.write_text(json.dumps({"sampling":{"scheduler":"moss","training_shift":2.}}))
    assert load_config(p)["sampling"]["training_shift"]==2.


def test_sft_prepare_json_binds_device_before_encoders(tmp_path):
    from moss_art.sft_prepare import main
    p=tmp_path/"prepare.json"
    p.write_text(json.dumps({"device":2,"manifest":"train.jsonl","cache":str(tmp_path/"cache"),
                            "posterior":"sample","seed":7,"conditioning_only":True}))
    with patch("moss_art.runtime.preflight"),patch("moss_art.runtime.configure_process") as bind,patch("moss_art.sft_prepare.prepare") as prepare:
        main(["--config",str(p)])
        assert bind.call_args.args[1]==2
        args,directory=prepare.call_args.args
        assert args.manifest=="train.jsonl" and args.posterior=="sample" and args.seed==7
        assert args.conditioning_only and directory==tmp_path/"cache"


def test_art_prepare_json_options(tmp_path):
    from moss_art.cli import main
    p=tmp_path/"prepare.json"
    p.write_text(json.dumps({"preparation":{"device":1,"max_sequence_length":2048,"limit":2},
                            "data":{"prompts":"prompts.jsonl","cache":"cache/test"}}))
    with patch("moss_art.cli.preflight"),patch("moss_art.runtime.configure_process") as bind,patch("moss_art.prepare.prepare_cache") as prepare:
        main(["prepare-text","--config",str(p)])
        assert bind.call_args.args[1]==1
        assert prepare.call_args.kwargs["max_sequence_length"]==2048
        assert prepare.call_args.kwargs["limit"]==2


def test_sft_json_train_resume_and_sample(tmp_path):
    import os,subprocess,sys
    from test_sft import fixture_cache
    from moss_art.sft import DEFAULTS
    fixture_cache(tmp_path/"cache")
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0)
    c["runtime"].update(platform="cpu",devices=[0],cpu_threads=1,output_dir=str(tmp_path/"train"))
    c["data"].update(cache=str(tmp_path/"cache"),context_length=8)
    c["lora"].update(rank=2,alpha=2.)
    c["training"].update(max_updates=1,batch_size=2,microbatch_size=2,checkpoint_every=1)
    c["sampling"].update(steps=2,skip_layers=[])
    path=tmp_path/"sft.json"
    env={**os.environ,"JAX_PLATFORMS":"cpu","JAX_NUM_CPU_DEVICES":"1"}
    def invoke(command):
        path.write_text(json.dumps(c))
        subprocess.run([sys.executable,"-m","moss_art.sft",command,"--config",str(path)],
                       env=env,check=True,capture_output=True,text=True,timeout=120)
    invoke("train")
    c["run"]["resume"]=str(tmp_path/"train")
    c["training"]["max_updates"]=2
    invoke("train")
    rows=[json.loads(line) for line in (tmp_path/"train/metrics.jsonl").read_text().splitlines()]
    assert [row["sample_cursor"] for row in rows]==[2,4]
    c["run"].update(resume="",checkpoint=str(tmp_path/"train/checkpoints/step_00000002"),sample_index=1,seed=19)
    c["runtime"]["output_dir"]=str(tmp_path/"sample")
    invoke("sample")
    import numpy as np
    latent=np.load(tmp_path/"sample/sample-0001-seed-19.npy")
    assert latent.shape==(1,8,2,2) and np.isfinite(latent).all()
    metadata=json.loads((tmp_path/"sample/sample-0001-seed-19.json").read_text())
    assert metadata["seed"]==19 and metadata["visual_tokens"]==2
    assert metadata["checkpoint"]==c["run"]["checkpoint"]


def test_sample_decode_uses_json_vae_policy_and_physical_device(tmp_path):
    import numpy as np
    from moss_art.sft import DEFAULTS,decode_sample
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(vae_precision="float16",vae_low_precision_stages=["up1","up2","up3"],
                       vae_conv_backend="cudnn_bounded",vae_conv_workspace_mib=128)
    latent=tmp_path/"latent.npy";png=tmp_path/"image.png"
    np.save(latent,np.zeros((1,128,1,1),np.float32))
    with patch("moss_art.sft.configure_process") as bind,patch("moss_art.evaluate.decode_latents",return_value=np.zeros((1,8,8,3),np.uint8)) as decode:
        decode_sample(c,str(latent),str(png),3)
        assert bind.call_args.args==(c,3)
        assert decode.call_args.kwargs==dict(precision="float16",low_precision_stages=["up1","up2","up3"],
            conv_backend="cudnn_bounded",conv_workspace_mib=128)
        assert png.is_file()
