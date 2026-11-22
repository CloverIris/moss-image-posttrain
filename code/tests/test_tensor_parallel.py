import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from moss_art.runtime import device_groups,configure_process
from moss_art.sft import DEFAULTS,validate

def test_two_device_groups_and_batch_validation():
    c=copy.deepcopy(DEFAULTS)
    c["runtime"].update(devices=[0,1,2,3],tensor_parallel_size=2)
    c["training"]["batch_size"]=2
    validate(c)
    assert device_groups(c)==[[0,1],[2,3]]
    c["runtime"]["devices"]=[0,3,1,2]
    assert device_groups(c)==[[0,3],[1,2]]
    c["training"]["batch_size"]=3
    with pytest.raises(ValueError,match="group count"): validate(c)
    for devices,size in (([0,1,2],2),([0,1],3),([0,1],True)):
        c["runtime"].update(devices=devices,tensor_parallel_size=size)
        with pytest.raises(ValueError): validate(c)

def test_tp_masks_gradients_cfg_shards_and_compilation(tmp_path):
    env=os.environ.copy()
    env.update(JAX_PLATFORMS="cpu",JAX_NUM_CPU_DEVICES="2",PYTHONPATH=str(Path.cwd()))
    output=tmp_path/"parity.json"
    subprocess.run([sys.executable,"scripts/diagnostics/check_tensor_parallel.py","--output",str(output)],
                   env=env,check=True,capture_output=True,text=True,timeout=120)
    result=json.loads(output.read_text())
    assert result["checkpoint_local_shapes"]==[[1,20,128],[1,20,128]]
    assert result["cfg_gradient_relative_rms"]<5e-5

def test_tp_sft_training_and_export_matches_single_group(tmp_path):
    from test_sft import fixture_cache
    from moss_art.checkpoint import load_checkpoint
    import yaml
    fixture_cache(tmp_path/"cache")
    c=copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny",dtype="float32",attention_backend="native",dit_autotune_level=0,checkpoint_interval=2)
    c["data"].update(cache=str(tmp_path/"cache"),context_length=16)
    c["sampling"]["skip_layers"]=[]
    c["runtime"].update(platform="cpu",cpu_threads=1,devices=[0,1],tensor_parallel_size=2)
    c["training"].update(max_updates=2,batch_size=2,microbatch_size=2,checkpoint_every=2)
    c["lora"].update(rank=2,alpha=2.)
    env=os.environ.copy();env.update(JAX_PLATFORMS="cpu",JAX_NUM_CPU_DEVICES="2")
    outputs=[]
    for size in (1,2):
        c["runtime"].update(tensor_parallel_size=size,devices=[0] if size==1 else [0,1],output_dir=str(tmp_path/f"tp{size}"))
        config=tmp_path/f"tp{size}.yaml";config.write_text(yaml.safe_dump(c))
        subprocess.run([sys.executable,"-m","moss_art.sft","train","--config",str(config)],
                       env=env,check=True,capture_output=True,text=True,timeout=120)
        checkpoint=tmp_path/f"tp{size}/checkpoints/step_00000002"
        outputs.append(load_checkpoint(checkpoint))
        assert (checkpoint/"adapter.safetensors").is_file()
    for key in outputs[0][0]:
        np.testing.assert_allclose(outputs[1][0][key],outputs[0][0][key],rtol=2e-5,atol=2e-7,err_msg=key)
    assert outputs[1][2]["sample_cursor"]==4


def test_physical_gpu_indices_resolve_to_uuid_in_requested_order(monkeypatch):
    from moss_art import runtime
    environment={"CUDA_VISIBLE_DEVICES":"stale-parent-mask", "CUDA_DEVICE_ORDER":"FASTEST_FIRST"}
    monkeypatch.setattr(runtime.os,"environ",environment)
    monkeypatch.setattr(runtime,"gpu_inventory",lambda: {i:{"uuid":f"GPU-physical-{i}"} for i in range(4)})
    if hasattr(runtime.os,"sched_getaffinity"):
        monkeypatch.setattr(runtime.os,"sched_getaffinity",lambda _pid:set(range(8)))
        monkeypatch.setattr(runtime.os,"sched_setaffinity",lambda _pid,_cores:None)
    c=copy.deepcopy(DEFAULTS)
    configure_process(c,[0,3])
    assert environment["CUDA_VISIBLE_DEVICES"]=="GPU-physical-0,GPU-physical-3"
    configure_process(c,[1,2])
    assert environment["CUDA_VISIBLE_DEVICES"]=="GPU-physical-1,GPU-physical-2"
    with pytest.raises(RuntimeError,match="do not exist"):
        configure_process(c,[0,9])
    c["runtime"]["platform"]="cpu"
    monkeypatch.setattr(runtime,"gpu_inventory",lambda:pytest.fail("CPU must not query NVIDIA"))
    configure_process(c,[0,3])
    assert environment["CUDA_VISIBLE_DEVICES"]==""
