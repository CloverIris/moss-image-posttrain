import numpy as np
import pytest
from moss_art.config import load_config, fingerprint
from moss_art.trainer import aggregate_gradients, select_records, make_payloads

def test_gradient_denominator_keeps_masked_rows():
    a = {"gradients":{"x":np.array([3.,6.])},"denominator":3,"loss_sum":6,"masked":1,"active":2}
    b = {"gradients":{"x":np.array([0.,0.])},"denominator":1,"loss_sum":0,"masked":0,"active":0}
    grad, metrics = aggregate_gradients([a,b])
    np.testing.assert_allclose(grad["x"],[.75,1.5])
    assert metrics["loss"] == 1.5 and metrics["items"] == 4

def test_prompt_cursor_is_resume_stable():
    records = list(range(7))
    whole = select_records(records,0,18,42)
    assert whole == select_records(records,0,8,42)+select_records(records,8,10,42)
    assert len(set(whole[:7])) == 7

def test_seeds_do_not_depend_on_worker_partition():
    c = load_config("configs/smoke_cpu.yaml")
    records = [{"sample_id":str(i),"prompt":str(i)} for i in range(4)]
    one = make_payloads(c,records,2)[0]["seeds"]
    c["runtime"]["devices"] = [0,1,2,3]
    four = sum([x["seeds"] for x in make_payloads(c,records,2)],[])
    assert one == four
    assert len(set(one)) == len(one)

def test_invalid_odd_group_and_unknown_key_rejected(tmp_path):
    p = tmp_path/"bad.yaml"
    p.write_text("sampling: {group_size: 3}")
    with pytest.raises(ValueError): load_config(p)
    p.write_text("runtime: {typo: true}")
    with pytest.raises(ValueError,match="Unknown"): load_config(p)

# Malformed values must fail on the host, before loading a 5B model.
def _changed(path, value):
    import copy
    from moss_art.config import DEFAULTS
    config = copy.deepcopy(DEFAULTS)
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return config


def test_config_malformed_scalars_lists_and_booleans_raise_valueerror():
    from moss_art.config import validate
    cases = [
        (("sampling", "shift"), "5.0"),
        (("sampling", "guidance_scale"), True),
        (("sampling", "steps"), 2.5),
        (("sampling", "train_timesteps"), 3),
        (("sampling", "train_timesteps"), [[0]]),
        (("sampling", "skip_layers"), "22"),
        (("runtime", "devices"), 0),
        (("runtime", "devices"), [[0]]),
        (("runtime", "cpu_threads"), 1.5),
        (("runtime", "cpu_threads"), True),
        (("runtime", "worker_timeout"), "120"),
        (("runtime", "allow_busy"), 1),
        (("runtime", "output_dir"), None),
        (("training", "keep_checkpoints"), False),
        (("training", "max_no_signal_rollouts"), 1.5),
        (("optimizer", "betas"), .9),
        (("optimizer", "betas"), [.9, True]),
        (("optimizer", "learning_rate"), None),
        (("lora", "alpha"), {"value": 4}),
        (("lora", "rank"), True),
        (("reward", "command"), "python score.py"),
        (("reward", "command"), ["python", 3]),
        (("reward", "threads"), 1.5),
        (("reward", "batch_size"), False),
        (("reward", "decode"), "false"),
        (("data", "negative_prompt"), None),
    ]
    for path, value in cases:
        with pytest.raises(ValueError):
            validate(_changed(path, value))


def test_config_scheduler_indices_seed_and_resource_bounds():
    from moss_art.config import validate
    cases = [
        (("sampling", "shift"), 0.), (("sampling", "training_shift"), -1.),
        (("sampling", "skip_layers"), [42]), (("sampling", "skip_layers"), [-1]),
        (("sampling", "skip_layers"), [22, 22]), (("sampling", "skip_layers"), [True]),
        (("sampling", "train_timesteps"), []), (("sampling", "train_timesteps"), [50]),
        (("sampling", "train_timesteps"), [0, 0]), (("sampling", "train_timesteps"), [False]),
        (("model", "tiny_seed"), -1), (("training", "seed"), 2**32),
        (("lora", "seed"), True), (("runtime", "devices"), []),
        (("runtime", "devices"), [0, 0]), (("runtime", "devices"), [True]),
        (("runtime", "worker_timeout"), 0.), (("runtime", "cpu_threads"), 0),
        (("runtime", "busy_memory_mib"), -1), (("runtime", "memory_fraction"), 1.01),
        (("training", "keep_checkpoints"), 0), (("training", "max_no_signal_rollouts"), 0),
        (("reward", "timeout"), -.5), (("reward", "threads"), 0),
        (("reward", "batch_size"), 0), (("reward", "metric"), "typo"),
        (("reward", "command"), ["python", "bad\x00argument"]),
        (("sampling", "shift"), float("nan")), (("art", "eps"), float("inf")),
        (("optimizer", "betas"), [.9, float("nan")]),
    ]
    for path, value in cases:
        with pytest.raises(ValueError):
            validate(_changed(path, value))


def test_config_valid_boundaries_and_command_argv():
    import copy
    from moss_art.config import DEFAULTS, validate
    config = copy.deepcopy(DEFAULTS)
    config["sampling"].update(skip_layers=[0, 41], shift=.01, training_shift=.01, scheduler="moss")
    config["runtime"].update(worker_timeout=.5, cpu_threads=1, memory_fraction=1., busy_memory_mib=0)
    config["reward"].update(timeout=.5, threads=1, batch_size=1)
    for section, key in (("model", "tiny_seed"), ("lora", "seed"), ("training", "seed")):
        config[section][key] = 2**32-1
    assert validate(config) is config
    config["model"]["kind"] = "tiny"
    config["sampling"]["skip_layers"] = [0, 1]
    config["reward"].update(kind="command", command=["python", "score.py", "--prompt", ""])
    assert validate(config) is config
    config["sampling"]["skip_layers"] = [2]
    with pytest.raises(ValueError, match="skip_layers"):
        validate(config)
    config["sampling"]["skip_layers"] = []
    for argv in ([], [""], ["  "]):
        config["reward"]["command"] = argv
        with pytest.raises(ValueError, match="argv"):
            validate(config)


def test_config_nonmapping_yaml_rejected_and_all_shipped_configs_valid(tmp_path):
    from pathlib import Path
    invalid_yaml = ["[]", "false", "1", "runtime: false", "sampling: 2",
                    "model: null", "optimizer: {betas: 0.9}", "reward: {command: echo}"]
    path = tmp_path/"malformed.yaml"
    for text in invalid_yaml:
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)
    for path in Path("configs").glob("*.yaml"):
        load_config(path)

def test_attention_backend_is_explicit_validated_and_checkpoint_bound():
    import copy
    from moss_art.config import validate
    native = load_config("configs/smoke_cpu.yaml")
    assert native["model"]["attention_backend"] == "native"
    bad = copy.deepcopy(native)
    bad["model"]["attention_backend"] = "auto"
    with pytest.raises(ValueError, match="attention_backend"):
        validate(bad)
    bad["model"]["attention_backend"] = "cudnn"
    with pytest.raises(ValueError, match="platform=cuda"):
        validate(bad)
    bad["runtime"]["platform"] = "cuda"
    with pytest.raises(ValueError, match="BF16"):
        validate(bad)
    bad["model"]["dtype"] = "bfloat16"
    validate(bad)
    reference = copy.deepcopy(bad)
    reference["model"]["attention_backend"] = "native"
    assert fingerprint(reference) != fingerprint(bad)
    assert load_config("configs/moss_lora_4gpu.yaml")["model"]["attention_backend"] == "cudnn"

def test_dit_autotune_is_scoped_validated_and_checkpoint_bound():
    import copy
    from moss_art.config import validate
    c = load_config("configs/smoke_cpu.yaml")
    assert c["model"]["dit_autotune_level"] == 0
    for value in (-1, 1, 5, True, 4.0, "4"):
        bad = copy.deepcopy(c)
        bad["model"]["dit_autotune_level"] = value
        with pytest.raises(ValueError):
            validate(bad)
    tuned = copy.deepcopy(c)
    tuned["model"]["dit_autotune_level"] = 4
    with pytest.raises(ValueError, match="platform=cuda"):
        validate(tuned)
    tuned["runtime"]["platform"] = "cuda"
    validate(tuned)
    reference = copy.deepcopy(tuned)
    reference["model"]["dit_autotune_level"] = 0
    assert fingerprint(reference) != fingerprint(tuned)
    assert load_config("configs/moss_lora_4gpu.yaml")["model"]["dit_autotune_level"] == 4


def test_sampler_configuration_and_checkpoint_identity():
    import copy
    from moss_art.config import DEFAULTS, validate
    current = copy.deepcopy(DEFAULTS)
    assert current["sampling"]["steps"] == 50
    assert current["sampling"]["sampler"] == "res_multistep"
    assert current["sampling"]["scheduler"] == "beta"
    validate(current)
    for key, value in (("sampler", "resmutistep"), ("scheduler", "invalid"),
                       ("beta_alpha", 0.), ("beta_beta", -1.), ("beta_beta", True),
                       ("beta_alpha", float("nan")), ("training_shift", 2.)):
        changed = copy.deepcopy(current)
        changed["sampling"][key] = value
        with pytest.raises(ValueError):
            validate(changed)
    for key, value in (("sampler", "euler"), ("scheduler", "moss"),
                       ("beta_alpha", .7), ("beta_beta", .8)):
        changed = copy.deepcopy(current)
        changed["sampling"][key] = value
        validate(changed)
        assert fingerprint(current) != fingerprint(changed)


def test_inventory_preserves_nvidia_index_uuid_and_bus_id(monkeypatch):
    from types import SimpleNamespace
    from moss_art import runtime
    captured=[]
    def query(argv, **kwargs):
        captured.append(argv)
        return SimpleNamespace(stdout="3, GB203(SM120), 12, 16303, GPU-last, 00000000:1D:00.0\n0, GB203(SM120), 266, 16303, GPU-first, 00000000:08:00.0\n")
    monkeypatch.setattr(runtime.subprocess,"run",query)
    inventory=runtime.gpu_inventory()
    assert inventory[3]["uuid"]=="GPU-last"
    assert inventory[0]["pci_bus_id"]=="00000000:08:00.0"
    assert inventory[0]["used_mib"]==266
    assert "uuid,pci.bus_id" in captured[0][1]
