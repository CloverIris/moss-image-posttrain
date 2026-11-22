"""Strict, portable experiment configuration; imports no accelerator framework."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from .config_io import read_mapping, RUN_DEFAULTS, validate_run
from .spectrum import DEFAULTS as SPECTRUM_DEFAULTS, validate_options as validate_spectrum

DEFAULTS = {
    "name": "moss-art",
    "run": copy.deepcopy(RUN_DEFAULTS),
    "preparation": {"backend":"jax","device":4,"max_sequence_length":4096,"full_weight_hash":False,"limit":0,"auto_prepare":True,"async_prefetch":False,"qwen_gpu_layers":16,"qwen_gpu_embedding":True,"qwen_gpu_vision":True,
                    "vllm":{"python":"$CONDA_PREFIX/envs/vllm_env/bin/python","devices":[1,2],"tensor_parallel_size":2,"pipeline_parallel_size":1,"data_parallel_size":1,"topology_aware":False,"storage_dtype":"float32","batch_size":8,"gpu_memory_utilization":0.85,"max_num_batched_tokens":8192,"max_num_seqs":8,"enforce_eager":False,"model_path":"","processor_dir":"$CODE_ROOT/moss-image/reference-lab/processor","max_reference_pixels":262144}},
    "model": {"kind": "moss", "path": "$MODEL_ROOT/MOSS-Image-SFT",
              "dtype": "bfloat16", "tiny_seed": 123, "attention_backend": "native", "dit_autotune_level": 0,
              "activation_storage": "cpu", "checkpoint_interval": 1,
              "vae_precision": "fp32", "vae_low_precision_stages": [],
              "vae_conv_backend": "xla", "vae_conv_workspace_mib": 256},
    "lora": {"rank": 4, "alpha": 4.0, "preset": "attn", "seed": 42},
    "data": {"prompts": "configs/prompts.jsonl", "cache": "cache/text", "negative_prompt": "", "context_length":4096, "short_context_length":0},
    "sampling": {"rollout_mode": "bf16", "vae_resident_rollout": False, "height": 256, "width": 256, "steps": 50, "group_size": 2, "rollout_batch_size": 1,
                 "prompts_per_rollout": 4, "split_group_across_workers": False, "pair_cached_points": False, "guidance_scale": 3.0, "skip_layers": [22],
                 "train_timesteps": [0, 10, 20, 30, 40], "shift": 5.0, "training_shift": 1.0,
                 "sampler": "res_multistep", "scheduler": "beta", "beta_alpha": 0.6, "beta_beta": 0.6,
                 "spectrum": copy.deepcopy(SPECTRUM_DEFAULTS)},
    "art": {"noise_level": 0.8, "kl_threshold": 1e-5, "eps": 1e-5, "feedback": "scalar"},
    "optimizer": {"name": "adamw", "momentum": 0.95, "nesterov": True, "ns_steps": 5, "rms_match": 0.2,
                  "learning_rate": 1e-5, "betas": [0.9, 0.999], "epsilon": 1e-8,
                  "weight_decay": 0.0, "max_grad_norm": 1.0},
    "training": {"max_updates": 20, "prompts_per_update": 0, "rollouts_per_prompt": 64, "training_points_per_trajectory": 4, "microbatch_size": 1, "rollouts_per_update": 1, "updates_per_rollout": 2, "seed": 42, "max_no_signal_rollouts": 20,
                 "checkpoint_every": 2, "keep_checkpoints": 3, "evaluate_every": 2, "save_rollout_images": False},
    "resolution_mixing": {"enabled": False, "profiles": []},
    "logging": {"tensorboard": False, "log_dir": ""},
    "reward": {"kind": "siglip", "command": [], "timeout": 900,
               "model_path": "$HOME/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
               "python": "$CONDA_PREFIX/envs/torch_env/bin/python",
               "device": "cpu", "batch_size": 1, "metric": "cosine", "threads": 4,
               "decode": True, "cpu_decode": False,
               "url": "", "http_max_retries": 2, "http_retry_backoff": 0.5,
               "http_bearer_token_env": "", "http_response_limit_bytes": 1048576,
               "backbone_path": "", "metadata_dir": "vendor/pixai_tagger/metadata", "dtype": "bfloat16", "attention_backend": "cudnn"},
    "runtime": {"execution_backend":"workers", "devices": [0, 1, 2, 3], "platform": "cuda", "allow_busy": False,
                "busy_memory_mib": 1500, "worker_timeout": 3600, "output_dir": "outputs/moss-art",
                "cpu_threads": 4, "memory_fraction": 0.9, "min_free_disk_gib": 0.0},
}

def _merge(base, new, prefix=""):
    if not isinstance(new, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in new.items():
        if key not in base:
            raise ValueError(f"Unknown configuration option: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value
    return base

def _validate_types(actual, defaults, prefix=""):
    """Reject malformed YAML values before comparisons or accelerator startup."""
    if not isinstance(actual, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    if set(actual) != set(defaults):
        raise ValueError(f"{prefix or 'config'} has missing or unknown fields")
    for key, example in defaults.items():
        name, value = prefix + key, actual[key]
        if isinstance(example, dict):
            _validate_types(value, example, name + ".")
        elif isinstance(example, bool):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        elif isinstance(example, list):
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
        elif isinstance(example, str):
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"{name} must be a string without NUL characters")
        elif isinstance(example, int) and name not in ("runtime.worker_timeout", "reward.timeout"):
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer (not boolean)")
        else:
            if type(value) not in (int, float):
                raise ValueError(f"{name} must be a finite number (not boolean)")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")


def _integer_list(values, name, *, minimum=0, maximum=None, nonempty=False):
    if not isinstance(values, list) or (nonempty and not values):
        raise ValueError(f"{name} must be {'a nonempty' if nonempty else 'an'} integer list")
    if any(type(value) is not int or value < minimum or
           (maximum is not None and value >= maximum) for value in values):
        raise ValueError(f"{name} contains an invalid integer index")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique indices")


def validate(c):
    _validate_types(c, DEFAULTS)
    validate_run(c["run"])
    p=c["preparation"]
    if p["backend"] not in ("jax", "vllm"):
        raise ValueError("preparation.backend must be jax or vllm")
    v=p["vllm"]
    _integer_list(v["devices"], "preparation.vllm.devices", nonempty=True)
    sizes = [v[name] for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")]
    if any(size < 1 for size in sizes) or math.prod(sizes) != len(v["devices"]):
        raise ValueError("preparation.vllm device count must equal DP * PP * TP")
    if v["topology_aware"] and (v["tensor_parallel_size"] != 1 or v["pipeline_parallel_size"] != 2):
        raise ValueError("topology_aware requires TP=1 and PP=2; device groups are PCIe pairs")
    if v["storage_dtype"] not in ("float32", "bfloat16"):
        raise ValueError("preparation.vllm.storage_dtype must be float32 or bfloat16")
    if v["batch_size"] < 1 or v["max_num_seqs"] < 1 or v["max_num_batched_tokens"] < 1:
        raise ValueError("preparation.vllm batch and token budgets must be positive")
    if not 0 < v["gpu_memory_utilization"] <= 1:
        raise ValueError("preparation.vllm.gpu_memory_utilization must lie in (0, 1]")
    if v["max_reference_pixels"] < 65536:
        raise ValueError("preparation.vllm.max_reference_pixels must be at least 65536")
    if not v["python"].strip() or not v["processor_dir"].strip():
        raise ValueError("preparation.vllm Python and processor paths must be nonempty")
    if p["backend"] == "vllm":
        if p["async_prefetch"]:
            raise ValueError("vLLM is an offline preprocessor; set preparation.async_prefetch=false")
        if c["model"]["kind"] != "moss" or c["runtime"]["platform"] != "cuda":
            raise ValueError("vLLM preparation requires the MOSS model and CUDA")
        if v["max_num_batched_tokens"] < c["data"]["context_length"] + 1:
            raise ValueError("preparation.vllm.max_num_batched_tokens must fit data.context_length + 1")
    if p["device"]<0 or p["max_sequence_length"]<1 or p["limit"]<0:
        raise ValueError("Invalid preparation device, token budget or limit")
    if not 0<=p["qwen_gpu_layers"]<=32:raise ValueError("qwen_gpu_layers must lie between0 and32")
    from .prompt_buckets import validate_lengths
    validate_lengths(c["data"]["short_context_length"],c["data"]["context_length"])
    if p["async_prefetch"] and c["runtime"]["platform"]=="cuda" and p["device"] in c["runtime"]["devices"]:
        raise ValueError("Async Qwen GPU must be separate from DiT devices")
    s, t, r, o, l = (c[x] for x in ("sampling", "training", "runtime", "optimizer", "lora"))
    validate_spectrum(s["spectrum"])
    model, reward = c["model"], c["reward"]
    from .batch_plan import resolve
    if not r["devices"]:raise ValueError("runtime.devices must be nonempty")
    resolve(c)
    if r["execution_backend"] not in ("mesh","workers"):raise ValueError("Unknown execution_backend")
    if s["rollout_mode"] not in ("bf16", "mxfp8", "mxfp8_calibrated"):
        raise ValueError("sampling.rollout_mode must be bf16, mxfp8 or mxfp8_calibrated")
    if s["rollout_mode"] != "bf16":
        if r["execution_backend"] != "mesh" or r["platform"] != "cuda" or model["dtype"] != "bfloat16":
            raise ValueError("MXFP8 rollout requires CUDA Mesh and a BF16 learner")
    if s["vae_resident_rollout"]:
        if s["rollout_mode"] not in ("mxfp8", "mxfp8_calibrated") or r["execution_backend"] != "mesh":
            raise ValueError("Resident rollout VAE requires MXFP8 Mesh rollout")
        if model["kind"] != "moss" or reward["cpu_decode"]:
            raise ValueError("Resident rollout VAE requires MOSS GPU decoding")
    if s["rollout_mode"] == "mxfp8_calibrated" and t["updates_per_rollout"] != 2:
        raise ValueError("Calibrated rollout requires two ART update windows")

    if r["execution_backend"]=="mesh":
        if r.get("tensor_parallel_size",1)!=1:raise ValueError("Mesh DP currently requires tensor_parallel_size=1")
        if model["activation_storage"]!="gpu":raise ValueError("Mesh DP requires GPU activation storage")
        if len(r["devices"])<2:raise ValueError("Mesh DP requires at least two devices")
        if not s["split_group_across_workers"] or s["prompts_per_rollout"]!=1:raise ValueError("Mesh DP requires split prompt groups")


    if model["activation_storage"] not in ("cpu", "gpu"):
        raise ValueError("model.activation_storage must be cpu or gpu")
    if model["checkpoint_interval"] < 1:
        raise ValueError("model.checkpoint_interval must be a positive integer")
    if model["kind"] not in ("moss", "tiny"):
        raise ValueError("model.kind must be moss or tiny")
    if model["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("model.dtype must be bfloat16 or float32")
    if model["vae_precision"] not in ("fp32", "bfloat16", "float16"):
        raise ValueError("model.vae_precision must be fp32, bfloat16 or float16")
    vae_stages = model["vae_low_precision_stages"]
    allowed_vae_stages = ("input", "mid", "up0", "up1", "up2", "up3", "output")
    if any(not isinstance(stage, str) or stage not in allowed_vae_stages for stage in vae_stages):
        raise ValueError("model.vae_low_precision_stages contains an unknown VAE stage")
    if len(set(vae_stages)) != len(vae_stages):
        raise ValueError("model.vae_low_precision_stages must be unique")
    if model["vae_precision"] == "fp32" and vae_stages:
        raise ValueError("FP32 VAE requires empty vae_low_precision_stages")
    if model["vae_precision"] != "fp32" and not vae_stages:
        raise ValueError("Low-precision VAE requires explicit vae_low_precision_stages")
    if model["vae_conv_backend"] not in ("xla", "cudnn_bounded", "tiled_gemm"):
        raise ValueError("model.vae_conv_backend must be xla, cudnn_bounded or tiled_gemm")
    if not 0 <= model["vae_conv_workspace_mib"] <= 512:
        raise ValueError("model.vae_conv_workspace_mib must be between 0 and 512")
    if model["vae_conv_backend"] == "cudnn_bounded":
        if r["platform"] != "cuda" or reward["cpu_decode"]:
            raise ValueError("Bounded cuDNN VAE convolution requires CUDA decoding")
    if model["vae_conv_backend"] != "xla" and model["vae_precision"] == "bfloat16":
        raise ValueError("Optimized VAE convolution supports FP32 or float16")
    if model["attention_backend"] not in ("native", "cudnn"):
        raise ValueError("model.attention_backend must be native or cudnn")
    if model["dit_autotune_level"] not in (0, 4):
        raise ValueError("model.dit_autotune_level must be 0 or 4")
    if model["dit_autotune_level"] and r["platform"] != "cuda":
        raise ValueError("DiT autotuning requires runtime.platform=cuda")
    if model["attention_backend"] == "cudnn":
        if r["platform"] != "cuda":
            raise ValueError("cuDNN attention requires runtime.platform=cuda; no fallback is performed")
        if model["dtype"] != "bfloat16":
            raise ValueError("cuDNN attention requires BF16 model activations")
    for section, key in (("model", "tiny_seed"), ("lora", "seed"), ("training", "seed")):
        if not 0 <= c[section][key] < 2**32:
            raise ValueError(f"{section}.{key} must be a uint32 integer")
    for section, key in (("model", "path"), ("data", "prompts"), ("data", "cache"),
                         ("runtime", "output_dir"), ("reward", "model_path"), ("reward", "python")):
        if not c[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty path")
    if not c["name"].strip():
        raise ValueError("name must be nonempty")
    for key in ("height", "width"):
        if s[key] <= 0 or s[key] % 16:
            raise ValueError(f"sampling.{key} must be a positive multiple of 16")
    if s["rollout_batch_size"] < 1 or c["training"]["microbatch_size"] < 1:
        raise ValueError("rollout_batch_size and microbatch_size must be positive")
    for key in ("steps", "group_size", "prompts_per_rollout"):
        if s[key] < 1:
            raise ValueError(f"sampling.{key} must be positive")
    if s["group_size"] < 2 or s["group_size"] % 2:
        raise ValueError("group_size must be even and at least two for two ART windows")
    for key in ("shift", "training_shift"):
        if s[key] <= 0:
            raise ValueError(f"sampling.{key} must be positive")
    if s["sampler"] not in ("euler", "res_multistep"):
        raise ValueError("sampling.sampler must be euler or res_multistep")
    if s["scheduler"] not in ("moss", "beta"):
        raise ValueError("sampling.scheduler must be moss or beta")
    if s["beta_alpha"] <= 0 or s["beta_beta"] <= 0:
        raise ValueError("sampling beta_alpha and beta_beta must be positive")
    if s["scheduler"] == "beta" and s["training_shift"] != 1.0:
        raise ValueError("sampling.training_shift is a legacy moss scheduler option; beta requires its default 1.0")
    if s["guidance_scale"] < 0:
        raise ValueError("sampling.guidance_scale must be nonnegative")
    layers = 42 if model["kind"] == "moss" else 2
    _integer_list(s["skip_layers"], "sampling.skip_layers", maximum=layers)
    _integer_list(s["train_timesteps"], "sampling.train_timesteps", maximum=s["steps"], nonempty=True)
    if t["rollouts_per_update"] < 1 or t["updates_per_rollout"] not in (1, 2):
        raise ValueError("rollouts_per_update must be positive; updates_per_rollout must be 1 or 2")
    boundary = t["updates_per_rollout"]
    if t["max_updates"] < boundary or t["max_updates"] % boundary:
        raise ValueError("max_updates must align with the optimizer-update boundary")
    for key in ("checkpoint_every", "evaluate_every"):
        if t[key] < 0 or t[key] % boundary:
            raise ValueError(f"training.{key} must align with the optimizer-update boundary")
    for key in ("max_no_signal_rollouts", "keep_checkpoints"):
        if t[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if r["platform"] not in ("cpu", "cuda"):
        raise ValueError("runtime.platform must be cpu or cuda")
    _integer_list(r["devices"], "runtime.devices", nonempty=True)
    if s["split_group_across_workers"]:
        if s["prompts_per_rollout"]!=1 or s["group_size"]%len(r["devices"]):
            raise ValueError("Split groups require one prompt and group_size divisible by worker count")
    elif s["prompts_per_rollout"] % len(r["devices"]):
        raise ValueError("prompts_per_rollout must divide evenly over worker count")
    if l["rank"] < 1 or l["alpha"] <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if l["preset"] not in ("attn", "attn-mlp"):
        raise ValueError("Supported LoRA presets are attn and attn-mlp")
    from .resolution_mixing import validate_mixing
    validate_mixing(c)
    from .optimizers import validate_optimizer
    validate_optimizer(o)
    if c["art"]["feedback"] not in ("scalar", "pairwise"):
        raise ValueError("feedback must be scalar or pairwise")
    if c["art"]["feedback"] == "pairwise" and s["group_size"] not in (2, 4, 8):
        raise ValueError("Pairwise feedback supports K=2,4,8")
    if any(c["art"][key] <= 0 for key in ("noise_level", "kl_threshold", "eps")):
        raise ValueError("ART noise level, threshold, and epsilon must be positive")
    if reward["kind"] not in ("toy_latent", "command", "siglip", "toy", "http", "pixai", "silva"):
        raise ValueError("unknown reward kind")
    if reward["kind"] == "pixai":
        if reward["metric"] not in ("coverage","iou") or reward["dtype"] not in ("bfloat16","float32") or reward["attention_backend"] not in ("native","cudnn"):
            raise ValueError("Invalid PixAI metric, dtype or attention backend")
        if reward["attention_backend"]=="cudnn" and (r["platform"]!="cuda" or reward["dtype"]!="bfloat16"):
            raise ValueError("PixAI cuDNN needs CUDA BF16")
    if reward["kind"] == "silva":
        if reward["metric"]!="calibrated" or not reward["backbone_path"].strip():
            raise ValueError("SILVA requires calibrated metric and backbone_path")
        if reward["dtype"] not in ("float32","bfloat16") or reward["attention_backend"] not in ("native","cudnn"):
            raise ValueError("Invalid SILVA dtype or attention backend")
        if reward["attention_backend"]=="cudnn" and (r["platform"]!="cuda" or reward["dtype"]!="bfloat16"):
            raise ValueError("SILVA cuDNN needs CUDA BF16")
    if reward["kind"] == "http":
        from .http_reward import validate_http_config
        validate_http_config(reward)
    if any(not isinstance(arg, str) or "\x00" in arg for arg in reward["command"]):
        raise ValueError("reward.command must be an argv list of strings without NUL characters")
    if reward["kind"] == "command" and (not reward["command"] or not reward["command"][0].strip()):
        raise ValueError("command reward needs a nonempty executable argv")
    if reward["timeout"] <= 0 or reward["batch_size"] < 1 or reward["threads"] < 1:
        raise ValueError("reward timeout, batch size and thread count must be positive")
    if reward["kind"] != "toy_latent" and not reward["decode"]:
        raise ValueError("Image rewards require reward.decode=true")
    if reward["kind"] == "siglip" and reward["device"] != "cpu":
        raise ValueError("Built-in SigLIP uses an isolated CPU process; use command for custom scheduling")
    if reward["kind"] == "siglip" and reward["metric"] not in ("cosine", "logit"):
        raise ValueError("reward.metric must be cosine or logit for SigLIP")
    if r["cpu_threads"] < 1 or not 0 < r["memory_fraction"] <= 1 or r["worker_timeout"] <= 0:
        raise ValueError("Invalid runtime resource budget")
    if r["min_free_disk_gib"] < 0:
        raise ValueError("runtime.min_free_disk_gib must be nonnegative")
    if r["busy_memory_mib"] < 0:
        raise ValueError("runtime.busy_memory_mib must be nonnegative")
    if model["kind"] == "moss" and reward["kind"] == "toy_latent" and t["max_updates"] > 4:
        raise ValueError("toy_latent is only a smoke reward; full-model smoke is limited to four updates")
    if model["kind"] == "moss" and model["dtype"] != "bfloat16":
        raise ValueError("The memory-bounded pretrained loader requires BF16 base weights")
    return c

def load_config(path, overrides=None):
    path = Path(path).resolve()
    content = read_mapping(path)
    # Legacy preparation limit aliases full; explicit data.context_length wins.
    if isinstance(content,dict) and "max_sequence_length" in content.get("preparation",{}) and "context_length" not in content.get("data",{}):
        content.setdefault("data",{})["context_length"]=content["preparation"]["max_sequence_length"]
    c = _merge(copy.deepcopy(DEFAULTS), {} if content is None else content)
    if overrides is not None:
        _merge(c, overrides)
    return validate(c)

def fingerprint(c):
    frozen = copy.deepcopy(c)
    # A disabled optional sampler must preserve fingerprints of old runs.
    if not frozen["sampling"].get("spectrum", {}).get("enabled", False):
        frozen["sampling"].pop("spectrum", None)
    frozen.pop("run", None)
    frozen.pop("preparation", None)
    frozen.pop("logging", None)
    # Moving an output directory or extending a finished run does not change the algorithm.
    frozen["runtime"].pop("output_dir", None)
    frozen["runtime"].pop("allow_busy", None)
    frozen["training"].pop("max_updates", None)
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
