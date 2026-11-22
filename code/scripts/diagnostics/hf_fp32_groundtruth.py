"""Standalone streamed HF 5.20 FP32 oracle for public synthetic Qwen fixtures."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare(actual, reference):
    import numpy as np
    x, y = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Comparison shape mismatch: {x.shape} versus {y.shape}")
    difference = x - y
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    tx, ty = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    cosine = np.sum(tx * ty, axis=1) / np.maximum(
        np.linalg.norm(tx, axis=1) * np.linalg.norm(ty, axis=1), 1e-30)
    relative = np.linalg.norm(tx - ty, axis=1) / np.maximum(np.linalg.norm(ty, axis=1), 1e-30)
    return {
        "shape": list(x.shape), "cosine": float(np.sum(x * y) / max(nx * ny, 1e-30)),
        "relative_l2": float(np.linalg.norm(difference) / max(ny, 1e-30)),
        "mae": float(np.mean(np.abs(difference))), "max_abs": float(np.max(np.abs(difference))),
        "per_token_cosine": {"min": float(cosine.min()), "median": float(np.median(cosine)),
                             "p05": float(np.quantile(cosine, .05)), "last": float(cosine[-1]),
                             "worst_index": int(cosine.argmin())},
        "per_token_relative_l2": {"max": float(relative.max()), "median": float(np.median(relative)),
                                  "last": float(relative[-1]), "worst_index": int(relative.argmax())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    physical = {int(index.strip()): uuid.strip() for index, uuid in (row.split(",", 1) for row in rows.splitlines())}
    os.environ["CUDA_VISIBLE_DEVICES"] = physical[c["gpu"]]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5 import modeling_qwen3_5 as source
    from tokenizers import Tokenizer

    if transformers.__version__ != "5.20.1":
        raise ValueError(f"Expected installed HF 5.20.1 oracle, got {transformers.__version__}")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    model = Path(c["model"])
    directory = model / "text_encoder"
    cfg = Qwen3_5Config.from_pretrained(directory, local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    report = {
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "source_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_gpu": c["gpu"], "dtypes": c.get("dtypes", ["float32"]), "weights": "original BF16; promoted for FP32 oracle",
        "attention": "SDPA MATH; TF32 disabled", "use_cache": False,
        "method": "HF decoder layers loaded one at a time; original final RMSNorm; all tokens", "cases": [],
    }

    def read(key, *, to_gpu=True):
        with safe_open(directory / index[key], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        return tensor.to(device=device, dtype=compute_dtype) if to_gpu else tensor

    # Only public synthetic cases, hard-coded independently of any dataset.
    public = {
        "short": "prompt:\nA white cat on a blue chair.\nnegative prompt:\nwatermark",
        "long": "A quiet landscape with trees, distant hills, and warm sunlight. " * 36,
    }
    expected_lengths = {"short": 28, "long": 478}
    if c.get("synthetic_examples"):
        example_file = Path(c["synthetic_examples"]).resolve()
        if example_file != (Path(c["numeric_validation"]) / "examples.json").resolve():
            raise ValueError("Synthetic examples must accompany the numeric-validation fixture")
        representative = {row["id"]: row for row in json.loads(example_file.read_text())}
        for name, length in (("natural", 340), ("tags", 267)):
            row = representative[name]
            if row.get("images") or not isinstance(row.get("prompt"), str):
                raise ValueError("This oracle extension accepts only synthetic text representatives")
            public[name] = row["prompt"]
            expected_lengths[name] = length
        report["synthetic_examples_sha256"] = hashlib.sha256(example_file.read_bytes()).hexdigest()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for dtype_name, case_name in [(dtype_name, case_name) for dtype_name in c.get("dtypes", ["float32"])
                                      for case_name in c["cases"]]:
            if dtype_name not in ("float32", "bfloat16"):
                raise ValueError("Supported oracle precisions are float32 and bfloat16")
            compute_dtype = getattr(torch, dtype_name)
            suffix = "hf_fp32" if dtype_name == "float32" else "hf_bf16_stream"
            layer_dir = output / f"{case_name}_{suffix}_layers"
            layer_dir.mkdir(parents=True, exist_ok=True)
            if case_name not in public:
                raise ValueError("This standalone oracle accepts only synthetic short/long text fixtures")
            formatted = "<|im_start|>user\n" + public[case_name].strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ids = np.asarray(tokenizer.encode(formatted, add_special_tokens=True).ids, np.int32)
            fixture = Path(c["numeric_validation"]) / f"{case_name}_vllm.npz"
            with np.load(fixture, allow_pickle=False) as saved:
                np.testing.assert_array_equal(ids, saved["input_ids"])
            if len(ids) != expected_lengths[case_name]:
                raise ValueError("Unexpected public fixture token length")
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            embedding = read("language_model.embed_tokens.weight", to_gpu=False)
            x = embedding[torch.from_numpy(ids.astype(np.int64))[None]].to(device=device, dtype=compute_dtype)
            del embedding
            np.save(layer_dir / "embedding.npy", x.float().cpu().numpy(), allow_pickle=False)
            positions = torch.arange(len(ids), device=device)[None, None].expand(3, 1, -1)
            rotary = source.Qwen3_5TextRotaryEmbedding(cfg).to(device)
            positional = rotary(x, positions)
            causal = torch.zeros(1, 1, len(ids), len(ids), device=device, dtype=compute_dtype)
            causal.masked_fill_(torch.triu(torch.ones(len(ids), len(ids), device=device, dtype=torch.bool), 1),
                                torch.finfo(compute_dtype).min)
            details = []
            for layer_index in range(cfg.num_hidden_layers):
                prefix = f"language_model.layers.{layer_index}."
                loaded_at = time.perf_counter()
                state = {key[len(prefix):]: read(key) for key in index if key.startswith(prefix)}
                with torch.device("meta"):
                    layer = source.Qwen3_5DecoderLayer(cfg, layer_index)
                layer.load_state_dict(state, assign=True)
                layer.eval()
                torch.cuda.synchronize()
                load_seconds = time.perf_counter() - loaded_at
                computed_at = time.perf_counter()
                x = layer(x, position_embeddings=positional, position_ids=None,
                          attention_mask=causal if cfg.layer_types[layer_index] == "full_attention" else None,
                          past_key_values=None)
                torch.cuda.synchronize()
                detail = {"layer": layer_index, "type": cfg.layer_types[layer_index], "dtype": dtype_name,
                          "load_seconds": load_seconds, "forward_seconds": time.perf_counter() - computed_at,
                          "finite": bool(torch.isfinite(x).all()), "rms": float(torch.mean(x.square()).sqrt())}
                details.append(detail)
                print(json.dumps({"case": case_name, **detail}), flush=True)
                if not detail["finite"]:
                    raise ValueError("Nonfinite FP32 HF oracle hidden state")
                np.save(layer_dir / f"layer_{layer_index:02d}.npy", x.float().cpu().numpy(), allow_pickle=False)
                del layer, state
                gc.collect()
            with torch.device("meta"):
                norm = source.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            norm.load_state_dict({"weight": read("language_model.norm.weight")}, assign=True)
            actual = norm(x).float().cpu().numpy().copy()
            np.savez(output / f"{case_name}_{suffix}.npz", embedding=actual, input_ids=ids)
            result = {"case": case_name, "dtype": dtype_name, "tokens": len(ids), "shape": list(actual.shape), "all_finite": bool(np.isfinite(actual).all()),
                      "seconds": time.perf_counter() - started, "layers": details,
                      "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "comparisons": {}}
            paths = {"vllm_compiled_batch": Path(c["numeric_validation"]) / f"{case_name}_vllm.npz"}
            for label, suffix_path in c.get("reference_suffixes", {
                "hf_520_bf16": "_reference_520.npz", "hf_513_bf16": "_reference.npz",
            }).items():
                paths[label] = Path(c["numeric_validation"]) / f"{case_name}{suffix_path}"
            if case_name == "long":
                paths["vllm_eager_single"] = Path(c["eager_long"])
            for label, path in paths.items():
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        result["comparisons"][label] = compare(saved["embedding"], actual)
            report["cases"].append(result)
            atomic_json(output / "reference.json", report)
            print(json.dumps({"case_complete": case_name, **{k: v for k, v in result.items() if k != "layers"}}), flush=True)
            del x, norm, positional, rotary, causal
            gc.collect()
            torch.cuda.empty_cache()
    report["completed"] = True
    atomic_json(output / "reference.json", report)


if __name__ == "__main__":
    main()
