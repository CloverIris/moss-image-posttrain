"""Offline vLLM prefill adapter for the original MOSS Qwen3.5 encoder.

No JAX imports. A small model view symlinks existing weights. The dedicated
vLLM plugin maps their names, ties an unused sampling head, and exports native
final-normalized states through the official hidden-state connector.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .data import atomic_json, model_identity
from .vllm_contract import CONDITIONING_CONTRACT, format_prompt

ARCHITECTURE = "MossQwen3_5ForConditionalGeneration"


def configure_tool_path():
    """An absolute Python executable does not activate its environment's tools."""
    paths = [str(Path(sys.executable).parent)]
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    if cuda.is_dir():
        paths.append(str(cuda))
    paths.extend(os.environ.get("PATH", "").split(os.pathsep))
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(paths))
    if shutil.which("ninja") is None:
        raise RuntimeError("vLLM kernel compilation requires ninja in the selected Python environment")


def make_model_view(model_dir, destination, processor_dir):
    """Construct only metadata/symlinks; never rewrite or duplicate weights."""
    model_dir, destination = Path(model_dir).resolve(), Path(destination).resolve()
    identity = model_identity(model_dir)
    stamp = {"version": 1, "source_fingerprint": identity["fingerprint"],
             "architecture": ARCHITECTURE, "normalization": "native_final_norm",
             "dummy_head": "tied_embedding_only_for_discarded_one_token_output"}
    marker = destination / "moss_vllm_view.json"
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != stamp:
            raise ValueError("Model view differs from the MOSS source; choose an empty view directory")
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "text_encoder/config.json").read_text())
    text = config["text_config"]
    if (config.get("model_type") != "qwen3_5" or text.get("num_hidden_layers") != 32
            or text.get("hidden_size") != 4096):
        raise ValueError("This MOSS adapter requires the 32-layer, 4096-hidden Qwen3.5 encoder")
    config = copy.deepcopy(config)
    config["architectures"] = [ARCHITECTURE]
    config["tie_word_embeddings"] = True
    config["text_config"]["tie_word_embeddings"] = True
    atomic_json(destination / "config.json", config)

    def link(source):
        name = source.name
        source = source.resolve()
        target = destination / name
        if target.is_symlink() and target.resolve() == source:
            return
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to replace model-view file {target}")
        target.symlink_to(source)

    weights = sorted((model_dir / "text_encoder").glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("No MOSS text encoder safetensors found")
    for source in [*weights, *(model_dir / "text_encoder").glob("*.safetensors.index.json"),
                   *(model_dir / "tokenizer").glob("*.json"),
                   *(model_dir / "tokenizer").glob("*.jinja")]:
        link(source)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        source = Path(processor_dir) / name
        if not source.is_file():
            raise FileNotFoundError(source)
        link(source)
    atomic_json(marker, stamp)
    return destination


def tokenizer_for(model_dir):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def text_tokens(tokenizer, prompt):
    return np.asarray(tokenizer.encode(format_prompt(prompt), add_special_tokens=True).ids,
                      dtype=np.int32)


def unpack_features(payload, expected_ids):
    """Check exact token alignment before producing an unpadded MOSS condition."""
    import torch
    ids = payload["token_ids"].detach().cpu().numpy().astype(np.int32)
    expected_ids = np.asarray(expected_ids, dtype=np.int32).reshape(-1)
    if ids.ndim != 1 or not np.array_equal(ids, expected_ids):
        raise ValueError("vLLM exported token ids differ from the original MOSS processor/tokenizer")
    hidden = payload["hidden_states"]
    if hidden.ndim != 3 or tuple(hidden.shape) != (len(ids), 1, 4096):
        raise ValueError(f"Unexpected extracted hidden shape: {tuple(hidden.shape)}")
    if hidden.dtype != torch.bfloat16:
        raise ValueError("MOSS prefill must export BF16 normalized features")
    embedding = hidden[:, 0, :].float().cpu().numpy()[None].copy()
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite vLLM condition")
    if not np.any(embedding):
        raise ValueError("All-zero vLLM features: hidden-state extraction did not produce a valid MOSS condition")
    return embedding


def unpack_token_embeddings(output, expected_ids):
    """Native token_embed: ALL tokens, final RMSNorm, no pooler activation."""
    import torch
    ids = np.asarray(output.prompt_token_ids, dtype=np.int32)
    if not np.array_equal(ids, np.asarray(expected_ids, dtype=np.int32)):
        raise ValueError("vLLM token_embed IDs differ from MOSS tokenization")
    hidden = output.outputs.data
    if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (len(ids), 4096):
        raise ValueError("vLLM token_embed must return all [sequence,4096] prompt features")
    if hidden.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Unexpected token_embed output dtype")
    value = hidden.detach().float().cpu().numpy()[None].copy()
    if not np.isfinite(value).all() or not np.any(value):
        raise ValueError("Nonfinite or all-zero token_embed features")
    return value


class VllmPrefill:
    def __init__(self, model_dir, config, work_dir, *, max_sequence_length):
        configure_tool_path()
        self.model_dir = Path(model_dir).resolve()
        self.config = dict(config)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer_for(model_dir)
        self.processor = None
        self.max_sequence_length = max_sequence_length
        from vllm import LLM, ModelRegistry
        # Plugin entry point repeats this lazy registration in spawned workers.
        ModelRegistry.register_model(ARCHITECTURE, "vllm_moss_plugin.model:" + ARCHITECTURE)
        view = make_model_view(model_dir, config.get("model_path") or self.work_dir / "model-view",
                               config["processor_dir"])
        max_tokens = int(config["max_num_batched_tokens"])
        if max_tokens < max_sequence_length + 1:
            raise ValueError("max_num_batched_tokens must fit full prompt length + one discarded token")
        options = dict(
            model=str(view), dtype="bfloat16", trust_remote_code=False,
            tensor_parallel_size=config["tensor_parallel_size"],
            pipeline_parallel_size=config.get("pipeline_parallel_size", 1),
            distributed_executor_backend="mp",
            max_model_len=max_sequence_length + 1,
            max_num_batched_tokens=max_tokens, max_num_seqs=config["max_num_seqs"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            enforce_eager=config["enforce_eager"],
            mm_processor_kwargs={"size": {"shortest_edge": 65536,
                                         "longest_edge": config["max_reference_pixels"]}},
            enable_prefix_caching=False, enable_chunked_prefill=False,
        )
        self.native_pooling = config.get("pipeline_parallel_size", 1) > 1
        if self.native_pooling:
            from vllm.config import PoolerConfig
            # The native pooling runner supports PP. It returns the same model
            # final-normalized states without a fake draft or sampling token.
            options.update(runner="pooling", convert="embed",
                           pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
                           hf_overrides={"head_dtype": "model"})
        else:
            from vllm.config.kv_transfer import KVTransferConfig
            options.update(speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [32]}}},
                           kv_transfer_config=KVTransferConfig(
                kv_connector="MossHiddenStatesConnector", kv_role="kv_producer",
                kv_connector_module_path="vllm_moss_plugin.connector",
                kv_connector_extra_config={"shared_storage_path": str(self.work_dir / "raw-hidden"),
                                           "use_synchronization_lock": True,
                                            "num_writer_threads": 1}))
        self.llm = LLM(**options)
        self.metadata = {
            "vllm_version": importlib.metadata.version("vllm"),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "architecture": ARCHITECTURE, "model_view": str(view),
            "normalization_location": "native_vllm_final_norm_in_model_wrapper",
            "feature_api": "token_embed" if self.native_pooling else "extract_hidden_states",
            "include_output_tokens": False, "max_tokens": 0 if self.native_pooling else 1,
            "kv_connector": None if self.native_pooling else "MossHiddenStatesConnector",
            "compute_config": {
                "dtype": "bfloat16",
                "tensor_parallel_size": config["tensor_parallel_size"],
                "pipeline_parallel_size": config.get("pipeline_parallel_size", 1),
                "enforce_eager": config["enforce_eager"],
                "max_sequence_length": max_sequence_length,
                "max_num_batched_tokens": max_tokens,
                "max_num_seqs": config["max_num_seqs"],
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "max_reference_pixels": config["max_reference_pixels"],
            },
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        plugin_source = importlib.util.find_spec("vllm_moss_plugin.model").origin
        self.metadata["plugin_sha256"] = hashlib.sha256(Path(plugin_source).read_bytes()).hexdigest()
        if not self.native_pooling:
            connector_source = importlib.util.find_spec("vllm_moss_plugin.connector").origin
            self.metadata["connector_sha256"] = hashlib.sha256(Path(connector_source).read_bytes()).hexdigest()
        if config.get("pipeline_parallel_size", 1) > 1 and not self.native_pooling:
            pp_source = importlib.util.find_spec("vllm_moss_plugin.pp_compat").origin
            self.metadata["pp_compat_sha256"] = hashlib.sha256(Path(pp_source).read_bytes()).hexdigest()

    def encode_token_lists(self, token_lists):
        requests = [{"prompt_token_ids": np.asarray(ids, dtype=np.int32).reshape(-1).tolist()}
                    for ids in token_lists]
        return self._generate(requests, token_lists)

    def _generate(self, requests, expected_ids):
        if self.native_pooling:
            from vllm import PoolingParams
            if not requests:
                return []
            if any(not 0 < len(ids) <= self.max_sequence_length for ids in expected_ids):
                raise ValueError("Prefill length exceeds full context limit; truncate is forbidden")
            started = time.perf_counter()
            outputs = self.llm.encode(requests, pooling_task="token_embed",
                                      pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
            if len(outputs) != len(expected_ids):
                raise ValueError("vLLM returned an unexpected token_embed request count")
            features = [unpack_token_embeddings(out, ids) for out, ids in zip(outputs, expected_ids)]
            print(json.dumps({"event": "vllm_prefill_batch", "feature_api": "token_embed",
                              "requests": len(requests), "tokens": sum(map(len, expected_ids)),
                              "seconds": time.perf_counter() - started}), flush=True)
            return features
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            load_hidden_states, cleanup_hidden_states,
        )
        if not requests:
            return []
        if any(not 0 < len(ids) <= self.max_sequence_length for ids in expected_ids):
            raise ValueError("Prefill length exceeds full context limit; truncate is forbidden")
        params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True,
                                extra_args={"kv_transfer_params": {"include_output_tokens": False}})
        started = time.perf_counter()
        outputs = self.llm.generate(requests, params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise ValueError("vLLM returned an unexpected request count")
        features = []
        for output, ids in zip(outputs, expected_ids):
            if output.prompt_token_ids != np.asarray(ids, dtype=np.int32).reshape(-1).tolist():
                raise ValueError("vLLM request output token ids do not match MOSS tokenization")
            path = (output.kv_transfer_params or {}).get("hidden_states_path")
            if not path:
                raise RuntimeError("vLLM did not return a hidden_states_path")
            payload = load_hidden_states(path)
            try:
                features.append(unpack_features(payload, ids))
            finally:
                cleanup_hidden_states(path)
        print(json.dumps({"event": "vllm_prefill_batch", "requests": len(requests),
                          "tokens": sum(len(x) for x in expected_ids),
                          "seconds": time.perf_counter() - started}), flush=True)
        return features

    def encode_requests(self, requests):
        """Public image+text helper; preserves all visual and textual prompt tokens."""
        from PIL import Image, ImageOps
        if self.processor is None:
            from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor, Qwen3VLProcessor
            processor_dir = self.config["processor_dir"]
            self.processor = Qwen3VLProcessor(
                image_processor=AutoImageProcessor.from_pretrained(processor_dir, local_files_only=True),
                video_processor=AutoVideoProcessor.from_pretrained(processor_dir, local_files_only=True),
                tokenizer=AutoTokenizer.from_pretrained(self.model_dir / "tokenizer", local_files_only=True),
            )
        prompts, expected = [], []
        for request in requests:
            text = request.get("prompt", "")
            refs = request.get("images", [])
            if not refs:
                ids = text_tokens(self.tokenizer, text)
                prompts.append({"prompt_token_ids": ids.tolist()})
            else:
                images = []
                for path in refs:
                    with Image.open(path) as im:
                        images.append(ImageOps.exif_transpose(im).convert("RGB").copy())
                rendered = ("<|im_start|>user\n" + "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
                            + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
                kwargs = {"size": {"shortest_edge": 65536,
                                   "longest_edge": self.config["max_reference_pixels"]}}
                processed = self.processor(text=rendered, images=images, return_tensors="pt", **kwargs)
                ids = processed["input_ids"][0].numpy().astype(np.int32)
                prompts.append({"prompt": rendered, "multi_modal_data": {"image": images},
                                "mm_processor_kwargs": kwargs})
            expected.append(ids)
        return self._generate(prompts, expected), expected
