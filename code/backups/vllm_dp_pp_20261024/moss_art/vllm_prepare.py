"""Config-driven, resumable offline vLLM preparation for ART prompt caches."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .data import CACHE_VERSION, PromptCache, atomic_json, load_prompts, model_identity, text_key
from .vllm_contract import CONDITIONING_CONTRACT, ENCODER


def select_devices(devices):
    """Resolve physical nvidia-smi indices to UUIDs before importing Torch."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {int(a.strip()): b.strip() for a, b in (line.split(",", 1) for line in rows.splitlines())}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapping[index] for index in devices)
    except KeyError as exc:
        raise ValueError(f"Unknown physical GPU index {exc.args[0]}") from exc


def prepare_cache(config):
    from .prompt_buckets import PromptBuckets, log_drop
    from .sft_data import write_npz
    from .vllm_qwen import VllmPrefill, tokenizer_for, text_tokens
    c = config
    directory = Path(c["data"]["cache"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    maximum = c["data"]["context_length"]
    records = load_prompts(c["data"]["prompts"], default_negative_prompt=c["data"]["negative_prompt"])
    if c["preparation"]["limit"]:
        records = records[:c["preparation"]["limit"]]
    tokenizer = tokenizer_for(c["model"]["path"])
    texts = list(dict.fromkeys(text.strip() for r in records for text in (r.prompt, r.negative_prompt)))
    ids_by_text = {text: text_tokens(tokenizer, text) for text in texts}
    kept, dropped, lengths = [], [], []
    for record in records:
        required = max(len(ids_by_text[text.strip()]) for text in (record.prompt, record.negative_prompt))
        lengths.append(required)
        if required > maximum:
            event = log_drop(record.id, required, maximum, "vllm_prepare_text")
            event["text_keys"] = [text_key(record.prompt), text_key(record.negative_prompt)]
            dropped.append(event)
        else:
            kept.append(record)
    buckets = PromptBuckets.resolve(lengths, c["data"]["short_context_length"], maximum)
    identity = model_identity(c["model"]["path"], full_weight_hash=c["preparation"]["full_weight_hash"])
    manifest_path = directory / "manifest.json"
    settings = dict(schema_version=CACHE_VERSION, encoder=ENCODER, model_identity=identity,
                    max_sequence_length=maximum, prefill_buckets={"short": buckets.short, "full": buckets.full},
                    conditioning_contract=CONDITIONING_CONTRACT)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, expected in settings.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Existing cache {key} differs; choose a new cache directory")
    else:
        manifest = {**settings, "embedding_storage_dtype": "float32 (from BF16)",
                    "padding": "vLLM packed prefill; stored valid prompt tokens only", "entries": {}}
        atomic_json(manifest_path, manifest)
    manifest["dropped_records"] = dropped
    atomic_json(manifest_path, manifest)
    (directory / "dropped.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dropped))
    texts = list(dict.fromkeys(text.strip() for r in kept for text in (r.prompt, r.negative_prompt)))
    cache = PromptCache(directory, model_dir=c["model"]["path"])
    missing = []
    for text in texts:
        if text_key(text) in manifest["entries"]:
            cache.load_text(text)
        else:
            missing.append(text)
    # Resume and fully prepared datasets must never allocate GPU model weights.
    missing.sort(key=lambda text: len(ids_by_text[text]))
    estimated_bytes = sum(len(ids_by_text[t]) * 4096 * 4 for t in missing)
    free_bytes = shutil.disk_usage(directory).free
    print(json.dumps({"event": "vllm_prefill_plan", "records": len(kept), "dropped": len(dropped),
                      "unique_texts": len(texts), "missing": len(missing),
                      "feature_bytes_uncompressed": estimated_bytes, "free_disk_bytes": free_bytes}), flush=True)
    if estimated_bytes > free_bytes:
        raise ValueError("Not enough free space for uncompressed conditioning cache")
    started = time.perf_counter()
    if missing:
        encoder = VllmPrefill(c["model"]["path"], c["preparation"]["vllm"],
                              directory / ".vllm-work", max_sequence_length=maximum)
        if "backend" in manifest and manifest["backend"] != encoder.metadata:
            raise ValueError("Partial-cache producer version/source changed; use a new cache directory")
        manifest["backend"] = encoder.metadata
        atomic_json(manifest_path, manifest)
        batch_size = c["preparation"]["vllm"]["batch_size"]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            arrays = encoder.encode_token_lists([ids_by_text[text] for text in batch])
            if len(arrays) != len(batch):
                raise RuntimeError("Incorrect prefill batch result count")
            for text, embedding in zip(batch, arrays):
                ids = ids_by_text[text][None]
                key = text_key(text)
                descriptor = write_npz(directory, f"{key}.npz", embedding=embedding,
                                       attention_mask=np.ones(ids.shape, dtype=bool), input_ids=ids)
                manifest["entries"][key] = {**descriptor, "shape": list(embedding.shape),
                                            "token_count": int(ids.shape[1]),
                                            "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
            atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "vllm_cache_progress", "encoded": offset + len(batch),
                              "total": len(missing)}), flush=True)
    # Keep the model inside this short-lived process; the caller starts DiT only
    # after process exit, which releases both vLLM workers and CUDA allocations.
    PromptCache(directory, model_dir=c["model"]["path"]).validate_prompts(kept)
    result = {"event": "text_cache_ready", "encoder": ENCODER, "cache_dir": str(directory),
              "records": len(kept), "dropped": len(dropped), "unique_texts": len(texts),
              "encoded_now": len(missing), "seconds": time.perf_counter() - started}
    atomic_json(directory / "preparation_report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    from .config import load_config
    c = load_config(args.config)
    if c["preparation"]["backend"] != "vllm":
        raise ValueError("vllm_prepare requires preparation.backend=vllm")
    select_devices(c["preparation"]["vllm"]["devices"])
    from .locking import output_lock
    with output_lock(c["data"]["cache"]):
        prepare_cache(c)


if __name__ == "__main__":
    main()
