"""Prepare reusable two-bucket native-JAX Qwen features on the encoder GPU.

python -m moss_art.prepare --config configs/moss_lora_4gpu.json
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from .data import CACHE_VERSION, PromptCache, atomic_json, load_prompts, model_identity, sha256_file, text_key


def prepare_cache(model_dir, prompts_path, cache_dir, *, max_sequence_length=4096,
                  short_sequence_length=0, default_negative_prompt="", full_weight_hash=False, limit=None):
    from .prompt_buckets import PromptBuckets,log_drop,validate_lengths
    from vendor.moss_jax.text_encoder import encode_prompt_tokens
    validate_lengths(short_sequence_length,max_sequence_length)
    records = load_prompts(prompts_path, default_negative_prompt=default_negative_prompt)
    if limit is not None:
        if limit<1: raise ValueError("limit must be positive")
        records=records[:limit]
    texts=list(dict.fromkeys(t.strip() for r in records for t in (r.prompt,r.negative_prompt)))
    # CPU-only complete tokenization BEFORE loading any weights.
    tokens={text:encode_prompt_tokens(model_dir,text,None) for text in texts}
    dropped=[];kept=[];lengths=[]
    for record in records:
        n=max(tokens[t.strip()][0].shape[1] for t in (record.prompt,record.negative_prompt))
        lengths.append(n)
        if n>max_sequence_length:
            event=log_drop(record.id,n,max_sequence_length,"prepare_text")
            event["text_keys"]=[text_key(record.prompt),text_key(record.negative_prompt)]
            dropped.append(event)
        else: kept.append(record)
    buckets=PromptBuckets.resolve(lengths,short_sequence_length,max_sequence_length)
    texts=list(dict.fromkeys(t.strip() for r in kept for t in (r.prompt,r.negative_prompt)))
    directory=Path(cache_dir);directory.mkdir(parents=True,exist_ok=True)
    identity=model_identity(model_dir,full_weight_hash=full_weight_hash)
    manifest_path=directory/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if (manifest.get("schema_version")!=CACHE_VERSION or manifest.get("encoder")!="moss-native-jax-qwen3.5"
            or manifest.get("model_identity",{}).get("fingerprint")!=identity["fingerprint"]
            or manifest.get("max_sequence_length")!=max_sequence_length
            or manifest.get("prefill_buckets")!={"short":buckets.short,"full":buckets.full}):
            raise ValueError("Existing cache has different encoder/settings; choose a new cache directory")
    else:
        manifest=dict(schema_version=CACHE_VERSION,encoder="moss-native-jax-qwen3.5",model_identity=identity,
            max_sequence_length=max_sequence_length,prefill_buckets=dict(short=buckets.short,full=buckets.full),
            embedding_storage_dtype="float32 (from BF16)",padding="two-shape prefill; stored valid tokens only",entries={})
        atomic_json(manifest_path,manifest)
    manifest["dropped_records"]=dropped
    directory.joinpath("dropped.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in dropped))
    cache=PromptCache(directory)
    missing=[]
    for text in texts:
        if text_key(text) in manifest["entries"]: cache.load_text(text)
        else: missing.append(text)
    if missing:
        from .qwen_prefill import JaxPrefill
        encoder=JaxPrefill(model_dir)
        for index,text in enumerate(missing):
            started=time.monotonic();ids,mask=tokens[text]
            embedding=encoder.encode(ids,buckets.capacity(ids.shape[1]))
            key=text_key(text)
            from .sft_data import write_npz
            descriptor=write_npz(directory,f"{key}.npz",embedding=embedding,attention_mask=mask.astype(bool),input_ids=ids)
            manifest["entries"][key]=dict(descriptor,shape=list(embedding.shape),token_count=int(ids.shape[1]),
                text=text,input_ids_sha256=__import__("hashlib").sha256(ids.tobytes()).hexdigest())
            atomic_json(manifest_path,manifest)
            print(json.dumps(dict(event="text_encoded",index=index+1,total=len(missing),tokens=int(ids.shape[1]),
                capacity=buckets.capacity(ids.shape[1]),seconds=round(time.monotonic()-started,2),key=key)),flush=True)
        del encoder;gc.collect()
    atomic_json(manifest_path,manifest)
    PromptCache(directory).validate_prompts(kept)
    result=dict(event="text_cache_ready",cache_dir=str(directory.resolve()),records=len(kept),dropped=len(dropped),
                unique_texts=len(texts),encoded_now=len(missing),model_fingerprint=identity["fingerprint"])
    print(json.dumps(result),flush=True)
    return result


def ensure_cache(c):
    """Fill missing immutable features in a GPU-4 child before DiT workers start.

    All rollout rounds read the same disk features; no per-sample Qwen pass.
    An existing incompatible cache fails explicitly rather than being overwritten.
    """
    if c["model"]["kind"]!="moss" or not c["preparation"]["auto_prepare"]: return
    import subprocess
    from .locking import output_lock
    directory=Path(c["data"]["cache"])
    with output_lock(directory):
        child=dict(c)
        # Public prepare command already handles process/GPU assignment.
        with tempfile.TemporaryDirectory(prefix="moss-prefill-") as temporary:
            config=Path(temporary)/"config.json"
            config.write_text(json.dumps(child))
            env=dict(os.environ)
            env.pop("JAX_PLATFORMS",None)
            subprocess.run([sys.executable,"-m","moss_art.prepare","--config",str(config)],env=env,check=True)

def main(argv=None):
    from .cli import main as cli_main
    import sys
    return cli_main(["prepare-text", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
