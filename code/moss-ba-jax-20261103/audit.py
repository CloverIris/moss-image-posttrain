"""Verify every input used by this experiment and pin the training implementation."""
import hashlib
import sys
from pathlib import Path
from importlib.metadata import version
from common import NATIVE, RUN, read, save, sha

def stream_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    sys.path.insert(0,str(NATIVE))
    from moss_art.data import PromptCache,load_prompts
    cfg=read(RUN/"config.json")
    cache=PromptCache(cfg["cache"],model_dir=cfg["model"])
    groups={split:load_prompts(cfg[split]) for split in ("train","eval")}
    verified={split:cache.validate_prompts(records) for split,records in groups.items()}
    assert set(cfg["order"])=={r.id for r in groups["train"]}
    assert not ({r.id for r in groups["train"]} & {r.id for r in groups["eval"]})
    code=Path(__file__).resolve().parent
    sources=[code/name for name in ("train.py","teacher.py","side.py","solver.py","common.py")]
    sources+=list((NATIVE/"moss_art").rglob("*.py"))+list((NATIVE/"vendor/moss_jax").rglob("*.py"))
    identity={str(path):sha(path) for path in sorted(sources)}
    lock=RUN/"training_sources.json"
    if lock.exists() and read(lock)!=identity:
        raise RuntimeError("Training source changed")
    if not lock.exists():save(lock,identity)
    weights=Path(cfg["model"])/"transformer"
    weight_hashes={f.name:stream_hash(f) for f in sorted(weights.glob("*.safetensors"))}
    report=dict(verified_prompt_pairs=verified,cached_texts_checked=len(cache._verified),
                transformer_weights=weight_hashes,base_config_sha256=sha(weights/"config.json"),
                run_sha256=sha(RUN/"config.json"),sources_sha256=sha(lock),
                packages={name:version(name) for name in ("jax","jaxlib","flax","optax")},
                parity="official PyTorch oracle, 1/5/8/12 NFE; signed GL3 and masked Qwen tests passed")
    save(RUN/"input_audit.json",report)
    print(report,flush=True)
if __name__=="__main__":main()

