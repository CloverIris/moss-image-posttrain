"""Command line interface; no JAX import until accelerator assignment."""
import argparse
import json
from pathlib import Path
from .config import load_config
from .runtime import preflight

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("rollout", "rollout-only"):
        from .rollout_only import main as rollout_main
        return rollout_main(argv[1:])
    if argv and argv[0] == "sft":
        from .sft import main as sft_main
        return sft_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "doctor", "prepare-text"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="JSON configuration file")
    args = parser.parse_args(argv)
    c = load_config(args.config)
    from .config_io import validate_run
    run=validate_run(c["run"],args.command)
    if args.command == "doctor":
        from .assets import experiment_identity
        from .trainer import read_records
        from importlib.metadata import version
        identity = experiment_identity(c)
        records = read_records(c)
        cache_entries = 0
        if c["model"]["kind"] == "moss" and not c["preparation"]["async_prefetch"]:
            from .data import PromptCache, PromptRecord
            cache = PromptCache(c["data"]["cache"], model_dir=c["model"]["path"])
            for row in records:
                cache.load(PromptRecord(id=row["sample_id"], prompt=row["prompt"],
                                        negative_prompt=row["negative_prompt"]))
                cache_entries += 1
        print(json.dumps({"configuration":"valid", "devices":preflight(c),
              "prompt_count":len(records), "validated_prompt_cache_entries":cache_entries,
              "identity_fingerprint":identity["fingerprint"],
              "versions":{name:version(name) for name in ("jax","jaxlib","optax","numpy","safetensors")}}, indent=2))
    elif args.command == "prepare-text":
        if c["preparation"]["backend"] == "vllm":
            from .prepare import launch_vllm_prepare
            return launch_vllm_prepare(c)
        from .runtime import configure_process
        p=c["preparation"]
        c["runtime"]["devices"] = [p["device"]]
        preflight(c)
        configure_process(c, p["device"])
        from .prepare import prepare_cache
        prepare_cache(c["model"]["path"],c["data"]["prompts"],c["data"]["cache"],
                      max_sequence_length=c["data"]["context_length"],
                      short_sequence_length=c["data"]["short_context_length"],
                      default_negative_prompt=c["data"]["negative_prompt"],
                      full_weight_hash=p["full_weight_hash"],limit=p["limit"] or None)
    elif args.command == "train":
        if c["staging"]["enabled"]:
            from .staged import run
            return run(c)
        from .trainer import train
        train(c, run["resume"] or None, run["init_adapter"] or None)
    else:
        from .trainer import evaluate_checkpoint, evaluate_adapter
        if run["adapter"]:
            evaluate_adapter(c, run["adapter"], run["decode"])
        else:
            evaluate_checkpoint(c, run["checkpoint"], run["decode"])

if __name__ == "__main__": main()
