"""Run identity and atomic outputs shared by the two GPU processes."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

BASE = Path("$MOSS_ROOT")
RUN = BASE / "distillation/ba_cnn_att_v2_1024_cfg3_20261103"
NATIVE = Path("$CODE_ROOT/moss-art-jax-warmup4")
REGISTRY = BASE / "datasets/next_training_dataset.json"
SOURCE = BASE / "runs/rank_sqrt_kawai_deepghs_w1_1_top1975v2_P24_K8_T4_1024_U200_calibrated_nol2_20261101_015727"
CAL = BASE / "calibration/conservative_v2_20261103_128_v1"
PYTHON = "$CONDA_PREFIX/envs/jax_env/bin/python"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def arrays(path, **data):
    if shutil.disk_usage(RUN).free < 10 * 1024**3:
        raise OSError("10 GiB NVMe reserve reached")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as stream:
        np.savez(stream, **data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def initialize():
    from random import Random
    registry = read(REGISTRY)
    policy = read(registry["training_precision_policy"])
    native = read(SOURCE / "config.json")
    train = CAL / "prospective_train_excluding_calibration_validation.jsonl"
    rows = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    train_rows = rows(train)
    ident = lambda row: row.get("id", row.get("sample_id"))
    calibration = set(read(CAL / "preparation_report.json")["validation_prompt_ids"])
    original = rows(registry["data"]["prompts"])
    assert [r for r in original if ident(r) not in calibration] == train_rows
    ids = [ident(r) for r in train_rows]
    plan = [r for r in read(SOURCE / "fixed_eval_5060/fixed_request_plan.json") if r["rollout_id"] == 0]
    withheld = {r["prompt_id"] for r in plan} | calibration
    for extra in registry["additional_evaluation_holdouts"]:
        withheld.update(extra["ids"])
    assert len(ids) == len(set(ids)) == 1971 and not set(ids) & withheld
    assert native["sampling"]["guidance_scale"] == 3 and native["sampling"]["skip_layers"] == [22]
    assert policy["fp8_training_required_rollout_mode"] == "mxfp8_calibrated"
    Random(20261103).shuffle(ids)
    cfg = dict(model=native["model"]["path"], cache=registry["data"]["cache"],
               train=str(train), eval=registry["fixed_eval"]["manifest"],
               train_sha=sha(train), eval_sha=sha(registry["fixed_eval"]["manifest"]),
               cache_manifest_sha=sha(Path(registry["data"]["cache"])/"manifest.json"),
               native_source=str(NATIVE), official_commit="c03aa8a92d7b98c419ba648175ed7a9ee6997829",
               sampling=dict(rollout_mode="bf16", guidance_scale=3., skip_layers=[22], shift=5., resolution=1024),
               reference_steps=48, nfe=[8,12], chain=4, updates_per_prompt=4,
               width=128, qwen_width=4096, lr=0.0003, seed=20261103,
               order=ids, eval_plan=plan, teacher_gpu=1, learner_gpu=2, teacher="frozen MOSS-Image-Base")
    path = RUN / "config.json"
    if path.exists():
        if read(path) != cfg:
            raise ValueError("Existing run has different frozen configuration")
    else:
        save(path,cfg)
    return cfg

