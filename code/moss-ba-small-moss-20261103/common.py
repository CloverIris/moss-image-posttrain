"""Run identity and atomic artifacts for the fresh 0.1B MOSS-shaped student."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

BASE=Path("$MOSS_ROOT")
REFERENCE=BASE/"distillation/ba_cnn_att_v2_1024_cfg3_20261103"
RUN=Path(os.environ.get("MOSS_BA_RUN",str(BASE/"distillation/ba_smallmoss_102m_dp4_bf16_1024_20261103")))
NATIVE=Path("$CODE_ROOT/moss-art-jax-warmup4")
PYTHON="$CONDA_PREFIX/envs/jax_env/bin/python"

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.part")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)

def arrays(path,**data):
    if shutil.disk_usage(RUN).free<10*1024**3:raise OSError("10GiB NVMe reserve")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.part")
    with tmp.open("wb") as stream:
        np.savez(stream,**data);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
