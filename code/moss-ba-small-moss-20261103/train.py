"""Training helpers shared by replica and coordinator."""
import os
import shutil
import numpy as np
from common import RUN

def atomic_bytes(path,data):
    if shutil.disk_usage(RUN).free<10*1024**3:raise OSError("10GiB NVMe reserve")
    tmp=path.with_suffix(".part")
    with tmp.open("wb") as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def sample_interval(rng,t,deploy):
    sign=1 if rng.random()<.5 else -1
    limit=t if sign>0 else 1-t
    if limit<1e-5:sign=-sign;limit=1-limit
    if rng.random()<.5:
        length=min(limit,float(rng.choice(deploy))*float(rng.uniform(.25,1.)))
    else:
        length=-np.log1p(rng.random()*np.expm1(-12.*limit))/12.
    return np.float32(sign*length)
