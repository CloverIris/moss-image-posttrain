"""Fresh Mesh startup, complete frozen-weight digest and synthetic forward."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    request=json.loads(Path(parser.parse_args().config).read_text())
    from moss_art.config import load_config
    from moss_art.runtime import preflight,configure_process
    c=load_config(request["training_config"])
    verify=request.get("verify",True)
    c["model"].update(mxfp8_weight_mode=request["mode"],mxfp8_cache_dir=request["cache_dir"],mxfp8_cache_verify=verify)
    preflight(c);configure_process(c,c["runtime"]["devices"])
    import jax
    import numpy as np
    from moss_art.mesh_runtime import MeshPool
    from moss_art.memory import memory_snapshot
    output=Path(request["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with MeshPool(c,base_precision="mxfp8") as pool:
        model=pool.worker.model
        loaded=time.monotonic()-started
        memory=memory_snapshot(pool.devices)
        h=hashlib.sha256()
        for key,value in sorted(model.base.items()):
            h.update((key+str(value.dtype)+str(value.shape)).encode())
            # Temporary copies avoid caching a second host model in JAX arrays.
            temporary=value.copy()
            array=np.asarray(temporary)
            h.update(array.tobytes())
            del temporary,array
        digest=h.hexdigest()
        rng=np.random.default_rng(29);batch=pool.size
        shape=(batch,model.latent_channels,request.get("resolution",768)//16,request.get("resolution",768)//16)
        state=pool.parallel.put(rng.normal(size=shape).astype(np.float32))
        context=pool.parallel.put(rng.normal(0,.1,size=(batch,288,model.text_dim)).astype(np.float32))
        negative=pool.parallel.put(np.zeros((batch,288,model.text_dim),np.float32))
        sigma=pool.parallel.put(np.full(batch,.6,np.float32))
        lengths=pool.parallel.put(np.full(batch,288,np.int32));nl=pool.parallel.put(np.full(batch,64,np.int32))
        before=time.monotonic()
        prediction=model.predict(pool.worker.lora,state,sigma,context,cfg_scale=3.,negative_context=negative,
                                 context_lengths=lengths,negative_context_lengths=nl)
        result=np.asarray(prediction).copy()
        if not np.isfinite(result).all():raise FloatingPointError("Nonfinite cache parity forward")
        np.save(output/'prediction.npy',result,allow_pickle=False)
        report=dict(mode=request["mode"],load_seconds=loaded,frozen_parameter_sha256=digest,
            base_storage_bytes=model.base_storage_bytes,quantized_modules=model.mxfp8_modules,
            precision_contract=model.precision_contract(),memory=memory,
            forward_seconds=time.monotonic()-before,prediction_shape=list(result.shape),
            synthetic_inputs=True,offline_checksum_verification=verify,
            cpu_bf16_host_map_retained="_mesh_host_params" in pool.config,devices=c["runtime"]["devices"])
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
