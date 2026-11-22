"""Fresh-process 1024 SFT activation storage benchmark, fixed identical jobs."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",type=int,required=True)
    p.add_argument("--tensor-parallel-size",type=int,choices=(1,2),default=1)
    p.add_argument("--storage",choices=("cpu","gpu"),required=True)
    p.add_argument("--interval",type=int,default=1)
    p.add_argument("--output",required=True)
    p.add_argument("--parity",action="store_true",help="Repeat the same job with CPU/CPU/GPU/GPU-segment checkpoints in one process")
    args=p.parse_args()
    from moss_art.sft import load_config
    from moss_art.runtime import configure_process,preflight
    c=load_config("configs/sft/smoke_1024.yaml")
    c["runtime"]["devices"]=list(range(args.device,args.device+args.tensor_parallel_size))
    c["runtime"]["tensor_parallel_size"]=args.tensor_parallel_size
    c["model"].update(activation_storage=args.storage,checkpoint_interval=args.interval)
    preflight(c)
    configure_process(c,args.device if args.tensor_parallel_size==1 else c["runtime"]["devices"])
    from moss_art.sft_worker import SFTWorker
    worker=SFTWorker(c,0)
    records=[]
    plan=[(args.storage,args.interval,step%2,123+step) for step in range(3)]
    if args.parity:
        plan=[("cpu",1,0,125),("cpu",1,0,125),("gpu",1,0,125),("gpu",6,0,125)]
    for step,(storage,interval,index,seed) in enumerate(plan):
        worker.model.offload_activations=storage=="cpu"
        worker.model.checkpoint_interval=interval
        start=time.monotonic()
        result=worker.gradient([{"index":index,"seed":seed}])
        seconds=time.monotonic()-start
        flat=np.concatenate([result["gradients"][k].ravel() for k in sorted(result["gradients"])])
        record=dict(step=step,storage=storage,interval=interval,seconds=seconds,loss=result["loss_sum"],gradient_norm=float(np.linalg.norm(flat)),
                    peak_bytes=result["memory_stats"].get("peak_bytes_in_use"),
                    current_bytes=result["memory_stats"].get("bytes_in_use"))
        records.append(record)
        print(json.dumps(record),flush=True)
        np.savez(Path(args.output).with_suffix(f".step{step}.npz"),gradient=flat)
        del result,flat
    report=dict(device=args.device,tensor_parallel_size=args.tensor_parallel_size,storage=args.storage,interval=args.interval,parity=args.parity,records=records,
                note="Fixed initial adapter, one sample per call, no optimizer; first call may compile.")
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__": main()
