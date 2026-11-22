"""Private persistent prefill subprocess; host processor communicates via NPZ."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

class PrefillClient:
    def __init__(self,python,model,vision_capacity):
        self.temporary=tempfile.TemporaryDirectory(prefix="moss-qwen-")
        self.directory=Path(self.temporary.name)
        try:
            self.process=subprocess.Popen([python,"-m","moss_art.qwen_service",model,str(vision_capacity)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            self._receive()
        except BaseException:
            self.close();raise
    def _receive(self):
        line=self.process.stdout.readline()
        if not line: raise RuntimeError(f"JAX Qwen process exited: {self.process.poll()}")
        reply=json.loads(line)
        if "error" in reply: raise RuntimeError(reply["error"])
        return reply
    def encode(self,ids,capacity,pixels=None,grids=()):
        source=self.directory/"request.npz";target=self.directory/"response.npy"
        np.savez(source,ids=ids,pixels=np.empty((0,)) if pixels is None else pixels,grids=np.asarray(grids,dtype=np.int32).reshape(-1,3))
        self.process.stdin.write(json.dumps(dict(source=str(source),target=str(target),capacity=capacity))+"\n")
        self.process.stdin.flush();self._receive()
        return np.load(target,allow_pickle=False)
    def close(self):
        if hasattr(self,"process"):
            if self.process.poll() is None:
                self.process.stdin.close()
                try:self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=30)
            self.process.stdout.close()
        self.temporary.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def main():
    import sys,traceback
    from .qwen_prefill import JaxPrefill
    try:
        encoder=JaxPrefill(sys.argv[1],vision_capacity=int(sys.argv[2]))
        print(json.dumps(dict(ready=True,device=str(encoder.device))),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            with np.load(request["source"],allow_pickle=False) as data:
                value=encoder.encode(data["ids"],request["capacity"],data["pixels"],data["grids"])
            np.save(request["target"],value)
            print(json.dumps(dict(ok=True,calls=encoder.calls)),flush=True)
    except BaseException:
        print(json.dumps(dict(error=traceback.format_exc())),flush=True)
        raise

if __name__=="__main__":main()
