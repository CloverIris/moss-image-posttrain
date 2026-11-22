"""Explicit build of bounded cuDNN FFI, using installed dependencies only."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def find(component,suffix):
 for item in sys.path:
  p=Path(item)/"nvidia"/component/suffix
  if p.exists():return p.resolve()
 raise FileNotFoundError(f"Installed nvidia/{component}/{suffix} not found")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default=str(ROOT/"outputs/native"));a=p.parse_args()
 import jax
 source=ROOT/"moss_art/native/cudnn_conv_ffi.cc";digest=hashlib.sha256(source.read_bytes()).hexdigest()
 compiler=shutil.which("g++")
 if compiler is None:raise RuntimeError("g++ required; dependencies are not installed automatically")
 ci,cl,ui,ul=find("cudnn","include"),find("cudnn","lib"),find("cu13","include"),find("cu13","lib")
 ji=Path(jax.ffi.include_dir()).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
 binary=out/f"moss_cudnn_conv_{digest[:16]}.so"
 with (out/".build.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);fd,tmp=tempfile.mkstemp(prefix=".cudnn-",suffix=".so",dir=out);os.close(fd)
  command=[compiler,"-std=c++17","-O2","-shared","-fPIC",str(source),"-o",tmp,"-I"+str(ji),"-I"+str(ci),"-I"+str(ui),"-L"+str(cl),"-l:libcudnn.so.9","-L"+str(ul),"-l:libcudart.so.13","-Wl,-rpath,"+str(cl),"-Wl,-rpath,"+str(ul)]
  try:subprocess.run(command,check=True);os.replace(tmp,binary)
  finally:
   if Path(tmp).exists():Path(tmp).unlink()
  manifest={"schema":1,"built_utc":datetime.now(timezone.utc).isoformat(),"source":str(source),"source_sha256":digest,"binary":str(binary),"binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),"jax_version":jax.__version__,"compiler":compiler,"command":command,"headers":{"jax_ffi":str(ji),"cudnn":str(ci),"cuda":str(ui)},"libraries":{"cudnn":str(cl),"cuda":str(ul)}}
  pending=out/".build.json.tmp";pending.write_text(json.dumps(manifest,indent=2)+"\n");os.replace(pending,out/"build.json")
 print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
