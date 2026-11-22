"""Two-slot shared-memory Qwen producer. No feature queue or feature Pipe.

Parent: FREE -> REQUESTED; child: WRITING -> READY; parent: READING -> FREE
only after every DiT consumer acknowledges its private copy. Generation tags
reject stale leases. Slot payloads are bounded, independent of rollout count.
"""
import copy
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import time
import traceback
import numpy as np

FREE,REQUESTED,WRITING,READY,READING=range(5)
HEADER_BYTES=64


def enabled(c):return c["preparation"].get("async_prefetch",False)


def plan_records(c):
    from .data import load_prompts
    from .prompt_buckets import log_drop
    source=load_prompts(c["data"]["prompts"],default_negative_prompt=c["data"]["negative_prompt"])
    scanner=None
    if c["model"]["kind"]!="tiny":
        from .prompt_lengths import LengthScanner
        scanner=LengthScanner(c["model"]["path"])
    rows=[]
    for r in source:
        lengths=[scanner.tokens(t)["tokens"] for t in (r.prompt,r.negative_prompt)] if scanner else [3,2]
        if max(lengths)>c["data"]["context_length"]:
            log_drop(r.id,max(lengths),c["data"]["context_length"],"async_catalog")
            continue
        rows.append(dict(sample_id=r.id,prompt=r.prompt,negative_prompt=r.negative_prompt,
                         metadata=r.metadata,condition_token_counts=lengths))
    if not rows:raise ValueError("All prompts exceed full length")
    return rows


class CachedEncoder:
    def __init__(self,c,buckets):
        from .data import CACHE_VERSION,model_identity,PromptCache,atomic_json
        self.c=c;self.buckets=buckets;self.encoder=None;self.calls=0
        self.directory=Path(c["data"]["cache"])
        self.directory.mkdir(parents=True,exist_ok=True)
        identity=model_identity(c["model"]["path"],full_weight_hash=c["preparation"]["full_weight_hash"])
        path=self.directory/"manifest.json"
        if path.exists():
            manifest=json.loads(path.read_text())
            if (manifest.get("schema_version")!=CACHE_VERSION or manifest.get("encoder")!="moss-native-jax-qwen3.5"
                or manifest.get("model_identity",{}).get("fingerprint")!=identity["fingerprint"]
                or manifest.get("max_sequence_length")!=buckets.full
                or manifest.get("prefill_buckets")!={"short":buckets.short,"full":buckets.full}):
                raise ValueError("Async cache encoder/settings mismatch; select a new cache directory")
        else:
            manifest=dict(schema_version=CACHE_VERSION,encoder="moss-native-jax-qwen3.5",model_identity=identity,
                max_sequence_length=buckets.full,prefill_buckets=dict(short=buckets.short,full=buckets.full),
                embedding_storage_dtype="float32 (from BF16)",padding="two-shape prefill; stored valid tokens only",entries={})
            atomic_json(path,manifest)
        self.cache=PromptCache(self.directory)

    def encode(self,record,branch):
        from .data import text_key,atomic_json
        text=record["prompt" if branch==0 else "negative_prompt"]
        key=text_key(text)
        if key in self.cache.manifest["entries"]:
            return self.cache.load_text(text)[0],False
        if self.encoder is None:
            from .qwen_prefill import JaxPrefill
            self.encoder=JaxPrefill(self.c["model"]["path"],gpu_layers=self.c["preparation"].get("qwen_gpu_layers",16),gpu_embedding=self.c["preparation"].get("qwen_gpu_embedding",True),gpu_vision=self.c["preparation"].get("qwen_gpu_vision",True))
        from vendor.moss_jax.text_encoder import encode_prompt_tokens
        ids,mask=encode_prompt_tokens(self.c["model"]["path"],text,None)
        embedding=self.encoder.encode(ids,self.buckets.capacity(ids.shape[1]))
        from .sft_data import write_npz
        descriptor=write_npz(self.directory,f"{key}.npz",embedding=embedding,attention_mask=mask.astype(bool),input_ids=ids)
        self.cache.manifest["entries"][key]=dict(descriptor,shape=list(embedding.shape),token_count=int(ids.shape[1]),
            text=text.strip(),input_ids_sha256=__import__("hashlib").sha256(ids.tobytes()).hexdigest())
        atomic_json(self.directory/"manifest.json",self.cache.manifest)
        self.calls+=1
        return embedding,True


class TinyEncoder:
    """Same deterministic conditions as the tiny synchronous worker."""
    def __init__(self,c,buckets):self.cache={}
    def encode(self,record,branch):
        import hashlib
        key=record["prompt"] if branch==0 else ""
        if key not in self.cache:
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],"little")
            self.cache[key]=(np.random.default_rng(seed).normal(size=(1,3,24)).astype(np.float32)*.2 if branch==0
                             else np.zeros((1,2,24),np.float32))
            return self.cache[key],True
        return self.cache[key],False


def _producer(c,records,buckets,names,shape,condition,states,generations,counts,indices,lengths,metrics,stop,failed,error,initialized,factory):
    handles=[]
    try:
        from .runtime import configure_process
        configure_process(c,c["preparation"]["device"])
        from .locking import output_lock
        from contextlib import nullcontext
        with (output_lock(c["data"]["cache"]) if c["model"]["kind"]!="tiny" else nullcontext()):
            encoder=(factory or (TinyEncoder if c["model"]["kind"]=="tiny" else CachedEncoder))(c,buckets)
            handles=[shared_memory.SharedMemory(name=name,track=False) for name in names]
            buffers=[np.ndarray(shape,np.float32,buffer=h.buf,offset=HEADER_BYTES) for h in handles]
            initialized.set()
            while not stop.is_set():
                with condition:
                    condition.wait_for(lambda:stop.is_set() or REQUESTED in states,timeout=1)
                    if stop.is_set():break
                    pending=[i for i in range(2) if states[i]==REQUESTED]
                    if not pending:continue
                    slot=min(pending,key=lambda i:generations[i]);states[slot]=WRITING
                    generation=generations[slot];count=counts[slot]
                    ids=list(indices[slot*shape[0]:slot*shape[0]+count])
                start=time.monotonic();encoded=hits=0
                for i,index in enumerate(ids):
                    for branch in (0,1):
                        if stop.is_set():return
                        value,new=encoder.encode(records[index],branch)
                        n=value.shape[1]
                        if value.shape!=(1,n,shape[3]) or not 0<n<=shape[2] or not np.isfinite(value).all():
                            raise ValueError("Invalid prefill shape/values")
                        if n!=records[index]["condition_token_counts"][branch]:
                            raise ValueError("Tokenizer catalog disagrees with encoded condition")
                        buffers[slot][i,branch,:n]=value[0]
                        lengths[slot*shape[0]*2+i*2+branch]=n
                        encoded+=int(new);hits+=int(not new)
                end=time.monotonic()
                np.ndarray((8,),np.int64,buffer=handles[slot].buf)[:4]=[generation,count,shape[2],shape[3]]
                with condition:
                    if states[slot]!=WRITING or generations[slot]!=generation:raise RuntimeError("Slot ownership changed during write")
                    metrics[slot*4:slot*4+4]=[start,end,encoded,hits]
                    states[slot]=READY;condition.notify_all()
                print(json.dumps(dict(event="qwen_buffer_ready",slot=slot,generation=generation,
                    started=start,finished=end,seconds=end-start,encoded=encoded,cache_hits=hits)),flush=True)
    except BaseException:
        message=traceback.format_exc().encode()[:len(error)-1]
        error[:len(message)]=message;error[len(message)]=0
        failed.set();initialized.set()
        with condition:condition.notify_all()
    finally:
        if "buffers" in locals():del buffers
        for h in handles:h.close()


def read_buffer(descriptor):
    """Consumer must copy before acknowledging; no views escape this function."""
    shape=tuple(descriptor["shape"])
    handle=shared_memory.SharedMemory(name=descriptor["name"],track=False)
    try:
        header=np.ndarray((8,),np.int64,buffer=handle.buf)
        expected=[descriptor["generation"],descriptor["count"],shape[2],shape[3]]
        if list(header[:4])!=expected:raise RuntimeError("Stale Qwen buffer generation")
        source=np.ndarray(shape,np.float32,buffer=handle.buf,offset=HEADER_BYTES)
        result=[]
        for i,pair in zip(descriptor["positions"],descriptor["lengths"]):
            if not 0<=i<descriptor["count"] or any(not 0<n<=shape[2] for n in pair):raise ValueError("Invalid buffer extent")
            result.append(tuple(source[i,b,:n][None].copy() for b,n in enumerate(pair)))
        if list(header[:4])!=expected:raise RuntimeError("Qwen buffer changed during read")
        return result
    finally:
        if "source" in locals():del source
        if "header" in locals():del header
        handle.close()


class DoubleBuffer:
    def __init__(self,c,records,*,factory=None):
        from .prompt_buckets import PromptBuckets
        from .runtime import preflight
        self.c=c;self.records=records;self.handles=[];self.process=None
        self.timeout=c["runtime"]["worker_timeout"]
        self.buckets=PromptBuckets.resolve([max(r["condition_token_counts"]) for r in records],
            c["data"]["short_context_length"],c["data"]["context_length"])
        self.lookup={r["sample_id"]:i for i,r in enumerate(records)}
        if len(self.lookup)!=len(records):raise ValueError("Duplicate catalog IDs")
        hidden=24 if c["model"]["kind"]=="tiny" else json.loads((Path(c["model"]["path"])/"text_encoder/config.json").read_text())["text_config"]["hidden_size"]
        self.shape=(c["sampling"]["prompts_per_rollout"],2,self.buckets.full,hidden)
        size=HEADER_BYTES+int(np.prod(self.shape))*4
        if Path("/dev/shm").exists():
            import shutil
            if 2*size>shutil.disk_usage("/dev/shm").free:raise ValueError("Insufficient /dev/shm space for two prompt buffers")
        encoder_config=copy.deepcopy(c);encoder_config["runtime"]["devices"]=[c["preparation"]["device"]]
        preflight(encoder_config)
        ctx=mp.get_context("spawn")
        self.condition=ctx.Condition();self.states=ctx.Array("i",[FREE,FREE],lock=False)
        self.generations=ctx.Array("q",[0,0],lock=False);self.counts=ctx.Array("i",[0,0],lock=False)
        self.indices=ctx.Array("q",2*self.shape[0],lock=False)
        self.lengths=ctx.Array("i",4*self.shape[0],lock=False);self.metrics=ctx.Array("d",8,lock=False)
        self.stop=ctx.Event();self.failed=ctx.Event();self.initialized=ctx.Event()
        self.error=ctx.Array("B",16384,lock=False);self.next_generation=1;self.next_slot=0
        try:
            for _ in range(2):self.handles.append(shared_memory.SharedMemory(create=True,size=size))
            self.process=ctx.Process(target=_producer,args=(c,records,self.buckets,[h.name for h in self.handles],
                self.shape,self.condition,self.states,self.generations,self.counts,self.indices,self.lengths,self.metrics,
                self.stop,self.failed,self.error,self.initialized,factory),name="moss-qwen-prefill",daemon=False)
            self.process.start()
            self._wait(lambda:self.initialized.is_set())
        except BaseException:self.close();raise

    def _check(self):
        if self.failed.is_set():raise RuntimeError("Qwen producer failed: "+bytes(self.error[:]).split(b"\0")[0].decode(errors="replace"))
        if self.process is not None and not self.process.is_alive():raise RuntimeError(f"Qwen producer exited ({self.process.exitcode})")

    def _wait(self,predicate):
        deadline=time.monotonic()+self.timeout
        with self.condition:
            while not predicate():
                self._check()
                if time.monotonic()>=deadline:raise TimeoutError("Qwen double buffer timed out")
                self.condition.wait(timeout=min(.1,max(0,deadline-time.monotonic())))
            self._check()

    def submit(self,records):
        if not 0<len(records)<=self.shape[0]:raise ValueError("Invalid prefetch batch size")
        ids=[self.lookup[r["sample_id"]] for r in records]
        if any(any(r.get(k,"")!=self.records[i].get(k,"") for k in ("prompt","negative_prompt")) for r,i in zip(records,ids)):
            raise ValueError("Request differs from immutable prompt catalog")
        with self.condition:
            self._check()
            free=[i for i in range(2) if self.states[i]==FREE]
            if not free:raise RuntimeError("Both prompt buffers are owned; release a consumed batch first")
            slot=self.next_slot if self.next_slot in free else free[0]
            self.next_slot=1-slot;generation=self.next_generation;self.next_generation+=1
            self.indices[slot*self.shape[0]:slot*self.shape[0]+len(ids)]=ids
            self.counts[slot]=len(ids);self.generations[slot]=generation;self.states[slot]=REQUESTED
            self.condition.notify_all()
        return (slot,generation)

    def acquire(self,ticket):
        slot,generation=ticket;start=time.monotonic()
        def ready():
            if self.generations[slot]!=generation:raise RuntimeError("Stale prompt buffer ticket")
            if self.states[slot] in (READING,FREE):raise RuntimeError("Prompt ticket already consumed/released")
            return self.states[slot]==READY
        self._wait(ready)
        with self.condition:
            self.states[slot]=READING
            count=self.counts[slot]
            descriptor=dict(name=self.handles[slot].name,shape=self.shape,generation=generation,count=count,
                positions=list(range(count)),lengths=[list(self.lengths[slot*self.shape[0]*2+i*2:slot*self.shape[0]*2+i*2+2]) for i in range(count)])
            start_fill,end_fill,encoded,hits=self.metrics[slot*4:slot*4+4]
        event=dict(event="qwen_buffer_acquired",slot=slot,generation=generation,wait_seconds=time.monotonic()-start,
                   fill_started=start_fill,fill_finished=end_fill,encoded=int(encoded),cache_hits=int(hits))
        print(json.dumps(event),flush=True)
        return descriptor,event

    def release(self,ticket):
        slot,generation=ticket
        with self.condition:
            if self.generations[slot]!=generation or self.states[slot]!=READING:raise RuntimeError("Invalid buffer release")
            self.states[slot]=FREE;self.condition.notify_all()

    def bind(self,pool,records,ticket=None):
        if ticket is None:ticket=self.submit(records)
        descriptor,event=self.acquire(ticket)
        slot=ticket[0]
        expected=[self.lookup[r["sample_id"]] for r in records]
        if list(self.indices[slot*self.shape[0]:slot*self.shape[0]+len(records)])!=expected:
            raise ValueError("Condition lease belongs to another batch")
        count=len(records);workers=len(pool.workers)
        split=self.c["sampling"].get("split_group_across_workers",False)
        if split and count!=1:raise ValueError("Split-group condition broadcast requires one prompt")
        if not split and count%workers:raise ValueError("Condition batch must divide over workers")
        width=1 if split else count//workers;payloads=[]
        for i in range(workers):
            lo,hi=(0,1) if split else (i*width,(i+1)*width)
            view=dict(descriptor,positions=descriptor["positions"][lo:hi],lengths=descriptor["lengths"][lo:hi])
            payloads.append(dict(descriptor=view,records=records[lo:hi],buckets=[self.buckets.short,self.buckets.full]))
        # This synchronous ACK is the sole point that permits slot reuse.
        acknowledgements=pool.call("bind_conditions",payloads)
        if len(acknowledgements)!=workers or any(a.get("generation")!=ticket[1] or a.get("copied")!=width for a in acknowledgements):
            raise RuntimeError("Incomplete condition-buffer consumer acknowledgement")
        self.release(ticket)
        return event

    def close(self):
        if hasattr(self,"stop"):
            self.stop.set()
            with self.condition:self.condition.notify_all()
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=10)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
            if self.process.is_alive():self.process.kill();self.process.join(timeout=5)
            self.process=None
        for h in self.handles:
            h.close();h.unlink()
        self.handles=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
