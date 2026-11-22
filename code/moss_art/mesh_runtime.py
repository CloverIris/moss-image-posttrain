"""One-controller, four-device JAX Mesh runtime. No training-worker IPC.

The MossModel is the graph; base/LoRA/optimizer values are explicit distributed
arrays. Compatibility result views preserve the old ART grouping/seed order.
"""
from __future__ import annotations
import copy,gc,json,os,time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .worker import NativeWorker
from .rollout_mixing import ROW_FIELDS


@partial(jax.jit,static_argnames=("noise","kl","eps","anchor_mode","reference_beta","return_details"))
def _loss_vjp_items(prediction,old,sigmas,directions,magnitudes,coefficients,anchors,valid,*,noise,kl,eps,anchor_mode="cached",
                    reference_prediction=None,reference_beta=0.,return_details=False):
    """Keep loss values and their VJP local to each data-parallel sample.

    Returning a globally summed scalar inside this JIT introduces a small
    cross-device all-reduce. On the deployed GPU runtime its first invocation
    corrupted parts of the normalized direction and the velocity cotangent.
    Retaining the per-item vectors avoids that execution path; an all-ones
    VJP is exactly the derivative of the original summed objective.
    """
    from .art import art_loss
    def objective(current):
        terms=art_loss(current,old,sigmas,directions,magnitudes,
            ode_velocity_coefficient=coefficients,restore_anchor_mask=anchors,
            q2_reference_sde_noise_level=noise,q2_reference_kl_threshold=kl,eps=eps,anchor_mode=anchor_mode,valid_mask=valid,
            reference_prediction=reference_prediction,reference_mse_beta=reference_beta)
        aux=(terms["masked"]&valid,terms["effective_keep_mask"]&valid)
        if return_details:
            aux += (terms["per_item_policy_loss"],terms["per_item_reference_mse"],terms["keep_mask"]&valid,
                    terms["per_item_policy_output_grad_sq"],terms["per_item_reference_output_grad_sq"],
                    terms["per_item_output_grad_dot"])
        return jnp.where(valid,terms["per_item_loss"],0),aux
    losses,pullback,aux=jax.vjp(objective,prediction,has_aux=True)
    return (losses,aux),pullback(jnp.ones_like(losses))[0]


def _loss_vjp(prediction,old,sigmas,directions,magnitudes,coefficients,anchors,valid,*,noise,kl,eps,anchor_mode="cached",
              reference_prediction=None,reference_beta=0.,return_details=False):
    # The caller already synchronizes these metrics per microbatch. Copy only
    # the tiny per-item vectors; cotangents and LoRA gradient reduction remain
    # on the GPU. No global XLA option or ART objective is changed.
    (losses,aux),cotangent=_loss_vjp_items(
        prediction,old,sigmas,directions,magnitudes,coefficients,anchors,valid,
        noise=noise,kl=kl,eps=eps,anchor_mode=anchor_mode,
        reference_prediction=reference_prediction,reference_beta=reference_beta,return_details=return_details)
    masked,kept=aux[:2]
    loss=float(np.sum(np.asarray(losses),dtype=np.float64))
    counts=(int(np.count_nonzero(np.asarray(masked))),int(np.count_nonzero(np.asarray(kept))))
    if return_details:
        policy,reference,policy_kept,policy_grad_sq,reference_grad_sq,grad_dot=aux[2:]
        real=np.asarray(valid,dtype=bool)
        policy_grad_sq=np.asarray(policy_grad_sq);reference_grad_sq=np.asarray(reference_grad_sq)
        details=dict(policy_loss_sum=float(np.sum(np.asarray(policy),dtype=np.float64)),
                     reference_mse_sum=float(np.sum(np.asarray(reference),dtype=np.float64)),
                     policy_active=int(np.count_nonzero(np.asarray(policy_kept))),
                     policy_output_grad_sq_sum=float(np.sum(policy_grad_sq,dtype=np.float64)),
                     reference_output_grad_sq_sum=float(np.sum(reference_grad_sq,dtype=np.float64)),
                     output_grad_dot_sum=float(np.sum(np.asarray(grad_dot),dtype=np.float64)),
                     policy_output_grad_zero_count=int(np.count_nonzero(real&(policy_grad_sq==0))),
                     reference_output_grad_zero_count=int(np.count_nonzero(real&(reference_grad_sq==0))))
        return (loss,counts),cotangent,details
    return (loss,counts),cotangent


class MeshPool:
    def __init__(self,c,initial_lora=None,base_precision="bf16"):
        self.config=copy.deepcopy(c)
        self.devices=jax.local_devices()
        self.size=len(c["runtime"]["devices"])
        if len(self.devices)!=self.size:raise ValueError("Visible devices must match Mesh configuration")
        self.config["_mesh_dp_size"]=self.size
        offline_rollout = base_precision=="mxfp8" and c["model"].get("mxfp8_weight_mode","online")=="offline"
        if c["model"]["kind"]=="moss" and not offline_rollout:
            from vendor.moss_jax.weights import load_host_params
            self.config["_mesh_host_params"]=load_host_params(c["model"]["path"],"transformer")
        self.workers=list(self.devices)  # logical data shards, never processes
        self.shards=None;self.prepared=None
        self._vae_decoder=None;self._vae_build_count=0
        self.parameter_version=0;self.anchor_version=None;self.anchor_cache=None;self.anchor_metrics={}
        self.worker=NativeWorker(self.config,0,initial_lora=initial_lora,base_precision=base_precision)
        self.parallel=self.worker.model.parallel
        info=self.worker.info()
        self.info=[dict(info,rank=i,device=str(device),lora=info["lora"] if i==0 else None) for i,device in enumerate(self.devices)]
        print(json.dumps(dict(event="mesh_runtime",pid=os.getpid(),devices=self.size,training_processes=1,gradient_reduction="device_collective")),flush=True)

    def memory_event(self,stage):
        from .memory import memory_snapshot
        print(json.dumps(dict(event="phase_memory",stage=stage,**memory_snapshot(self.devices))),flush=True)

    def place_state(self,state):
        return jax.tree.map(lambda x:jax.device_put(x,self.parallel.replicated) if hasattr(x,"dtype") else x,state)

    def _split_cache(self,cache):
        n=len(cache["endpoints"])
        result=[]
        offset=0
        for ids in np.array_split(np.arange(n),self.size):
            part=slice(offset,offset+len(ids));offset+=len(ids)
            local=dict(cache)
            for k in ("states","old","endpoints","groups"):local[k]=cache[k][part]
            local["contexts"]=cache["contexts"][part]
            for key in ROW_FIELDS:
                if key in cache: local[key]=cache[key][part]
            result.append(local)
        return result

    def call(self,op,payloads=None):
        ps=payloads if payloads is not None else [{} for _ in self.workers]
        if len(ps)!=self.size:raise ValueError("One logical payload per mesh shard required")
        if op=="rollout":
            self._ensure_precision("bf16" if self.config["sampling"].get("rollout_mode","bf16")=="bf16" else "mxfp8")
            self.ensure_rollout_vae()
            first=ps[0]
            if any([r["sample_id"] for r in p["records"]]!=[r["sample_id"] for r in first["records"]] for p in ps):
                raise ValueError("Mesh rollout currently broadcasts a complete prompt group")
            seeds=[seed for p in ps for seed in p["seeds"]]
            result=self.worker.rollout(first["records"],seeds,keep_cache=first.get("keep_cache",True),samples_per_prompt=len(seeds))
            identity=dict(parameter_version=self.parameter_version,rollout_precision=self.worker.model.base_precision,
                          precision_contract=self.worker.model.precision_contract())
            result.update(identity)
            if first.get("keep_cache",True):self.worker.cache.update(identity)
            parts=np.array_split(np.arange(len(seeds)),self.size)
            return [{key:(value[ids] if isinstance(value,np.ndarray) and key in ("endpoints","groups") else
                         [value[i] for i in ids] if key in ("seeds","prompts","sample_ids","target_tags",*ROW_FIELDS) else value)
                     for key,value in result.items()} for ids in parts]
        if op=="export_replay":return self._split_cache(self.worker.export_replay())
        if op=="install_replay":
            self.shards=[p["cache"] for p in ps]
            if self.worker is not None:self.worker.clear_replay()
            return [{"rows":len(v["endpoints"])} for v in self.shards]
        if op=="prepare_update":
            self.release_vae(reason="prepare_update")
            if self.shards is None:self.shards=self._split_cache(self.worker.export_replay())
            self.prepared=ps
            from .replay import replay_identity
            identity=replay_identity(self.shards)
            if "parameter_version" not in identity:raise ValueError("Mesh replay is missing its source parameter version")
            if identity["parameter_version"]!=self.parameter_version:
                raise ValueError("Replay parameter version does not match learner theta0")
            self._ensure_precision("bf16")
            self.anchor_cache=None;self.anchor_version=None;self.anchor_metrics={}
            if self.config["sampling"].get("rollout_mode")=="mxfp8_calibrated":self._prepare_anchors()
            return [dict(active_groups=int(np.asarray(p["stats"]["group_active"]).sum()),
                         groups=len(p["stats"]["group_active"]),items_per_window=[len(w) for w in p["windows"]]) for p in ps]
        if op=="gradient":
            if len({p["window"] for p in ps})!=1:raise ValueError("Mesh gradient windows must be synchronous")
            return [self._gradient(ps[0]["window"])]
        if op=="set_lora":
            self.worker.set_lora(ps[0]["lora"])
            self.parameter_version+=1
            return [{"loaded":len(self.worker.lora)} for _ in self.workers]
        if op=="clear_replay":
            self.shards=None;self.prepared=None
            self.anchor_cache=None;self.anchor_version=None;self.anchor_metrics={}
            result=self.worker.clear_replay()
            return [result for _ in self.workers]
        if op in ("bind_conditions","set_sampling_profile"):
            result=getattr(self.worker,op)(**ps[0])
            return [result for _ in self.workers]
        raise ValueError("Unsupported mesh operation: "+op)

    def _ensure_precision(self, precision):
        if precision=="bf16":self.release_vae(reason="bf16_learner")
        if self.worker is not None and self.worker.model.base_precision==precision:return
        started=time.monotonic()
        if self.worker is not None:self.pause()
        self.resume(base_precision=precision)
        print(json.dumps(dict(event="dit_precision_switch",precision=precision,parameter_version=self.parameter_version,
                              seconds=time.monotonic()-started,contract=self.worker.model.precision_contract())),flush=True)

    def _window_batches(self,window,*,include_inactive=False):
        """Same padded per-device microbatches for reference and learner forwards."""
        settings=self.config["art"];micro=self.config["training"]["microbatch_size"]
        buckets=[{} for _ in self.workers]
        for rank,(cache,payload) in enumerate(zip(self.shards,self.prepared)):
            stats=payload["stats"]
            for row,slot in payload["windows"][window]:
                group=int(cache["groups"][row]);direction=stats["group_directions"][group]
                magnitude=stats["group_guidance_magnitudes"][group];anchor=False
                if settings["feedback"]=="pairwise":
                    magnitude=stats["row_guidance_magnitudes"][row];anchor=bool(stats["row_restore_anchor"][row])
                if not include_inactive and not anchor and (magnitude==0 or not np.any(direction)):continue
                pair=cache["contexts"][row];key=(pair[0].shape[1],pair[1].shape[1])
                buckets[rank].setdefault(key,[]).append((int(row),int(slot),group,float(magnitude),anchor))
        keys=sorted({key for bucket in buckets for key in bucket})
        for key in keys:
            batches=max((len(bucket.get(key,[]))+micro-1)//micro for bucket in buckets)
            for batch in range(batches):
                arrays={k:[] for k in ("state","old","direction","sigma","coef","magnitude","anchor","valid","positive","negative","pl","nl")}
                locations=[]
                prototype=next(cache for cache in self.shards if len(cache["endpoints"]))
                for rank,(cache,payload) in enumerate(zip(self.shards,self.prepared)):
                    stats=payload["stats"];entries=buckets[rank].get(key,[])[batch*micro:(batch+1)*micro]
                    for index in range(micro):
                        if index<len(entries):
                            row,slot,group,magnitude,anchor=entries[index];step=int(cache["selected"][slot])
                            positive,negative=cache["contexts"][row];pl,nl=cache["condition_lengths"][group]
                            values=dict(state=cache["states"][row,slot],old=cache["old"][row,slot],direction=stats["group_directions"][group],
                                sigma=cache["sigmas"][step],coef=cache["ode_velocity_coefficients"][step],magnitude=magnitude,anchor=anchor,valid=True,
                                positive=np.asarray(positive)[0],negative=np.asarray(negative)[0],pl=int(pl[0]),nl=int(nl[0]))
                            locations.append((rank,row,slot))
                        else:
                            shape=prototype["endpoints"].shape[1:];text_dim=self.worker.model.text_dim
                            values=dict(state=np.zeros(shape,np.float32),old=np.zeros(shape,np.float32),direction=np.zeros(shape,np.float32),
                                sigma=.5,coef=-.01,magnitude=0.,anchor=False,valid=False,positive=np.zeros((key[0],text_dim),np.float32),
                                negative=np.zeros((key[1],text_dim),np.float32),pl=1,nl=1)
                            locations.append(None)
                        for name,value in values.items():arrays[name].append(value)
                bools={"anchor","valid"};ints={"pl","nl"}
                host={k:np.asarray(v,dtype=bool if k in bools else np.int32 if k in ints else np.float32) for k,v in arrays.items()}
                yield host,locations

    def _prepare_anchors(self):
        if any(len(p["windows"])!=2 for p in self.prepared):raise ValueError("Calibration needs A/B windows")
        self.anchor_version=self.parameter_version
        self.anchor_cache=[{} for _ in self.workers]
        started=time.monotonic();calls=rows=0;differences=[]
        for host,locations in self._window_batches(1,include_inactive=True):
            values={k:self.parallel.put(v) for k,v in host.items()}
            reference=self.worker.model.predict_reference(self.worker.lora,values["state"],values["sigma"],values["positive"],
                **self.worker._kwargs(values["negative"],(values["pl"],values["nl"])))
            reference=np.asarray(reference,dtype=np.float32).copy()
            if not np.isfinite(reference).all():raise FloatingPointError("Nonfinite theta0 BF16 anchor")
            for index,location in enumerate(locations):
                if location is None:continue
                rank,row,slot=location;value=reference[index].copy();value.setflags(write=False)
                self.anchor_cache[rank][(row,slot)]=value;rows+=1
                e=host["old"][index]-value
                # Common pre-mask set: diagnostics never select by algorithm-specific mask.
                direction=host["direction"][index]
                G=direction/max(float(np.sqrt(np.mean(direction**2))),self.config["art"]["eps"])*host["magnitude"][index]
                erms=float(np.sqrt(np.mean(e.astype(np.float64)**2)));grms=float(np.sqrt(np.mean(G.astype(np.float64)**2)))
                dot=float(np.mean(e.astype(np.float64)*G.astype(np.float64)))
                differences.append(dict(sigma=float(host["sigma"][index]),e_rms=erms,g_rms=grms,
                    e_over_g=erms/max(grms,1e-12),cosine=dot/max(erms*grms,1e-12),active=bool(grms>0 or host["anchor"][index])))
            calls+=1
            del reference,values
        self.anchor_diagnostics=differences
        active=[d for d in differences if d["active"]]
        self.anchor_metrics=dict(reference_seconds=time.monotonic()-started,reference_batches=calls,reference_rows=rows,
            reference_host_bytes=sum(v.nbytes for cache in self.anchor_cache for v in cache.values()),
            reference_parameter_version=self.anchor_version,
            e_over_g_mean=float(np.mean([d["e_over_g"] for d in active])) if active else 0.,
            e_g_cosine_mean=float(np.mean([d["cosine"] for d in active])) if active else 0.)
        print(json.dumps(dict(event="anchor_prepared",**self.anchor_metrics)),flush=True)

    def _gradient(self,window):
        self._ensure_precision("bf16")
        settings=self.config["art"]
        calibrated=self.config["sampling"].get("rollout_mode")=="mxfp8_calibrated"
        if calibrated:
            if self.anchor_cache is None:raise RuntimeError("B references must precede any A update")
            if window==0 and self.parameter_version!=self.anchor_version:raise RuntimeError("A self-anchor requires rollout theta0")
            if window==1 and self.parameter_version not in (self.anchor_version,self.anchor_version+1):raise RuntimeError("B learner must follow at most one A update")
        anchor_mode=("self" if window==0 else "calibrated") if calibrated else "cached"
        reference_beta=settings.get("reference_mse_beta",0.)
        denominator=sum(len(p["windows"][window]) for p in self.prepared)
        gradients=jax.tree.map(jnp.zeros_like,self.worker.lora)
        loss_sum=0.;masked=active=0
        policy_loss_sum=reference_mse_sum=reference_seconds=0.;policy_active=0
        output_grad_sums={k:0. for k in ("policy_output_grad_sq_sum","reference_output_grad_sq_sum","output_grad_dot_sum",
                                       "policy_output_grad_zero_count","reference_output_grad_zero_count")}
        output_grad_real_items=0
        # Calibrated B uses exactly the reference prepass batch layout, even for inactive rows.
        for host,locations in self._window_batches(window,include_inactive=calibrated or reference_beta>0):
            output_grad_real_items+=int(np.count_nonzero(host["valid"]))
            if calibrated and window==1:
                for index,location in enumerate(locations):
                    if location is not None:
                        rank,row,slot=location;host["old"][index]=self.anchor_cache[rank][(row,slot)]
            values={k:self.parallel.put(v) for k,v in host.items()}
            reference=None
            if reference_beta>0:
                started=time.monotonic()
                # Run before retaining the learner tape, bounding peak memory.
                # A/B contain disjoint points: consume this small batch directly
                # instead of keeping a second full replay cache on the host.
                reference=self.worker.model.predict_base(values["state"],values["sigma"],values["positive"],
                    **self.worker._kwargs(values["negative"],(values["pl"],values["nl"])))
                reference_seconds+=time.monotonic()-started
            prediction,pullback=self.worker.model.predict_with_pullback(self.worker.lora,values["state"],values["sigma"],values["positive"],
                **self.worker._kwargs(values["negative"],(values["pl"],values["nl"])))
            (loss,aux),cotangent,details=_loss_vjp(prediction,values["old"],values["sigma"],values["direction"],values["magnitude"],values["coef"],values["anchor"],values["valid"],
                noise=settings["noise_level"],kl=settings["kl_threshold"],eps=settings["eps"],anchor_mode=anchor_mode,
                reference_prediction=reference,reference_beta=reference_beta,return_details=True)
            if not np.isfinite(float(loss)):raise FloatingPointError("Nonfinite mesh loss")
            loss_sum+=float(loss);masked+=int(aux[0]);active+=int(aux[1])
            policy_loss_sum+=details["policy_loss_sum"];reference_mse_sum+=details["reference_mse_sum"]
            policy_active+=details["policy_active"]
            for key in output_grad_sums:output_grad_sums[key]+=details[key]
            if int(aux[1]):
                gradient=pullback(cotangent)
                gradients=jax.tree.map(lambda a,b:a+b,gradients,gradient)
                jax.block_until_ready(gradients)
                del gradient
            del prediction,pullback,cotangent,values,reference
        # Uncomputed inactive genuine items have zero cotangents in both terms.
        skipped_output_items=denominator-output_grad_real_items
        output_grad_sums["policy_output_grad_zero_count"]+=skipped_output_items
        output_grad_sums["reference_output_grad_zero_count"]+=skipped_output_items
        return dict(gradients=gradients,loss_sum=loss_sum,denominator=denominator,masked=masked,active=active,
                    policy_loss_sum=policy_loss_sum,reference_mse_sum=reference_mse_sum,policy_active=policy_active,
                    reference_beta=reference_beta,reference_seconds=reference_seconds,**output_grad_sums,
                    memory_stats={"peak_bytes_in_use":max((d.memory_stats() or {}).get("peak_bytes_in_use",0) for d in self.devices)})

    def pause(self):
        self.release_vae(reason="pause")
        if self.worker is None:return
        # Keep one LoRA state, one host replay and no serialized worker snapshots.
        from .replay import host_snapshot
        self.memory_event("before_dit_release")
        w=self.worker
        self.saved_state={key:host_snapshot(getattr(w,key)) for key in ("cache","buckets","text_cache","updates_seen")}
        self.saved_state["bound_conditions"]=getattr(w,"bound_conditions",{})
        self.saved_state["lora"]=w.lora
        w.model.release_base()
        self.worker=None
        del w
        gc.collect()
        self.memory_event("after_dit_release")

    def resume(self,base_precision="bf16"):
        if self.worker is not None:raise RuntimeError("Mesh model is already loaded")
        self.worker=NativeWorker(self.config,0,base_precision=base_precision,initial_lora=self.saved_state["lora"])
        self.worker.import_state(self.saved_state)
        del self.saved_state
        self.parallel=self.worker.model.parallel
        self.memory_event("after_dit_restore")

    def _new_vae_decoder(self,*,resident):
        from .evaluate import MeshLatentDecoder
        m=self.config["model"];started=time.monotonic()
        decoder=MeshLatentDecoder(m["path"],self.parallel,precision=m["vae_precision"],
            low_precision_stages=m["vae_low_precision_stages"],conv_backend=m["vae_conv_backend"],
            conv_workspace_mib=m["vae_conv_workspace_mib"])
        self._vae_build_count+=1
        parameter_bytes=sum(int(x.nbytes) for x in jax.tree.leaves(decoder.params))
        print(json.dumps(dict(event="vae_loaded",resident_rollout=resident,build_count=self._vae_build_count,
            parameter_bytes_per_device=parameter_bytes,devices=self.size,seconds=time.monotonic()-started)),flush=True)
        self.memory_event("after_rollout_vae_load" if resident else "after_vae_load")
        return decoder

    def ensure_rollout_vae(self):
        """Keep one replicated decoder beside the MXFP8 base for this rollout phase."""
        if not self.config["sampling"].get("vae_resident_rollout",False):return None
        if self.worker is None or self.worker.model.base_precision!="mxfp8":
            raise RuntimeError("Resident rollout VAE requires the active MXFP8 DiT")
        if self._vae_decoder is None:self._vae_decoder=self._new_vae_decoder(resident=True)
        return self._vae_decoder

    def release_vae(self,*,reason="phase_transition"):
        """Drop parameters and compiled decoder buffers before reward or BF16 training."""
        decoder=getattr(self,"_vae_decoder",None)
        if decoder is None:return
        self._vae_decoder=None
        started=time.monotonic()
        decoder.close()
        del decoder
        gc.collect()
        print(json.dumps(dict(event="vae_released",reason=reason,build_count=self._vae_build_count,
                              seconds=time.monotonic()-started)),flush=True)
        self.memory_event("after_rollout_vae_release")

    def decode(self,endpoints):
        sizes=[len(x) for x in endpoints]
        resident=self.worker is not None and self.config["sampling"].get("vae_resident_rollout",False)
        if resident:
            decoder=self.ensure_rollout_vae()
            images=decoder(np.concatenate(endpoints))
        else:
            # The original staged BF16 path constructs VAE only after DiT pause.
            with self._new_vae_decoder(resident=False) as decoder:
                images=decoder(np.concatenate(endpoints))
        offsets=np.cumsum([0]+sizes)
        self.memory_event("after_vae")
        return [dict(images=images[offsets[i]:offsets[i+1]],memory_stats=self.devices[i].memory_stats() or {}) for i in range(self.size)]

    def score(self,images,target_tags=None):
        self.release_vae(reason="reward")
        reward=self.config["reward"]
        if reward["kind"]=="silva":
            from .silva_jax import SilvaScorerJax
            model=SilvaScorerJax.from_pretrained(reward["model_path"],reward["backbone_path"],dtype=reward["dtype"],
                attention_backend=reward["attention_backend"],data_parallel_size=self.size)
            try:result=model.score(images,batch_size=reward["batch_size"])
            finally:model.close()
        elif reward["kind"]=="pixai":
            from .pixai_tagger import PixAITagger
            from .tag_reward import hard_scores
            model=PixAITagger.from_pretrained(reward["model_path"],metadata_dir=reward["metadata_dir"],dtype=reward["dtype"],
                attention_backend=reward["attention_backend"],data_parallel_size=self.size)
            try:
                probs=model.probabilities(images,batch_size=reward["batch_size"])
                result=hard_scores(probs,target_tags,reward["metadata_dir"],reward["metric"])
            finally:model.close()
        else:raise ValueError("Unsupported mesh reward: "+reward["kind"])
        self.memory_event("after_reward")
        return result

    def close(self):
        self.release_vae(reason="close")
        if self.worker is not None:
            self.worker.model.release_base()
            self.worker=None
        self.shards=None;self.prepared=None
        if hasattr(self,"saved_state"):del self.saved_state
        for info in self.info:info["lora"]=None
        self.config.pop("_mesh_host_params",None)
        gc.collect()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
