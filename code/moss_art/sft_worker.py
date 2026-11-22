"""Native JAX BF16 flow-matching LoRA worker, with bounded true microbatches."""
import traceback
import numpy as np

def training_example(latent, seed, options):
    rng = np.random.default_rng(seed)
    if options["sigma_distribution"] == "logit_normal":
        value = np.clip(rng.normal(options["logit_mean"],options["logit_std"]),-30,30)
        sigma = 1/(1+np.exp(-value))
    else:
        sigma = rng.uniform()
    shift = options["sigma_shift"]
    sigma = np.float32(np.clip(shift*sigma/(1+(shift-1)*sigma),1e-5,1-1e-5))
    noise = rng.standard_normal(latent.shape,dtype=np.float32)
    target = noise-latent
    state = latent+sigma*target
    dropped = rng.random() < options["conditioning_dropout"]
    return state, target, sigma, dropped

def worker_main(conn, config, rank, device):
    from .runtime import configure_process
    configure_process(config,device)
    try:
        worker = SFTWorker(config,rank)
        conn.send({"result":worker.info()})
        while True:
            request = conn.recv()
            op = request.pop("op")
            if op == "close":
                break
            conn.send({"result":getattr(worker,op)(**request)})
    except EOFError:
        pass
    except BaseException as exc:
        conn.send({"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
    finally:
        conn.close()

class SFTWorker:
    def __init__(self,c,rank):
        import jax
        import jax.numpy as jnp
        from .model import MossModel
        from .sft_data import SFTCache
        self.c,self.rank = c,rank
        self.cache = SFTCache(c["data"]["cache"],require_targets=False)
        self.empty = self.cache.empty_context()
        from .prompt_buckets import configure_sft_cache
        self.buckets = configure_sft_cache(self.cache,c["data"],emit=rank==0)
        self.context_capacity = self.buckets.full
        options = dict(rank=c["lora"]["rank"],alpha=c["lora"]["alpha"],
            targets={"attn":"attention","attn-mlp":"all"}[c["lora"]["preset"]],
            attention_backend=c["model"]["attention_backend"],
            dit_autotune_level=c["model"]["dit_autotune_level"],
            offload_activations=c["model"]["activation_storage"]=="cpu",
            checkpoint_interval=c["model"]["checkpoint_interval"],
            tensor_parallel_size=c["runtime"]["tensor_parallel_size"],
            base_precision=c["model"]["base_precision"],mxfp8_dgrad=c["model"]["mxfp8_dgrad"])
        if c["model"]["kind"]!="tiny" and c["model"].get("mxfp8_weight_mode","online")=="offline":
            options.update(mxfp8_weight_mode="offline",mxfp8_cache_dir=c["model"]["mxfp8_cache_dir"],
                           mxfp8_cache_verify=c["model"].get("mxfp8_cache_verify",True))
        self.model = (MossModel.tiny(seed=c["model"]["tiny_seed"],dtype=getattr(jnp,c["model"]["dtype"]),**options)
                      if c["model"]["kind"]=="tiny" else MossModel.from_pretrained(c["model"]["path"],**options))
        self.lora = self.model.init_lora(c["lora"]["seed"])
        self.device = self.model.device
        self._compiled_shapes = set()

    def info(self):
        import jax
        tp=self.model.tensor_parallel_size
        return dict(rank=self.rank,device=str(self.device),devices=[str(d) for d in jax.local_devices()],
                    base_precision=self.model.base_precision,mxfp8_dgrad=self.model.mxfp8_dgrad,
                    base_storage_bytes=self.model.base_storage_bytes,mxfp8_modules=self.model.mxfp8_modules,
                    parallelism=dict(tensor_parallel_size=tp,hidden_per_device=self.model.config["dim"]//tp,
                                     heads_per_device=self.model.config["num_heads"]//tp),model_config=self.model.config,
                    lora={k:np.asarray(v) for k,v in self.lora.items()} if self.rank==0 else None)

    def set_lora(self,lora):
        import jax
        self.lora = {k:self.model._put(v) for k,v in lora.items()}
        jax.block_until_ready(self.lora)
        return {"loaded":len(self.lora)}

    def _manage_compilation_cache(self,state_shape,context_shape,limit=16):
        # Context buffers have fixed capacity; only image resolutions vary.
        # Dynamic valid lengths preserve visual tokens and original RoPE.
        # Between samples there is no live pullback, so old executables can
        # be released without dropping weights or accumulated host gradients.
        import jax
        import gc
        signature=(tuple(state_shape),tuple(context_shape))
        if signature not in self._compiled_shapes and len(self._compiled_shapes)>=limit:
            jax.clear_caches()
            gc.collect()
            self._compiled_shapes.clear()
        self._compiled_shapes.add(signature)

    def gradient(self,jobs):
        import jax
        import jax.numpy as jnp
        grads = {k:np.zeros(v.shape,np.float32) for k,v in self.lora.items()}
        losses, sigmas, dropped_count = [], [], 0
        # Keep job/seed order. Resolution changes flush a partial microbatch;
        # fixed prompt capacity allows different valid lengths in the same batch.
        batches=[]
        for job in jobs:
            row=self.cache.rows[job["index"]]
            shape=(row["height"],row["width"],self.buckets.capacity(row["sequence_length"]))
            if not batches or len(batches[-1][1]) >= self.c["training"]["microbatch_size"] or batches[-1][0] != shape:
                batches.append((shape,[]))
            batches[-1][1].append(job)
        batch_sizes=[]
        for batch_shape,batch_jobs in batches:
            capacity=batch_shape[2]
            states,targets,contexts,lengths,batch_sigmas=[],[],[],[],[]
            for job in batch_jobs:
                row = self.cache.rows[job["index"]]
                latent,context = self.cache.load(row)
                state,target,sigma,dropped = training_example(latent,job["seed"],self.c["sft"])
                if dropped:
                    context=self.empty
                from .conditioning import pad_context
                context,valid_lengths = pad_context(context,capacity)
                states.append(state);targets.append(target);contexts.append(context)
                lengths.append(valid_lengths);batch_sigmas.append(sigma)
                sigmas.append(float(sigma));dropped_count += int(dropped)
            state,target,context=[np.concatenate(values,axis=0) for values in (states,targets,contexts)]
            valid_lengths=np.concatenate(lengths)
            sigma=np.asarray(batch_sigmas,np.float32)
            batch_size=len(batch_jobs)
            batch_sizes.append(batch_size)
            self._manage_compilation_cache(state.shape,context.shape)
            state,context,target = [self.model._put(value) for value in (state,context,target)]
            # No CFG, reward, rollout, skipped layers or gradient through encoders.
            prediction,pullback = self.model.predict_with_pullback(self.lora,state,sigma,context,context_lengths=valid_lengths)
            residual = prediction.astype(jnp.float32)-target
            # Sum per-example means; the coordinator divides exactly once by
            # GLOBAL sample count, including partial microbatches.
            loss = jnp.sum(jnp.mean(residual**2,axis=(1,2,3)))
            grad = pullback((2.*batch_size/residual.size)*residual)
            loss_value=float(loss)
            if not np.isfinite(loss_value):
                raise FloatingPointError("Nonfinite SFT loss")
            for key,value in grad.items():
                host=np.asarray(value,np.float32)
                if not np.isfinite(host).all():
                    raise FloatingPointError(f"Nonfinite gradient: {key}")
                grads[key] += host
            losses.append(loss_value)
            del prediction,pullback,residual,grad,state,context,target
        return dict(gradients=grads,loss_sum=sum(losses),denominator=len(jobs),
                    sigmas=sigmas,dropped=dropped_count,microbatch_sizes=batch_sizes,memory_stats=self.device.memory_stats() or {},
                    device_memory_stats=[d.memory_stats() or {} for d in jax.local_devices()])

    def sample(self,index,seed):
        import jax
        import jax.numpy as jnp
        from .sampling import schedule, step_coefficients, sample_step
        row=self.cache.rows[index]
        _,context=self.cache.load(row,require_target=False)
        from .conditioning import pad_context
        capacity=self.buckets.capacity(max(context.shape[1],self.empty.shape[1]))
        context,valid_lengths=pad_context(context,capacity)
        negative,negative_lengths=pad_context(self.empty,capacity)
        context=self.model._put(context)
        negative=self.model._put(negative)
        s=self.c["sampling"]
        _,sigmas=schedule(s["steps"],scheduler=s["scheduler"],shift=s["shift"],
                         training_shift=s["training_shift"],
                         beta_alpha=s["beta_alpha"],beta_beta=s["beta_beta"])
        coefficients=step_coefficients(sigmas,sampler=s["sampler"])
        state=self.model._put(np.random.default_rng(seed).standard_normal(
            (1,self.model.latent_channels,row["height"]//16,row["width"]//16),dtype=np.float32))
        history=None
        for step in range(len(sigmas)-1):
            velocity=self.model.predict(self.lora,state,sigmas[step],context,cfg_scale=s["guidance_scale"],
                negative_context=negative,negative_skip_layers=tuple(s["skip_layers"]),
                context_lengths=valid_lengths,negative_context_lengths=negative_lengths)
            state,history=sample_step(state,velocity,coefficients[step],history)
            state.block_until_ready()
        return {"latent":np.asarray(state),"id":row["id"],"steps":len(sigmas)-1}
