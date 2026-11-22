"""Native JAX model worker. Environment is selected BEFORE importing JAX."""
from __future__ import annotations
import hashlib
from collections import OrderedDict
import traceback
import numpy as np

def worker_main(conn, config, rank, device):
    from .runtime import configure_process
    configure_process(config, device)
    try:
        worker = NativeWorker(config, rank)
        conn.send({"result": worker.info()})
        while True:
            request = conn.recv()
            op = request.pop("op")
            if op == "close":
                break
            result = getattr(worker, op)(**request)
            conn.send({"result": result})
    except EOFError:
        pass
    except BaseException as exc:
        try:
            conn.send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()

class NativeWorker:
    def __init__(self, config, rank, *, base_precision="bf16", initial_lora=None):
        import jax
        import jax.numpy as jnp
        from .model import MossModel
        self.c, self.rank = config, rank
        options = dict(data_parallel_size=config.get("_mesh_dp_size",1),rank=config["lora"]["rank"], alpha=config["lora"]["alpha"],
                       base_precision=base_precision,
                       offload_activations=config["model"]["activation_storage"] == "cpu",
                       checkpoint_interval=config["model"]["checkpoint_interval"],
                       attention_backend=config["model"]["attention_backend"],
                       dit_autotune_level=config["model"]["dit_autotune_level"])
        targets = {"attn": "attention", "attn-mlp": "all"}[config["lora"]["preset"]]
        if config["model"]["kind"] == "tiny":
            self.model = MossModel.tiny(seed=config["model"]["tiny_seed"], targets=targets, dtype=getattr(jnp, config["model"]["dtype"]), **options)
        else:
            targets = {"attn": "attention", "attn-mlp": "all"}.get(config["lora"]["preset"], config["lora"]["preset"])
            host = config.get("_mesh_host_params")
            if config.get("_host_base"):
                from .host_weights import attach
                self._host_memory, host = attach(config["_host_base"])
            self.model = MossModel.from_pretrained(config["model"]["path"], host_params=host, targets=targets, **options)
        self.device = jax.devices()[0]
        if initial_lora is None:self.lora=self.model.init_lora(config["lora"]["seed"])
        else:self.set_lora(initial_lora)
        self.cache = None
        # CPU embeddings are bounded; active rollout contexts have independent
        # JAX references and remain valid if their host cache entry is evicted.
        self.text_cache = OrderedDict()
        self._prompt_cache = None
        self._compiled_text_lengths = set()
        self.updates_seen = 0
        self.buckets = None

    def info(self):
        return {"rank": self.rank, "device": str(self.device),
                "activation_storage": "cpu" if self.model.offload_activations else "gpu",
                "checkpoint_interval": self.model.checkpoint_interval,
                "base_precision": self.model.base_precision,
                "attention_backend": self.model.attention_backend,
                "dit_autotune_level": self.model.dit_autotune_level,
                "sampling": {key: self.c["sampling"][key] for key in
                             ("sampler", "scheduler", "steps", "shift", "beta_alpha", "beta_beta")},
                "model_config": dict(self.model.config),
                "lora": (self.lora if self.c.get("_mesh_dp_size",1)>1 else {k: np.asarray(v) for k, v in self.lora.items()}) if self.rank == 0 else None}

    def export_state(self):
        import jax
        # Includes sampler replay state and text conditions, never model weights.
        state = {key: getattr(self, key) for key in
                 ("cache", "lora", "buckets", "text_cache", "updates_seen")}
        state["bound_conditions"] = getattr(self, "bound_conditions", {})
        from .replay import host_snapshot
        return host_snapshot(state)

    def import_state(self, state):
        self.set_lora(state["lora"])
        for key, value in state.items():
            if key != "lora": setattr(self, key, value)
        return {"restored": True}

    def set_sampling_profile(self, sampling, microbatch_size):
        import jax
        changed = any(self.c["sampling"][key] != sampling[key] for key in ("height", "width", "rollout_batch_size"))
        self.c["sampling"].update(sampling)
        self.c["training"]["microbatch_size"] = microbatch_size
        self.cache = None
        if changed:
            # At update/evaluation boundaries only; old shape executables cannot
            # consume the GPU memory needed by the next resolution. Disk cache remains.
            jax.clear_caches()
        return {"height": sampling["height"], "width": sampling["width"], "shape_changed": changed}

    def set_lora(self, lora):
        import jax
        placement=self.model.parallel.replicated if self.c.get("_mesh_dp_size",1)>1 else self.device
        self.lora = {k: jax.device_put(v if isinstance(v,jax.Array) else np.asarray(v,dtype=np.float32),placement) for k,v in lora.items()}
        jax.block_until_ready(self.lora)
        return {"loaded": len(self.lora)}

    def bind_conditions(self,descriptor,records,buckets):
        from .prefetch import read_buffer
        from .prompt_buckets import PromptBuckets
        values=read_buffer(descriptor)
        if len(values)!=len(records):raise ValueError("Condition buffer record count mismatch")
        self.bound_conditions={r["sample_id"]:pair for r,pair in zip(records,values)}
        self.buckets=PromptBuckets(*buckets)
        # Copies are private host arrays. Subsequent device uploads and the two
        # ART pullbacks cannot observe a reused producer slot.
        return {"copied":len(records),"generation":descriptor["generation"]}

    def _conditioning(self, record):
        import jax
        key = record["sample_id"]
        if self.c.get("preparation",{}).get("async_prefetch",False):
            if key not in getattr(self,"bound_conditions",{}):raise RuntimeError("Missing asynchronous condition binding")
            return self.bound_conditions[key]
        if key not in self.text_cache:
            if self.c["model"]["kind"] == "tiny":
                seed = int.from_bytes(hashlib.sha256(record["prompt"].encode()).digest()[:4], "little")
                rng = np.random.default_rng(seed)
                dim = self.model.config["text_dim"]
                positive = rng.normal(size=(1, 3, dim)).astype(np.float32) * .2
                negative = np.zeros((1, 2, dim), np.float32)
            else:
                from .data import PromptCache, PromptRecord
                entry = PromptRecord(id=key, prompt=record["prompt"],
                                     negative_prompt=record.get("negative_prompt", ""), metadata=record.get("metadata", {}))
                if self._prompt_cache is None:
                    self._prompt_cache = PromptCache(self.c["data"]["cache"], model_dir=self.c["model"]["path"])
                cached = self._prompt_cache.load(entry)
                positive, negative = cached["positive"], cached["negative"]
            self.text_cache[key] = (np.asarray(positive), np.asarray(negative))
            while len(self.text_cache) > 32:
                self.text_cache.popitem(last=False)
        else:
            self.text_cache.move_to_end(key)
        positive, negative = self.text_cache[key]
        return (jax.device_put(positive, self.device), jax.device_put(negative, self.device))

    def _kwargs(self, negative, lengths=None):
        return {**({"context_lengths":lengths[0],"negative_context_lengths":lengths[1]} if lengths else {}), "cfg_scale": self.c["sampling"]["guidance_scale"],
                "negative_context": negative,
                "negative_skip_layers": tuple(self.c["sampling"]["skip_layers"])}

    def _manage_compilation_cache(self, conditioning):
        """Clear old executables only at a rollout boundary, with no live VJP.

        Production uses two padded capacities with dynamic valid lengths.
        This guard also bounds executables for direct diagnostic callers.
        clear_caches releases compiled executables, never model or LoRA arrays.
        """
        import jax
        import gc
        lengths = {int(value.shape[1]) for pair in conditioning for value in pair}
        if len(self._compiled_text_lengths | lengths) > 16:
            jax.clear_caches()
            gc.collect()
            self._compiled_text_lengths = lengths
            return True
        self._compiled_text_lengths.update(lengths)
        return False

    def rollout(self, records, seeds, keep_cache=True, samples_per_prompt=None):
        import jax
        import jax.numpy as jnp
        from .sampling import schedule, step_coefficients, sample_step
        s = self.c["sampling"]
        count=s["group_size"] if samples_per_prompt is None else samples_per_prompt
        if type(count) is not int or not 1<=count<=s["group_size"] or len(seeds)!=len(records)*count:
            raise ValueError("Invalid local rollout sample count/seeds")
        if keep_cache:
            self.cache = None  # previous rollout completed both update windows
        conditioning = [self._conditioning(record) for record in records]
        from .prompt_buckets import PromptBuckets
        from .conditioning import pad_context
        if getattr(self,"buckets",None) is None:
            if self.c["model"]["kind"]=="moss" and getattr(self,"_prompt_cache",None) is not None:
                from .data import load_prompts,text_key
                records_all=load_prompts(self.c["data"]["prompts"],default_negative_prompt=self.c["data"]["negative_prompt"])
                entries=self._prompt_cache.manifest["entries"]
                lengths=[max(entries[text_key(t)]["token_count"] for t in (r.prompt,r.negative_prompt))
                         for r in records_all if all(text_key(t) in entries for t in (r.prompt,r.negative_prompt))]
            else:
                lengths=[max(p.shape[1],n.shape[1]) for p,n in conditioning]
            self.buckets=PromptBuckets.resolve(lengths,self.c["data"]["short_context_length"],self.c["data"]["context_length"],emit=self.rank==0 if hasattr(self,"rank") else False)
        padded=[]
        condition_lengths=[]
        for positive,negative in conditioning:
            capacity=self.buckets.capacity(max(positive.shape[1],negative.shape[1]))
            pos,pl=pad_context(positive,capacity)
            neg,nl=pad_context(negative,capacity)
            padded.append((jax.device_put(pos,self.device),jax.device_put(neg,self.device)))
            condition_lengths.append((pl,nl))
        conditioning=padded
        self._manage_compilation_cache(conditioning)
        _, sigmas = schedule(s["steps"], scheduler=s["scheduler"], shift=s["shift"],
                             training_shift=s["training_shift"], beta_alpha=s["beta_alpha"],
                             beta_beta=s["beta_beta"])
        coefficients = step_coefficients(sigmas, sampler=s["sampler"])
        velocity_coefficients = np.asarray([item.ode_velocity_coefficient for item in coefficients], np.float32)
        selected = tuple(s["train_timesteps"])
        if max(selected) >= len(sigmas) - 1:
            raise ValueError("Selected training step is absent after scheduler grid deduplication")
        states, velocities, endpoints, row_records, groups, contexts = [], [], [], [], [], []
        channels = self.model.config["latent_channels"]
        shape = (1, channels, s["height"] // 16, s["width"] // 16)
        for group, record in enumerate(records):
            positive, negative = conditioning[group]
            batch_size = s.get("rollout_batch_size", 1)*self.c.get("_mesh_dp_size",1)
            for begin in range(0, count, batch_size):
                batch_seeds = seeds[group * count + begin:group * count + min(begin + batch_size, count)]
                size = len(batch_seeds)
                z = jnp.concatenate([jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float32)
                                     for seed in batch_seeds], axis=0)
                if self.c.get("_mesh_dp_size",1)>1:z=self.model._put(z)
                pos = jnp.repeat(positive, size, axis=0) if size > 1 else positive
                neg = jnp.repeat(negative, size, axis=0) if size > 1 else negative
                lengths = tuple(np.repeat(value, size, axis=0) for value in condition_lengths[group])
                batch_states, batch_velocities = {}, {}
                previous_denoised = None
                for index in range(len(sigmas) - 1):
                    velocity = self.model.predict(self.lora, z, sigmas[index], pos,
                                                  **self._kwargs(neg, lengths)).astype(jnp.float32)
                    if index in selected and keep_cache:
                        batch_states[index] = np.asarray(z).copy()
                        batch_velocities[index] = np.asarray(velocity).copy()
                    z, previous_denoised = sample_step(z, velocity, coefficients[index], previous_denoised)
                batch_endpoints = np.asarray(z).copy()
                if not np.isfinite(batch_endpoints).all():
                    raise FloatingPointError("Nonfinite rollout endpoint")
                for row in range(size):
                    endpoints.append(batch_endpoints[row])
                    groups.append(group)
                    row_records.append(record)
                    contexts.append((positive, negative))
                    if keep_cache:
                        states.append([batch_states[i][row] for i in selected])
                        velocities.append([batch_velocities[i][row] for i in selected])
        endpoints = np.stack(endpoints)
        if keep_cache:
            self.cache = {"states": np.asarray(states), "old": np.asarray(velocities), "endpoints": endpoints,
                          "groups": np.asarray(groups, np.int32), "records": row_records, "contexts": contexts,
                          "sigmas": sigmas, "selected": selected,
                          "ode_velocity_coefficients": velocity_coefficients.copy(), "condition_lengths":condition_lengths}
        return {"endpoints": endpoints, "seeds": list(seeds), "prompts": [r["prompt"] for r in row_records],
                "sample_ids": [r["sample_id"] for r in row_records],
                "target_tags": [r.get("metadata",{}).get("target_tags") for r in row_records], "groups": np.asarray(groups),
                "sigmas": sigmas, "ode_velocity_coefficients": velocity_coefficients,
                "sampler": s["sampler"], "scheduler": s["scheduler"], "actual_steps": len(coefficients)}

    def decode(self, endpoints):
        from .evaluate import decode_latents
        images = decode_latents(self.c["model"]["path"], np.asarray(endpoints), device=self.device,
                                precision=self.c["model"]["vae_precision"],
                                low_precision_stages=self.c["model"]["vae_low_precision_stages"],
                                conv_backend=self.c["model"]["vae_conv_backend"],
                                conv_workspace_mib=self.c["model"]["vae_conv_workspace_mib"])
        return {"images": images}

    def export_replay(self):
        import jax
        keys = ("states", "old", "endpoints", "groups", "contexts", "sigmas",
                "selected", "ode_velocity_coefficients", "condition_lengths")
        from .replay import host_snapshot
        keys += tuple(k for k in ("parameter_version","rollout_precision","precision_contract") if k in self.cache)
        return host_snapshot({key: self.cache[key] for key in keys})

    def install_replay(self, cache):
        import jax
        self.cache = dict(cache)
        shared={}
        contexts=[]
        for group,(positive,negative) in zip(cache["groups"],cache["contexts"]):
            group=int(group)
            if group not in shared:
                shared[group]=(jax.device_put(positive,self.device),jax.device_put(negative,self.device))
            contexts.append(shared[group])
        self.cache["contexts"]=contexts
        return {"rows": len(cache["endpoints"])}

    def clear_replay(self):
        import gc
        self.cache=None
        self.stats={}
        self.windows=[]
        gc.collect()
        return {"cleared":True}

    def prepare_update(self, rewards=None, seed=0, stats=None, windows=None):
        import jax
        from .art import covariance, prompt_windows
        cache = self.cache
        a = self.c["art"]
        stats = stats if stats is not None else covariance(cache["endpoints"], np.asarray(rewards, np.float32), cache["groups"],
                           mode=a["feedback"], group_size=self.c["sampling"]["group_size"],
                           seed=seed, eps=a["eps"])
        self.stats = {k: np.asarray(v) for k, v in stats.items()}
        self.windows = windows if windows is not None else prompt_windows(cache["groups"], len(cache["selected"]), seed=seed)
        return {"active_groups": int(np.asarray(stats["group_active"]).sum()),
                "groups": int(len(stats["group_active"])), "items_per_window": [len(x) for x in self.windows]}

    def gradient(self, window):
        if self.c["training"].get("microbatch_size", 1) > 1:
            return self._gradient_batched(window)
        import jax
        import jax.numpy as jnp
        from .art import art_loss
        a, cache = self.c["art"], self.cache
        gradients = {k: np.zeros(v.shape, np.float32) for k, v in self.lora.items()}
        loss_sum, mask_sum, active_sum = 0., 0, 0
        items = self.windows[window]
        for row, slot in items:
            group = cache["groups"][row]
            direction = self.stats["group_directions"][group]
            magnitude = self.stats["group_guidance_magnitudes"][group]
            anchor = False
            if a["feedback"] == "pairwise":
                magnitude = self.stats.get("row_guidance_magnitudes", self.stats["group_guidance_magnitudes"][cache["groups"]])[row]
                anchor = bool(self.stats.get("row_restore_anchor", np.zeros(len(cache["groups"]), bool))[row])
            if not anchor and (magnitude == 0 or not np.any(direction)):
                continue  # Still contributes to the ORIGINAL denominator below.
            state = jax.device_put(cache["states"][row:row+1, slot], self.device)
            old = jax.device_put(cache["old"][row:row+1, slot], self.device)
            direction = jax.device_put(direction[None], self.device)
            step = cache["selected"][slot]
            sigma = cache["sigmas"][step]
            # History is frozen from rollout. Its additive term cancels between
            # old/new local ODE means; retain the coefficient of the RAW velocity.
            velocity_coefficient = cache["ode_velocity_coefficients"][step]
            positive, negative = cache["contexts"][row]
            prediction, pullback = self.model.predict_with_pullback(self.lora, state, sigma, positive, **self._kwargs(negative,cache["condition_lengths"][group]))
            def loss_fn(current):
                terms = art_loss(current, old, jnp.asarray([sigma]), direction, jnp.asarray([magnitude]),
                                 ode_velocity_coefficient=jnp.asarray([velocity_coefficient]), restore_anchor_mask=jnp.asarray([anchor]),
                                 q2_reference_sde_noise_level=a["noise_level"],
                                 q2_reference_kl_threshold=a["kl_threshold"], eps=a["eps"])
                return terms["per_item_loss"].sum(), (terms["masked"].sum(), terms["keep_mask"].sum())
            (loss, aux), cotangent = jax.value_and_grad(loss_fn, has_aux=True)(prediction)
            if not np.isfinite(np.asarray(loss)):
                raise FloatingPointError("Nonfinite ART loss")
            loss_sum += float(loss)
            mask_sum += int(aux[0])
            active_sum += int(aux[1])
            if int(aux[1]):
                grad = pullback(cotangent)
                for key in gradients:
                    value = np.asarray(grad[key], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite gradient: {key}")
                    gradients[key] += value
                del grad
            del pullback, prediction, cotangent
        return {"gradients": gradients, "loss_sum": loss_sum, "denominator": len(items),
                "masked": mask_sum, "active": active_sum,
                "memory_stats": self.device.memory_stats() or {}}

    def _gradient_batched(self, window):
        """Sum per-item ART losses; the coordinator owns the original denominator."""
        import jax
        import jax.numpy as jnp
        from .art import art_loss
        a, cache = self.c["art"], self.cache
        items = self.windows[window]
        batch_size = self.c["training"]["microbatch_size"]
        gradients = {k: np.zeros(v.shape, np.float32) for k, v in self.lora.items()}
        loss_sum, mask_sum, active_sum = 0., 0, 0
        # Separate padded capacities, so both established Qwen buckets remain valid.
        buckets = {}
        for row, slot in items:
            group = int(cache["groups"][row])
            direction = self.stats["group_directions"][group]
            magnitude = self.stats["group_guidance_magnitudes"][group]
            anchor = False
            if a["feedback"] == "pairwise":
                magnitude = self.stats["row_guidance_magnitudes"][row]
                anchor = bool(self.stats["row_restore_anchor"][row])
            if not anchor and (magnitude == 0 or not np.any(direction)):
                continue  # Retain inactive items in len(items).
            positive, negative = cache["contexts"][row]
            key = (positive.shape[1], negative.shape[1])
            buckets.setdefault(key, []).append((int(row), int(slot), group, magnitude, anchor))
        for entries in buckets.values():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start:start+batch_size]
                rows, slots, groups, magnitudes, anchors = map(list, zip(*chunk))
                state = jax.device_put(cache["states"][rows, slots], self.device)
                old = jax.device_put(cache["old"][rows, slots], self.device)
                direction = jax.device_put(self.stats["group_directions"][groups], self.device)
                steps = np.asarray(cache["selected"])[slots]
                sigmas = np.asarray(cache["sigmas"])[steps]
                coefficients = np.asarray(cache["ode_velocity_coefficients"])[steps]
                positive = jnp.concatenate([cache["contexts"][row][0] for row in rows], axis=0)
                negative = jnp.concatenate([cache["contexts"][row][1] for row in rows], axis=0)
                lengths = tuple(np.concatenate([cache["condition_lengths"][g][branch] for g in groups])
                                for branch in range(2))
                prediction, pullback = self.model.predict_with_pullback(
                    self.lora, state, sigmas, positive, **self._kwargs(negative, lengths))
                def loss_fn(current):
                    terms = art_loss(current, old, sigmas, direction, magnitudes,
                                     ode_velocity_coefficient=coefficients, restore_anchor_mask=anchors,
                                     q2_reference_sde_noise_level=a["noise_level"],
                                     q2_reference_kl_threshold=a["kl_threshold"], eps=a["eps"])
                    return terms["per_item_loss"].sum(), (terms["masked"].sum(), terms["keep_mask"].sum())
                (loss, aux), cotangent = jax.value_and_grad(loss_fn, has_aux=True)(prediction)
                if not np.isfinite(np.asarray(loss)):
                    raise FloatingPointError("Nonfinite ART loss")
                loss_sum += float(loss); mask_sum += int(aux[0]); active_sum += int(aux[1])
                if int(aux[1]):
                    grad = pullback(cotangent)
                    for key in gradients:
                        value = np.asarray(grad[key], dtype=np.float32)
                        if not np.isfinite(value).all():
                            raise FloatingPointError("Nonfinite batched gradient")
                        gradients[key] += value
                    del grad
                del pullback, prediction, cotangent
        return {"gradients": gradients, "loss_sum": loss_sum, "denominator": len(items),
                "masked": mask_sum, "active": active_sum,
                "memory_stats": self.device.memory_stats() or {}}
