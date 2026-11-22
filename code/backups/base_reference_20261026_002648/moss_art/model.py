"""Memory-bounded, blockwise JAX LoRA training of the native MOSS transformer.

The 5B frozen BF16 base is resident once per worker, optionally tensor-sharded. Shared per-block JITs avoid
unrolling 42 layers. Training retains sparse block-boundary checkpoints on GPU or CPU,
then reconstructs each segment and recomputes blocks in their VJPs; gradients are never requested
for base weights, text embeddings, or time-conditioning weights.

Do not wrap MossModel methods in an outer jax.jit: Python controls the host
activation checkpoints. Use predict_with_pullback to reuse a rollout/replay
forward when computing an external differentiable velocity loss.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from vendor.moss_jax import dit
from vendor.moss_jax.weights import load_config, load_host_params
from . import lora as lora_ops
from .attention import attention
from .conditioning import padded_rope, condition_block


def _dit_block(base, adapters, x, rope, modulation, heads: int, scale: float,
               attention_backend: str = "native", sequence_lengths=None, hidden_sharding=None):
    linear = lambda prefix, value: lora_ops.linear(base, adapters, "block." + prefix, value, scale)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = [modulation[:, i, None] for i in range(6)]
    y = dit._add(dit._mul(dit.rms_norm(x, base["block.norm1.weight"]), dit._add(scale_a, 1.0)), shift_a)
    if hidden_sharding is not None:
        # One shared gather for the three column-parallel projections, rather
        # than allowing the partitioner to gather the same activation 3 times.
        replicated = jax.sharding.NamedSharding(hidden_sharding.mesh,jax.sharding.PartitionSpec())
        y = jax.lax.with_sharding_constraint(y,replicated)
    q, k, v = linear("proj.q", y), linear("proj.k", y), linear("proj.v", y)
    if hidden_sharding is not None:
        q,k,v = (jax.lax.with_sharding_constraint(z,hidden_sharding) for z in (q,k,v))
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = dit.rms_norm(q.reshape(shape), base["block.q_norm.weight"]).reshape(q.shape)
    k = dit.rms_norm(k.reshape(shape), base["block.k_norm.weight"]).reshape(k.shape)
    x = dit._add(x, dit._mul(linear("proj.o", attention(q, k, v, rope, heads, backend=attention_backend,
        **({"sequence_lengths": sequence_lengths} if sequence_lengths is not None else {}))), gate_a))
    y = dit._add(dit._mul(dit.rms_norm(x, base["block.norm2.weight"]), dit._add(scale_f, 1.0)), shift_f)
    hidden, gate = jnp.split(linear("proj.ffn_up_gate", y), 2, axis=-1)
    if hidden_sharding is not None:
        hidden,gate = (jax.lax.with_sharding_constraint(z,hidden_sharding) for z in (hidden,gate))
    return dit._add(x, dit._mul(linear("proj.ffn_down", dit.situ(hidden, gate)), gate_f))


def _block_vjp_impl(base, adapters, x, rope, modulation, cotangent, heads, scale,
                    attention_backend="native", sequence_lengths=None, hidden_sharding=None):
    # base is an explicit JIT argument, not a closed-over constant; argnums of
    # vjp excludes it, so XLA never materializes a 5B parameter-gradient tree.
    _, pullback = jax.vjp(lambda a, z: _dit_block(base, a, z, rope, modulation, heads, scale, attention_backend, sequence_lengths, hidden_sharding), adapters, x)
    return pullback(cotangent)


# Keep the original callable names and default behavior for reference tests and
# CPU execution. The schema's level 0 means no local compiler override: XLA
# uses its default autotuning policy or explicitly supplied process flags.
_block_forward = jax.jit(_dit_block, static_argnums=(5, 6, 7))
_block_vjp = jax.jit(_block_vjp_impl, static_argnums=(6, 7, 8))


def _validate_dit_autotune_level(level, device=None):
    if type(level) is not int or level not in (0, 4):
        raise ValueError("dit_autotune_level must be the integer 0 or 4.")
    if level == 4 and device is not None and device.platform != "gpu":
        raise ValueError("dit_autotune_level=4 requires a CUDA GPU; use 0 on CPU.")


@lru_cache(maxsize=2)
def _get_block_kernels(autotune_level):
    """Reuse block JITs; 0 inherits XLA defaults, 4 requests local autotuning."""
    _validate_dit_autotune_level(autotune_level)
    if autotune_level == 0:
        return _block_forward, _block_vjp
    # Per-JIT options only apply to a top-level compilation. Compile the raw
    # functions directly; wrapping an existing jit would not scope tuning
    # correctly. VJP differentiates the raw block inside this one compilation.
    # These options affect only these DiT compilations. Process-wide XLA
    # defaults and any caller-supplied XLA_FLAGS remain unchanged.
    options = {"xla_gpu_autotune_level": autotune_level}
    return (
        jax.jit(_dit_block, static_argnums=(5, 6, 7), compiler_options=options),
        jax.jit(_block_vjp_impl, static_argnums=(6, 7, 8), compiler_options=options),
    )


@jax.jit
def _final_vjp(base, x, head, cotangent):
    _, pullback = jax.vjp(lambda z: dit._final(base, z, head)[0], x)
    return pullback(cotangent)[0]


def _view(params, prefix, replacement="block"):
    return {replacement + key[len(prefix):]: value for key, value in params.items()
            if key.startswith(prefix + ".")}


def _tree_add(a, b):
    return jax.tree.map(lambda x, y: x + y, a, b)


def _validate_attention_backend(backend, device=None, base=None):
    """Fail before weight transfers or loading when the backend is unusable."""
    if backend not in ("native", "cudnn"):
        raise ValueError(f"Unknown attention backend {backend!r}; expected 'native' or 'cudnn'.")
    if backend == "native":
        return
    if device is not None and device.platform != "gpu":
        raise ValueError("cuDNN attention requires a CUDA GPU; use attention_backend='native' on CPU.")
    if base is not None:
        for name, value in base.items():
            if name != "rope.bands" and value.dtype not in (jnp.bfloat16, jnp.float16):
                raise ValueError(
                    f"cuDNN attention requires BF16 or FP16 base parameters; {name} has dtype {value.dtype}."
                )


class MossModel:
    """Single-device or two-device TP MOSS with trainable DiT attention/MLP LoRA.

    state: [1, latent_channels, H, W], context: [1, tokens, text_dim]; optional context_lengths marks a padded prefix.
    sigma is already scheduler-shifted; the MOSS timestep is exactly 1000*sigma.
    predict output is FP32. CFG is uncond + scale*(cond-uncond); skip-layer 22
    applies to the negative branch only, matching native MOSS defaults.
    """
    def __init__(self, base, config=None, *, rank=8, alpha=None, targets="all",
                 device=None, offload_activations=True, attention_backend="native", dit_autotune_level=0,
                 checkpoint_interval=1, tensor_parallel_size=1, base_precision="bf16", mxfp8_dgrad="bf16", data_parallel_size=1):
        _validate_attention_backend(attention_backend)
        _validate_dit_autotune_level(dit_autotune_level)
        self.config = dit._config(config)
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if type(rank) is not int or rank <= 0 or not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("rank and alpha must be positive; rank must be an integer.")
        self.scale = self.alpha / self.rank
        self.targets = targets
        self.device = device or jax.devices()[0]
        if type(tensor_parallel_size) is not int or tensor_parallel_size not in (1,2):
            raise ValueError("tensor_parallel_size supports 1 or 2")
        if base_precision not in ("bf16", "mxfp8") or mxfp8_dgrad not in ("bf16", "mxfp8"):
            raise ValueError("Invalid frozen-base precision or MXFP8 dgrad precision")
        if base_precision == "mxfp8" and tensor_parallel_size != 1:
            raise ValueError("MXFP8 currently supports DP (tensor_parallel_size=1), not TP")
        if base_precision == "mxfp8" and self.device.platform == "gpu":
            capability = str(getattr(self.device, "compute_capability", "0.0"))
            if float(capability) < 10:
                raise ValueError("MXFP8 cuDNN block-scaled GEMM requires a Blackwell GPU")
        self.base_precision, self.mxfp8_dgrad = base_precision, mxfp8_dgrad
        self.tensor_parallel_size = tensor_parallel_size
        self.parallel = None
        self.data_parallel_size=data_parallel_size
        if data_parallel_size>1:
            if tensor_parallel_size!=1:raise ValueError("DP and TP meshes cannot be combined in this backend")
            if offload_activations:raise ValueError("Mesh DP requires GPU activation checkpoints")
            from .data_parallel import get_data_parallel
            self.parallel=get_data_parallel(data_parallel_size,dit_autotune_level)
        if tensor_parallel_size != 1:
            from .tensor_parallel import TensorParallel
            self.parallel = TensorParallel(tensor_parallel_size,self.config,dit_autotune_level)
        _validate_attention_backend(attention_backend, self.device, base)
        _validate_dit_autotune_level(dit_autotune_level, self.device)
        self.dit_autotune_level = dit_autotune_level
        self._block_forward, self._block_vjp = (
            self.parallel.kernels if self.parallel else _get_block_kernels(dit_autotune_level))
        self.attention_backend = attention_backend
        self.offload_activations = bool(offload_activations)
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        self.checkpoint_interval = checkpoint_interval
        # Bound CUDA host staging during multi-worker model loading. Enqueuing
        # the whole 5B model at once can retain gigabytes of pinned host buffers
        # per process, competing with activation offload on four GPUs.
        self.base = {}
        self._released = False
        for key, value in base.items():
            placement = self.parallel.parameter_sharding(key,value) if self.parallel else self.device
            # This model owns its buffers. Phase release must not invalidate a
            # caller-owned JAX input in tiny/reference runs. Host mmap inputs
            # still require exactly one device upload.
            dense = jax.device_put(value, placement, may_alias=False)
            dense.block_until_ready()
            from .mxfp8 import eligible, quantize, SCALE_SUFFIX, TRANSPOSE_SUFFIX, TRANSPOSE_SCALE_SUFFIX
            if base_precision == "mxfp8" and eligible(key, value):
                prefix = key[:-7]
                q, scales = quantize(dense)
                jax.block_until_ready((q, scales))
                self.base[key] = q
                self.base[prefix + SCALE_SUFFIX] = scales
                if mxfp8_dgrad == "mxfp8":
                    qt, st = quantize(dense.T)
                    jax.block_until_ready((qt, st))
                    self.base[prefix + TRANSPOSE_SUFFIX] = qt
                    self.base[prefix + TRANSPOSE_SCALE_SUFFIX] = st
                # The source may be a caller-owned JAX array (tiny tests).
                # Drop this reference, never explicitly delete caller buffers.
                del dense
            else:
                self.base[key] = dense
        self.base_storage_bytes = sum(v.nbytes for v in self.base.values())
        self.mxfp8_modules = sum(k.endswith(".mx_scale") for k in self.base)
        self.latent_channels = self.config["latent_channels"]
        self.text_dim = self.config["text_dim"]
        self._blocks = [_view(self.base, f"layers.{i}") for i in range(self.config["num_layers"])]
        self._time = {k: v for k, v in self.base.items() if k.startswith(
            ("time_embedding.", "shared_block_modulation.", "video_head_modulation."))}
        self._final = {k: v for k, v in self.base.items()
                       if k.startswith(("final_norm_video.", "final_linear_video."))}
        self._modules = lora_ops.target_modules(self.base, self.targets)
        jax.block_until_ready(self.base)

    def precision_contract(self):
        """Describe the actual frozen-base path; no mutable quantizer history.

        LoRA is additive FP32 in both paths and is never merged into the frozen
        quantized base. Rebuilding the sampler with the same LoRA therefore
        synchronizes theta without recalibrating scales or rounding adapters.
        """
        from .mxfp8 import quantization_contract
        return dict(base_precision=self.base_precision,
                    base_storage_bytes=self.base_storage_bytes,
                    quantized_modules=self.mxfp8_modules,
                    lora_master_dtype="float32", lora_merge="additive_unmerged",
                    output_dtype="float32", mxfp8_dgrad=self.mxfp8_dgrad,
                    quantization=quantization_contract() if self.base_precision == "mxfp8" else None)

    def release_base(self):
        """Release the owned frozen base before loading the next compute phase.

        The caller retains LoRA/optimizer/replay separately. Recreate this model
        from the original host weights to change precision; never dequantize an
        FP8 base and call that the BF16 learner. All derived dictionaries alias
        base leaves and must be cleared together. Existing pullbacks become
        invalid after release; there is no hidden second base or GPU cache.
        """
        if self._released:
            return
        jax.block_until_ready(self.base)
        leaves = tuple(self.base.values())
        self.base = {}
        self._blocks = []
        self._time = {}
        self._final = {}
        self.base_storage_bytes = 0
        self.mxfp8_modules = 0
        self._released = True
        for value in leaves:
            value.delete()

    @classmethod
    def from_pretrained(cls, model_dir, *, host_params=None, **kwargs):
        """Read the released transformer safetensors directly, without Torch."""
        import ml_dtypes
        backend = kwargs.get("attention_backend", "native")
        autotune_level = kwargs.get("dit_autotune_level", 0)
        _validate_attention_backend(backend)
        _validate_dit_autotune_level(autotune_level)
        if backend == "cudnn" or autotune_level == 4:
            kwargs["device"] = kwargs.get("device") or jax.devices()[0]
            _validate_attention_backend(backend, kwargs["device"])
            _validate_dit_autotune_level(autotune_level, kwargs["device"])
        config = load_config(model_dir, "transformer")
        host = load_host_params(model_dir, "transformer") if host_params is None else host_params
        # A memmap view is retained until each asynchronous device_put completes.
        # There is no stacked second model copy and no FP32 base/optimizer.
        host = {k: (v if k == "rope.bands" else np.asarray(v, dtype=ml_dtypes.bfloat16))
                for k, v in host.items()}
        return cls(host, config, **kwargs)

    @classmethod
    def tiny(cls, seed=0, *, rank=2, alpha=None, targets="all", device=None,
             offload_activations=True, dtype=jnp.float32, config=None, attention_backend="native",
             dit_autotune_level=0, checkpoint_interval=1, tensor_parallel_size=1, base_precision="bf16", mxfp8_dgrad="bf16", data_parallel_size=1):
        _validate_attention_backend(attention_backend)
        _validate_dit_autotune_level(dit_autotune_level)
        if attention_backend == "cudnn" or dit_autotune_level == 4:
            device = device or jax.devices()[0]
            _validate_attention_backend(attention_backend, device)
            _validate_dit_autotune_level(dit_autotune_level, device)
        cfg = dict(dim=32, ffn_dim=48, num_heads=4, num_layers=2,
                   text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
        cfg.update(config or {})
        return cls(tiny_base(cfg, seed, dtype), cfg, rank=rank, alpha=alpha,
                   targets=targets, device=device, offload_activations=offload_activations,
                   attention_backend=attention_backend, dit_autotune_level=dit_autotune_level,
                   checkpoint_interval=checkpoint_interval,tensor_parallel_size=tensor_parallel_size,
                   base_precision=base_precision,mxfp8_dgrad=mxfp8_dgrad,data_parallel_size=data_parallel_size)

    def _put(self, value, *, hidden=False):
        if self.data_parallel_size>1:return self.parallel.put(value)
        placement = (self.parallel.hidden if hidden else self.parallel.replicated) if self.parallel else self.device
        return jax.device_put(value,placement)

    def init_lora(self, seed=0):
        placement = self.parallel.replicated if self.parallel else self.device
        return lora_ops.init_lora(self.base, self.rank, seed=seed, targets=self.targets, device=placement)

    def _check(self, adapters, state, sigma, context):
        if self._released:
            raise RuntimeError("The model base has been released; reconstruct from original host weights.")
        if state.ndim != 4 or state.shape[0] < 1 or state.shape[1] != self.latent_channels or min(state.shape[2:]) <= 0:
            raise ValueError(f"state must have shape [B,{self.latent_channels},H,W].")
        if context.ndim != 3 or context.shape[0] != state.shape[0] or context.shape[1] == 0 or context.shape[2] != self.text_dim:
            raise ValueError(f"context must have shape [B,positive_length,{self.text_dim}].")
        if np.shape(sigma) not in ((), (1,), (state.shape[0],)):
            raise ValueError("sigma must be scalar or have shape [batch].")
        if not isinstance(sigma, jax.core.Tracer):
            sigma_host = np.asarray(sigma)
            if not np.isfinite(sigma_host).all() or not ((sigma_host >= 0) & (sigma_host <= 1)).all():
                raise ValueError("sigma must be finite and within [0,1].")
        expected = {module + suffix for module in self._modules for suffix in (lora_ops.DOWN, lora_ops.UP)}
        if set(adapters) != expected:
            raise ValueError("LoRA keys do not match this model's configured targets.")
        for module in self._modules:
            out_dim, in_dim = self.base[module + ".weight"].shape
            if adapters[module + lora_ops.DOWN].shape != (self.rank, in_dim) or adapters[module + lora_ops.UP].shape != (out_dim, self.rank):
                raise ValueError(f"LoRA shape mismatch for {module}")
            if any(adapters[module + suffix].dtype != jnp.float32 for suffix in (lora_ops.DOWN, lora_ops.UP)):
                raise ValueError("LoRA factors must be FP32 master parameters.")

    def _checkpoint(self, x):
        # Explicit device_get synchronizes and allows all within-block temporaries
        # to be freed before the next layer. Store BF16 as BF16 (not FP32).
        return np.array(jax.device_get(x), copy=True) if self.offload_activations else x

    def _forward(self, adapters, state, sigma, context, skip_layers=(), *, record=False, context_lengths=None,
                 capture_features=False):
        self._check(adapters, state, sigma, context)
        if context_lengths is not None:
            lengths_host = np.asarray(context_lengths)
            if (lengths_host.shape != (state.shape[0],) or lengths_host.dtype.kind not in "iu"
                    or not ((lengths_host > 0) & (lengths_host <= context.shape[1])).all()):
                raise ValueError("context_lengths must contain the valid nonempty prefix length")
            context_lengths = self._put(np.asarray(lengths_host,np.int32))
        skip_layers = frozenset(skip_layers)
        if any(type(i) is not int or not 0 <= i < self.config["num_layers"] for i in skip_layers):
            raise ValueError(f"skip_layers must be in [0,{self.config['num_layers']}).")
        state = self._put(jnp.asarray(state, jnp.float32))
        context = self._put(context)
        batch, _, height, width = state.shape
        image_tokens = height * width
        _, modulation, head = dit._compiled_time_conditioning(
            self._time, jnp.asarray(sigma, jnp.float32) * 1000, batch, self.config["freq_dim"])
        video = state.reshape(batch, self.latent_channels, image_tokens).transpose(0, 2, 1)
        video = dit._compiled_linear(_view(self.base, "video_embedder", "linear"), video)
        if context_lengths is None:
            rope = dit._compiled_rope(height, width, context.shape[1], self.config["dim"] // self.config["num_heads"],
                                     self.config["rope_base"], dtype=video.dtype, bands=self.base.get("rope.bands"))
        else:
            rope = padded_rope(height, width, context.shape[1], self.config["dim"] // self.config["num_heads"],
                               self.config["rope_base"], context_lengths, dtype=video.dtype, bands=self.base.get("rope.bands"))
        text = dit._compiled_linear(_view(self.base, "text_embedder.proj", "linear"), context)
        for i in range(2):
            parameters = _view(self.base, f"text_embedder.blocks.{i}")
            if context_lengths is None:
                text = dit._compiled_text_block(parameters, text, rope[:, image_tokens:], self.config["num_heads"])
            else:
                text = condition_block(parameters, text, rope[:, image_tokens:], self.config["num_heads"],
                                       context_lengths, self.attention_backend)
        sequence_lengths = None if context_lengths is None else context_lengths + jnp.int32(image_tokens)
        x = self._put(jnp.concatenate((video, text), axis=1),hidden=True)
        checkpoints = {}
        active = [i for i in range(self.config["num_layers"]) if i not in skip_layers]
        segments = [active[start:start+self.checkpoint_interval]
                    for start in range(0,len(active),self.checkpoint_interval)]
        starts = {segment[0] for segment in segments}
        for i in active:
            if record and i in starts:
                checkpoints[i] = self._checkpoint(x)
            local_lora = _view(adapters, f"layers.{i}")
            x = self._block_forward(self._blocks[i], local_lora, x, rope, modulation, self.config["num_heads"],
                               self.scale, self.attention_backend, sequence_lengths)
            if record:
                # Bound live temporaries without sending activations across PCIe.
                x.block_until_ready()
        final_x = x[:, :image_tokens]
        if record:
            final_x = self._checkpoint(final_x)
        output = dit._compiled_final(self._final, self._put(final_x,hidden=True), head)[0]
        output = output.transpose(0, 2, 1).reshape(batch, self.latent_channels, height, width).astype(jnp.float32)
        tape = dict(checkpoints=checkpoints, final_x=final_x, rope=rope, modulation=modulation,
                    head=head, shape=state.shape, sequence_length=x.shape[1], skip_layers=skip_layers,
                    valid_lengths=sequence_lengths, segments=segments) if record else None
        return output, final_x if capture_features else tape

    def _backward(self, adapters, tape, cotangent):
        if self._released:
            raise RuntimeError("A pullback cannot outlive its model base.")
        if tuple(cotangent.shape) != tuple(tape["shape"]):
            raise ValueError("cotangent must have exactly the predicted velocity shape.")
        batch, channels, height, width = tape["shape"]
        image_tokens = height * width
        dz = self._put(jnp.asarray(cotangent, jnp.float32))
        dz = dz.reshape(batch, channels, image_tokens).transpose(0, 2, 1)
        # JAX VJP cotangent dtype must equal the original (BF16) output dtype.
        final_x = self._put(tape["final_x"],hidden=True)
        dz = dz.astype(self._final["final_linear_video.weight"].dtype)
        dx = _final_vjp(self._final, final_x, tape["head"], dz)
        dx = self._put(jnp.pad(dx, ((0, 0), (0, tape["sequence_length"] - image_tokens), (0, 0))),hidden=True)
        gradients = {}
        for i in tape["skip_layers"]:
            prefix = f"layers.{i}"
            gradients.update({prefix + key[len("block"):]: jnp.zeros_like(value)
                              for key,value in _view(adapters,prefix).items()})
        for segment in reversed(tape["segments"]):
            # Sparse permanent checkpoints + at most one segment of temporary
            # inputs. interval=1 keeps every block input and does no extra replay.
            x = self._put(tape["checkpoints"][segment[0]],hidden=True)
            inputs = [x]
            for i in segment[:-1]:
                x = self._block_forward(self._blocks[i],_view(adapters,f"layers.{i}"),
                    x,tape["rope"],tape["modulation"],self.config["num_heads"],
                    self.scale,self.attention_backend,tape["valid_lengths"])
                x.block_until_ready()
                inputs.append(x)
            del x
            for i in reversed(segment):
                x = inputs.pop()
                prefix = f"layers.{i}"
                local_gradients, dx = self._block_vjp(self._blocks[i],_view(adapters,prefix),
                    x,tape["rope"],tape["modulation"],dx,self.config["num_heads"],
                    self.scale,self.attention_backend,tape["valid_lengths"])
                jax.block_until_ready((local_gradients,dx))
                gradients.update({prefix + key[len("block"):]: value for key,value in local_gradients.items()})
                del x,local_gradients
        return gradients

    def predict(self, adapters, state, sigma, context, *, cfg_scale=1.0, negative_context=None,
                negative_skip_layers=(22,), skip_layers=(), context_lengths=None, negative_context_lengths=None):
        if not np.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        positive, _ = self._forward(adapters, state, sigma, context, skip_layers, context_lengths=context_lengths)
        if cfg_scale == 1.0:
            return positive
        if negative_context is None:
            raise ValueError("negative_context is required when cfg_scale != 1.")
        negative, _ = self._forward(adapters, state, sigma, negative_context, negative_skip_layers, context_lengths=negative_context_lengths)
        return negative + jnp.float32(cfg_scale) * (positive - negative)

    def predict_with_features(self, adapters, state, sigma, context, *, cfg_scale=1.0,
                              negative_context=None, negative_skip_layers=(22,), skip_layers=(),
                              context_lengths=None, negative_context_lengths=None):
        """Actual rollout forward plus head inputs, separately for each CFG branch.

        These device arrays contain image tokens only. No training tape or text
        tokens are retained; the caller owns this trajectory's forecast history.
        """
        if not np.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        if cfg_scale != 1.0 and negative_context is None:
            raise ValueError("negative_context is required when cfg_scale != 1.")
        positive, hp = self._forward(adapters, state, sigma, context, skip_layers,
            context_lengths=context_lengths, capture_features=True)
        if cfg_scale == 1.0:
            return positive, (hp,)
        negative, hn = self._forward(adapters, state, sigma, negative_context, negative_skip_layers,
            context_lengths=negative_context_lengths, capture_features=True)
        return negative + jnp.float32(cfg_scale) * (positive - negative), (hp, hn)

    def predict_from_features(self, features, sigma, state_shape, *, cfg_scale=1.0):
        """Current timestep's original head on forecast image tokens; no DiT blocks."""
        if self._released:
            raise RuntimeError("The model base has been released.")
        if not np.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        batch, channels, height, width = state_shape
        if channels != self.latent_channels or len(features) != (1 if cfg_scale == 1.0 else 2):
            raise ValueError("Forecast CFG branches or latent shape do not match the model.")
        expected = (batch, height * width, self.config["dim"])
        if any(tuple(h.shape) != expected for h in features):
            raise ValueError("Forecast features must match [batch,image_tokens,hidden_dim].")
        if np.shape(sigma) not in ((), (1,), (batch,)) or not np.isfinite(sigma).all() or not ((np.asarray(sigma) >= 0) & (np.asarray(sigma) <= 1)).all():
            raise ValueError("sigma must be finite, in [0,1], and scalar or [batch].")
        # Reuse exactly the same kernels/rounding as the actual MOSS forward.
        _, _, head = dit._compiled_time_conditioning(self._time,
            jnp.asarray(sigma, jnp.float32) * 1000, batch, self.config["freq_dim"])
        outputs = []
        for feature in features:
            value = dit._compiled_final(self._final, self._put(feature, hidden=True), head)[0]
            outputs.append(value.transpose(0, 2, 1).reshape(state_shape).astype(jnp.float32))
        if cfg_scale == 1.0:
            return outputs[0]
        return outputs[1] + jnp.float32(cfg_scale) * (outputs[0] - outputs[1])

    def predict_with_pullback(self, adapters, state, sigma, context, *, cfg_scale=1.0,
                              negative_context=None, negative_skip_layers=(22,), skip_layers=(), context_lengths=None, negative_context_lengths=None):
        """Return (FP32 velocity, pullback(cotangent)->FP32 LoRA gradients).

        This is the efficient ART replay interface: evaluate value_and_grad of
        the ART objective with respect to the returned velocity, then pass its
        cotangent to pullback. Forward checkpoints stay on host until released.
        The callable may be used multiple times with different cotangents.
        """
        if not np.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        positive, positive_tape = self._forward(adapters, state, sigma, context, skip_layers, record=True, context_lengths=context_lengths)
        if cfg_scale == 1.0:
            return positive, lambda cotangent: self._backward(adapters, positive_tape, cotangent)
        if negative_context is None:
            raise ValueError("negative_context is required when cfg_scale != 1.")
        negative, negative_tape = self._forward(adapters, state, sigma, negative_context, negative_skip_layers, record=True, context_lengths=negative_context_lengths)
        def pullback(cotangent):
            gp = self._backward(adapters, positive_tape, jnp.float32(cfg_scale) * cotangent)
            gn = self._backward(adapters, negative_tape, jnp.float32(1 - cfg_scale) * cotangent)
            return _tree_add(gp, gn)
        return negative + jnp.float32(cfg_scale) * (positive - negative), pullback

    def predict_reference(self, adapters, state, sigma, context, **kwargs):
        """Frozen BF16 old prediction using the exact learner forward path.

        This is one forward at an already saved state, not another rollout.
        CFG, negative skip layers, padding lengths and time conversion are
        supplied unchanged by the caller. Discarding the pullback releases its
        activation checkpoints immediately; only the FP32 velocity is kept.
        The returned value is detached from adapters and base parameters.
        """
        if self.base_precision != "bf16":
            raise ValueError("The calibrated old reference requires the BF16 learner path.")
        prediction, pullback = self.predict_with_pullback(adapters, state, sigma, context, **kwargs)
        prediction.block_until_ready()
        del pullback
        return jax.lax.stop_gradient(prediction)

    def vjp(self, adapters, state, sigma, context, cotangent, **kwargs):
        """Convenience recompute-forward VJP; prefer predict_with_pullback."""
        _, pullback = self.predict_with_pullback(adapters, state, sigma, context, **kwargs)
        return pullback(cotangent)

    def loss_and_grad(self, adapters, state, sigma, context, loss_fn, **kwargs):
        """Run a user-supplied scalar velocity loss with one retained forward."""
        prediction, pullback = self.predict_with_pullback(adapters, state, sigma, context, **kwargs)
        value, cotangent = jax.value_and_grad(loss_fn)(prediction)
        return value, pullback(cotangent)


def tiny_base(config=None, seed=0, dtype=jnp.float32):
    """Random, small native-layout fixture; executes the full actual architecture."""
    fixture_config = dict(dim=32, ffn_dim=48, num_heads=4, num_layers=2,
                          text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
    fixture_config.update(config or {})
    cfg = dit._config(fixture_config)
    d, f, h, c, t, q = (cfg[k] for k in ("dim", "ffn_dim", "num_heads", "latent_channels", "text_dim", "freq_dim"))
    rng = np.random.default_rng(seed)
    result = {}
    def linear(prefix, out_dim, in_dim, bias=True):
        result[prefix + ".weight"] = jnp.asarray(rng.normal(0, 0.35 / np.sqrt(in_dim), (out_dim, in_dim)), dtype=dtype)
        if bias:
            result[prefix + ".bias"] = jnp.asarray(rng.normal(0, 0.02, out_dim), dtype=dtype)
    def norm(prefix, dim):
        result[prefix + ".weight"] = jnp.ones((dim,), dtype=dtype)
    linear("video_embedder", d, c)
    linear("text_embedder.proj", d, t)
    linear("time_embedding.0", d, q)
    linear("time_embedding.2", d, d)
    linear("shared_block_modulation", 6 * d, d)
    # Keep both attention/FFN gates open in this fixture.
    mod = np.asarray(result["shared_block_modulation.bias"]).reshape(6, d).copy()
    mod[2] += 0.5
    mod[5] += 0.5
    result["shared_block_modulation.bias"] = jnp.asarray(mod.reshape(-1), dtype=dtype)
    linear("video_head_modulation", 2 * d, d)
    for i in range(2):
        prefix = f"text_embedder.blocks.{i}"
        norm(prefix + ".norm1", d)
        norm(prefix + ".norm2", d)
        norm(prefix + ".q_norm", d // h)
        norm(prefix + ".k_norm", d // h)
        linear(prefix + ".qkv", 3 * d, d)
        linear(prefix + ".proj", d, d)
        linear(prefix + ".ffn_up", f, d)
        linear(prefix + ".ffn_gate", f, d)
        linear(prefix + ".ffn_down", d, f)
    for i in range(cfg["num_layers"]):
        prefix = f"layers.{i}"
        norm(prefix + ".norm1", d)
        norm(prefix + ".norm2", d)
        norm(prefix + ".q_norm", d // h)
        norm(prefix + ".k_norm", d // h)
        for name in ("q", "k", "v", "o"):
            linear(prefix + ".proj." + name, d, d)
        linear(prefix + ".proj.ffn_up_gate", 2 * f, d)
        linear(prefix + ".proj.ffn_down", d, f)
    norm("final_norm_video", d)
    linear("final_linear_video", c, d)
    return result
