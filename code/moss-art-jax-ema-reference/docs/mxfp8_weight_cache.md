# MXFP8 online and offline weights

There are two independent quantization decisions. Frozen weights can be
quantized at model load (`online`) or exported once and loaded directly
(`offline`). Activations depend on the actual input and remain dynamically
quantized on every invocation in both modes. Offline weights do not mean fixed
activation scales or a calibration-dataset approximation.

Both modes call the same quantizer: E4M3FN data, one E8M0FNU power-of-two scale
per 32 contraction elements. The scope stays limited to eligible transformer
block projections; other parameters retain their original loader's BF16 cast
policy (with the existing exception for RoPE bands). CUDA GEMMs remain JAX's
scaled_matmul/cuDNN blockScaledDot. No arithmetic kernel is replaced.

The default preserves current behavior:

```json
"model": {
  "mxfp8_weight_mode": "online",
  "mxfp8_cache_dir": "",
  "mxfp8_cache_verify": true
}
```

Online mode uploads source BF16 tensors and quantizes frozen projection weights
once each time the MXFP8 model is constructed. It does not requantize weights
at every denoising step. The live model retains compressed weights/scales.

Export a reusable cache with:

```bash
cd $CODE_ROOT/moss-art-jax
$CONDA_PREFIX/envs/jax_env/bin/python run.py quantize-mxfp8 \
  --config configs/preparation/mxfp8_weights.json
```

The export JSON specifies `model_dir`, `output_dir`, `platform` (`cpu` or
`cuda`), `device` and `include_transpose`. The exporter processes one tensor at
a time using the unchanged JAX quantizer and commits immutable numeric NPY
files plus a JSON manifest. It needs neither prompts nor calibration images.
Running the same command again verifies and reuses a valid existing cache;
it never replaces a conflicting/incomplete output directory.

Use the prepared cache as follows:

```json
"model": {
  "mxfp8_weight_mode": "offline",
  "mxfp8_cache_dir": "$MOSS_ROOT/cache/mxfp8/MOSS-Image-Base-v1",
  "mxfp8_cache_verify": true
}
```

Offline mode validates source model identity, quantizer contract/source,
logical dtypes/shapes, manifest and tensor checksums. It mmaps saved raw FP8
bits and scales and uploads them directly; it never routes them back through
the BF16 casting loader or weight quantizer. A staged MXFP8 collector skips
the shared CPU BF16 host map as well. An invalid or absent cache fails rather
than silently falling back to online quantization.

`mxfp8_cache_verify=false` skips full tensor SHA256 scans at runtime but still
checks the manifest, file size/mtime, dtype, shape and source provenance. Full
verification is the default. Source weight provenance follows the project's
existing header/size/mtime/path identity policy, not a full source-payload hash.

LoRA factors are always separate and unmerged. Updating them does not require
exporting another base cache. BF16 training, calibrated theta0 references and
frozen-base regularization always read the original BF16 weights, never a
dequantized copy of this cache. These I/O settings do not change the algorithm
fingerprint; ordinary source/model/data checkpoint compatibility checks remain.

The forward-only cache omits transposes. If MXFP8 dgrad is explicitly used by
the model API, export with `include_transpose=true`; W.T is separately quantized
with its own contraction-axis scales. A forward-only caller can read such a
cache without uploading the unused transpose arrays.

Ready configuration examples are `configs/art/silva_staged_mxfp8_online.json`
and `configs/art/silva_staged_mxfp8_offline.json`. They retain the same run and
training parameters, differing only in weight-loading policy. No training run
is started by preparing or selecting either configuration.

The same options are forwarded by the pretrained SFT worker. Existing SFT
MXFP8 dgrad options are unchanged; a BF16 learner always loads source BF16.

## Verified real-model example, 2026-10-26

The prepared MOSS-Image-Base cache contains 252 converted projections and is
5,557,230,459 bytes (5.56 GB, approximately 5.18 GiB). Its one-time GPU export
took 19.80 seconds, or 20.46 seconds including the exporter process.

Four-device MXFP8 model construction was measured in separate processes with
warm file caches. This timer starts before Mesh/model creation, after Python
imports; it is not a complete rollout or training-step measurement.

| Weight loading mode | Model construction time |
| --- | ---: |
| Online conversion from BF16 | 6.16 s |
| Offline with full tensor checksums | 7.39 s |
| Offline with manifest/file-metadata verification | 4.25 s |

Offline storage does not automatically make startup faster: full SHA256 scans
of 5.56 GB cost time. `configs/art/silva_staged_mxfp8_offline_fast.json` selects
metadata verification on this already exported immutable local cache. The
ordinary offline example keeps full verification enabled.

All three real-model runs produced the same SHA256 over every frozen parameter
and scale. Their synthetic 768x768 latent predictions were bitwise identical.
CPU tests also cover both dgrad variants, changed LoRA factors, corruption,
source mismatches and restoring original BF16 weights. No user prompts or
generated images were used for these parity measurements.

Offline rollout construction avoids retaining the BF16 host map. In the load
probe, process RSS fell from 12.43 to 2.93 GiB; most of this difference was
reclaimable file-backed mapping, not anonymous memory. System MemAvailable
remained similar, so RSS should not be interpreted as equivalent RAM savings.
