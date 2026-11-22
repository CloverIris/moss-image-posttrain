# vLLM 0.24.0 hidden-state export with PP

The MOSS target is vLLM's Qwen3.5 implementation and already supports pipeline
parallelism. The current PP2 implementation uses native token pooling and does
not activate any compatibility patches. The extractor fallback below needs
additional compatibility work in this installed vLLM release. It changes Python objects in the worker
process; it never edits vLLM's installed files.

## Audited lifecycle

In `vllm/v1/worker/gpu_model_runner.py`, the constructor creates the extractor
only on the final PP stage. Three subsequent methods checked the speculative
method without checking stage ownership: `_dummy_run`,
`_check_and_update_cudagraph_mode`, and `initialize_kv_cache`.
`vllm_moss_plugin/pp_compat.py` adds `get_pp_group().is_last_rank` to those exact
extractor predicates. Normal target KV initialization, Mamba state allocation,
graph configuration and dummy runs continue on every stage.

Before workers start, `SpeculativeConfig._verify_args` also validates the
synthetic draft model with the target PP configuration. The weightless
`ExtractHiddenStatesModel` is a final-stage cache sink, not a partitioned draft
decoder. The explicit fallback config function exempts only the exact architecture
list `['ExtractHiddenStatesModel']` from the PP capability branch in
`ModelConfig.verify_with_parallel_config`. This fallback function is explicit
and is not invoked by normal plugin registration. All target-model capability checks,
other draft models and the remaining parallel validation stay intact. This
does not register the cache sink as a general-purpose pipeline model.

The first stage does not own a cache-only attention layer. The MOSS connector
therefore marks non-final stages passive, without binding a nonexistent hidden
cache or claiming the final writer's filesystem locks. The final stage retains
the normal writer and physical-block-size correction. Exported features remain
the already normalized final Qwen output; no extra BF16 residual RMSNorm is
introduced.

## Fail-closed compatibility

The config patch requires vLLM `0.24.0` and the complete audited AST hash of its
validation method. The worker patches additionally require complete AST hashes
of all three lifecycle methods and exactly one matching extractor predicate in
each. A different version or source implementation raises before replacements
are assigned. Only an actual extractor speculative configuration with PP>1 can
install the worker patches. Pooling and PP1 install no worker lifecycle patches; their numerical
execution is unchanged. `compatibility_metadata()` reports config and runner
installation separately. The adapter records the plugin source SHA in its
cache metadata.

The host tests cover passive first-stage behavior, final-stage exports, target
KV/Mamba lifecycle preservation, unsupported targets/drafts, other speculative
methods, remaining parallel validation, lazy model registration, repeated
installation and rejected source/version changes. On 2026-10-24, the plugin's
17 tests and compatibility's 15 tests passed locally and on the server. A real
vLLM import and registration matched the config source guard without initializing
CUDA. These are host checks; real PP output comparison is a separate validation.

## BF16 parameter footprint (checkpoint headers only)

The source is `$MODEL_ROOT/MOSS-Image-Base/text_encoder`. Reading only
the safetensors JSON headers found 759 tensors totaling 16,785,390,048 bytes:

| Component | Bytes | GiB |
| --- | ---: | ---: |
| 32 decoder layers | 13,839,123,456 | 12.889 |
| Token embedding | 2,034,237,440 | 1.895 |
| Vision tower and merger | 912,020,960 | 0.849 |
| Final norm | 8,192 | <0.001 |

In this vLLM version, `Qwen3_5Model.__init__` creates token embedding on every
stage (`qwen3_5.py:227`), and its multimodal wrapper creates the vision tower on
every stage (`qwen3_5.py:587`). `_mark_tower_model` skips it only when all relevant
multimodal limits are zero, not according to PP rank. The adapter's unused tied
sampling head shares that stage's embedding parameter and adds no new matrix.

Thus the current 16/16 partition has approximately 9.188 GiB of checkpoint
weights on each GPU, including the duplicated embedding and vision tower;
activations, target KV/Mamba states, export cache, runtime workspaces and padding
are additional. It is already balanced by weight bytes. If a future change
places the vision tower only on the first stage, 15/17 becomes closer by weight
bytes (8.798/8.730 GiB), but that is not the current construction and is not
assumed by the runner.

`VLLM_PP_LAYER_PARTITION='16,16'` can explicitly select stage layer counts. The
installed implementation is `vllm/distributed/utils.py:get_pp_indices`, lines
109–154; it validates the count and sum. No partition override is needed for
the present default. This setting belongs to vLLM and does not change XLA.
