"""Process-local PP guards for vLLM 0.24.0's hidden-state extractor.

The installed runner creates an extractor only on the final PP stage, but
three later lifecycle branches check the global speculative method without
checking stage ownership. Rewrite only those exact calls in memory; retain
normal target KV/Mamba initialization and every other runner operation.
Nothing edits site-packages or imports Torch while this module is imported.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import textwrap


SUPPORTED_VERSION = "0.24.0"
PATCH_ID = "moss_hidden_state_pp_v1"
_METHOD_AST_SHA256 = {
    "_dummy_run": "0927d1a22e7d7224c59a0d6f560e5ee7182f6766c2359c59252c63abc913040e",
    "_check_and_update_cudagraph_mode": "94bf9da09d957b68fad824aad3397c2066929eb838dc0aa7359d972ba1e2ee9f",
    "initialize_kv_cache": "d3b2207729d98c0e08a2d333e948c60ed170df33ad0261e5caadfc38cbf5c14f",
}
_installation = {"installed": False, "reason": "not requested"}
_CONFIG_AST_SHA256 = "c7dd2b6aa2c23eb851b12e3caac31eeb6768675d866c562cb1767a55f5a37eb5"
_config_installation = {"installed": False, "reason": "not requested"}


def _method_node(method):
    original = inspect.unwrap(method)
    source = textwrap.dedent(inspect.getsource(original))
    module = ast.parse(source)
    node = next(item for item in module.body if isinstance(item, ast.FunctionDef))
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    return original, node, digest


class _GuardExtractorCalls(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "uses_extract_hidden_states"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "speculative_config"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and not node.args and not node.keywords):
            self.count += 1
            stage = ast.Attribute(value=ast.Call(func=ast.Name(id="get_pp_group", ctx=ast.Load()),
                                                args=[], keywords=[]),
                                  attr="is_last_rank", ctx=ast.Load())
            return ast.copy_location(ast.BoolOp(op=ast.And(), values=[node, stage]), node)
        return node


class _GuardCacheSinkConfig(ast.NodeTransformer):
    """Exempt the non-pipelined, final-stage-only cache sink from one check."""

    def __init__(self):
        self.count = 0

    def visit_If(self, node):
        self.generic_visit(node)
        if not (isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And)):
            return node
        values = node.test.values
        if len(values) != 2:
            return node
        size, capability = values
        if not (isinstance(size, ast.Compare) and isinstance(size.left, ast.Name)
                and size.left.id == "pipeline_parallel_size" and len(size.ops) == 1
                and isinstance(size.ops[0], ast.Gt) and len(size.comparators) == 1
                and isinstance(size.comparators[0], ast.Constant) and size.comparators[0].value == 1
                and isinstance(capability, ast.UnaryOp) and isinstance(capability.op, ast.Not)
                and isinstance(capability.operand, ast.Call)
                and isinstance(capability.operand.func, ast.Attribute)
                and capability.operand.func.attr == "is_pp_supported_model"):
            return node
        # This exact architecture is a weightless cache consumer instantiated
        # exclusively on the last PP worker by the audited GPUModelRunner. It
        # does not receive/send pipeline intermediate tensors and is not the
        # target Qwen model. All other parallel checks remain untouched.
        is_sink = ast.Compare(
            left=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                               attr="architectures", ctx=ast.Load()),
            ops=[ast.Eq()], comparators=[ast.List(elts=[ast.Constant("ExtractHiddenStatesModel")],
                                                 ctx=ast.Load())])
        node.test.values.insert(1, ast.UnaryOp(op=ast.Not(), operand=is_sink))
        self.count += 1
        return node


def _compile_method(original, node, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = dict(original.__globals__)
    exec(compile(module, filename, "exec"), namespace)
    return namespace[node.name]


def _patch_model_config_class(config_class, version, expected_ast_sha=None):
    existing = getattr(config_class, "_moss_hidden_state_pp_config_patch", None)
    if existing:
        if existing.get("patch_id") != PATCH_ID or existing.get("vllm_version") != version:
            raise RuntimeError("An incompatible MOSS PP config patch is already installed")
        return existing
    if version != SUPPORTED_VERSION:
        raise RuntimeError(f"MOSS hidden-state PP config compatibility supports vLLM {SUPPORTED_VERSION}; "
                           f"found {version}. Re-audit the cache-sink config before enabling PP.")
    original, node, digest = _method_node(config_class.verify_with_parallel_config)
    expected = _CONFIG_AST_SHA256 if expected_ast_sha is None else expected_ast_sha
    if digest != expected:
        raise RuntimeError(f"MOSS PP config source guard rejected {config_class.__name__}."
                           f"verify_with_parallel_config: AST SHA256 {digest} differs from audited vLLM.")
    transformer = _GuardCacheSinkConfig()
    node = transformer.visit(copy.deepcopy(node))
    if transformer.count != 1:
        raise RuntimeError(f"MOSS PP expected one config capability branch; got {transformer.count}")
    replacement = _compile_method(original, node, "<moss-pp-compat:verify_with_parallel_config>")
    metadata = {"installed": True, "patch_id": PATCH_ID, "vllm_version": version,
                "original_ast_sha256": digest, "guarded_predicates": transformer.count,
                "exempt_architectures": ["ExtractHiddenStatesModel"], "site_packages_modified": False}
    config_class.verify_with_parallel_config = replacement
    config_class._moss_hidden_state_pp_config_patch = metadata
    return metadata


def install_config_compat():
    """Explicit fallback only, before building an extractor+PP EngineArgs.

    Normal plugin registration deliberately does not call this function;
    native pooling needs no config or worker patches.
    """
    global _config_installation
    from vllm.config.model import ModelConfig

    _config_installation = _patch_model_config_class(ModelConfig, importlib.metadata.version("vllm"))
    return _config_installation


def _patch_runner_class(runner_class, version, expected_ast_shas=None):
    """Validate all targets before atomically assigning any replacement."""
    existing = getattr(runner_class, "_moss_hidden_state_pp_patch", None)
    if existing:
        if existing.get("patch_id") != PATCH_ID or existing.get("vllm_version") != version:
            raise RuntimeError("An incompatible MOSS PP runner patch is already installed")
        return existing
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"MOSS hidden-state PP compatibility supports vLLM {SUPPORTED_VERSION}; "
            f"found {version}. Re-audit the extractor lifecycle before enabling PP."
        )
    expected = _METHOD_AST_SHA256 if expected_ast_shas is None else expected_ast_shas
    replacements, methods = {}, {}
    for name in _METHOD_AST_SHA256:
        method = getattr(runner_class, name)
        original, node, digest = _method_node(method)
        if digest != expected.get(name):
            raise RuntimeError(
                f"MOSS PP source guard rejected {runner_class.__name__}.{name}: "
                f"AST SHA256 {digest} differs from audited vLLM {SUPPORTED_VERSION}."
            )
        transformer = _GuardExtractorCalls()
        node = transformer.visit(copy.deepcopy(node))
        if transformer.count != 1:
            raise RuntimeError(f"MOSS PP expected one extractor predicate in {name}; got {transformer.count}")
        replacements[name] = _compile_method(original, node, f"<moss-pp-compat:{name}>")
        methods[name] = {"original_ast_sha256": digest, "guarded_predicates": transformer.count}
    filename = inspect.getsourcefile(inspect.unwrap(getattr(runner_class, "initialize_kv_cache")))
    source_sha = hashlib.sha256(Path(filename).read_bytes()).hexdigest() if filename else None
    metadata = {"installed": True, "patch_id": PATCH_ID, "vllm_version": version,
                "original_runner_source": filename, "original_runner_source_sha256": source_sha,
                "methods": methods, "site_packages_modified": False}
    for name, replacement in replacements.items():
        setattr(runner_class, name, replacement)
    runner_class._moss_hidden_state_pp_patch = metadata
    return metadata


def is_extract_hidden_states_pp(vllm_config):
    if vllm_config is None:
        return False
    parallel = getattr(vllm_config, "parallel_config", None)
    speculative = getattr(vllm_config, "speculative_config", None)
    return bool(parallel is not None and parallel.pipeline_parallel_size > 1
                and speculative is not None and speculative.uses_extract_hidden_states())


def install(vllm_config):
    """Called during MOSS model construction, after runner/distributed init."""
    global _installation
    if not is_extract_hidden_states_pp(vllm_config):
        _installation = {"installed": False, "reason": "No extractor with PP>1; execution is unchanged"}
        return _installation
    from vllm.distributed import get_pp_group

    if get_pp_group().world_size == 1:
        _installation = {"installed": False, "reason": "PP=1; existing execution is unchanged"}
        return _installation
    # This import is intentionally deferred until worker-side model creation.
    # GPUModelRunner is already imported then; plugin discovery stays lazy.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _installation = _patch_runner_class(GPUModelRunner, importlib.metadata.version("vllm"))
    return _installation


def compatibility_metadata():
    return {"runner": copy.deepcopy(_installation), "config": copy.deepcopy(_config_installation)}
