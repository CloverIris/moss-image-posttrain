"""No implicit process-wide compiler restrictions; local JIT options stay explicit."""
import ast
from copy import deepcopy
from pathlib import Path
import re

import pytest

from moss_art.config import DEFAULTS
from moss_art import runtime

ROOT = Path(__file__).resolve().parents[1]
COMPILER_POLICY_ENV = {
    "XLA_FLAGS", "TF_XLA_FLAGS", "JAX_DISABLE_JIT",
    "JAX_DISABLE_MOST_OPTIMIZATIONS", "JAX_OPTIMIZATION_LEVEL",
    "JAX_DEFAULT_MATMUL_PRECISION", "JAX_COMPILER_ENABLE_REMAT_PASS",
    "TF_FP16_CONV_USE_FP32_COMPUTE", "TF_CUDNN_USE_AUTOTUNE",
    "TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32", "TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32",
    "NVIDIA_TF32_OVERRIDE",
}


@pytest.mark.parametrize("platform,device", [("cpu", None), ("cuda", 2)])
@pytest.mark.parametrize("inherited", [
    {}, {"XLA_FLAGS": ""},
    {"XLA_FLAGS": " --xla_gpu_autotune_level=4 --xla_dump_to=/tmp/user-dump "},
    {"XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=constant_folding"},
    {"TF_FP16_CONV_USE_FP32_COMPUTE": "0"},
    {"TF_FP16_CONV_USE_FP32_COMPUTE": "1"},
    {"JAX_DISABLE_JIT": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest"},
])
def test_configure_process_preserves_compiler_policy_and_only_sets_resources(
    monkeypatch, platform, device, inherited,
):
    # No real environment, CPU-affinity, JAX initialization or CUDA side effect.
    environment = dict(inherited)
    monkeypatch.setattr(runtime.os, "environ", environment)
    if hasattr(runtime.os, "sched_getaffinity"):
        monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda _pid: set(range(8)))
        monkeypatch.setattr(runtime.os, "sched_setaffinity", lambda _pid, _cores: None)
    monkeypatch.setattr(runtime, "gpu_inventory", lambda: {2: {"uuid": "GPU-test-2"}})
    config = deepcopy(DEFAULTS)
    config["runtime"].update(platform=platform, memory_fraction=.75, cpu_threads=2)
    runtime.configure_process(config, device=device)
    assert {k: v for k, v in environment.items() if k in COMPILER_POLICY_ENV} == inherited
    assert environment["CUDA_VISIBLE_DEVICES"] == ("" if device is None else "GPU-test-2")
    assert environment["JAX_PLATFORMS"] == platform
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.75"
    assert environment["OMP_NUM_THREADS"] == environment["OPENBLAS_NUM_THREADS"] == "2"
    # Numeric type availability and cache location do not disable XLA passes.
    assert environment["JAX_ENABLE_X64"] == "true"
    assert environment["JAX_COMPILATION_CACHE_DIR"]


def _literal_string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _environment_writes(source):
    """Find explicit env writes, including child-process copies and update kwargs."""
    tree = ast.parse(source)
    environment_names = {"os.environ", "environ"}
    # Resolve simple environment aliases/copies without executing entry points.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is None:
                    continue
                expression = ast.unparse(value)
                is_environment = expression in environment_names
                if isinstance(value, ast.Call):
                    is_environment |= (
                        isinstance(value.func, ast.Attribute)
                        and value.func.attr == "copy"
                        and ast.unparse(value.func.value) in environment_names
                    )
                    is_environment |= (
                        isinstance(value.func, ast.Name) and value.func.id == "dict"
                        and bool(value.args) and ast.unparse(value.args[0]) in environment_names
                    )
                if is_environment:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    environment_names.update(ast.unparse(target) for target in targets)
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and ast.unparse(target.value) in environment_names:
                    key = _literal_string(target.slice)
                    if key in COMPILER_POLICY_ENV:
                        writes.append((node.lineno, key))
        if not isinstance(node, ast.Call):
            continue
        method = node.func
        if isinstance(method, ast.Attribute) and ast.unparse(method.value) in environment_names:
            if method.attr in ("setdefault", "__setitem__") and node.args:
                key = _literal_string(node.args[0])
                if key in COMPILER_POLICY_ENV:
                    writes.append((node.lineno, key))
            if method.attr == "update":
                keys = [keyword.arg for keyword in node.keywords]
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        keys.extend(_literal_string(key) for key in arg.keys)
                writes.extend((node.lineno, key) for key in keys if key in COMPILER_POLICY_ENV)
        if ast.unparse(method) in ("os.putenv", "putenv") and node.args:
            key = _literal_string(node.args[0])
            if key in COMPILER_POLICY_ENV:
                writes.append((node.lineno, key))
        # The JAX config API can express the same unwanted global override.
        if ast.unparse(method) in ("jax.config.update", "config.update") and node.args:
            key = _literal_string(node.args[0])
            if key and key.upper() in COMPILER_POLICY_ENV:
                writes.append((node.lineno, key))
    return sorted(set(writes))


def test_live_python_entry_points_do_not_inject_global_compiler_policy():
    # Historical reports, measured source snapshots and ignored outputs are
    # deliberately excluded: their original experimental conditions are evidence.
    sources = sorted((ROOT / "moss_art").rglob("*.py"))
    sources += sorted((ROOT / "scripts").rglob("*.py"))
    violations = [
        f"{path.relative_to(ROOT)}:{line}: {key}"
        for path in sources for line, key in _environment_writes(path.read_text())
    ]
    assert not violations, "\n".join(violations)


def test_shell_entry_points_do_not_export_global_compiler_policy():
    assignments = re.compile(
        r"(?:^|[;\s])(?:export\s+)?(" + "|".join(sorted(COMPILER_POLICY_ENV)) + r")\s*="
    )
    violations = []
    for path in sorted((ROOT / "scripts").rglob("*.sh")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.lstrip().startswith("#") and assignments.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not violations, "\n".join(violations)
