"""Adapter contract tests using stand-ins; neither Torch nor CUDA is imported."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "integrations" / "vllm_plugin"


class FakeWeightsMapper:
    def __init__(self, *, orig_to_new_prefix):
        self.orig_to_new_prefix = orig_to_new_prefix

    def _map_name(self, name):
        for prefix, replacement in self.orig_to_new_prefix.items():
            if name.startswith(prefix):
                name = name.replace(prefix, replacement, 1)
        return name


class FakeQwenModel:
    _processor_factory = object()

    def __init__(self, *, num_layers=32, result=None, vllm_config=None):
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(num_hidden_layers=num_layers)
        )
        self.result = result
        self.selected_layers = None
        self.forward_kwargs = None

    def set_aux_hidden_state_layers(self, layers):
        self.selected_layers = layers

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.result


def load_adapter():
    fake_qwen = types.ModuleType("vllm.model_executor.models.qwen3_5")
    fake_qwen.Qwen3_5ForConditionalGeneration = FakeQwenModel
    fake_utils = types.ModuleType("vllm.model_executor.models.utils")
    fake_utils.WeightsMapper = FakeWeightsMapper
    fake_pp_compat = types.ModuleType("vllm_moss_plugin.pp_compat")
    fake_pp_compat.install = Mock()
    fake_pp_compat.is_extract_hidden_states_pp = lambda config: False
    modules = {
        "vllm.model_executor.models.qwen3_5": fake_qwen,
        "vllm.model_executor.models.utils": fake_utils,
        "vllm_moss_plugin.pp_compat": fake_pp_compat,
    }
    path = PLUGIN_ROOT / "vllm_moss_plugin" / "model.py"
    spec = importlib.util.spec_from_file_location("_moss_plugin_model_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.MossQwen3_5ForConditionalGeneration


class PluginContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter()

    def test_language_prefix_mapping_preserves_visual(self):
        mapper = self.adapter.hf_to_vllm_mapper
        self.assertEqual(
            mapper._map_name("language_model.layers.31.mlp.down_proj.weight"),
            "language_model.model.layers.31.mlp.down_proj.weight",
        )
        self.assertEqual(
            mapper._map_name("visual.merger.norm.weight"),
            "visual.merger.norm.weight",
        )

    def test_reuses_multimodal_processor_factory(self):
        self.assertIs(self.adapter._processor_factory, FakeQwenModel._processor_factory)

    def test_constructor_installs_guards_only_for_extracting_pipeline(self):
        compatibility_path = PLUGIN_ROOT / "vllm_moss_plugin" / "pp_compat.py"
        spec = importlib.util.spec_from_file_location("_moss_plugin_compat_gate", compatibility_path)
        real_compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_compat)
        for pp, extract in [(1, True), (2, False), (2, True), (2, None)]:
            with self.subTest(pp=pp, extract=extract):
                config = types.SimpleNamespace(
                    parallel_config=types.SimpleNamespace(pipeline_parallel_size=pp),
                    speculative_config=None if extract is None else types.SimpleNamespace(
                        uses_extract_hidden_states=lambda: extract))
                install = Mock()
                with patch.dict(self.adapter.__init__.__globals__, {
                    "is_extract_hidden_states_pp": real_compat.is_extract_hidden_states_pp,
                    "install_pp_compat": install,
                }):
                    self.adapter(vllm_config=config)
                if pp > 1 and extract:
                    install.assert_called_once_with(config)
                else:
                    install.assert_not_called()

    def test_only_final_layer_is_accepted_and_forwarded(self):
        model = self.adapter()
        model.set_aux_hidden_state_layers((32,))
        self.assertEqual(model.selected_layers, (32,))
        for layers in [(), (0,), (31,), (31, 32), (32, 32)]:
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(ValueError, "exactly auxiliary layer"):
                    model.set_aux_hidden_state_layers(layers)

    def test_rejects_other_backbone_depth(self):
        model = self.adapter(num_layers=40)
        with self.assertRaisesRegex(ValueError, "requires a 32-layer"):
            model.set_aux_hidden_state_layers((32,))

    def test_exports_native_normalized_output_without_another_norm(self):
        normalized, raw_aux = object(), object()
        model = self.adapter(result=(normalized, [raw_aux]))
        main, exported = model.forward("tokens", "positions", foo="bar")
        self.assertIs(main, normalized)
        self.assertIs(exported[0], normalized)
        self.assertIsNot(exported[0], raw_aux)
        self.assertEqual(model.forward_kwargs["input_ids"], "tokens")
        self.assertEqual(model.forward_kwargs["positions"], "positions")
        self.assertEqual(model.forward_kwargs["foo"], "bar")

    def test_forwards_intermediate_tensors_and_embeddings(self):
        model = self.adapter(result=object())
        model.forward(None, "positions", intermediate_tensors="state", inputs_embeds="emb")
        self.assertEqual(model.forward_kwargs["intermediate_tensors"], "state")
        self.assertEqual(model.forward_kwargs["inputs_embeds"], "emb")

    def test_no_auxiliary_output_is_unchanged_for_profiling(self):
        for result in [object(), {"hidden_states": object()}, (object(), [])]:
            with self.subTest(result=type(result).__name__):
                model = self.adapter(result=result)
                self.assertIs(model.forward(None, None), result)

    def test_unexpected_auxiliary_count_fails_closed(self):
        model = self.adapter(result=(object(), [object(), object()]))
        with self.assertRaisesRegex(ValueError, "expected one auxiliary"):
            model.forward(None, None)

    def test_registration_is_lazy_and_repeatable(self):
        path = PLUGIN_ROOT / "vllm_moss_plugin" / "__init__.py"
        spec = importlib.util.spec_from_file_location("_moss_plugin_registration_test", path)
        module = importlib.util.module_from_spec(spec)
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.ModelRegistry = types.SimpleNamespace(register_model=Mock())
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            spec.loader.exec_module(module)
            fake_vllm.ModelRegistry.register_model.assert_not_called()
            module.register()
            module.register()
        self.assertEqual(fake_vllm.ModelRegistry.register_model.call_count, 2)
        fake_vllm.ModelRegistry.register_model.assert_called_with(
            "MossQwen3_5ForConditionalGeneration",
            "vllm_moss_plugin.model:MossQwen3_5ForConditionalGeneration",
        )

    def test_entry_point_is_declared_for_worker_processes(self):
        import tomllib

        with (PLUGIN_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(
            config["project"]["entry-points"]["vllm.general_plugins"]["moss_qwen_prefill"],
            "vllm_moss_plugin:register",
        )


class ConnectorContractTests(unittest.TestCase):
    @staticmethod
    def connector_class(is_last_stage=True):
        class FakeExampleConnector:
            def __init__(self, logical_block_size=272):
                self._block_size = logical_block_size
                self._kv_cache = None
                self._is_tp_rank_zero = True
                self.locks_created = 0
                self.writes_submitted = 0

            def register_kv_caches(self, caches):
                self._kv_cache = caches["hidden"]

            def wait_for_save(self):
                if self._is_tp_rank_zero:
                    self.locks_created += 1

            def _submit_async_write(self, pending):
                if self._is_tp_rank_zero:
                    self.writes_submitted += 1

        name = "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector"
        fake_module = types.ModuleType(name)
        fake_module.ExampleHiddenStatesConnector = FakeExampleConnector
        fake_distributed = types.ModuleType("vllm.distributed")
        fake_distributed.get_pp_group = lambda: types.SimpleNamespace(is_last_rank=is_last_stage)
        path = PLUGIN_ROOT / "vllm_moss_plugin" / "connector.py"
        spec = importlib.util.spec_from_file_location("_moss_connector_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: fake_module, "vllm.distributed": fake_distributed}):
            spec.loader.exec_module(module)
        # Calls import distributed at runtime; retain a direct loader test hook
        # through the module-level function below instead of importing vLLM.
        original_register = module.MossHiddenStatesConnector.register_kv_caches
        def register(self, caches):
            with patch.dict(sys.modules, {"vllm.distributed": fake_distributed}):
                return original_register(self, caches)
        module.MossHiddenStatesConnector.register_kv_caches = register
        return module.MossHiddenStatesConnector

    def test_uses_actual_kernel_block_size_after_binding(self):
        connector = self.connector_class()()
        cache = types.SimpleNamespace(ndim=4, shape=(1208, 68, 1, 4096))
        connector.register_kv_caches({"hidden": cache})
        self.assertIs(connector._kv_cache, cache)
        self.assertEqual(connector._block_size, 68)

    def test_non_final_pipeline_stage_is_passive_without_hidden_cache(self):
        connector = self.connector_class(is_last_stage=False)()
        # The inherited implementation would fail without a hidden cache.
        connector.register_kv_caches({})
        self.assertEqual(connector.cache_layers, [])
        self.assertIsNone(connector._kv_cache)
        self.assertFalse(connector._is_tp_rank_zero)
        self.assertTrue(connector._moss_passive_pp_stage)
        self.assertEqual(connector._block_size, 272)
        connector.wait_for_save()
        connector._submit_async_write(object())
        self.assertEqual(connector.locks_created, 0)
        self.assertEqual(connector.writes_submitted, 0)

    def test_final_pipeline_stage_keeps_normal_writer_behavior(self):
        connector = self.connector_class()()
        connector.register_kv_caches({"hidden": types.SimpleNamespace(ndim=4, shape=(10, 68, 1, 4096))})
        self.assertFalse(connector._moss_passive_pp_stage)
        self.assertTrue(connector._is_tp_rank_zero)
        connector.wait_for_save()
        connector._submit_async_write(object())
        self.assertEqual(connector.locks_created, 1)
        self.assertEqual(connector.writes_submitted, 1)

    def test_kernel_block_ids_expand_across_physical_block_boundary(self):
        connector = self.connector_class()()
        connector.register_kv_caches({
            "hidden": types.SimpleNamespace(ndim=4, shape=(1208, 68, 1, 4096))
        })
        # Match the inherited writer's block_id * _block_size + offsets.
        block_ids = [8, 16]
        slots = [
            block * connector._block_size + offset
            for block in block_ids
            for offset in range(connector._block_size)
        ]
        self.assertEqual(slots[:2], [544, 545])
        self.assertEqual(slots[67:70], [611, 1088, 1089])
        # Validate which physical block/token each exported slot addresses.
        selected = [(slot // 68, slot % 68) for slot in slots]
        self.assertEqual(selected[67:70], [(8, 67), (16, 0), (16, 1)])
        self.assertNotEqual(slots[0], 8 * 272)

    def test_scheduler_keeps_configured_logical_block_size(self):
        connector = self.connector_class()()
        # A scheduler connector does not bind worker cache tensors.
        self.assertEqual(connector._block_size, 272)

    def test_rejects_wrong_cache_geometry(self):
        for cache in [None, types.SimpleNamespace(ndim=3, shape=(68, 1, 4096)),
                      types.SimpleNamespace(ndim=4, shape=(10, 0, 1, 4096))]:
            with self.subTest(cache=cache):
                connector = self.connector_class()()
                with self.assertRaises(ValueError):
                    connector.register_kv_caches({"hidden": cache})


if __name__ == "__main__":
    unittest.main()
