"""Host-only lifecycle guards; no Torch, CUDA or model weights are imported."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "integrations/vllm_plugin/vllm_moss_plugin/pp_compat.py"
SPEC = importlib.util.spec_from_file_location("_moss_pp_compat_tests", PATH)
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)
_GROUP = types.SimpleNamespace(is_last_rank=False, world_size=2)


def get_pp_group():
    return _GROUP


class FakeRunner:
    def initialize_kv_cache(self, cache, is_profiling=False):
        self.events.append(("target_cache", cache, is_profiling))
        self.events.append(("mamba", cache))
        if self.speculative_config and self.speculative_config.uses_extract_hidden_states():
            self.drafter.validate_same_kv_cache_group(cache)
        self.events.append(("connector_registration", cache))

    def _check_and_update_cudagraph_mode(self, mode):
        self.events.append(("target_graph", mode))
        if self.speculative_config and (
            self.speculative_config.use_eagle() or self.speculative_config.uses_extract_hidden_states()
        ):
            self.drafter.initialize_cudagraph_keys(mode)

    def _dummy_run(self, tokens):
        self.events.append(("target_dummy", tokens))
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            self.drafter.dummy_run(tokens)
        return tokens


def runner_class():
    class LocalRunner(FakeRunner):
        pass
    return LocalRunner


def digest_map(cls):
    return {name: compat._method_node(getattr(cls, name))[2] for name in compat._METHOD_AST_SHA256}


def make_runner(cls, last=False, method="extract_hidden_states"):
    _GROUP.is_last_rank = last
    instance = cls()
    instance.events = []
    instance.speculative_config = types.SimpleNamespace(
        uses_extract_hidden_states=lambda: method == "extract_hidden_states",
        use_eagle=lambda: method == "eagle", uses_draft_model=lambda: method == "draft_model")
    if last or method != "extract_hidden_states":
        instance.drafter = types.SimpleNamespace(validate_same_kv_cache_group=Mock(),
                                                 initialize_cudagraph_keys=Mock(), dummy_run=Mock())
    return instance


class PipelineGuardTests(unittest.TestCase):
    def test_first_stage_skips_only_extractor_and_keeps_target_lifecycle(self):
        cls = runner_class()
        compat._patch_runner_class(cls, "0.24.0", digest_map(cls))
        runner = make_runner(cls, last=False)
        runner.initialize_kv_cache("cache", is_profiling=True)
        runner._check_and_update_cudagraph_mode("graph")
        self.assertEqual(runner._dummy_run(37), 37)
        self.assertFalse(hasattr(runner, "drafter"))
        self.assertEqual(runner.events, [("target_cache", "cache", True), ("mamba", "cache"),
                                         ("connector_registration", "cache"), ("target_graph", "graph"),
                                         ("target_dummy", 37)])

    def test_last_stage_keeps_extractor_and_target_behavior(self):
        cls = runner_class()
        compat._patch_runner_class(cls, "0.24.0", digest_map(cls))
        runner = make_runner(cls, last=True)
        runner.initialize_kv_cache("cache")
        runner._check_and_update_cudagraph_mode("graph")
        runner._dummy_run(11)
        runner.drafter.validate_same_kv_cache_group.assert_called_once_with("cache")
        runner.drafter.initialize_cudagraph_keys.assert_called_once_with("graph")
        runner.drafter.dummy_run.assert_called_once_with(11)

    def test_other_speculative_methods_remain_unchanged(self):
        cls = runner_class()
        compat._patch_runner_class(cls, "0.24.0", digest_map(cls))
        runner = make_runner(cls, last=False, method="eagle")
        runner.initialize_kv_cache("cache")
        runner._check_and_update_cudagraph_mode("graph")
        runner._dummy_run(4)
        runner.drafter.validate_same_kv_cache_group.assert_not_called()
        runner.drafter.initialize_cudagraph_keys.assert_called_once_with("graph")
        runner.drafter.dummy_run.assert_called_once_with(4)

    def test_repeated_install_is_idempotent(self):
        cls = runner_class()
        first = compat._patch_runner_class(cls, "0.24.0", digest_map(cls))
        method = cls.initialize_kv_cache
        second = compat._patch_runner_class(cls, "0.24.0", {})
        self.assertIs(first, second)
        self.assertIs(method, cls.initialize_kv_cache)

    def test_wrong_source_fails_before_any_assignment(self):
        cls = runner_class()
        expected = digest_map(cls)
        expected["initialize_kv_cache"] = "wrong-source"
        before = {name: getattr(cls, name) for name in expected}
        with self.assertRaisesRegex(RuntimeError, "source guard rejected"):
            compat._patch_runner_class(cls, "0.24.0", expected)
        for name, method in before.items():
            self.assertIs(getattr(cls, name), method)
        self.assertFalse(hasattr(cls, "_moss_hidden_state_pp_patch"))

    def test_unknown_version_fails_closed(self):
        cls = runner_class()
        with self.assertRaisesRegex(RuntimeError, "supports vLLM 0.24.0"):
            compat._patch_runner_class(cls, "0.25.0", digest_map(cls))

    def test_pp1_does_not_import_runner_or_check_new_version(self):
        distributed = types.ModuleType("vllm.distributed")
        distributed.get_pp_group = lambda: types.SimpleNamespace(world_size=1)
        with patch.dict(sys.modules, {"vllm.distributed": distributed}), \
                patch.object(compat.importlib.metadata, "version", side_effect=AssertionError("not needed")):
            config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(pipeline_parallel_size=1),
                                           speculative_config=types.SimpleNamespace(uses_extract_hidden_states=lambda: True))
            report = compat.install(config)
        self.assertFalse(report["installed"])
        self.assertIn("unchanged", report["reason"])

    def test_pooling_and_other_speculation_do_not_install_any_patch(self):
        parallel = types.SimpleNamespace(pipeline_parallel_size=2)
        configurations = [None, types.SimpleNamespace(parallel_config=parallel, speculative_config=None),
                          types.SimpleNamespace(parallel_config=parallel, speculative_config=
                                                types.SimpleNamespace(uses_extract_hidden_states=lambda: False))]
        with patch.object(compat.importlib.metadata, "version", side_effect=AssertionError("not needed")):
            for config in configurations:
                with self.subTest(config=config):
                    self.assertFalse(compat.is_extract_hidden_states_pp(config))
                    self.assertFalse(compat.install(config)["installed"])

    def test_only_extractor_with_pp2_matches_install_gate(self):
        config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(pipeline_parallel_size=2),
                                       speculative_config=types.SimpleNamespace(uses_extract_hidden_states=lambda: True))
        self.assertTrue(compat.is_extract_hidden_states_pp(config))


class FakeModelConfig:
    def verify_with_parallel_config(self, parallel_config):
        self.events.append("heads")
        if self.total_heads % parallel_config.tensor_parallel_size:
            raise ValueError("invalid tensor parallel heads")
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
            self.architectures, self
        ):
            raise NotImplementedError("target requires SupportsPP")
        self.events.append("remaining_parallel_checks")
        if parallel_config.invalid_context_parallel:
            raise ValueError("invalid context parallel")


class ConfigGuardTests(unittest.TestCase):
    def config_class(self):
        class LocalConfig(FakeModelConfig):
            pass
        digest = compat._method_node(LocalConfig.verify_with_parallel_config)[2]
        compat._patch_model_config_class(LocalConfig, "0.24.0", digest)
        return LocalConfig

    def make_config(self, architecture, *, supported=False):
        config = self.config_class()()
        config.architectures = architecture
        config.total_heads = 16
        config.events = []
        config.registry = types.SimpleNamespace(is_pp_supported_model=Mock(return_value=supported))
        return config

    def parallel(self, pp=2, tp=1, invalid_context=False):
        return types.SimpleNamespace(pipeline_parallel_size=pp, tensor_parallel_size=tp,
                                     invalid_context_parallel=invalid_context)

    def test_final_stage_cache_sink_skips_only_pipeline_capability(self):
        config = self.make_config(["ExtractHiddenStatesModel"])
        config.verify_with_parallel_config(self.parallel())
        self.assertEqual(config.events, ["heads", "remaining_parallel_checks"])
        config.registry.is_pp_supported_model.assert_not_called()

    def test_target_and_other_drafts_still_require_pipeline_capability(self):
        for architecture in [["MossQwen3_5ForConditionalGeneration"], ["OtherDraft"],
                             ["ExtractHiddenStatesModel", "OtherDraft"]]:
            with self.subTest(architecture=architecture):
                config = self.make_config(architecture)
                with self.assertRaisesRegex(NotImplementedError, "SupportsPP"):
                    config.verify_with_parallel_config(self.parallel())
                config.registry.is_pp_supported_model.assert_called_once()

    def test_supported_target_keeps_remaining_checks(self):
        config = self.make_config(["MossQwen3_5ForConditionalGeneration"], supported=True)
        config.verify_with_parallel_config(self.parallel())
        self.assertEqual(config.events, ["heads", "remaining_parallel_checks"])

    def test_cache_sink_still_validates_tensor_and_context_parallel(self):
        config = self.make_config(["ExtractHiddenStatesModel"])
        with self.assertRaisesRegex(ValueError, "tensor parallel"):
            config.verify_with_parallel_config(self.parallel(tp=3))
        with self.assertRaisesRegex(ValueError, "context parallel"):
            config.verify_with_parallel_config(self.parallel(invalid_context=True))

    def test_pp1_keeps_validation_and_does_not_query_capability(self):
        config = self.make_config(["OtherModel"])
        config.verify_with_parallel_config(self.parallel(pp=1))
        config.registry.is_pp_supported_model.assert_not_called()
        self.assertEqual(config.events, ["heads", "remaining_parallel_checks"])

    def test_config_guard_is_idempotent_and_source_checked(self):
        cls = self.config_class()
        original = cls.verify_with_parallel_config
        compat._patch_model_config_class(cls, "0.24.0", "unused-when-already-patched")
        self.assertIs(cls.verify_with_parallel_config, original)
        class NewConfig(FakeModelConfig):
            pass
        original = NewConfig.verify_with_parallel_config
        with self.assertRaisesRegex(RuntimeError, "config source guard rejected"):
            compat._patch_model_config_class(NewConfig, "0.24.0", "wrong")
        self.assertIs(NewConfig.verify_with_parallel_config, original)


if __name__ == "__main__":
    unittest.main()
