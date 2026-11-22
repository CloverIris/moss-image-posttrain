"""Real subprocess integration tests of the tiny native-MOSS ART training loop.

Runs CPU only, in temporary output directories. Exercises fresh and resumed CLI
training, one/four synchronous workers, fixed-seed evaluation and exported LoRA.
The tiny model uses the same native forward/manual reverse and coordinator code
as the 5B model; these checks establish mechanics, not image-quality gains.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _run_cli(arguments, *, log, timeout=240, expect_failure=None):
    # Test cases describe overrides; materialize them in JSON before invoking
    # the public CLI, which accepts only an action and --config.
    arguments=list(map(str,arguments))
    config=json.loads(Path(arguments[arguments.index("--config")+1]).read_text())
    fields={"--max-updates":("training","max_updates",int),
            "--output":("runtime","output_dir",str),
            "--resume":("run","resume",str),"--init-adapter":("run","init_adapter",str),
            "--checkpoint":("run","checkpoint",str),"--adapter":("run","adapter",str)}
    for i in range(1,len(arguments),2):
        flag,value=arguments[i:i+2]
        if flag=="--config": continue
        section,key,cast=fields[flag]
        config.setdefault(section,{})[key]=cast(value)
    config_path=Path(log).with_suffix(".config.json")
    config_path.write_text(json.dumps(config))
    arguments=[arguments[0],"--config",str(config_path)]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="0")
    # Each run chooses its own CPU affinity from the original available cores.
    env.pop("MOSS_ART_CPUSET", None)
    command = [sys.executable, "-m", "moss_art.cli", *map(str, arguments)]
    print(f"[e2e] {Path(log).stem}", flush=True)
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill only the subprocess session created by THIS test, including its
        # spawned workers; never leave orphan training jobs after a test timeout.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        Path(log).write_text(output, encoding="utf-8")
        raise AssertionError(f"CLI timeout after {timeout}s: {command}\n{output[-12000:]}")
    Path(log).write_text(output, encoding="utf-8")
    if expect_failure is not None:
        if process.returncode == 0 or expect_failure not in output:
            raise AssertionError(f"Expected rejection {expect_failure!r}, got {process.returncode}: {output[-12000:]}")
        if '"event": "starting"' in output:
            raise AssertionError("Rejected run had already started workers")
    elif process.returncode:
        raise AssertionError(f"CLI failed with {process.returncode}: {command}\n{output[-16000:]}")
    return output


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.temporary = tempfile.TemporaryDirectory(prefix="moss-art-e2e-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.prompts = cls.directory/"prompts.jsonl"
        cls.prompts.write_text("\n".join(json.dumps({
            "sample_id": f"test-{i}", "prompt": prompt})
            for i, prompt in enumerate(("a red bird", "a blue fish", "a green tree", "a yellow flower",
                                        "a small house", "a white cloud", "a grey cat", "a black dog")))+"\n")
        config = {
            "name": "tiny-art-integration",
            "model": {"kind": "tiny", "dtype": "float32", "tiny_seed": 71},
            "lora": {"rank": 2, "alpha": 2., "preset": "attn", "seed": 11},
            "data": {"prompts": str(cls.prompts), "cache": str(cls.directory/"unused-cache")},
            "sampling": {"height": 32, "width": 32, "steps": 3, "group_size": 2,
                         "prompts_per_rollout": 4, "guidance_scale": 1., "skip_layers": [],
                         "train_timesteps": [0, 1]},
            "optimizer": {"learning_rate": .0001},
            "training": {"max_updates": 4, "seed": 97, "checkpoint_every": 2,
                         "evaluate_every": 0, "keep_checkpoints": 3},
            "reward": {"kind": "toy_latent", "decode": False},
            "runtime": {"platform": "cpu", "devices": [0], "cpu_threads": 1,
                        "worker_timeout": 180, "output_dir": str(cls.directory/"continuous")},
        }
        cls.config = cls.directory/"one-worker.json"
        cls.config.write_text(json.dumps(config))
        cls.continuous = cls.directory/"continuous"
        cls.resumed = cls.directory/"resumed"
        cls.parallel = cls.directory/"parallel"
        _run_cli(["train", "--config", cls.config], log=cls.directory/"continuous.log")
        _run_cli(["train", "--config", cls.config, "--max-updates", "2", "--output", cls.resumed],
                 log=cls.directory/"first-half.log")
        _run_cli(["train", "--config", cls.config, "--max-updates", "4", "--output", cls.resumed,
                  "--resume", cls.resumed], log=cls.directory/"resume.log")
        config["runtime"].update(devices=[0, 1, 2, 3], output_dir=str(cls.parallel))
        cls.parallel_config = cls.directory/"four-workers.json"
        cls.parallel_config.write_text(json.dumps(config))
        _run_cli(["train", "--config", cls.parallel_config], log=cls.directory/"parallel.log")
        # Load after all subprocesses finished; checkpoint arrays stay on host.
        from moss_art.checkpoint import load_checkpoint
        def load(output):
            latest = json.loads((output/"latest.json").read_text())
            return load_checkpoint(latest["checkpoint"])
        cls.full_state = load(cls.continuous)
        cls.resume_state = load(cls.resumed)
        cls.parallel_state = load(cls.parallel)
        cls.evaluation_outputs = [cls.directory/"evaluation_a", cls.directory/"evaluation_b"]
        for number, output in enumerate(cls.evaluation_outputs):
            _run_cli(["evaluate", "--config", cls.config, "--checkpoint", cls.continuous,
                      "--output", output], log=cls.directory/f"evaluation-{number}.log")


        cls.adapter_evaluation = cls.directory/"adapter_evaluation"
        _run_cli(["evaluate", "--config", cls.config, "--adapter", cls.continuous/"adapter.safetensors",
                  "--output", cls.adapter_evaluation], log=cls.directory/"adapter-evaluation.log")
        warm_config = json.loads(cls.config.read_text())
        cls.warm_started = cls.directory/"warm-started"
        warm_config["runtime"]["output_dir"] = str(cls.warm_started)
        warm_config["training"].update(max_updates=2, evaluate_every=2)
        cls.warm_config = cls.directory/"warm-start.json"
        cls.warm_config.write_text(json.dumps(warm_config))
        _run_cli(["train", "--config", cls.warm_config, "--init-adapter", cls.continuous/"adapter.safetensors"],
                 log=cls.directory/"warm-start.log")
        cls.warm_state = load(cls.warm_started)

    def assertTrees(self, expected, actual, *, exact):
        import jax
        e, et = jax.tree.flatten(expected)
        a, at = jax.tree.flatten(actual)
        self.assertEqual(et, at)
        self.assertEqual(len(e), len(a))
        for index, (x, y) in enumerate(zip(e, a)):
            x, y = np.asarray(x), np.asarray(y)
            self.assertEqual(x.dtype, y.dtype, f"leaf {index} dtype")
            self.assertEqual(x.shape, y.shape, f"leaf {index} shape")
            if exact or x.dtype.kind not in "fc":
                np.testing.assert_array_equal(x, y, err_msg=f"leaf {index}")
            else:
                np.testing.assert_allclose(x, y, atol=3e-7, rtol=3e-5, err_msg=f"leaf {index}")

    @staticmethod
    def _updates(output):
        return [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()
                if json.loads(line).get("event") == "update"]

    def test_resume_matches_continuous_parameters_and_adamw_bitwise(self):
        self.assertTrees(self.full_state[:2], self.resume_state[:2], exact=True)
        for name in ("update", "rollout", "prompt_cursor", "seed", "config_fingerprint", "data_sha256"):
            self.assertEqual(self.full_state[2][name], self.resume_state[2][name], name)
        self.assertEqual(self.full_state[2]["update"], 4)
        self.assertEqual(self.full_state[2]["rollout"], 2)
        self.assertEqual(self.full_state[2]["prompt_cursor"], 8)
        self.assertEqual([event["update"] for event in self._updates(self.resumed)], [1, 2, 3, 4])
        for a, b in zip(self._updates(self.continuous), self._updates(self.resumed)):
            for key in ("loss", "grad_norm", "adapter_delta_norm", "reward_mean", "items",
                        "masked_fraction", "active_fraction"):
                self.assertEqual(a[key], b[key], key)

    def test_four_cpu_workers_match_single_worker_synchronous_updates(self):
        self.assertTrees(self.full_state[:2], self.parallel_state[:2], exact=False)
        one, four = self._updates(self.continuous), self._updates(self.parallel)
        self.assertEqual(len(one), 4)
        self.assertEqual(len(four), 4)
        for a, b in zip(one, four):
            self.assertEqual(a["items"], 8)
            self.assertEqual(a["items"], b["items"])
            for key in ("loss", "grad_norm", "adapter_delta_norm", "reward_mean"):
                np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=5e-5, err_msg=key)
        self.assertGreater(max(event["grad_norm"] for event in one), 0.)
        self.assertGreater(sum(event["adapter_delta_norm"] for event in one), 0.)
        info = json.loads((self.parallel/"run_info.json").read_text())
        self.assertEqual(len(info["worker_devices"]), 4)
        self.assertTrue(info["host_reduction"])

    def test_fixed_seed_evaluation_repeats_exactly(self):
        results = [json.loads((output/"evaluation"/"step_00000004"/"metrics.json").read_text())
                   for output in self.evaluation_outputs]
        self.assertEqual(results[0], results[1])
        self.assertTrue(results[0]["fixed_seeds"])
        self.assertEqual(len(results[0]["rewards"]), 8)
        self.assertTrue(np.isfinite(results[0]["rewards"]).all())

    def test_exported_safetensors_matches_checkpoint_adapter(self):
        from safetensors import safe_open
        from safetensors.numpy import load_file
        from moss_art.lora import import_safetensors
        lora = self.full_state[0]
        adapter = self.continuous/"adapter.safetensors"
        self.assertTrue(adapter.is_file())
        tensors = load_file(str(adapter))
        for key, value in lora.items():
            exported_key = "diffusion_model."+key
            self.assertIn(exported_key, tensors)
            np.testing.assert_array_equal(tensors[exported_key], value)
        alpha_keys = [key for key in tensors if key.endswith(".alpha")]
        self.assertEqual(len(alpha_keys), len(lora)//2)
        for key in alpha_keys:
            self.assertEqual(float(tensors[key]), 2.)
        with safe_open(str(adapter), framework="numpy") as f:
            self.assertEqual(f.metadata()["scale_rule"], "alpha/rank")
        imported = import_safetensors(adapter, lora, alpha=2.)
        self.assertTrees(lora, imported, exact=True)
        completion = json.loads((self.continuous/"complete.json").read_text())
        self.assertEqual(completion["update"], 4)
        self.assertEqual(Path(completion["adapter"]), adapter)



    def test_adapter_cli_evaluation_and_fresh_optimizer_warm_start(self):
        import hashlib
        import jax
        expected = json.loads((self.evaluation_outputs[0]/"evaluation"/"step_00000004"/"metrics.json").read_text())
        adapter_result = json.loads((self.adapter_evaluation/"adapter_evaluation.json").read_text())
        initial_eval = json.loads((self.warm_started/"evaluation"/"step_00000000"/"metrics.json").read_text())
        self.assertEqual(adapter_result["result"]["rewards"], expected["rewards"])
        self.assertEqual(initial_eval["rewards"], expected["rewards"])
        source = self.continuous/"adapter.safetensors"
        identity = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        self.assertEqual(adapter_result["source_adapter"], identity)
        self.assertEqual(self.warm_state[2]["initial_adapter"], identity)
        info = json.loads((self.warm_started/"run_info.json").read_text())
        self.assertEqual(info["initial_adapter"], identity)
        self.assertEqual(self.warm_state[2]["update"], 2)
        self.assertEqual(self.warm_state[2]["rollout"], 1)
        self.assertEqual(self.warm_state[2]["prompt_cursor"], 4)
        # Adam count must be 2 (fresh), not 6 (continued from source checkpoint).
        counts = [int(np.asarray(value)) for value in jax.tree.leaves(self.warm_state[1])
                  if np.asarray(value).shape == () and np.asarray(value).dtype.kind in "iu"]
        self.assertEqual(counts, [2])
        self.assertTrue(any(not np.array_equal(self.warm_state[0][k], v)
                            for k, v in self.full_state[0].items()))

    def _output_snapshot(self):
        names = ("config.json", "metrics.jsonl", "run_info.json", "latest.json",
                 "complete.json", "adapter.safetensors")
        return {name: (self.continuous/name).read_bytes() for name in names}

    def test_rejected_resume_preserves_existing_output(self):
        from moss_art.checkpoint import save_checkpoint
        before = self._output_snapshot()
        changed = json.loads(self.config.read_text())
        changed["lora"]["rank"] = 3
        changed_config = self.directory/"incompatible-config.json"
        changed_config.write_text(json.dumps(changed))
        _run_cli(["train", "--config", changed_config, "--resume", self.continuous],
                 log=self.directory/"rejected-config.log", expect_failure="incompatible resume")
        self.assertEqual(self._output_snapshot(), before)
        # A validly serialized checkpoint claiming different numerical source is
        # an asset mismatch, independently of tensor checksum corruption.
        lora, optimizer, metadata = self.full_state
        changed_metadata = copy.deepcopy(metadata)
        changed_metadata["experiment"]["source_files"]["moss_art/model.py"]["sha256"] = "0"*64
        changed_checkpoint = save_checkpoint(self.directory/"incompatible-assets", lora, optimizer, changed_metadata)
        _run_cli(["train", "--config", self.config, "--resume", changed_checkpoint],
                 log=self.directory/"rejected-assets.log", expect_failure="incompatible resume")
        self.assertEqual(self._output_snapshot(), before)

    def test_active_output_writer_rejects_second_process(self):
        from moss_art.locking import output_lock
        before = self._output_snapshot()
        with output_lock(self.continuous):
            _run_cli(["train", "--config", self.config, "--resume", self.continuous],
                     log=self.directory/"rejected-lock.log", expect_failure="active writer")
        self.assertEqual(self._output_snapshot(), before)

    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"integration": "tiny-native-moss-art", "cpu_only": True,
                          "seconds": round(time.monotonic()-cls.started, 2),
                          "continuous_updates": 4, "resume_split": [2, 2],
                          "worker_counts": [1, 4], "fixed_seed_evaluations": 2}), flush=True)


if __name__ == "__main__":
    unittest.main()
