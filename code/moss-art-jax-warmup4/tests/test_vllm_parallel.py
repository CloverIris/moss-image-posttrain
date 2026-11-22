"""Host-only DP orchestration/merge tests; no vLLM, Torch or GPU is loaded."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from moss_art.data import atomic_json, sha256_file
from moss_art.vllm_contract import storage_contract, encode_embedding_arrays, validate_conditioning_contract
from moss_art.vllm_parallel import balanced_shards, merge_manifests, normalize_jobs, replica_groups, run_parallel, validate_entry, _stop_process


def job(name, length=4):
    return {"key": hashlib.sha256(name.encode()).hexdigest(), "input_ids": list(range(1, length + 1))}


def manifest(storage="float32"):
    contract = storage_contract(storage)
    return {"schema_version": 1, "encoder": "moss-vllm-qwen3.5",
            "model_identity": {"fingerprint": "fixed-model-tokenizer"},
            "max_sequence_length": 256, "prefill_buckets": {"short": 32, "full": 256},
            "conditioning_contract": contract, "embedding_storage_dtype": storage,
            "padding": "valid tokens only", "entries": {}, "complete": False,
            "dropped_records": [{"id": "metadata-only", "required_tokens": 300}]}


def producer(path, version="fixed"):
    return {"vllm_version": version, "adapter_sha256": "same-source",
            "model_view": str(path), "compute_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 2}}


def append_entry(directory, target_manifest, item, fill=1.0):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    ids = np.asarray(item["input_ids"], dtype=np.int32)[None]
    shape = (1, len(item["input_ids"]), target_manifest["conditioning_contract"]["hidden_size"])
    storage = validate_conditioning_contract(target_manifest["conditioning_contract"])
    floats = np.full(shape, fill, np.float32)
    raw = encode_embedding_arrays(floats, storage)
    filename = directory / (item["key"] + ".npz")
    np.savez_compressed(filename, embedding=raw, attention_mask=np.ones(ids.shape, bool), input_ids=ids)
    target_manifest["entries"][item["key"]] = {
        "file": filename.name, "sha256": sha256_file(filename), "shape": list(shape),
        "token_count": ids.shape[1], "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "embedding_storage_dtype": storage}


def write_shard(directory, parent, items, *, version="fixed", fill=1.0):
    child = {key: copy.deepcopy(value) for key, value in parent.items()
             if key not in ("entries", "complete", "dropped_records", "parallel_producers", "parallel_execution", "backend")}
    child.update(entries={}, backend=producer(Path(directory) / "view", version), complete=True)
    for item in items:
        append_entry(directory, child, item, fill)
    atomic_json(Path(directory) / "manifest.json", child)
    return child


class FinishedProcess:
    def __init__(self, code=0):
        self.returncode = code

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {"preparation": {"limit": 0, "vllm": {
                    "python": "unused-python", "devices": [1, 2, 0, 3], "data_parallel_size": 2,
                    "pipeline_parallel_size": 2, "tensor_parallel_size": 1, "model_path": str(self.root / "views")}},
                "data": {"cache": str(self.root), "prompts": "SOURCE_MUST_NOT_BE_OPENED.jsonl",
                         "context_length": 256, "short_context_length": 32},
                "runtime": {"devices": [0, 1, 2, 3]}}

    def test_balance_dedupe_and_gpu_contract(self):
        items = [job("a", 100), job("b", 90), job("c", 20), job("d", 10)]
        shards = balanced_shards(items + [items[0]], 2)
        self.assertEqual(sorted(sum(len(row["input_ids"]) for row in shard) for shard in shards), [110, 110])
        self.assertEqual(sum(map(len, shards)), 4)
        self.assertEqual(shards, balanced_shards(list(reversed(items)), 2))
        self.assertEqual(replica_groups(self.config()), [[1, 2], [0, 3]])
        bad = self.config(); bad["preparation"]["vllm"]["devices"] = [0, 1, 1, 2]
        with self.assertRaises(ValueError): replica_groups(bad)
        with self.assertRaises(ValueError): normalize_jobs([items[0], {**items[0], "input_ids": [9]}])
        with self.assertRaises(ValueError): normalize_jobs([{**items[0], "prompt": "not a token job"}])

    def test_strict_merge_allows_only_view_path_difference_and_bf16_storage(self):
        parent = manifest("bfloat16")
        items = [job("a", 3), job("b", 5)]
        a, b = self.root / "a", self.root / "b"
        write_shard(a, parent, [items[0]])
        write_shard(b, parent, [items[1]])
        result = merge_manifests(parent, self.root / "cache", [a, b], items)
        self.assertEqual(set(result["entries"]), {row["key"] for row in items})
        self.assertNotIn("model_view", result["backend"])
        self.assertEqual(len(result["parallel_producers"]), 2)
        self.assertEqual(parent["entries"], {})
        for item in items:
            entry = result["entries"][item["key"]]
            validate_entry(self.root / "cache", item["key"], entry, parent["conditioning_contract"], item)
            with np.load(self.root / "cache" / entry["file"]) as arrays:
                self.assertEqual(arrays["embedding"].dtype, np.uint16)

    def test_duplicate_same_content_merges_and_conflict_keeps_manifest(self):
        parent = manifest(); item = job("same")
        a, b, destination = self.root / "a", self.root / "b", self.root / "cache"
        write_shard(a, parent, [item]); write_shard(b, parent, [item])
        # NPZ ZIP timestamps can change its byte hash, so use the same actual
        # bytes for this deliberate duplicate-content test.
        target = b / (item["key"] + ".npz")
        target.write_bytes((a / target.name).read_bytes())
        bmanifest = json.loads((b / "manifest.json").read_text())
        bmanifest["entries"][item["key"]]["sha256"] = sha256_file(target)
        atomic_json(b / "manifest.json", bmanifest)
        merged = merge_manifests(parent, destination, [a, b], [item])
        before = (destination / "manifest.json").read_bytes()
        write_shard(b, parent, [item], fill=2.0)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            merge_manifests(merged, destination, [b], [item])
        self.assertEqual((destination / "manifest.json").read_bytes(), before)

    def test_bad_checksum_or_producer_never_commits_root_manifest(self):
        parent = manifest(); items = [job("a"), job("b")]
        a, b, destination = self.root / "a", self.root / "b", self.root / "cache"
        write_shard(a, parent, [items[0]]); write_shard(b, parent, [items[1]], version="different-source")
        atomic_json(destination / "manifest.json", parent)
        before = (destination / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "producer"):
            merge_manifests(parent, destination, [a, b], items)
        self.assertEqual((destination / "manifest.json").read_bytes(), before)
        write_shard(b, parent, [items[1]])
        (b / (items[1]["key"] + ".npz")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupted"):
            merge_manifests(parent, destination, [a, b], items)
        self.assertEqual((destination / "manifest.json").read_bytes(), before)

    def test_rejects_escaping_files_token_mismatch_and_zero_features(self):
        parent = manifest(); item = job("a"); shard = self.root / "shard"
        child = write_shard(shard, parent, [item])
        entry = child["entries"][item["key"]]
        with self.assertRaisesRegex(ValueError, "planned"):
            validate_entry(shard, item["key"], entry, parent["conditioning_contract"], [99] * 4)
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate_entry(shard, item["key"], {**entry, "file": "../outside.npz"}, parent["conditioning_contract"])
        zero = write_shard(shard, parent, [item], fill=0)
        with self.assertRaises(ValueError):
            validate_entry(shard, item["key"], zero["entries"][item["key"]], parent["conditioning_contract"])

    def test_missing_coverage_and_shared_metadata_mismatch_do_not_commit(self):
        parent = manifest(); items = [job("a"), job("b")]
        shard, destination = self.root / "shard", self.root / "cache"
        atomic_json(destination / "manifest.json", parent)
        before = (destination / "manifest.json").read_bytes()
        partial = write_shard(shard, parent, items[:1])
        with self.assertRaisesRegex(ValueError, "cover"):
            merge_manifests(parent, destination, [shard], items)
        self.assertEqual((destination / "manifest.json").read_bytes(), before)
        partial["model_identity"]["fingerprint"] = "different-model"
        atomic_json(shard / "manifest.json", partial)
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            merge_manifests(parent, destination, [shard], items[:1])
        self.assertEqual((destination / "manifest.json").read_bytes(), before)

    def test_worker_failure_cleans_up_active_replica(self):
        parent = manifest(); items = [job("a"), job("b")]
        processes = []

        class ActiveProcess(FinishedProcess):
            def __init__(self):
                super().__init__(None)

        def launch(command, **kwargs):
            process = ActiveProcess() if not processes else FinishedProcess(7)
            processes.append(process)
            return process

        # The process-control primitive is checked separately by the OS. Here
        # ensure every launched child reaches cleanup when a peer fails, with
        # no fake PIDs sent to real killpg during this host-only test.
        with patch("moss_art.vllm_parallel._stop_process") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "replica 1"):
                run_parallel(self.config(), items, parent, self.root, popen_factory=launch)
        self.assertEqual([call.args[0] for call in cleanup.call_args_list], processes)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_exited_leader_still_cleans_its_owned_group_and_refuses_parent_group(self):
        leader = FinishedProcess(1)
        with patch("moss_art.vllm_parallel.os.name", "posix"), \
                patch("moss_art.vllm_parallel.os.getpgrp", return_value=123, create=True), \
                patch("moss_art.vllm_parallel.signal.SIGKILL", 9, create=True), \
                patch("moss_art.vllm_parallel.os.killpg", create=True) as kill:
            # Descendants ignore TERM; group cleanup must not stop at the
            # already exited direct child's status and must escalate.
            _stop_process(leader, owned_pgid=456, timeout=0)
            self.assertEqual(kill.call_args_list[0].args, (456, signal.SIGTERM))
            self.assertEqual(kill.call_args_list[-1].args, (456, signal.SIGKILL))
            self.assertTrue(all(call.args[0] == 456 for call in kill.call_args_list))
            kill.reset_mock()
            with self.assertRaisesRegex(ValueError, "unowned/parent"):
                _stop_process(leader, owned_pgid=123, timeout=0)
            kill.assert_not_called()

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux process-group regression")
    def test_real_exited_parent_does_not_leave_sleeping_grandchild(self):
        # Isolated CPU-only process tree: its session leader exits first,
        # reproducing the case where a failed vLLM parent leaves PP workers.
        source = ("import subprocess,sys; "
                  "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                  "print(child.pid,flush=True)")
        leader = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE,
                                  text=True, start_new_session=True)
        try:
            child_pid = int(leader.stdout.readline())
            self.assertEqual(leader.wait(timeout=5), 0)
            _stop_process(leader, owned_pgid=leader.pid, timeout=0.2)
            deadline = time.monotonic() + 2
            while True:
                stat = Path(f"/proc/{child_pid}/stat")
                # A reparented zombie is already terminated; only the init
                # reaper can collect it. A live/sleeping child is the failure.
                stopped = not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
                if stopped or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
            self.assertTrue(stopped, "owned grandchild survived group cleanup")
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            leader.stdout.close()

    def test_sigterm_during_spawn_registers_child_then_cleans_and_restores_handler(self):
        before = signal.getsignal(signal.SIGTERM)
        processes, logs = [], []

        def launch(command, **kwargs):
            process = FinishedProcess()
            processes.append(process); logs.append(kwargs["stdout"])
            # Deliver the Python callback inside Popen's critical section:
            # the child must be registered before the unwind takes effect.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return process

        with patch("moss_art.vllm_parallel._stop_process") as cleanup:
            with self.assertRaises(SystemExit) as error:
                run_parallel(self.config(), [job("a"), job("b")], manifest(), self.root, popen_factory=launch)
        self.assertEqual(error.exception.code, 128 + signal.SIGTERM)
        self.assertEqual([call.args[0] for call in cleanup.call_args_list], processes)
        self.assertEqual(len(processes), 1)
        self.assertTrue(all(log.closed for log in logs))
        self.assertIs(signal.getsignal(signal.SIGTERM), before)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_background_thread_entry_does_not_change_signal_handlers(self):
        errors = []

        def launch(command, **kwargs):
            rows = [json.loads(line) for line in Path(command[command.index("--worker-jobs") + 1]).read_text().splitlines()]
            write_shard(Path(command[command.index("--worker-cache") + 1]), rows[0]["settings"], rows[1:])
            return FinishedProcess()

        def run():
            try:
                run_parallel(self.config(), [job("a"), job("b")], manifest(), self.root, popen_factory=launch)
            except BaseException as error:
                errors.append(error)

        with patch("moss_art.vllm_parallel.signal.signal", side_effect=AssertionError("thread changed a process signal")):
            thread = threading.Thread(target=run)
            thread.start(); thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(len(json.loads((self.root / "manifest.json").read_text())["entries"]), 2)

    def test_cleanup_failure_still_attempts_other_replica_and_closes_logs(self):
        processes, logs = [], []

        def launch(command, **kwargs):
            rows = [json.loads(line) for line in Path(command[command.index("--worker-jobs") + 1]).read_text().splitlines()]
            write_shard(Path(command[command.index("--worker-cache") + 1]), rows[0]["settings"], rows[1:])
            process = FinishedProcess(); processes.append(process); logs.append(kwargs["stdout"])
            return process

        with patch("moss_art.vllm_parallel._stop_process", side_effect=[OSError("first cleanup failed"), None]) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "Could not clean up"):
                run_parallel(self.config(), [job("a"), job("b")], manifest(), self.root, popen_factory=launch)
        self.assertEqual([call.args[0] for call in cleanup.call_args_list], processes)
        self.assertTrue(all(log.closed for log in logs))

    def test_separate_jobs_configs_and_completed_cache_does_not_spawn(self):
        parent = manifest(); items = [job(str(index), 3 + index) for index in range(5)]
        atomic_json(self.root / "manifest.json", parent)
        config = self.config(); original = copy.deepcopy(config); launches = []

        def launch(command, **kwargs):
            child = json.loads(Path(command[command.index("--config") + 1]).read_text())
            rows = [json.loads(line) for line in Path(command[command.index("--worker-jobs") + 1]).read_text().splitlines()]
            cache = Path(command[command.index("--worker-cache") + 1])
            self.assertEqual(set(rows[0]), {"settings"})
            self.assertTrue(all(set(row) == {"key", "input_ids"} for row in rows[1:]))
            self.assertEqual(child["data"]["prompts"], command[command.index("--worker-jobs") + 1])
            self.assertEqual(child["preparation"]["vllm"]["data_parallel_size"], 1)
            self.assertIn(child["preparation"]["vllm"]["devices"], [[1, 2], [0, 3]])
            self.assertEqual(child["runtime"]["devices"], child["preparation"]["vllm"]["devices"])
            self.assertTrue(child["preparation"]["vllm"]["model_path"].endswith("dp-" + str(len(launches))))
            launches.append((cache, rows[1:]))
            write_shard(cache, rows[0]["settings"], rows[1:])
            return FinishedProcess()

        result = run_parallel(config, items, parent, self.root, popen_factory=launch)
        self.assertEqual(len(result["entries"]), len(items))
        report = json.loads((self.root / "parallel_report.json").read_text())
        self.assertEqual(report["launched_workers"], 2)
        self.assertEqual(config, original)
        self.assertFalse({x["key"] for x in launches[0][1]} & {x["key"] for x in launches[1][1]})
        complete = json.loads((self.root / "manifest.json").read_text())
        with patch("moss_art.vllm_parallel.subprocess.Popen", side_effect=AssertionError("must not spawn")):
            again = run_parallel(config, items, complete, self.root)
        self.assertEqual(again["entries"], complete["entries"])
        report = json.loads((self.root / "parallel_report.json").read_text())
        self.assertEqual(report["encoded_now"], 0)

    def test_failure_and_resume_reuses_completed_rank_and_partial_files(self):
        parent = manifest(); items = [job(str(index), 4 + index) for index in range(6)]
        atomic_json(self.root / "manifest.json", parent)
        before = (self.root / "manifest.json").read_bytes()
        launches = []

        def launch(command, **kwargs):
            rows = [json.loads(line) for line in Path(command[command.index("--worker-jobs") + 1]).read_text().splitlines()]
            cache = Path(command[command.index("--worker-cache") + 1])
            rank = int(cache.name.rsplit("-", 1)[1]); launches.append(rank)
            if rank == 1 and launches.count(1) == 1:
                write_shard(cache, rows[0]["settings"], rows[1:2])
                return FinishedProcess(1)
            write_shard(cache, rows[0]["settings"], rows[1:])
            return FinishedProcess()

        with self.assertRaises(RuntimeError):
            run_parallel(self.config(), items, parent, self.root, popen_factory=launch)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        result = run_parallel(self.config(), items, parent, self.root, popen_factory=launch)
        self.assertEqual(launches, [0, 1, 1])
        self.assertEqual(len(result["entries"]), len(items))
        report = json.loads((self.root / "parallel_report.json").read_text())
        self.assertEqual(report["launched_workers"], 1)
        self.assertEqual(report["encoded_now"], 2)
        self.assertEqual(len(json.loads((self.root / "manifest.json").read_text())["entries"]), 6)


if __name__ == "__main__":
    unittest.main()
