"""CPU-only transactional/protocol tests; no model or CUDA imports."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_reward_with_validation as wrapper


class FakePipeline:
    """Exercise original-pipeline delegation without invoking a model."""
    def __init__(self):
        self.calls = []
        self.fail_once = False

    @staticmethod
    def digest(value):
        return wrapper.digest(value)

    @staticmethod
    def provenance(cfg):
        return {"fake_existing_scorer": "same-code-sha256"}

    @staticmethod
    def canonical_items(request):
        rows = []
        ids = set()
        for i, item in enumerate(request["items"]):
            row = copy.deepcopy(item)
            row["id"] = str(row.get("id", i))
            if row["id"] in ids:
                raise ValueError("Duplicate fake input IDs")
            ids.add(row["id"])
            row["prompt_id"] = wrapper.group_id(row)
            row["image"] = str(Path(row["image"]).resolve())
            row["image_sha256"] = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
            rows.append(row)
        return rows

    def run(self, cfg, request, output):
        self.calls.append(copy.deepcopy(request))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated model stage interruption")
        fingerprint, archive = wrapper.pipeline_identity(self, cfg, request)
        rows = []
        for i, item in enumerate(self.canonical_items(request)):
            aesthetic=("kawai","deepghs","hps") if cfg.get("kawai_enabled",True) else ("deepghs","hps")
            models = aesthetic + (("sdg_artifact",) if cfg.get("sdg_enabled", True) else ())
            rows.append(dict(id=item["id"], image=item["image"], prompt_id=item["prompt_id"],
                             reward=i / 100, raw_scores={name: i / 100 for name in
                                 models},
                             normalized_scores={name: i / 100 for name in aesthetic},
                             contributions={name: i / 100 for name in
                                 models}))
        result = dict(scores=[row["reward"] for row in rows], details=rows,
                      archive=str(archive), fingerprint=fingerprint)
        wrapper.save(output, result)
        return result


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = self.root / "service"
        self.outbox = self.service / "outbox"
        self.outbox.mkdir(parents=True)
        self.step = self.root / "training" / "step_00000007"
        self.step.mkdir(parents=True)
        self.batch = dict(hps=dict(batch_size=2, replica_devices=[[0, 3], [1, 2]]))
        wrapper.save(self.root / "batch.json", self.batch)
        wrapper.save(self.root / "sdg.json", {"dtype": "bfloat16"})
        wrapper.save(self.root / "calibration.json", {"frozen": True})
        self.cfg = dict(batch_config=str(self.root / "batch.json"),
                        batch_root=str(self.root), sdg_config=str(self.root / "sdg.json"),
                        calibration=str(self.root / "calibration.json"), archive_root=str(self.step),
                        allowed_archive_root=str(self.root),
                        checkpoint_validation=dict(enabled=True, service_root=str(self.service)))
        self.request = dict(version=1, context=dict(step=7, step_dir=str(self.step)),
                            items=self.make_items(self.step, "training", 24, 8))
        self.pipeline = FakePipeline()

    def tearDown(self):
        self.temp.cleanup()

    def make_items(self, directory, prefix, prompts, repeats):
        directory.mkdir(parents=True, exist_ok=True)
        result = []
        for prompt in range(prompts):
            for repeat in range(repeats):
                key = f"{prefix}:p{prompt:03}:r{repeat:02}"
                image = directory / f"{prefix}_{prompt}_{repeat}.png"
                image.write_bytes(b"fixture-image-" + key.encode())
                result.append(dict(id=key, prompt_id=f"{prefix}-prompt-{prompt}",
                                   image=str(image), prompt=f"prompt:\nexact full text {prompt}",
                                   negative_prompt="do not modify", roll_id=repeat, seed=repeat))
        return result

    def ready(self, name="ckpt_00000006", step=6, prompts=4, repeats=4):
        directory = self.service / "jobs" / name
        items = self.make_items(directory / "images", "eval", prompts, repeats)
        request = dict(version=1, context=dict(checkpoint_id=name, source_step=step), items=items)
        path = directory / "eval_request.json"
        wrapper.save(path, request)
        ready = dict(version=1, status="ready_for_scoring", job_id=name, checkpoint_id=name, source_step=step,
                     checkpoint_manifest_sha256="a" * 64,
                     request_path=str(path), result_dir=str(directory), image_count=len(items),
                     request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        wrapper.save(self.outbox / (name + ".ready.json"), ready)
        return ready, request

    def invoke(self, output="reward.json"):
        return wrapper.run(self.cfg, self.request, self.step / output,
                           pipeline=self.pipeline, splitter=wrapper.split_groups)

    def test_append_once_return_only_train_and_ack(self):
        ready, eval_request = self.ready()
        result = self.invoke()
        self.assertEqual(len(self.pipeline.calls), 1)
        combined = self.pipeline.calls[0]
        self.assertEqual(len(combined["items"]), 208)
        self.assertEqual([r["id"] for r in combined["items"][:192]],
                         [r["id"] for r in self.request["items"]])
        self.assertEqual([r["prompt"] for r in combined["items"][-16:]],
                         [r["prompt"] for r in eval_request["items"]])
        self.assertEqual([r["negative_prompt"] for r in combined["items"][-16:]],
                         [r["negative_prompt"] for r in eval_request["items"]])
        self.assertEqual(len(result["scores"]), 192)
        self.assertEqual(len(result["details"]), 192)
        self.assertTrue(all(not r["id"].startswith(wrapper.EVAL_PREFIX) for r in result["details"]))
        self.assertTrue((self.outbox / (ready["job_id"] + ".ready.json")).exists())
        self.assertTrue((self.outbox / (ready["job_id"] + ".ack.json")).exists())
        ack = wrapper.read_json(self.outbox / (ready["job_id"] + ".ack.json"))
        self.assertEqual(ack["status"], "complete")
        self.assertEqual(len(ack["scores"]), 16)
        self.assertEqual(ack["checkpoint_manifest_sha256"], "a" * 64)
        evaluation = wrapper.read_json(Path(ready["result_dir"]) / "result.json")
        self.assertEqual(len(evaluation["scores"]), 16)
        self.assertEqual([r["original_id"] for r in evaluation["details"]],
                         [r["id"] for r in eval_request["items"]])
        summary = wrapper.read_json(Path(ready["result_dir"]) / "summary.json")
        self.assertEqual(len(summary["groups"]), 4)
        domains = wrapper.read_json(Path(result["archive"]) / "evaluation_domains.json")
        self.assertEqual((domains["training_count"], domains["validation_count"]), (192, 16))
        self.assertEqual(len(domains["train_ids"]), 192)
        training_summary = wrapper.read_json(Path(result["archive"]) / "training_summary.json")
        self.assertEqual(training_summary["images"], 192)
        self.assertTrue(evaluation["hps_batch_plan"]["batch_companions_preserved"])
        again = self.invoke("second_response.json")
        self.assertEqual(again, result)
        self.assertEqual(len(self.pipeline.calls), 1, "A committed request must not rescore")

    def test_failure_keeps_ready_claim_and_fixed_manifest(self):
        first, _ = self.ready()
        self.pipeline.fail_once = True
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        self.assertFalse((self.outbox / (first["job_id"] + ".ack.json")).exists())
        self.assertTrue((self.outbox / (first["job_id"] + ".claim.json")).exists())
        self.ready("ckpt_00000007", 7)
        self.invoke()
        self.assertEqual(self.pipeline.calls[0], self.pipeline.calls[1])
        self.assertEqual(len(self.pipeline.calls[1]["items"]), 208)
        self.assertFalse((self.outbox / "ckpt_00000007.claim.json").exists())

    def test_no_ready_does_not_wait_or_modify_train_scores(self):
        result = self.invoke()
        self.assertEqual(len(self.pipeline.calls[0]["items"]), 192)
        self.assertEqual(len(result["scores"]), 192)
        self.assertEqual(result["checkpoint_validation"]["jobs"], [])

    def test_incomplete_ready_is_deferred_without_blocking_train(self):
        ready, _ = self.ready(repeats=3)
        result = self.invoke()
        self.assertEqual(result["checkpoint_validation"]["checkpoint_images"], 0)
        self.assertFalse((self.outbox / (ready["job_id"] + ".claim.json")).exists())

    def test_existing_ack_is_never_scored_in_next_step(self):
        ready, _ = self.ready()
        self.invoke()
        self.request["context"]["step"] = 8
        self.invoke("next_step_response.json")
        self.assertEqual(len(self.pipeline.calls[-1]["items"]), 192)

    def test_id_prefixes_prevent_same_prompt_id_from_merging(self):
        ready, eval_request = self.ready()
        # Deliberately make a source eval prompt id collide with a training id.
        for row in eval_request["items"][:4]:
            row["prompt_id"] = self.request["items"][0]["prompt_id"]
        path = Path(ready["request_path"])
        wrapper.save(path, eval_request)
        ready["request_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        wrapper.save(self.outbox / (ready["job_id"] + ".ready.json"), ready)
        self.invoke()
        combined = self.pipeline.calls[0]
        self.assertNotEqual(combined["items"][0]["prompt_id"], combined["items"][192]["prompt_id"])

    def test_hps_pairing_protected_for_current_192_plus_16(self):
        ready, _ = self.ready()
        canonical_train = self.pipeline.canonical_items(self.request)
        record = wrapper.ready_record(self.outbox / (ready["job_id"] + ".ready.json"),
                                      self.cfg["checkpoint_validation"])
        plan = wrapper.hps_plan(canonical_train, record["items"], self.batch)
        self.assertEqual([len(shard) for shard in plan["train_batches"]], [48, 48])
        self.assertEqual([len(shard) for shard in plan["combined_batches"]], [52, 52])
        self.assertEqual([len(shard) for shard in plan["eval_batches"]], [4, 4])

    def test_hps_unsafe_batch_composition_is_deferred(self):
        self.ready()
        # Uneven groups sorted around the eval groups can mix a singleton train
        # image into a capability-conditioned batch with an eval image.
        self.request["items"] = (self.request["items"][:1] + self.request["items"][8:9]
                                 + self.request["items"][16:21])
        result = self.invoke()
        self.assertEqual(len(result["scores"]), 7)
        self.assertEqual(result["checkpoint_validation"]["checkpoint_images"], 0)

    def test_checksum_changed_claim_is_rejected(self):
        ready, _ = self.ready()
        ready["request_sha256"] = "invalid"
        wrapper.save(self.outbox / (ready["job_id"] + ".ready.json"), ready)
        result = self.invoke()
        self.assertEqual(result["checkpoint_validation"]["checkpoint_images"], 0)

    def test_source_request_mutation_on_retry_does_not_recombine(self):
        self.ready()
        self.pipeline.fail_once = True
        with self.assertRaises(RuntimeError):
            self.invoke()
        # A later ready mutation cannot alter the frozen combined request.
        path = next(self.outbox.glob("*.ready.json"))
        value = wrapper.read_json(path)
        value["checkpoint_id"] = "changed-but-never-used-on-retry"
        wrapper.save(path, value)
        self.invoke()
        self.assertEqual(self.pipeline.calls[0], self.pipeline.calls[1])

    def test_resume_repairs_orphan_without_deleting_raw_evidence(self):
        archive = self.root / "reward_models" / "fake"
        directory = archive / "deepghs_hps.work" / "hps" / "dp00"
        directory.mkdir(parents=True)
        raw = b'{"id":"a","result":{"mu":1}}\n{"id":"b","result":{"mu":2}}\n{"id":"c","result":{"mu":3}}\n{"id":'
        (directory / "hps.jsonl").write_bytes(raw)
        wrapper.save(directory / "hps.metrics.json", {"stale": True})
        changes = wrapper.repair_hps_resume(archive, dict(combined_batches=[[("a", "b"), ("c", "d")]]))
        self.assertEqual(len(changes), 1)
        self.assertEqual(Path(changes[0]["preserved"]).read_bytes(), raw)
        remaining = [json.loads(line)["id"] for line in (directory / "hps.jsonl").read_text().splitlines()]
        self.assertEqual(remaining, ["a", "b"])
        self.assertFalse((directory / "hps.metrics.json").exists())
        self.assertEqual(wrapper.repair_hps_resume(archive, dict(combined_batches=[[("a", "b"), ("c", "d")]])), [])

    def test_sdg_disabled_has_no_synthetic_component_or_config_read(self):
        self.ready()
        self.cfg.update(sdg_enabled=False, artifact_weight=0)
        self.cfg.pop("sdg_config")
        result = self.invoke()
        train_summary = wrapper.read_json(Path(result["archive"]) / "training_summary.json")
        self.assertFalse(train_summary["sdg_enabled"])
        self.assertEqual(set(train_summary["raw_component_mean"]), {"kawai", "deepghs", "hps"})
        self.assertEqual(set(train_summary["contribution_mean"]), {"kawai", "deepghs", "hps"})
        ack = wrapper.read_json(next(self.outbox.glob("*.ack.json")))
        self.assertFalse(ack["summary"]["sdg_enabled"])
        self.assertNotIn("sdg_artifact", ack["details"][0]["raw_scores"])
        self.assertEqual(ack["details"][0]["seed"], 0)

    def test_only_deepghs_hps_summary_and_ack_preserve_enabled_manifest(self):
        self.ready();self.cfg.update(sdg_enabled=False,kawai_enabled=False,artifact_weight=0,
            fusion_weights=dict(deepghs=2/3,hps=1/3))
        result=self.invoke()
        summary=wrapper.read_json(Path(result['archive'])/'training_summary.json')
        self.assertEqual(summary['enabled_models'],['deepghs','hps'])
        self.assertFalse(summary['kawai_enabled']);self.assertFalse(summary['sdg_enabled'])
        for key in ('raw_component_mean','normalized_component_mean','contribution_mean'):
            self.assertEqual(set(summary[key]),{'deepghs','hps'})
        ack=wrapper.read_json(next(self.outbox.glob('*.ack.json')))
        self.assertEqual(ack['enabled_models'],['deepghs','hps'])
        self.assertEqual(len(result['scores']),192);self.assertEqual(len(ack['scores']),16)

    def test_disabled_model_placeholder_or_undeclared_missing_model_rejected(self):
        row=dict(id='row',reward=1.,prompt_id='p',image='x',raw_scores=dict(deepghs=.4,hps=.6),
            normalized_scores=dict(deepghs=.4,hps=.6),contributions=dict(deepghs=.4,hps=.6))
        with self.assertRaisesRegex(ValueError,'enabled models'):
            wrapper.summary_for([row],dict(sdg_enabled=False))
        good=wrapper.summary_for([row],dict(kawai_enabled=False,sdg_enabled=False))
        self.assertEqual(good['enabled_models'],['deepghs','hps'])
        row['raw_scores']['kawai']=0.
        with self.assertRaisesRegex(ValueError,'enabled models'):
            wrapper.summary_for([row],dict(kawai_enabled=False,sdg_enabled=False))


if __name__ == "__main__":
    unittest.main()
