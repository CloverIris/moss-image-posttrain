"""HTTP reward protocol tests: real sockets and PNG bytes, no accelerator/model."""
from __future__ import annotations

import base64
from contextlib import ExitStack
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from moss_art.config import DEFAULTS, load_config, validate
from moss_art.http_reward import validate_http_config
from moss_art.rewards import score_images


class Endpoint:
    """Small real HTTP server with programmable responses and captured requests."""

    def __init__(self, replies):
        self.replies = replies
        self.requests = []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                request = {
                    "method": "POST",
                    "path": self.path,
                    "body": raw,
                    "json": json.loads(raw),
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                }
                owner.requests.append(request)
                index = len(owner.requests) - 1
                reply = (owner.replies(request, index) if callable(owner.replies)
                         else owner.replies[min(index, len(owner.replies) - 1)])
                if reply.get("delay"):
                    time.sleep(reply["delay"])
                if reply.get("disconnect"):
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                body = reply.get("body")
                if body is None:
                    body = json.dumps(reply.get("json"), ensure_ascii=False).encode("utf-8")
                self.send_response(reply.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                for key, value in reply.get("headers", {}).items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    # Expected when the client enforces a timeout/response limit.
                    pass

            def do_GET(self):
                owner.requests.append({"method": "GET", "path": self.path})
                self.send_error(500, "Unexpected redirected GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.url = f"http://127.0.0.1:{self.server.server_port}/score?version=1"
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class HTTPRewardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        }))
        self.images = np.random.default_rng(11).integers(
            0, 256, size=(5, 7, 9, 3), dtype=np.uint8)
        self.prompts = ["猫咪，灯光", "night sky", "庭院 🌸", "four", "five"]

    def endpoint(self, replies):
        return self.stack.enter_context(Endpoint(replies))

    @staticmethod
    def config(endpoint, **overrides):
        return {
            "kind": "http", "url": endpoint.url, "timeout": 2,
            "batch_size": 2, "http_max_retries": 0,
            "http_retry_backoff": 0, **overrides,
        }

    def score(self, endpoint, count=1, **overrides):
        return score_images(
            self.images[:count], self.prompts[:count], self.config(endpoint, **overrides))

    def test_wire_png_unicode_batching_order_and_headers(self):
        endpoint = self.endpoint(lambda req, _n: {
            "json": {"rewards": [100 - item["index"] * 0.25 for item in req["json"]["items"]]},
        })
        result = self.score(endpoint, count=5)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, [100, 99.75, 99.5, 99.25, 99])
        self.assertEqual([len(r["json"]["items"]) for r in endpoint.requests], [2, 2, 1])
        keys = []
        for request in endpoint.requests:
            self.assertEqual(request["method"], "POST")
            self.assertEqual(request["path"], "/score?version=1")
            self.assertEqual(set(request["json"]), {"version", "items"})
            self.assertEqual(request["json"]["version"], 1)
            self.assertEqual(request["headers"]["content-type"], "application/json")
            self.assertEqual(request["headers"]["accept"], "application/json")
            self.assertNotIn("authorization", request["headers"])
            keys.append(request["headers"]["idempotency-key"])
            for item in request["json"]["items"]:
                index = item["index"]
                self.assertEqual(set(item), {"index", "prompt", "image_base64", "mime_type"})
                self.assertEqual(item["prompt"], self.prompts[index])
                self.assertEqual(item["mime_type"], "image/png")
                raw = base64.b64decode(item["image_base64"], validate=True)
                self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
                with Image.open(BytesIO(raw)) as decoded:
                    self.assertEqual(decoded.mode, "RGB")
                    np.testing.assert_array_equal(np.asarray(decoded), self.images[index])
        self.assertEqual(len(set(keys)), 3)

    def test_float_image_conversion_is_transmitted_as_png(self):
        endpoint = self.endpoint([{"json": {"reward": 0.75}}])
        floats = np.array([[[[0., .5, 1.]]]], dtype=np.float32)
        result = score_images(floats, ["颜色"], self.config(endpoint, batch_size=1))
        np.testing.assert_array_equal(result, [.75])
        png = base64.b64decode(endpoint.requests[0]["json"]["items"][0]["image_base64"])
        with Image.open(BytesIO(png)) as image:
            np.testing.assert_array_equal(np.asarray(image), [[[0, 128, 255]]])

    def test_accepted_response_formats(self):
        for payload, expected in [
            ({"rewards": [0.25]}, .25), ({"scores": [-3]}, -3),
            ({"reward": .125}, .125), (5.5, 5.5),
        ]:
            with self.subTest(payload=payload):
                endpoint = self.endpoint([{"json": payload}])
                np.testing.assert_array_equal(self.score(endpoint, batch_size=1), [expected])

    def test_invalid_response_types_counts_and_nonfinite_fail_without_retry(self):
        payloads = [
            {"rewards": []}, {"rewards": [1, 2]}, {"rewards": "1"},
            {"rewards": [True]}, {"rewards": ["0.2"]}, {"rewards": [[1]]},
            {"rewards": [None]}, {"rewards": [float("nan")]},
            {"rewards": [float("inf")]}, {"rewards": [-float("inf")]},
            {"rewards": [1e100]}, {"reward": False}, {"reward": "0.25"},
            {"scores": [1], "rewards": [2]}, {"unrelated": [1]},
            [1], None, True,
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                endpoint = self.endpoint([{"json": payload}])
                with self.assertRaises(ValueError):
                    self.score(endpoint, batch_size=1, http_max_retries=2)
                self.assertEqual(len(endpoint.requests), 1)

    def test_scalar_response_rejected_for_multiple_images(self):
        for payload in ({"reward": .25}, .25):
            with self.subTest(payload=payload):
                endpoint = self.endpoint([{"json": payload}])
                with self.assertRaises(ValueError):
                    self.score(endpoint, count=2)
                self.assertEqual(len(endpoint.requests), 1)

    def test_malformed_json_and_utf8_are_not_retried(self):
        for raw in (b'{"rewards":', b"\xff\xfe", b""):
            with self.subTest(raw=raw):
                endpoint = self.endpoint([{"body": raw}])
                with self.assertRaisesRegex(ValueError, "JSON"):
                    self.score(endpoint, http_max_retries=2)
                self.assertEqual(len(endpoint.requests), 1)

    def test_transient_statuses_retry_same_request_and_key(self):
        for status in (429, 502, 503, 504):
            with self.subTest(status=status):
                endpoint = self.endpoint([
                    {"status": status, "json": {"error": "temporary"}},
                    {"json": {"rewards": [1, 2]}},
                ])
                np.testing.assert_array_equal(
                    self.score(endpoint, count=2, http_max_retries=2), [1, 2])
                self.assertEqual(len(endpoint.requests), 2)
                first, second = endpoint.requests
                self.assertEqual(first["body"], second["body"])
                self.assertEqual(first["headers"]["idempotency-key"],
                                 second["headers"]["idempotency-key"])

    def test_retry_budget_is_bounded(self):
        endpoint = self.endpoint([{"status": 503, "json": {"error": "unavailable"}}])
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.score(endpoint, http_max_retries=2)
        self.assertEqual(len(endpoint.requests), 3)
        self.assertEqual(len({r["body"] for r in endpoint.requests}), 1)
        self.assertEqual(len({r["headers"]["idempotency-key"] for r in endpoint.requests}), 1)

    def test_disconnected_transport_retries(self):
        endpoint = self.endpoint([{"disconnect": True}, {"json": {"reward": -1.5}}])
        np.testing.assert_array_equal(
            self.score(endpoint, batch_size=1, http_max_retries=1), [-1.5])
        self.assertEqual(len(endpoint.requests), 2)
        self.assertEqual(endpoint.requests[0]["body"], endpoint.requests[1]["body"])
        self.assertEqual(endpoint.requests[0]["headers"]["idempotency-key"],
                         endpoint.requests[1]["headers"]["idempotency-key"])

    def test_permanent_statuses_are_not_retried_and_errors_redact_secrets(self):
        token = "test-auth-secret-do-not-log"
        self.stack.enter_context(patch.dict(os.environ, {"MOSS_TEST_REWARD_TOKEN": token}))
        for status in (400, 401, 403, 404, 422, 500):
            with self.subTest(status=status):
                endpoint = self.endpoint([{"status": status, "body": token.encode() * 5000}])
                with self.assertRaises(RuntimeError) as caught:
                    self.score(endpoint, http_max_retries=2,
                               http_bearer_token_env="MOSS_TEST_REWARD_TOKEN")
                error = str(caught.exception)
                self.assertIn(str(status), error)
                self.assertNotIn(token, error)
                self.assertNotIn(endpoint.url, error)
                self.assertLess(len(error), 500)
                self.assertEqual(len(endpoint.requests), 1)

    def test_redirect_never_forwards_post_or_bearer_token(self):
        target = self.endpoint([{"json": {"reward": 1}}])
        token = "do-not-forward-this-token"
        self.stack.enter_context(patch.dict(os.environ, {"MOSS_TEST_REWARD_TOKEN": token}))
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                endpoint = self.endpoint([{
                    "status": status, "headers": {"Location": target.url}, "body": b"",
                }])
                with self.assertRaisesRegex(RuntimeError, str(status)):
                    self.score(endpoint, http_max_retries=2,
                               http_bearer_token_env="MOSS_TEST_REWARD_TOKEN")
                self.assertEqual(len(endpoint.requests), 1)
        self.assertEqual(target.requests, [])

    def test_timeout_is_bounded_and_fails_without_fake_reward(self):
        endpoint = self.endpoint([{"delay": .2, "json": {"reward": 1}}])
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.score(endpoint, timeout=.03, http_max_retries=0)
        self.assertEqual(len(endpoint.requests), 1)

    def test_response_limit_rejects_large_body_without_retry(self):
        endpoint = self.endpoint([{"body": b'{"rewards":[1]}' + b" " * 2048}])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.score(endpoint, http_response_limit_bytes=64, http_max_retries=2)
        self.assertEqual(len(endpoint.requests), 1)

    def test_token_resolved_at_each_call_and_never_sent_in_body(self):
        endpoint = self.endpoint([{"json": {"reward": .5}}])
        cfg = self.config(endpoint, batch_size=1, http_bearer_token_env="MOSS_TEST_REWARD_TOKEN")
        for token in ("first-token", "rotated-token"):
            with patch.dict(os.environ, {"MOSS_TEST_REWARD_TOKEN": token}):
                np.testing.assert_array_equal(
                    score_images(self.images[:1], self.prompts[:1], cfg), [.5])
            request = endpoint.requests[-1]
            self.assertEqual(request["headers"]["authorization"], "Bearer " + token)
            self.assertNotIn(token.encode(), request["body"])
            self.assertNotIn(token, json.dumps(cfg))
        self.assertEqual(len(endpoint.requests), 2)

    def test_missing_or_invalid_token_fails_before_any_post(self):
        endpoint = self.endpoint([{"json": {"reward": 1}}])
        cfg = self.config(endpoint, http_bearer_token_env="MOSS_TEST_REWARD_TOKEN")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                score_images(self.images[:1], self.prompts[:1], cfg)
        for token in ("", "bad token", "bad\r\nHeader: injected", "不可用"):
            with self.subTest(token=token), patch.dict(
                os.environ, {"MOSS_TEST_REWARD_TOKEN": token}
            ):
                with self.assertRaises(ValueError):
                    score_images(self.images[:1], self.prompts[:1], cfg)
        self.assertEqual(endpoint.requests, [])

    def test_later_batch_failure_does_not_return_partial_reward(self):
        endpoint = self.endpoint([
            {"json": {"rewards": [1, 2]}},
            {"status": 422, "json": {"error": "bad second batch"}},
        ])
        with self.assertRaises(RuntimeError):
            self.score(endpoint, count=5)
        self.assertEqual([len(r["json"]["items"]) for r in endpoint.requests], [2, 2])

    def test_trainer_decodes_workers_and_preserves_global_order_for_http(self):
        from moss_art.trainer import reward_round

        images, prompts = self.images[:4], self.prompts[:4]
        endpoints = np.arange(4 * 2 * 2 * 2, dtype=np.float32).reshape(4, 2, 2, 2)
        outputs = [
            {"endpoints": endpoints[:2], "prompts": prompts[:2]},
            {"endpoints": endpoints[2:], "prompts": prompts[2:]},
        ]
        requests = []

        class FakePool:
            def call(self, method, payloads):
                requests.append((method, payloads))
                return [{"images": images[:2]}, {"images": images[2:]}]

        endpoint = self.endpoint(lambda request, _n: {
            "json": {"rewards": [float(item["index"]) + .5
                                 for item in request["json"]["items"]]},
        })
        config = copy.deepcopy(DEFAULTS)
        config["reward"].update(self.config(endpoint, batch_size=3))
        with tempfile.TemporaryDirectory() as temporary:
            rewards, decoded = reward_round(
                config, FakePool(), outputs, image_dir=Path(temporary) / "evaluation")
            np.testing.assert_array_equal(rewards, [.5, 1.5, 2.5, 3.5])
            np.testing.assert_array_equal(decoded, images)
            self.assertEqual(requests[0][0], "decode")
            for index, payload in enumerate(requests[0][1]):
                np.testing.assert_array_equal(payload["endpoints"], outputs[index]["endpoints"])
            saved = Path(temporary) / "evaluation"
            self.assertEqual(json.loads((saved / "prompts.json").read_text()), prompts)
            for index, pixels in enumerate(images):
                with Image.open(saved / f"{index:04d}.png") as image:
                    np.testing.assert_array_equal(np.asarray(image), pixels)
        items = [item for request in endpoint.requests for item in request["json"]["items"]]
        self.assertEqual([item["prompt"] for item in items], prompts)
        self.assertEqual([item["index"] for item in items], [0, 1, 2, 3])
        for item, pixels in zip(items, images):
            with Image.open(BytesIO(base64.b64decode(item["image_base64"]))) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)


class HTTPConfigTests(unittest.TestCase):
    def test_config_accepts_http_url_without_contacting_remote(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "http.yaml"
            path.write_text(
                "reward:\n"
                "  kind: http\n"
                "  url: https://reward.example.test/predict\n"
                "  batch_size: 8\n"
                "  timeout: 12.5\n"
                "  http_max_retries: 3\n"
                "  http_retry_backoff: 0.125\n"
                "  http_bearer_token_env: MY_REWARD_TOKEN\n",
                encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["reward"]["kind"], "http")
        self.assertEqual(config["reward"]["url"], "https://reward.example.test/predict")
        self.assertEqual(config["reward"]["batch_size"], 8)
        self.assertEqual(config["reward"]["timeout"], 12.5)
        self.assertEqual(config["reward"]["http_response_limit_bytes"], 1024 * 1024)

    def test_invalid_urls_and_retry_budgets_rejected_by_preflight(self):
        base = {"url": "http://127.0.0.1:54321/reward"}
        invalid = [
            {"url": ""}, {"url": "/reward"}, {"url": "ftp://reward.example/"},
            {"url": "http://"}, {"url": "http://host:bad/reward"},
            {"url": "http://user:password@host/reward"}, {"url": "http://host/#fragment"},
            {"url": "http://host/a path"}, {"url": "http://host/\n"},
            {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5},
            {"timeout": 0}, {"timeout": -1}, {"timeout": float("nan")},
            {"http_max_retries": -1}, {"http_max_retries": 11},
            {"http_max_retries": True}, {"http_max_retries": .5},
            {"http_retry_backoff": -1}, {"http_retry_backoff": float("inf")},
            {"http_response_limit_bytes": 0}, {"http_response_limit_bytes": True},
            {"http_bearer_token_env": "actual token"}, {"http_bearer_token_env": "BAD\nENV"},
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_http_config({**base, **override})

    def test_http_requires_decoding_and_disallows_literal_token_config(self):
        config = copy.deepcopy(DEFAULTS)
        config["reward"].update({"kind": "http", "url": "http://localhost:1/reward", "decode": False})
        with self.assertRaisesRegex(ValueError, "decode"):
            validate(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "http.yaml"
            path.write_text(
                "reward:\n  kind: http\n  url: http://localhost:1/reward\n"
                "  bearer_token: literal-secret\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_config(path)

    def test_cli_json_reward_dispatches_training_and_evaluation(self):
        from moss_art.cli import main
        source = Path(__file__).resolve().parents[1] / "configs/smoke_cpu.json"
        url = "https://reward.example.test/predict"
        cases = [("train",{},"moss_art.trainer.train"),
                 ("evaluate",{"checkpoint":"/unused/checkpoint"},"moss_art.trainer.evaluate_checkpoint"),
                 ("evaluate",{"adapter":"/unused/adapter.safetensors"},"moss_art.trainer.evaluate_adapter")]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"run.json"
            for command,run,target in cases:
                config=json.loads(source.read_text())
                config.update(run=run,reward={"kind":"http","url":url,"decode":True})
                path.write_text(json.dumps(config))
                with self.subTest(command=command,run=run), patch(target) as invoked, patch(
                    "urllib.request.OpenerDirector.open",
                    side_effect=AssertionError("CLI must not make scoring requests"),
                ):
                    main([command,"--config",str(path)])
                    invoked.assert_called_once()
                    actual=invoked.call_args.args[0]
                    self.assertEqual(actual["reward"]["url"],url)
                    self.assertEqual(actual["reward"]["kind"],"http")
            config["run"]={}
            config["reward"]["url"]="file:///bad"
            path.write_text(json.dumps(config))
            with patch("moss_art.trainer.train") as train, self.assertRaises(ValueError):
                main(["train","--config",str(path)])
            train.assert_not_called()

    def test_http_tiny_identity_needs_no_local_reward_weights_or_model(self):
        from moss_art.assets import experiment_identity

        config = copy.deepcopy(DEFAULTS)
        config["model"].update({"kind": "tiny", "path": "/missing/model"})
        config["data"]["cache"] = "/missing/cache"
        config["reward"].update({
            "kind": "http", "url": "https://reward.example.test/predict",
            "model_path": "/missing/siglip", "python": "/missing/python",
        })
        with patch("urllib.request.OpenerDirector.open",
                   side_effect=AssertionError("Identity checks must not contact the endpoint")):
            identity = experiment_identity(config)
        self.assertEqual(identity["reward_assets"]["kind"], "http")
        self.assertNotIn("components", identity)
        self.assertNotIn("text_cache_manifest", identity)
        self.assertIn("http_reward.py", " ".join(identity["source_files"]))
        self.assertIn("not available", identity["reward_assets"]["external_state_verification"])


if __name__ == "__main__":
    unittest.main()
