"""HTTP JSON reward transport; only PNG pixels and prompts leave the trainer."""
from __future__ import annotations

import base64
from http.client import HTTPException
from io import BytesIO
import json
import math
import os
import re
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import uuid

import numpy as np
from PIL import Image

RETRYABLE_STATUS = frozenset((429, 502, 503, 504))
TOKEN_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_http_config(config):
    """Also used by configuration preflight, without contacting the endpoint."""
    url = config.get("url", "")
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError("reward.url must be an HTTP(S) URL without whitespace/control characters")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        valid = (parsed.scheme in ("http", "https") and parsed.hostname
                 and parsed.username is None and parsed.password is None and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("reward.url must be an absolute HTTP(S) URL without userinfo or fragment")
    for key, default, minimum in (("batch_size", 1, 1), ("http_max_retries", 2, 0),
                                  ("http_response_limit_bytes", 1048576, 1)):
        value = config.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"reward.{key} must be an integer >= {minimum}")
    # Bound accidental billion-attempt retry loops before any network request.
    if config.get("http_max_retries", 2) > 10:
        raise ValueError("reward.http_max_retries must be at most 10")
    for key, default, minimum in (("timeout", 900, 0), ("http_retry_backoff", .5, 0)):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"reward.{key} must be a finite nonnegative number")
        if key == "timeout" and value == 0:
            raise ValueError("reward.timeout must be positive")
    env_name = config.get("http_bearer_token_env", "")
    if not isinstance(env_name, str) or (env_name and TOKEN_ENV_PATTERN.fullmatch(env_name) is None):
        raise ValueError("reward.http_bearer_token_env must be empty or an environment variable name")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect could turn POST into GET or forward credentials elsewhere.
        return None


def _response_scores(payload, count):
    if isinstance(payload, dict):
        fields = [key for key in ("rewards", "scores", "reward") if key in payload]
        if len(fields) != 1:
            raise ValueError("HTTP reward response must contain exactly one of rewards, scores, or reward")
        field = fields[0]
        values = payload[field]
        if field == "reward":
            if count != 1:
                raise ValueError("A scalar reward response requires batch_size=1")
            values = [values]
    elif type(payload) in (int, float) and count == 1:
        values = [payload]
    else:
        raise ValueError("HTTP reward response must be a JSON reward object (or a scalar for one image)")
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"HTTP reward response must return exactly {count} scores in request order")
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("HTTP rewards must be finite numbers, not strings, booleans, or nested arrays")
    try:
        finite = all(math.isfinite(v) for v in values)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("HTTP rewards must be finite numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            result = np.asarray(values, dtype=np.float32)
        except (OverflowError, ValueError):
            raise ValueError("HTTP rewards must be representable as finite float32") from None
    if not np.isfinite(result).all():
        raise ValueError("HTTP rewards must be representable as finite float32")
    return result


def _post(opener, url, body, headers, config, count):
    retries = config.get("http_max_retries", 2)
    timeout = float(config.get("timeout", 900))
    limit = config.get("http_response_limit_bytes", 1048576)
    backoff = float(config.get("http_retry_backoff", .5))
    # Reuse one key and the exact request body across all attempts of this batch.
    headers = {**headers, "Idempotency-Key": str(uuid.uuid4())}
    for attempt in range(retries + 1):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                if not 200 <= status < 300:
                    raise RuntimeError(f"HTTP reward returned non-success status {status}")
                raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError(f"HTTP reward response exceeds {limit} bytes")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                raise ValueError("HTTP reward returned invalid UTF-8 JSON") from None
            return _response_scores(payload, count)
        except HTTPError as error:
            status = error.code
            error.close()
            if status not in RETRYABLE_STATUS or attempt == retries:
                raise RuntimeError(f"HTTP reward returned status {status} after {attempt + 1} attempt(s)") from None
        except (URLError, TimeoutError, OSError, HTTPException) as error:
            reason = getattr(error, "reason", error)
            # Retrying a rejected certificate cannot repair the configuration.
            if isinstance(reason, ssl.SSLCertVerificationError) or attempt == retries:
                raise RuntimeError(f"HTTP reward transport failed after {attempt + 1} attempt(s) "
                                   f"({type(reason).__name__})") from None
        if backoff:
            time.sleep(min(30., backoff * 2**attempt))
    raise AssertionError("unreachable HTTP retry state")


def score_http(images, prompts, config):
    """POST batches in order and return finite FP32 rewards, higher is better.

    Request: {version:1,items:[{index,prompt,image_base64,mime_type}]}.
    Base64 is raw PNG (no data-URL prefix); indices refer to the full input batch.
    Response: {rewards:[...]}, {scores:[...]}, or {reward:number} for one item.
    No temporary server-side file paths, checkpoint state or model weights travel.
    """
    validate_http_config(config)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    env_name = config.get("http_bearer_token_env", "")
    if env_name:
        token = os.environ.get(env_name, "")
        if not token or not token.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in token):
            raise ValueError(f"Bearer token environment variable {env_name} is missing or invalid")
        headers["Authorization"] = "Bearer " + token
    opener = build_opener(_NoRedirect())
    batch_size = config.get("batch_size", 1)
    chunks = []
    for start in range(0, len(images), batch_size):
        items = []
        for index in range(start, min(start + batch_size, len(images))):
            buffer = BytesIO()
            Image.fromarray(images[index]).save(buffer, format="PNG")
            items.append({"index": index, "prompt": prompts[index],
                          "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                          "mime_type": "image/png"})
        body = json.dumps({"version": 1, "items": items}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        chunks.append(_post(opener, config["url"], body, headers, config, len(items)))
    return np.concatenate(chunks)
