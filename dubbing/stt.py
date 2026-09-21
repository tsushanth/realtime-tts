"""STT step: source audio -> (text, duration_s). Thin wrapper around the existing worker-stt
service (see worker-stt/DESIGN.md, worker-stt/test_client.py) - this repo already has a working
realtime STT worker; dubbing does not reimplement it.

Mint/use pattern matches worker-stt/test_client.py's `mint()` exactly (HMAC-SHA256 session token,
same scheme as gateway/keys.js) - requires STT_SECRET_FILE (or STT_SECRET) pointing at the
worker's signing secret, and STT_BASE for the deployed Modal URL. Neither is provisioned in this
environment (checked via `env`), so SttClient.transcribe() will raise clearly rather than fake a
transcript.

Per the task's explicit scope-reduction option, TextInput below lets the pipeline skip STT
entirely and take source text + source language directly - this is what pipeline.py's CLI uses
for the actual end-to-end test run in this environment, since no STT secret is available.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class SttResult:
    text: str
    language: str
    duration_s: float


def mint(secret: str, key_id: str = "dubbing-mvp-test", ttl_ms: int = 600_000) -> str:
    """Identical scheme to worker-stt/test_client.py's mint() - same secret format, same token
    shape, so it works against the real worker unmodified."""
    payload = base64.urlsafe_b64encode(
        json.dumps({"id": key_id, "exp": int(time.time() * 1000) + ttl_ms}).encode()
    ).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    return f"{payload}.{sig}"


class SttClient:
    """Real backend against the deployed worker-stt Modal app. Test keys only (per task
    constraints): mint a short-lived (10 min TTL) token per run, use it for exactly this
    transcription call, then let it expire - there is no explicit revoke endpoint for these
    session tokens (they're stateless HMAC, not stored server-side), so "revoke" here means "mint
    a short TTL and don't reuse it" rather than a server-side delete call, unlike the gateway
    key's /admin/keys DELETE flow."""

    def __init__(self, base_url: str | None = None, secret: str | None = None):
        self.base_url = (base_url or os.environ.get("STT_BASE", "")).rstrip("/")
        secret_file = os.environ.get("STT_SECRET_FILE")
        self.secret = secret or os.environ.get("STT_SECRET") or (
            open(secret_file).read().strip() if secret_file and os.path.exists(secret_file) else None
        )
        if not self.base_url or not self.secret:
            raise RuntimeError(
                "worker-stt is not reachable from this environment: STT_BASE and "
                "STT_SECRET_FILE/STT_SECRET are not set. Use TextInput to bypass STT with a "
                "manually supplied source text instead, or provision these against a real "
                "worker-stt deployment (non-production test instance) to test this step live."
            )

    def transcribe(self, wav_path: str, language: str = "auto") -> SttResult:
        token = mint(self.secret)
        boundary = "dubbingmvpboundary"
        with open(wav_path, "rb") as f:
            audio = f.read()
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="format"\r\n\r\nauto\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n'
        ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/stt",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        return SttResult(text=data["text"], language=data.get("language", language), duration_s=data["duration"])


@dataclass
class TextInput:
    """Scope-reduction bypass: source text + language supplied directly, no audio/STT involved.
    duration_s is optional - if omitted, translate.py budgets off character count instead."""
    text: str
    language: str
    duration_s: float | None = None

    def as_result(self) -> SttResult:
        return SttResult(text=self.text, language=self.language, duration_s=self.duration_s or 0.0)
