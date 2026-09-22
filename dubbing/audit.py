"""Structured audit logging + usage reporting for the dubbing job pipeline - the Python-side
counterpart to gateway/audit.js's `auditLog` and gateway/server.js's `/admin/usage/report`, so
dubbing jobs are observable the same way voice-api/audio-isolation jobs already are, instead of
being the one engine with no audit trail and no usage accounting.

Hard rule, same as gateway/audit.js: NEVER log a raw API key, shared secret, or session token.
Only a key `id` (from gateway's /dubbing/authorize response - safe, non-secret) is logged.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from . import gateway_auth


def audit_log(event: str, **fields) -> None:
    """One JSON line per event on stdout - same shape as gateway/audit.js's auditLog, so both
    halves of this pipeline (Node gateway, Python job server) ship audit lines in one consistent
    format for whatever's tailing/aggregating stdout (matches gateway/audit.js's own comment that
    this is ground-floor compliance groundwork, not a tamper-evident log)."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit": True,
        "event": event,
        **fields,
    }
    print(json.dumps(record), file=sys.stdout, flush=True)


def report_usage(key_id: str | None, chars: int, engine: str = "dubbing", audio_seconds: float = 0) -> bool:
    """Reports usage back to gateway/server.js's /admin/usage/report - the same callback
    worker-modal-readaloud/app.py fires after a synthesize call completes - so a dubbing job's
    character/audio-second usage counts against the same per-key billing ledger (keys.js) as every
    other engine, rather than being invisible to billing entirely.

    Best-effort and fire-and-forget in spirit (same posture as the Modal caller): a failed usage
    report is logged but never raises, so a billing-reporting hiccup can't fail an otherwise-
    successful dubbing job. Requires MODAL_USAGE_REPORT_SECRET (same secret
    gateway/server.js's requireUsageReportSecret checks) and DUBBING_GATEWAY_URL; without either,
    this is a documented no-op, not a silent success.
    """
    if not key_id or (not chars and not audio_seconds):
        return False
    base = gateway_auth.gateway_url()
    secret = os.environ.get("MODAL_USAGE_REPORT_SECRET")
    if not base or not secret:
        audit_log("usage_report_skipped", reason="gateway_or_secret_not_configured", id=key_id, engine=engine)
        return False
    req = urllib.request.Request(
        base.rstrip("/") + "/admin/usage/report",
        data=json.dumps({"id": key_id, "chars": chars, "engine": engine, "audio_seconds": audio_seconds}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {secret}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        audit_log("usage_report_failed", id=key_id, engine=engine, error=str(e))
        return False
    audit_log("usage_reported", id=key_id, engine=engine, chars=chars, audio_seconds=audio_seconds)
    return ok
