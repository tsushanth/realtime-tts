"""Validates a customer's gateway API key before a dubbing job is queued.

This is the dubbing-side half of the gating gap flagged for this MVP: unlike voice-api's
/tts/authorize and /audio/authorize (gateway/server.js), the async job API originally accepted
any request carrying the shared `DUBBING_JOB_SECRET` server-to-server bearer token, with no check
that the underlying request was made on behalf of a real, billing-enabled customer key. That
shared secret is still required (it's the "is this even the trusted backend calling" boundary,
same as voice-pipeline/intake.py's INTAKE_SECRET) - this module adds the second, per-request gate:
"does the customer's own API key exist, and can it still afford to use paid engines".

Implemented as an outbound HTTP call to a new `/dubbing/authorize` endpoint on gateway/server.js
(same file, same `keys.js` key store `/tts/authorize` and `/audio/authorize` already use) rather
than re-implementing key validation/billing logic in Python against gateway's key store file
directly - that would create two independent, driftable implementations of "is this key valid and
billing-enabled" in two languages. One source of truth (keys.js), reached over loopback HTTP from
this process, exactly like dubbing's own tts.py/stt.py already reach the gateway/worker-stt over
HTTP for their own calls.

Fails closed: no DUBBING_GATEWAY_URL configured, or the gateway unreachable, means dubbing jobs
are refused - never silently treated as "auth not required".
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class AuthorizeError(Exception):
    """Raised with the HTTP status code + message the job_server should return to the caller."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def gateway_url() -> str | None:
    return os.environ.get("DUBBING_GATEWAY_URL")


def authorize(api_key: str | None) -> dict:
    """POSTs {"key": api_key} to <DUBBING_GATEWAY_URL>/dubbing/authorize. Returns the decoded
    {"authorized": true, "id": ...} body on success. Raises AuthorizeError otherwise - callers
    should map `.status`/`.message` straight onto the HTTP response they send back.
    """
    if not api_key:
        raise AuthorizeError(400, "api_key is required")
    base = gateway_url()
    if not base:
        # Same fail-closed posture as job_server._authorized(): an unconfigured dependency means
        # nobody is authorized, never "skip the check".
        raise AuthorizeError(503, "dubbing authorization gateway (DUBBING_GATEWAY_URL) is not configured")

    req = urllib.request.Request(
        base.rstrip("/") + "/dubbing/authorize",
        data=json.dumps({"key": api_key}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except (json.JSONDecodeError, ValueError):
            payload = {}
        raise AuthorizeError(e.code, payload.get("error", "unauthorized")) from e
    except urllib.error.URLError as e:
        raise AuthorizeError(503, f"could not reach dubbing authorization gateway: {e.reason}") from e
    except TimeoutError as e:
        raise AuthorizeError(503, "dubbing authorization gateway timed out") from e
