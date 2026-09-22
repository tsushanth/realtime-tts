"""Wire-protocol test for dubbing/gateway_auth.py's authorize() against a real loopback HTTP
server (stdlib http.server, not gateway/server.js itself - Node isn't available as a dependency
of the Python test suite) - separate from job_server tests, which inject a fake authorize_fn and
never touch this module's actual request/response handling.

This exercises the real thing job_server.py depends on: does authorize() send the right method/
path/body, and does it correctly translate each HTTP status gateway/server.js's /dubbing/authorize
can return (200/401/402/503-unreachable) into AuthorizeError with the right .status/.message.
"""
from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from dubbing import gateway_auth

RESPONSES = {
    "valid-key": (200, {"authorized": True, "id": "key-abc123"}),
    "bad-key": (401, {"error": "invalid or missing API key"}),
    "exhausted-key": (402, {"error": "Free tier exhausted for this key. Add a payment method in your dashboard to continue."}),
}


class FakeGatewayHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/dubbing/authorize"
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length) or b"{}")
        status, payload = RESPONSES.get(body.get("key"), (401, {"error": "invalid or missing API key"}))
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


class GatewayAuthWireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeGatewayHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls._prev = os.environ.get("DUBBING_GATEWAY_URL")
        os.environ["DUBBING_GATEWAY_URL"] = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        if cls._prev is None:
            os.environ.pop("DUBBING_GATEWAY_URL", None)
        else:
            os.environ["DUBBING_GATEWAY_URL"] = cls._prev

    def test_authorize_success_returns_decoded_body(self):
        result = gateway_auth.authorize("valid-key")
        self.assertEqual(result, {"authorized": True, "id": "key-abc123"})

    def test_authorize_invalid_key_raises_401(self):
        with self.assertRaises(gateway_auth.AuthorizeError) as ctx:
            gateway_auth.authorize("bad-key")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.message, "invalid or missing API key")

    def test_authorize_billing_exhausted_raises_402(self):
        with self.assertRaises(gateway_auth.AuthorizeError) as ctx:
            gateway_auth.authorize("exhausted-key")
        self.assertEqual(ctx.exception.status, 402)

    def test_authorize_missing_key_raises_400_without_a_network_call(self):
        with self.assertRaises(gateway_auth.AuthorizeError) as ctx:
            gateway_auth.authorize(None)
        self.assertEqual(ctx.exception.status, 400)

    def test_authorize_fails_closed_when_gateway_url_unset(self):
        prev = os.environ.pop("DUBBING_GATEWAY_URL")
        try:
            with self.assertRaises(gateway_auth.AuthorizeError) as ctx:
                gateway_auth.authorize("valid-key")
            self.assertEqual(ctx.exception.status, 503)
        finally:
            os.environ["DUBBING_GATEWAY_URL"] = prev

    def test_authorize_fails_closed_when_gateway_unreachable(self):
        prev = os.environ["DUBBING_GATEWAY_URL"]
        os.environ["DUBBING_GATEWAY_URL"] = "http://127.0.0.1:1"  # nothing listens here
        try:
            with self.assertRaises(gateway_auth.AuthorizeError) as ctx:
                gateway_auth.authorize("valid-key")
            self.assertEqual(ctx.exception.status, 503)
        finally:
            os.environ["DUBBING_GATEWAY_URL"] = prev


if __name__ == "__main__":
    unittest.main()
