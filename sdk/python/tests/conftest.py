import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from websockets.sync.server import serve


class Mock:
    """Local fake of the API: /tts/authorize, a WS synth endpoint and an HTTP endpoint."""
    behavior = "ok"      # ok | capacity_ws | error_voice | slow
    with_http = False
    stops = 0
    last_request = None
    last_http = None
    last_auth = None


@pytest.fixture
def mock():
    m = Mock()
    m.stops = 0

    def ws_handler(ws):
        req = json.loads(ws.recv())
        m.last_request = req
        if m.behavior == "capacity_ws":
            ws.send(json.dumps({"type": "error", "message": "at capacity, retry shortly"}))
            ws.close(1013)
            return
        if req["voice"] == "custom:nope":
            ws.send(json.dumps({"type": "error", "message": "unknown voice"}))
            return
        for i in range(50 if m.behavior == "slow" else 3):
            ws.send(json.dumps({"type": "chunk_meta", "i": i}))
            ws.send(bytes([i]) * 4)
            if m.behavior == "slow":
                try:
                    ws.recv(timeout=0.02)
                    m.stops += 1
                    ws.send(json.dumps({"type": "cancelled"}))
                    return
                except TimeoutError:
                    pass
        ws.send(json.dumps({"type": "done"}))

    wss = serve(ws_handler, "127.0.0.1", 0)
    wport = wss.socket.getsockname()[1]
    threading.Thread(target=wss.serve_forever, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _json(self, code, body, headers=()):
            b = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/tts/authorize":
                if body["key"] == "bad":
                    return self._json(401, {"error": "invalid key"})
                if body["key"] == "poor":
                    return self._json(402, {"error": "free tier exhausted"})
                out = {"token": "tok", "url": f"ws://127.0.0.1:{wport}/"}
                if m.with_http:
                    out["http_url"] = f"http://127.0.0.1:{self.server.server_port}/v1/tts/stream"
                return self._json(200, out)
            if self.path == "/v1/tts/stream":
                m.last_http = body
                m.last_auth = self.headers["Authorization"]
                if m.behavior == "capacity_http":
                    return self._json(503, {"error": "busy"}, [("Retry-After", "2")])
                if m.last_auth != "Bearer tok":
                    return self._json(401, {"error": "bad token"})
                self.send_response(200)
                self.send_header("X-Sample-Rate", "24000")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in (b"abcd", b"efgh"):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                self.wfile.write(b"0\r\n\r\n")

    http = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    m.base = f"http://127.0.0.1:{http.server_port}"
    yield m
    http.shutdown()
    wss.shutdown()
