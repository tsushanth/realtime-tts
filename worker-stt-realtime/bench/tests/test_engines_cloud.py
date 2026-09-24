import contextlib
import json
import threading
import time

import numpy as np
import pytest
from websockets.sync.server import serve

import engines_cloud as ec


@contextlib.contextmanager
def _serve(handler):
    """Local fake ws server; always shut down, even when an assertion fails."""
    srv = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.socket.getsockname()[1]
    finally:
        srv.shutdown()


def _until(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_gate_refuses_unconfirmed_calls_but_allows_public_clips(tmp_path):
    confirmed = tmp_path / "confirmed.txt"
    confirmed.write_text("a\n")
    ec.assert_allowed([{"call_id": None}, {"call_id": "a"}], str(confirmed))
    with pytest.raises(PermissionError, match="b"):
        ec.assert_allowed([{"call_id": "a"}, {"call_id": "b"}], str(confirmed))
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": "a"}], str(tmp_path / "missing.txt"))


def test_deepgram_flux_turn_flow():
    def handler(ws):
        first = True
        for msg in ws:
            if isinstance(msg, bytes):
                if first:
                    first = False
                    ws.send(json.dumps({"type": "TurnInfo", "event": "StartOfTurn", "transcript": ""}))
                    ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "hello"}))
            elif json.loads(msg).get("type") == "CloseStream":
                ws.send(json.dumps({"type": "TurnInfo", "event": "EndOfTurn", "transcript": "hello world"}))
                ws.close()

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "hello")
        assert not st.native_eos()
        assert st.final() == "hello world"
        assert st.native_eos()


def test_elevenlabs_partial_then_manual_commit_fallback(monkeypatch):
    monkeypatch.setattr(ec._ELStream, "AUTO_COMMIT_WAIT_S", 0.05)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started", "session_id": "s"}))
        for msg in ws:
            m = json.loads(msg)
            if m["message_type"] != "input_audio_chunk":
                continue
            if m["commit"]:
                ws.send(json.dumps({"message_type": "committed_transcript", "text": "testing one two"}))
            else:
                ws.send(json.dumps({"message_type": "partial_transcript", "text": "testing"}))

    with _serve(handler) as port:
        st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "testing")
        assert st.final() == "testing one two"
        assert st.native_eos()


def test_elevenlabs_auth_error_raises_on_new_stream():
    def handler(ws):
        ws.send(json.dumps({"message_type": "auth_error", "error": "bad key"}))
        ws.close()

    with _serve(handler) as port:
        with pytest.raises(RuntimeError, match="bad key"):
            ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
