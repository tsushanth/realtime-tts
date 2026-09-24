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
    ec.assert_allowed([{"call_id": None}, {"call_id": "a"}], str(confirmed), "tel")
    with pytest.raises(PermissionError, match="b"):
        ec.assert_allowed([{"call_id": "a"}, {"call_id": "b"}], str(confirmed), "real")
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": "a"}], str(tmp_path / "missing.txt"), "real")


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
        with pytest.raises(RuntimeError, match="auth_error"):
            ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()


def _conf(tmp_path, text):
    p = tmp_path / "confirmed.txt"
    p.write_text(text)
    return str(p)


def test_gate_fails_closed_on_missing_call_id_outside_public_sets(tmp_path):
    c = _conf(tmp_path, "a\n")
    for public in ("clean", "tel", "call"):
        ec.assert_allowed([{"call_id": None}], c, public)
    for other in ("real", "weird", ""):
        with pytest.raises(PermissionError, match="no call_id"):
            ec.assert_allowed([{"call_id": None, "ref": "SECRET WORDS"}], c, other)
        with pytest.raises(PermissionError, match="no call_id") as ei:
            ec.assert_allowed([{"call_id": "", "ref": "SECRET WORDS"}], c, other)
        assert "SECRET" not in str(ei.value)
    # missing confirmed file with only-None clips in "real": refused
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": None}], str(tmp_path / "nope.txt"), "real")
    # public set + unconfirmed non-None id still refused
    with pytest.raises(PermissionError, match="zzz"):
        ec.assert_allowed([{"call_id": "zzz"}], c, "tel")


def test_gate_exact_match_and_file_parsing(tmp_path):
    c = _conf(tmp_path, "# header\n\n  ab  \n#a\nx1\n")
    ec.assert_allowed([{"call_id": "ab"}, {"call_id": "x1"}], c, "real")
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": "a"}], c, "real")      # prefix of a confirmed id
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": "#a"}], c, "real")     # comment line is not an id
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": ""}], c, "real")       # empty id


def test_is_third_party():
    import engines
    assert engines.is_third_party("dg-flux") and engines.is_third_party("el-scribe")
    assert not any(engines.is_third_party(n) for n in ("dg-x", "el-", "moonshine-tiny", "fw-small", "zip-en-int8"))


def test_rtbench_main_gates_before_building_engine(tmp_path, monkeypatch):
    import json as _json
    import soundfile as sf
    import engines
    import rtbench
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    (tmp_path / "real").mkdir()
    sf.write(str(tmp_path / "real" / "c1.wav"), np.zeros(16000, dtype="float32"), 16000)
    (tmp_path / "real" / "manifest.json").write_text(_json.dumps([{"id": "c1", "ref": "hi", "call_id": "unconfirmed"}]))
    (tmp_path / "confirmed_calls.txt").write_text("other\n")
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("engines.build called before the gate")
    monkeypatch.setattr(engines, "build", boom)
    for eng in ("dg-flux", "el-scribe"):
        monkeypatch.setattr("sys.argv", ["rtbench", "--engine", eng, "--set", "real", "--out", str(tmp_path / "real__o__real.json")])
        with pytest.raises(PermissionError):
            rtbench.main()


def test_elevenlabs_error_after_start_raises_on_push_and_final():
    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started"}))
        for _ in ws:
            ws.send(json.dumps({"message_type": "quota_exceeded", "error": "out of credits"}))

    with _serve(handler) as port:
        st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert _until(lambda: st.error is not None)
            with pytest.raises(RuntimeError, match="quota_exceeded"):
                st.push(np.zeros(640, dtype="float32"))
            with pytest.raises(RuntimeError, match="quota_exceeded"):
                st.final()
        finally:
            st.close()


def test_deepgram_error_message_raises_on_push_and_final():
    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "Error", "code": "BAD_AUDIO", "description": "bad frame"}))

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert _until(lambda: st.error is not None)
            with pytest.raises(RuntimeError, match="deepgram error: BAD_AUDIO$"):
                st.push(np.zeros(640, dtype="float32"))
            with pytest.raises(RuntimeError, match="BAD_AUDIO"):
                st.final()
        finally:
            st.close()


def test_connection_loss_mid_stream_is_an_error():
    def handler(ws):
        for msg in ws:
            ws.close()     # drop on first audio
            return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert _until(lambda: st.error is not None)
            assert st.error.startswith("connection lost")
            with pytest.raises(RuntimeError, match="connection lost"):
                st.final()
        finally:
            st.close()


def test_close_is_idempotent_and_not_reported_as_loss():
    def handler(ws):
        for _ in ws:
            pass

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.close()
        st.close()
        assert _until(st.closed.is_set)
        assert st.error is None


class _FakeStream:
    def __init__(self, fail=False):
        self.closes, self.fail = 0, fail

    def push(self, x):
        if self.fail:
            raise RuntimeError("boom")

    def text(self):
        return ""

    def native_eos(self):
        return False

    def final(self):
        return "hi"

    def close(self):
        self.closes += 1


class _FakeEngine:
    def __init__(self, fail=False):
        self.streams, self.fail = [], fail

    def new_stream(self):
        self.streams.append(_FakeStream(self.fail))
        return self.streams[-1]


class _StubVAD:
    def reset(self):
        pass

    def push(self, x):
        return False

    def silence_ms(self):
        return 0.0


def test_run_clip_closes_stream_when_it_raises():
    import rtbench
    eng = _FakeEngine(fail=True)
    clip = {"audio": np.zeros(3200, dtype="float32"), "dur": 0.2, "ref": ""}
    with pytest.raises(RuntimeError, match="boom"):
        rtbench.run_clip(eng, _StubVAD(), clip, ["commit"], 0)
    assert [s.closes for s in eng.streams] == [1]


def test_rtbench_main_closes_warmup_and_clip_streams(tmp_path, monkeypatch):
    import engines
    import rtbench
    eng = _FakeEngine()
    monkeypatch.setattr(engines, "build", lambda *a, **k: eng)
    monkeypatch.setattr(rtbench, "SilenceVAD", _StubVAD)
    monkeypatch.setattr(rtbench, "CH", 0.001)
    clip = {"id": "c", "ref": "hi", "call_id": None, "keyterms": [], "dur": 0.2,
            "audio": np.zeros(3200, dtype="float32")}
    monkeypatch.setattr(rtbench, "load", lambda *a, **k: [clip, dict(clip, id="d")])
    monkeypatch.setattr("sys.argv", ["rtbench", "--engine", "fw-tiny-cpu", "--out", str(tmp_path / "o.json")])
    rtbench.main()
    assert len(eng.streams) == 3 and all(s.closes >= 1 for s in eng.streams)


def test_error_text_never_includes_server_prose():
    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "Error", "code": "BAD_AUDIO", "description": "SECRET prose"}))

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert _until(lambda: st.error is not None)
            assert "SECRET" not in st.error and "prose" not in st.error
        finally:
            st.close()

    def el(ws):
        ws.send(json.dumps({"message_type": "auth_error", "error": "SECRET prose"}))
        ws.close()

    with _serve(el) as port:
        with pytest.raises(RuntimeError) as ei:
            ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        assert "SECRET" not in str(ei.value)


def test_deepgram_drop_during_final_raises():
    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))
            else:
                ws.close(1011)     # CloseStream received: abnormal close with no EndOfTurn
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="connection lost"):
            st.final()


def test_deepgram_endofturn_then_close_returns_text_repeatedly():
    def handler(ws):
        for msg in ws:
            if not isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "EndOfTurn", "transcript": "done"}))
                ws.close()
                return

    for _ in range(5):     # the reader-exit / wait race must never produce a false error
        with _serve(handler) as port:
            st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
            st.push(np.zeros(640, dtype="float32"))
            assert st.final() == "done"


def test_elevenlabs_drop_during_final_raises(monkeypatch):
    monkeypatch.setattr(ec._ELStream, "AUTO_COMMIT_WAIT_S", 0.05)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started"}))
        for msg in ws:
            if json.loads(msg)["commit"]:
                ws.close()     # manual commit received: drop with no committed_transcript
                return
            ws.send(json.dumps({"message_type": "partial_transcript", "text": "par"}))

    with _serve(handler) as port:
        st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="connection lost"):
            st.final()


def test_elevenlabs_commit_then_close_returns_text_repeatedly(monkeypatch):
    monkeypatch.setattr(ec._ELStream, "AUTO_COMMIT_WAIT_S", 0.05)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started"}))
        for msg in ws:
            if json.loads(msg)["commit"]:
                ws.send(json.dumps({"message_type": "committed_transcript", "text": "all done"}))
                ws.close()
                return
            ws.send(json.dumps({"message_type": "partial_transcript", "text": "par"}))

    for _ in range(5):
        with _serve(handler) as port:
            st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
            st.push(np.zeros(640, dtype="float32"))
            assert st.final() == "all done"


def _rtbench_setup(tmp_path, monkeypatch, manifest):
    import soundfile as sf
    import engines
    import rtbench
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    (tmp_path / "real").mkdir()
    sf.write(str(tmp_path / "real" / "c1.wav"), np.zeros(16000, dtype="float32"), 16000)
    (tmp_path / "real" / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "confirmed_calls.txt").write_text("other\n")
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("engines.build called")
    monkeypatch.setattr(engines, "build", boom)
    return rtbench


def test_rtbench_main_refuses_real_clip_without_call_id(tmp_path, monkeypatch):
    rtbench = _rtbench_setup(tmp_path, monkeypatch, [{"id": "c1", "ref": "hi"}])
    monkeypatch.setattr("sys.argv", ["rtbench", "--engine", "dg-flux", "--set", "real", "--out", str(tmp_path / "real__o__real.json")])
    with pytest.raises(PermissionError, match="no call_id"):
        rtbench.main()


def test_rtbench_main_native_ms_with_third_party_is_an_arg_error(tmp_path, monkeypatch):
    rtbench = _rtbench_setup(tmp_path, monkeypatch, [{"id": "c1", "ref": "hi", "call_id": "other"}])
    monkeypatch.setattr("sys.argv", ["rtbench", "--engine", "el-scribe", "--set", "real", "--native-ms", "500",
                                     "--out", str(tmp_path / "real__o__real.json")])
    with pytest.raises(SystemExit) as ei:
        rtbench.main()
    assert ei.value.code == 2


def test_code_sanitiser_is_strict():
    f = ec._Stream._code
    assert f("BAD_AUDIO") == "BAD_AUDIO" and f("auth_error") == "auth_error"
    assert f("a" * 32) == "a" * 32
    for bad in ("0123456789abcdef0123456789abcdef01234567", "1abc", "a-b", "a.b", "a" * 33, "", None, 5):
        assert f(bad) == "unknown"


def test_deepgram_final_timeout_raises(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 0.2)

    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="final timeout: no terminal event"):
            st.final()


def test_elevenlabs_final_timeout_raises(monkeypatch):
    monkeypatch.setattr(ec._ELStream, "AUTO_COMMIT_WAIT_S", 0.05)
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 0.2)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started"}))
        for msg in ws:
            if not json.loads(msg)["commit"]:
                ws.send(json.dumps({"message_type": "partial_transcript", "text": "par"}))

    with _serve(handler) as port:
        st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="final timeout: no terminal event"):
            st.final()


def test_final_with_turn_already_done_does_not_wait(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 5.0)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started"}))
        for msg in ws:
            ws.send(json.dumps({"message_type": "committed_transcript", "text": "auto"}))

    with _serve(handler) as port:
        st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(st.native_eos)
        t0 = time.time()
        assert st.final() == "auto"
        assert time.time() - t0 < 1.0


def test_deepgram_clean_close_after_closestream_returns_partial(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 5.0)

    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "full text"}))
            else:
                ws.close(1000)     # normal close, no EndOfTurn
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "full text")
        t0 = time.time()
        assert st.final() == "full text"
        assert time.time() - t0 < 2.0     # far below FLUSH_WAIT_S=5
        assert st.error is None


def test_deepgram_abnormal_close_after_closestream_raises(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 5.0)

    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))
            else:
                ws.close(1011)
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="connection lost"):
            st.final()


def test_deepgram_tcp_drop_after_closestream_raises(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 5.0)

    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))
            else:
                ws.socket.close()     # no close frame
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        with pytest.raises(RuntimeError, match="connection lost"):
            st.final()


def test_deepgram_clean_close_before_closestream_still_an_error():
    def handler(ws):
        for msg in ws:
            ws.close(1000)
            return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert _until(lambda: st.error is not None)
            with pytest.raises(RuntimeError, match="connection lost"):
                st.push(np.zeros(640, dtype="float32"))
            with pytest.raises(RuntimeError, match="connection lost"):
                st.final()
        finally:
            st.close()


def test_deepgram_clean_close_observed_before_final_raises():
    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))
                ws.close(1000)     # normal close BEFORE any CloseStream
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        try:
            st.push(np.zeros(640, dtype="float32"))
            assert st.closed.wait(2)     # reader has observed the close; only then call final()
            with pytest.raises(RuntimeError, match="connection lost"):
                st.final()
        finally:
            st.close()


def test_deepgram_send_failure_in_final_records_error_and_flag_stays_unset():
    class _DeadWS:
        def send(self, data):
            raise OSError("SECRET socket text")

        def close(self):
            pass

    with _serve(lambda ws: [None for _ in ws]) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        real = st.ws
        st.ws = _DeadWS()
        try:
            with pytest.raises(RuntimeError, match="connection lost: OSError") as ei:
                st.final()
            assert "SECRET" not in str(ei.value)
            assert st._flush_sent is False
        finally:
            real.close()


def test_deepgram_going_away_1001_after_closestream_returns_partial(monkeypatch):
    monkeypatch.setattr(ec._Stream, "FLUSH_WAIT_S", 5.0)

    def handler(ws):
        for msg in ws:
            if isinstance(msg, bytes):
                ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "par"}))
            else:
                ws.close(1001)
                return

    with _serve(handler) as port:
        st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
        st.push(np.zeros(640, dtype="float32"))
        assert _until(lambda: st.text() == "par")
        t0 = time.time()
        assert st.final() == "par"
        assert time.time() - t0 < 2.0
