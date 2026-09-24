"""Streaming ASR engine adapters with one tiny interface.

Stream API (all audio is float32 mono 16 kHz):
  push(x)        feed a chunk (may block while decoding; the harness measures that)
  text()         current best hypothesis for the whole utterance so far (lowercase-insensitive)
  final()        finish the utterance NOW (client signalled end / endpointer fired) and return final text.
                 Wall time spent in here is part of the measured latency.
  native_eos()   True when the engine's own endpointer says the turn ended (or None if it has none)
"""
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("STT_MODELS", os.path.join(HERE, "..", "models"))


class Engine:
    name = "?"
    def new_stream(self): raise NotImplementedError


# ---------------------------------------------------------------- sherpa-onnx transducers
class SherpaEngine(Engine):
    def __init__(self, name, model_dir, prefix_enc, prefix_dec, prefix_join, threads=1,
                 native_rule2=None, **kw):
        import sherpa_onnx as so
        d = os.path.join(MODELS, model_dir)
        self.name = name
        self.native = native_rule2
        self.rec = so.OnlineRecognizer.from_transducer(
            tokens=os.path.join(d, "tokens.txt"),
            encoder=os.path.join(d, prefix_enc), decoder=os.path.join(d, prefix_dec),
            joiner=os.path.join(d, prefix_join),
            num_threads=threads, sample_rate=16000, decoding_method="greedy_search",
            enable_endpoint_detection=native_rule2 is not None,
            rule1_min_trailing_silence=native_rule2 if native_rule2 else 2.4,
            rule2_min_trailing_silence=native_rule2 if native_rule2 else 1.2,
            rule3_min_utterance_length=300.0, **kw)

    def new_stream(self):
        return _SherpaStream(self)


class _SherpaStream:
    def __init__(self, eng):
        self.e = eng
        self.s = eng.rec.create_stream()
        self._txt = ""
        self._ep_latched = False
        self._last_len = 0

    def push(self, x):
        r = self.e.rec
        self.s.accept_waveform(16000, x)
        while r.is_ready(self.s):
            r.decode_stream(self.s)
        self._txt = r.get_result(self.s)
        if not isinstance(self._txt, str):
            self._txt = self._txt.text
        if len(self._txt) != self._last_len:
            self._last_len = len(self._txt)
            self._ep_latched = False

    def text(self):
        return self._txt

    def native_eos(self):
        if self.e.native is None:
            return None
        if self._ep_latched:
            return False
        if self.e.rec.is_endpoint(self.s):
            self._ep_latched = True
            return True
        return False

    def final(self):
        r = self.e.rec
        # flush the encoder's right context: 0.32 s of zeros (the documented "tail padding")
        self.s.accept_waveform(16000, np.zeros(int(0.32 * 16000), dtype=np.float32))
        self.s.input_finished()
        while r.is_ready(self.s):
            r.decode_stream(self.s)
        t = r.get_result(self.s)
        return t if isinstance(t, str) else t.text


# ---------------------------------------------------------------- Moonshine (MIT) streaming
class MoonshineEngine(Engine):
    def __init__(self, arch_name, update_interval=0.2):
        from moonshine_voice import ModelArch
        from moonshine_voice.download import find_model_info, download_model_from_info
        arch = getattr(ModelArch, arch_name)
        path, arch = download_model_from_info(find_model_info("en", arch))
        self.path, self.arch, self.ui = path, arch, update_interval
        self.name = f"moonshine-{arch_name.lower()}"
        from moonshine_voice import Transcriber
        self.tr = Transcriber(path, arch, update_interval=update_interval)

    def new_stream(self):
        return _MoonshineStream(self)


class _MoonshineStream:
    def __init__(self, eng):
        self.e = eng
        self.st = eng.tr.create_stream(update_interval=eng.ui)
        self.st.start()
        self._lines = {}
        self._done = False
        self._pending_eos = False
        self._n_complete = 0
        from moonshine_voice import TranscriptEventListener

        me = self
        class L(TranscriptEventListener):
            def on_line_started(s, ev): pass
            def on_line_updated(s, ev): me._lines[ev.line.line_id] = ev.line.text
            def on_line_text_changed(s, ev): me._lines[ev.line.line_id] = ev.line.text
            def on_line_completed(s, ev):
                me._lines[ev.line.line_id] = ev.line.text
                me._n_complete += 1
                me._pending_eos = True
        self.st.add_listener(L())

    def push(self, x):
        self.st.add_audio(x.tolist(), 16000)
        # update_transcription is driven by add_audio at update_interval in the wrapper

    def text(self):
        return " ".join(t for _, t in sorted(self._lines.items()) if t)

    def native_eos(self):
        if self._pending_eos:
            self._pending_eos = False
            return True
        return False

    def final(self):
        self.st.stop()   # forces the current line to complete
        return self.text()


# ---------------------------------------------------------------- faster-whisper re-decode
class WhisperEngine(Engine):
    """Whisper is not streaming: re-decode the growing utterance every `step` s for partials,
    one last decode of the whole utterance for the final (what worker-stt did)."""
    def __init__(self, size="tiny.en", compute="int8", device="cpu", threads=4, step=1.0):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device=device, compute_type=compute, cpu_threads=threads)
        self.name = f"fw-{size}-{device}-{compute}"
        self.step = step
        self.m.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)

    def new_stream(self):
        return _WhisperStream(self)


class _WhisperStream:
    def __init__(self, e):
        self.e, self.buf, self._txt, self._last = e, [], "", 0.0
        self.n = 0

    def _dec(self):
        a = np.concatenate(self.buf) if self.buf else np.zeros(1600, np.float32)
        segs, _ = self.e.m.transcribe(a, language="en", beam_size=1, vad_filter=False,
                                      condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs)

    def push(self, x):
        self.buf.append(x); self.n += len(x)
        t = self.n / 16000
        if t - self._last >= self.e.step and t > 0.6:
            self._last = t
            self._txt = self._dec()

    def text(self): return self._txt
    def native_eos(self): return None
    def final(self):
        self._txt = self._dec(); return self._txt


def build(name, threads=1):
    """Registry. Model dirs under STT_MODELS."""
    if name in ("dg-flux", "el-scribe"):
        import engines_cloud
        return engines_cloud.DeepgramFluxEngine() if name == "dg-flux" else engines_cloud.ElevenLabsRealtimeEngine()
    if name == "zip-en-int8":     # Apache-2.0, LibriSpeech-only, chunk 320 ms
        s = "chunk-16-left-128.int8.onnx"
        return SherpaEngine(name, "sherpa-onnx-streaming-zipformer-en-2023-06-26",
                            f"encoder-epoch-99-avg-1-{s}", f"decoder-epoch-99-avg-1-{s}", f"joiner-epoch-99-avg-1-{s}", threads)
    if name.startswith("nemo-"):  # nemo-480 / nemo-80  (CC-BY-4.0 FastConformer transducer)
        ms = name.split("-")[1]
        return SherpaEngine(name, f"sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-{ms}ms-int8",
                            "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", threads)
    if name.startswith("moonshine-"):
        return MoonshineEngine(name.split("-", 1)[1].upper())
    if name.startswith("fw-"):
        _, size, dev = name.split("-")[0], name.split("-")[1], (name.split("-")[2] if len(name.split("-")) > 2 else "cpu")
        return WhisperEngine(size, "float16" if dev == "cuda" else "int8", dev, threads=4 if dev == "cpu" else 1)
    raise ValueError(name)
