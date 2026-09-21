"""TTS step: target-language text -> WAV, via Piper (CPU, multi-language) on this repo's existing
gateway. Kokoro is intentionally not used here - it's English-only (see voices/catalog.json /
eval/README.md), so it can't serve as a *target*-language dubbing voice; Piper is the only engine
that covers the catalog's non-English languages.

Mirrors eval/synth.mjs's client pattern (POST /tts/authorize with a gateway key -> short-lived
session token -> WS synth), reimplemented in Python for this pipeline. Requires:
  - TTS_GATEWAY_API_KEY: a billing-enabled key for gateway/server.js's /tts/authorize, same as
    eval/README.md's "How to re-run" section.

This environment does not have TTS_GATEWAY_API_KEY (or ADMIN_SECRET to mint one against
gateway/keys.js's /admin/keys) provisioned - checked directly via `env`, not assumed. Real Piper
synthesis also needs the actual voice model files (.onnx, ~60-80MB each per voices/catalog.json),
which live on a remote volume ("house-voices:") this pipeline talks to over the gateway, not
locally; local disk in this environment had ~140MB free at build time, too little to safely
install/run Piper standalone either. So: real network path is implemented below, but untestable
here for those two independent reasons. NullTTS is provided as an explicit, clearly-labeled stub
for exercising retime.py's mechanics without a real voice.
"""
from __future__ import annotations

import json
import os
import wave
import struct
import urllib.request
import urllib.error
from abc import ABC, abstractmethod

from . import catalog

GATEWAY_BASE = os.environ.get("TTS_GATEWAY_BASE", "https://api.readaloudai.org")


class TTSBackend(ABC):
    @abstractmethod
    def synth(self, text: str, language: str, out_wav_path: str) -> str:
        """Writes a WAV to out_wav_path, returns the voice id used."""
        ...


class PiperGatewayTTS(TTSBackend):
    """Real backend: authorizes against gateway/server.js, then POSTs to the worker's HTTP TTS
    endpoint (see gateway/server.js's /tts/authorize -> http_url). Requires TTS_GATEWAY_API_KEY.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TTS_GATEWAY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TTS_GATEWAY_API_KEY not set. Provision a billing-enabled test key for "
                "gateway/server.js (see eval/README.md's 'How to re-run' for the same "
                "requirement) - mint via /admin/keys with ADMIN_SECRET, use it for this test run "
                "only, then revoke it, matching eval/'s mint-use-revoke pattern. Not done "
                "automatically here because ADMIN_SECRET is also not provisioned in this "
                "environment."
            )

    def _authorize(self, engine: str = "piper") -> dict:
        req = urllib.request.Request(
            f"{GATEWAY_BASE}/tts/authorize",
            data=json.dumps({"key": self.api_key, "engine": engine}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"authorize failed: {e.code} {e.read().decode(errors='replace')}") from e

    def synth(self, text: str, language: str, out_wav_path: str) -> str:
        catalog.validate_target_language(language)
        voice = catalog.best_voice_for(language)
        auth = self._authorize("piper")
        http_url = auth["http_url"]
        req = urllib.request.Request(
            http_url,
            data=json.dumps({"text": text, "voice": f"custom:{voice['id']}", "token": auth["token"]}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            audio_bytes = r.read()
        with open(out_wav_path, "wb") as f:
            f.write(audio_bytes)
        return voice["id"]


class NullTTS(TTSBackend):
    """Explicit stub: writes a silent WAV of a length proportional to the text, purely so
    downstream retiming/stretch logic can be exercised mechanically without real credentials. Logs
    loudly on every call so it can never be mistaken for real audio."""

    CHARS_PER_SECOND = 15.0  # rough speaking rate used only to size the placeholder

    def synth(self, text: str, language: str, out_wav_path: str) -> str:
        catalog.validate_target_language(language)
        voice = catalog.best_voice_for(language)
        duration_s = max(0.3, len(text) / self.CHARS_PER_SECOND)
        sample_rate = 22050
        n_samples = int(duration_s * sample_rate)
        with wave.open(out_wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            # Not silence: a few short low-amplitude blips so silencedetect() in retime.py has
            # something realistic (speech-like non-silence separated by real silence) to find,
            # rather than one flat silent block it would treat as a single giant pause.
            frames = bytearray()
            blip_len = int(0.4 * sample_rate)
            gap_len = int(0.15 * sample_rate)
            pos = 0
            toggle = True
            while pos < n_samples:
                seg = blip_len if toggle else gap_len
                seg = min(seg, n_samples - pos)
                for i in range(seg):
                    val = int(2000 * ((i % 50) / 50.0 - 0.5)) if toggle else 0
                    frames += struct.pack("<h", val)
                pos += seg
                toggle = not toggle
            w.writeframes(bytes(frames))
        print(f"[NullTTS] STUB synth: no real TTS credentials/models available in this "
              f"environment; wrote a {duration_s:.2f}s placeholder tone, NOT real speech, to "
              f"{out_wav_path} (would have used voice {voice['id']} for {language})")
        return voice["id"]


def get_default_backend() -> TTSBackend:
    if os.environ.get("TTS_GATEWAY_API_KEY"):
        return PiperGatewayTTS()
    return NullTTS()
