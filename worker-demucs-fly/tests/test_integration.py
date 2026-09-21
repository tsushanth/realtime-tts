"""Integration test that actually runs Demucs on a small real audio clip via the HTTP endpoint
(fastapi TestClient, in-process, no separate server process). Downloads pretrained htdemucs
weights on first run if not already cached (~80MB) and takes tens of seconds on CPU - skip with
SKIP_DEMUCS_INTEGRATION=1 if that's not wanted (e.g. no network egress in CI).
Run: cd worker-demucs-fly && .venv/bin/python -m pytest tests/test_integration.py -v -s
"""
import io
import os
import unittest

os.environ.setdefault("AUTH_TOKEN", "test-token")

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(os.environ.get("SKIP_DEMUCS_INTEGRATION") == "1", "SKIP_DEMUCS_INTEGRATION=1")
class IsolateEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import server
        cls.server = server
        cls.client = TestClient(server.app)

    def _tone_wav_bytes(self, seconds=2.0, sr=8000, freq=440.0):
        t = np.linspace(0, seconds, int(sr * seconds), dtype=np.float32)
        tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, tone, sr, format="WAV")
        return buf.getvalue()

    def test_missing_auth_rejected(self):
        data = self._tone_wav_bytes()
        r = self.client.post("/v1/isolate", files={"file": ("clip.wav", data, "audio/wav")})
        self.assertEqual(r.status_code, 401)

    def test_bad_extension_rejected(self):
        data = self._tone_wav_bytes()
        r = self.client.post(
            "/v1/isolate", files={"file": ("clip.mp3", data, "audio/mpeg")},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(r.status_code, 415)

    def test_empty_file_rejected(self):
        r = self.client.post(
            "/v1/isolate", files={"file": ("clip.wav", b"", "audio/wav")},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(r.status_code, 400)

    def test_unknown_stem_rejected(self):
        data = self._tone_wav_bytes()
        r = self.client.post(
            "/v1/isolate", files={"file": ("clip.wav", data, "audio/wav")},
            data={"stem": "saxophone"},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(r.status_code, 400)

    def test_real_clip_end_to_end(self):
        """Uses the verification/speech.wav fixture (macOS `say` output) end-to-end through the
        real HTTP endpoint and Demucs model - the slow, real-thing test."""
        speech_path = os.path.join(os.path.dirname(HERE), "verification", "speech.wav")
        if not os.path.exists(speech_path):
            self.skipTest("verification/speech.wav not present - see verification/make_and_verify.py")
        with open(speech_path, "rb") as f:
            data = f.read()
        r = self.client.post(
            "/v1/isolate", files={"file": ("speech.wav", data, "audio/wav")},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        out, sr = sf.read(io.BytesIO(r.content))
        self.assertGreater(len(out), 0)
        self.assertEqual(sr, 44100)  # htdemucs native rate
        # isolated speech should carry real signal, not silence
        self.assertGreater(np.std(out), 1e-4)


if __name__ == "__main__":
    unittest.main()
