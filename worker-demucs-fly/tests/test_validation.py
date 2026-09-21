"""Unit tests for the request-validation logic in server.py, in worker-piper-fly/tests style
(unittest, no model/network needed - these do NOT load Demucs). Run:
  cd worker-demucs-fly && .venv/bin/python -m pytest tests/test_validation.py -v
"""
import base64
import hashlib
import hmac
import json
import os
import time
import unittest

os.environ.setdefault("AUTH_TOKEN", "test-token")
os.environ.setdefault("SESSION_SECRET", "test-secret")

import numpy as np

import server


class Stem(unittest.TestCase):
    def test_default_is_vocals(self):
        self.assertEqual(server.validate_stem(None), "vocals")

    def test_valid_stems_pass_through(self):
        for s in ("vocals", "drums", "bass", "other"):
            self.assertEqual(server.validate_stem(s), s)

    def test_unknown_stem_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_stem("saxophone")
        self.assertEqual(ctx.exception.status, 400)


class Filename(unittest.TestCase):
    def test_wav_allowed(self):
        server.validate_filename("clip.wav")  # no exception

    def test_missing_extension_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_filename("clip")
        self.assertEqual(ctx.exception.status, 415)

    def test_empty_filename_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_filename("")
        self.assertEqual(ctx.exception.status, 415)

    def test_disallowed_extension_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_filename("clip.mp3")
        self.assertEqual(ctx.exception.status, 415)

    def test_case_insensitive(self):
        server.validate_filename("CLIP.WAV")  # no exception


class Size(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_size(0)
        self.assertEqual(ctx.exception.status, 400)

    def test_over_limit_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_size(server.MAX_UPLOAD_BYTES + 1)
        self.assertEqual(ctx.exception.status, 413)

    def test_at_limit_ok(self):
        server.validate_size(server.MAX_UPLOAD_BYTES)  # no exception

    def test_small_ok(self):
        server.validate_size(1024)  # no exception


class Duration(unittest.TestCase):
    def test_too_short_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_duration(1, 44100)  # ~0.00002s
        self.assertEqual(ctx.exception.status, 400)

    def test_too_long_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.validate_duration(int((server.MAX_DURATION_S + 5) * 44100), 44100)
        self.assertEqual(ctx.exception.status, 413)

    def test_within_range_ok(self):
        self.assertAlmostEqual(server.validate_duration(44100, 44100), 1.0)


class DecodeAudio(unittest.TestCase):
    def test_garbage_bytes_rejected(self):
        with self.assertRaises(server.ValidationError) as ctx:
            server.decode_audio(b"not a real wav file at all")
        self.assertEqual(ctx.exception.status, 400)

    def test_valid_wav_decodes(self):
        import io
        import soundfile as sf
        t = np.linspace(0, 1, 8000, dtype=np.float32)
        tone = 0.1 * np.sin(2 * np.pi * 440 * t)
        buf = io.BytesIO()
        sf.write(buf, tone, 8000, format="WAV")
        audio, sr = server.decode_audio(buf.getvalue())
        self.assertEqual(sr, 8000)
        self.assertEqual(audio.shape[0], 8000)


class SessionAuth(unittest.TestCase):
    def _make_token(self, secret: str, payload: dict) -> str:
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = base64.urlsafe_b64encode(
            hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
        ).decode().rstrip("=")
        return f"{payload_b64}.{sig}"

    def test_valid_unexpired_token_accepted(self):
        token = self._make_token("test-secret", {"id": "key1", "exp": (time.time() + 60) * 1000})
        key_id, uid = server.verify_session_claims(token)
        self.assertEqual(key_id, "key1")

    def test_expired_token_rejected(self):
        token = self._make_token("test-secret", {"id": "key1", "exp": (time.time() - 60) * 1000})
        key_id, _ = server.verify_session_claims(token)
        self.assertIsNone(key_id)

    def test_wrong_secret_rejected(self):
        token = self._make_token("wrong-secret", {"id": "key1", "exp": (time.time() + 60) * 1000})
        key_id, _ = server.verify_session_claims(token)
        self.assertIsNone(key_id)

    def test_malformed_token_rejected(self):
        key_id, _ = server.verify_session_claims("not-a-valid-token")
        self.assertIsNone(key_id)

    def test_empty_token_rejected(self):
        key_id, _ = server.verify_session_claims("")
        self.assertIsNone(key_id)

    def test_uid_passthrough(self):
        token = self._make_token("test-secret", {"id": "key1", "uid": "user9", "exp": (time.time() + 60) * 1000})
        key_id, uid = server.verify_session_claims(token)
        self.assertEqual(key_id, "key1")
        self.assertEqual(uid, "user9")


if __name__ == "__main__":
    unittest.main()
