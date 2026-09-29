import io
import wave
import zipfile

import pytest

from orpheus_clone_prod.prepare_dataset import DatasetTooSmallError, prepare_dataset


def _silent_wav_bytes(duration_s=2.0, sr=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(duration_s * sr))
    return buf.getvalue()


def _zip_of(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_prepare_dataset_raises_on_empty_zip(tmp_path):
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(_zip_of({}), str(tmp_path), voice_tag="testvoice", min_clips=1)


def test_prepare_dataset_raises_when_below_min_clips(tmp_path, monkeypatch):
    # Stub transcription so this test doesn't need a real Whisper model.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "hello there",
    )
    zip_bytes = _zip_of({"clip1.wav": _silent_wav_bytes()})
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=5)


def test_prepare_dataset_skips_corrupt_files_and_keeps_valid_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "a valid transcript",
    )
    zip_bytes = _zip_of({
        "good1.wav": _silent_wav_bytes(),
        "good2.wav": _silent_wav_bytes(),
        "corrupt.wav": b"not actually a wav file",
    })
    rows = prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=2)
    assert len(rows) == 2
    assert all(r["text"].startswith("testvoice: ") for r in rows)
    assert all(r["audio"].startswith("clips/") for r in rows)


def test_prepare_dataset_rejects_clips_that_transcribe_empty(tmp_path, monkeypatch):
    # An unusable clip (silence, noise) that Whisper transcribes as empty
    # text must not become a training row with an empty transcript.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "",
    )
    zip_bytes = _zip_of({"silent.wav": _silent_wav_bytes()})
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=1)
