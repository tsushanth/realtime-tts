import io
import os
import wave
import zipfile

import pytest

from orpheus_clone_prod.prepare_dataset import (
    DatasetTooSmallError,
    TranscriptionUnavailableError,
    prepare_dataset,
)


@pytest.fixture(autouse=True)
def _no_real_whisper(monkeypatch):
    # prepare_dataset eagerly loads the transcription model before the
    # per-clip loop; stub it so these tests never touch faster-whisper.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.load_transcription_model",
        lambda: object(),
    )


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


def test_prepare_dataset_skips_corrupt_non_wav_files(tmp_path, monkeypatch):
    # Corrupt MP3 or FLAC files should be skipped gracefully, not crash the batch.
    # This tests the fix for the issue where only WAV files had validation.
    def mock_transcribe(path):
        if "corrupt" in path:
            raise ValueError(f"Failed to decode audio file: {path}")
        return "a valid transcript"

    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        mock_transcribe,
    )
    zip_bytes = _zip_of({
        "good1.wav": _silent_wav_bytes(),
        "good2.wav": _silent_wav_bytes(),
        "corrupt.mp3": b"not actually an mp3 file",
        "corrupt.flac": b"not actually a flac file",
    })
    rows = prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=2)
    assert len(rows) == 2
    assert all(r["text"].startswith("testvoice: ") for r in rows)
    assert all(r["audio"].startswith("clips/") for r in rows)


def test_prepare_dataset_cleans_up_clips_on_error(tmp_path, monkeypatch):
    # When DatasetTooSmallError is raised, the clips directory should be cleaned up
    # so that a retry doesn't have orphaned files from the failed attempt.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "a valid transcript",
    )
    zip_bytes = _zip_of({
        "clip1.wav": _silent_wav_bytes(),
    })
    clips_dir = os.path.join(str(tmp_path), "clips")

    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=5)

    # After the error, clips_dir should be removed or empty
    assert not os.path.exists(clips_dir) or len(os.listdir(clips_dir)) == 0


def test_prepare_dataset_model_load_failure_is_not_reported_as_bad_data(tmp_path, monkeypatch):
    # A systemic failure loading Whisper (download failure, missing cuDNN,
    # ...) must propagate as TranscriptionUnavailableError, NOT be swallowed
    # per-clip and surface as "only 0 usable clips" (blaming the customer).
    def failing_load():
        raise TranscriptionUnavailableError("transcription service unavailable: libcudnn not found")

    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.load_transcription_model",
        failing_load,
    )
    transcribed = []
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: transcribed.append(path) or "text",
    )
    zip_bytes = _zip_of({f"clip{i}.wav": _silent_wav_bytes() for i in range(3)})
    with pytest.raises(TranscriptionUnavailableError) as exc_info:
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=1)
    assert not isinstance(exc_info.value, DatasetTooSmallError)
    assert "usable clips" not in str(exc_info.value)
    assert transcribed == []  # never reached the per-clip loop


def test_real_load_transcription_model_wraps_import_failure(monkeypatch):
    # The real loader (not the autouse stub) converts any load failure into
    # TranscriptionUnavailableError. faster_whisper is not installed in the
    # test env, or is forced missing here.
    import sys

    from orpheus_clone_prod import prepare_dataset as mod

    monkeypatch.undo()  # drop the autouse stub for this test
    monkeypatch.setattr(mod, "_WHISPER_MODEL", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import -> ImportError
    with pytest.raises(TranscriptionUnavailableError):
        mod.load_transcription_model()
