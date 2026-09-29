import json
import os

import pytest

from orpheus_clone_prod.storage import VoiceRecordStore
from orpheus_clone_prod.train_job import run_training_job


def test_run_training_job_marks_failed_on_dataset_prep_error(tmp_path, monkeypatch):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    # No dataset uploaded at all -- prepare_dataset should raise, and the
    # job must record "failed" with a message rather than crash silently.
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"",  # empty zip
    )
    with pytest.raises(Exception):
        run_training_job("v-abc1234567", root=str(tmp_path))

    status = store.read_status("v-abc1234567")
    assert status["status"] == "failed"
    assert "error" in status


def test_run_training_job_writes_training_then_ready_on_success(tmp_path, monkeypatch):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})

    seen_statuses = []
    orig_write_status = store.write_status
    def tracking_write_status(vid, status, **fields):
        seen_statuses.append(status)
        return orig_write_status(vid, status, **fields)

    monkeypatch.setattr("orpheus_clone_prod.train_job.VoiceRecordStore", lambda root: store)
    monkeypatch.setattr(store, "write_status", tracking_write_status)
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"fake-zip-bytes",
    )
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job.prepare_dataset",
        lambda zip_bytes, dataset_dir, voice_tag, min_clips=20: [
            {"text": f"{voice_tag}: hello", "audio": "clips/a.wav"}
        ] * 20,
    )
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._run_lora_finetune_and_merge",
        lambda rows, dataset_dir, voice_tag: os.makedirs(f"{dataset_dir}/fake_merged", exist_ok=True) or f"{dataset_dir}/fake_merged",
    )

    run_training_job("v-abc1234567", root=str(tmp_path))

    assert seen_statuses == ["training", "ready"]
    final = store.read_status("v-abc1234567")
    assert final["status"] == "ready"


def test_run_training_job_preserves_original_error_when_status_write_fails(tmp_path, monkeypatch):
    """Regression test: if write_status(vid, "failed", ...) itself raises,
    the original exception should propagate, not the status-write failure."""
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})

    # Make the initial "training" status write succeed, but the "failed" status
    # write fail by having write_status raise when status == "failed"
    orig_write_status = store.write_status
    def selective_write_status(vid, status, **fields):
        if status == "failed":
            raise ValueError("Voice record 'v-abc1234567' does not exist. Call create() first.")
        return orig_write_status(vid, status, **fields)

    monkeypatch.setattr("orpheus_clone_prod.train_job.VoiceRecordStore", lambda root: store)
    monkeypatch.setattr(store, "write_status", selective_write_status)
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"fake-zip-bytes",
    )
    # Make prepare_dataset raise to trigger the exception handler
    original_error_msg = "Dataset preparation failed: invalid audio format"
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job.prepare_dataset",
        lambda zip_bytes, dataset_dir, voice_tag, min_clips=20: (_ for _ in ()).throw(RuntimeError(original_error_msg)),
    )

    # The original RuntimeError should propagate, not the ValueError from write_status
    with pytest.raises(RuntimeError, match=original_error_msg):
        run_training_job("v-abc1234567", root=str(tmp_path))
