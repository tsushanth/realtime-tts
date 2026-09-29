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

    # The API already committed "training" before spawning; the job must not
    # rewrite the manifest at startup (see train_job.py docstring).
    assert seen_statuses == ["ready"]
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


def _stub_training(monkeypatch):
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


def test_run_training_job_does_not_resurrect_voice_deleted_mid_run(tmp_path, monkeypatch):
    """A voice deleted (by the web container) while training must not come
    back: no manifest, no merged/ checkpoint, no leftover voice dir for the
    trailing Volume commit to push, and no exception."""
    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")
    _stub_training(monkeypatch)

    reloads = []

    def reload_sees_remote_delete():
        # Simulates the web container's committed delete becoming visible
        # in this container on reload().
        reloads.append(1)
        VoiceRecordStore(root=str(tmp_path)).delete(vid)

    run_training_job(vid, root=str(tmp_path), reload_store=reload_sees_remote_delete)

    assert reloads == [1]
    assert store.read_status(vid) is None
    assert not os.path.exists(store.checkpoint_dir(vid))
    assert not os.path.exists(os.path.join(str(tmp_path), vid))


def test_run_training_job_failure_after_delete_does_not_resurrect_or_raise(tmp_path, monkeypatch):
    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"",  # empty zip -> prepare_dataset raises
    )

    run_training_job(vid, root=str(tmp_path), reload_store=lambda: store.delete(vid))

    assert store.read_status(vid) is None
    assert not os.path.exists(os.path.join(str(tmp_path), vid))


def test_run_training_job_completes_deferred_delete_on_success(tmp_path, monkeypatch):
    """A delete requested while training was in flight (delete_requested
    marker file written by the API's 202 path) must be completed once training reaches
    its normal success exit point: the voice is actually gone, and no
    checkpoint is saved for a voice the customer asked to delete."""
    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")
    store.request_delete(vid)
    _stub_training(monkeypatch)

    run_training_job(vid, root=str(tmp_path))

    assert store.read_status(vid) is None
    assert not os.path.exists(store.checkpoint_dir(vid))
    assert not os.path.exists(os.path.join(str(tmp_path), vid))


def test_run_training_job_completes_deferred_delete_on_failure(tmp_path, monkeypatch):
    """Same as the success case, but training itself fails: the voice must
    end up gone, not marked 'failed'."""
    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")
    store.request_delete(vid)
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"",  # empty zip -> prepare_dataset raises
    )

    run_training_job(vid, root=str(tmp_path))

    assert store.read_status(vid) is None
    assert not os.path.exists(os.path.join(str(tmp_path), vid))


def test_run_training_job_reload_failure_falls_back_to_local_view(tmp_path, monkeypatch):
    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")
    _stub_training(monkeypatch)

    def failing_reload():
        raise RuntimeError("there are open files preventing the operation")

    run_training_job(vid, root=str(tmp_path), reload_store=failing_reload)

    assert store.read_status(vid)["status"] == "ready"
    assert os.path.isdir(store.checkpoint_dir(vid))
