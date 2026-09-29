import json
import os
import shutil
import tempfile

import pytest

from orpheus_clone_prod.storage import VoiceRecordStore


@pytest.fixture
def store(tmp_path):
    return VoiceRecordStore(root=str(tmp_path))


def test_create_writes_manifest_with_consent_and_awaiting_dataset_status(store, tmp_path):
    consent = {
        "speaker_name": "Jane Doe",
        "attested_by": "Jane Doe",
        "consent": True,
        "consent_text_version": "2026-09-v1",
        "consent_statement": "I am authorized...",
    }
    store.create("v-abc1234567", consent)
    manifest = json.load(open(tmp_path / "v-abc1234567" / "manifest.json"))
    assert manifest["speaker_name"] == "Jane Doe"
    assert manifest["status"] == "awaiting_dataset"
    assert "created_at" in manifest


def test_write_status_then_read_status_round_trips(store):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "training", clip_count=42)
    status = store.read_status("v-abc1234567")
    assert status["status"] == "training"
    assert status["clip_count"] == 42


def test_read_status_on_unknown_voice_returns_none(store):
    assert store.read_status("v-doesnotexist") is None


def test_save_checkpoint_dir_is_atomic_no_partial_merged_dir_visible(store, tmp_path):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    src = tmp_path / "staging_checkpoint"
    src.mkdir()
    (src / "config.json").write_text("{}")

    # Simulate a container killed mid-copy by making save_checkpoint_dir fail partway,
    # then confirm no half-written merged/ dir was left for read_status to see as ready.
    orig_rename = os.rename
    def failing_rename(a, b):
        raise OSError("simulated crash mid-move")
    os.rename = failing_rename
    try:
        with pytest.raises(OSError):
            store.save_checkpoint_dir("v-abc1234567", str(src))
    finally:
        os.rename = orig_rename

    assert not (tmp_path / "v-abc1234567" / "merged").exists()

    # Now do it for real and confirm it succeeds and is visible.
    store.save_checkpoint_dir("v-abc1234567", str(src))
    assert (tmp_path / "v-abc1234567" / "merged" / "config.json").exists()


def test_delete_removes_the_whole_voice_directory(store, tmp_path):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.delete("v-abc1234567")
    assert not (tmp_path / "v-abc1234567").exists()


def test_write_status_on_unknown_vid_raises_valueerror(store):
    """Fix #2: write_status should raise ValueError on unknown vid instead of fabricating a manifest."""
    with pytest.raises(ValueError, match="Voice record .* does not exist"):
        store.write_status("v-doesnotexist", "training")


def test_mark_dataset_uploaded_preserves_existing_status(store):
    """Fix #1: mark_dataset_uploaded should preserve existing status, not reset to awaiting_dataset."""
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    # Move status to "training"
    store.write_status("v-abc1234567", "training", clip_count=42)
    # Call mark_dataset_uploaded and confirm status is still "training"
    store.mark_dataset_uploaded("v-abc1234567")
    manifest = store.read_status("v-abc1234567")
    assert manifest["status"] == "training", "Status should remain 'training', not reset to 'awaiting_dataset'"
    assert manifest["dataset_uploaded"] is True, "dataset_uploaded flag should be True"
    assert manifest["clip_count"] == 42, "Other fields should be preserved"
