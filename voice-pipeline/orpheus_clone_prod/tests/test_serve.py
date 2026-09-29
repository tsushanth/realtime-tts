import pytest

from orpheus_clone_prod.serve import UnknownVoiceError, resolve_voice_dir
from orpheus_clone_prod.storage import VoiceRecordStore


def test_resolve_voice_dir_rejects_wrong_prefix(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom:v-abc1234567", store)  # Piper prefix, not custom-fast:


def test_resolve_voice_dir_rejects_malformed_id(tmp_path):
    """Test that malformed voice IDs are rejected. Near-miss cases verify the
    regex pattern ^v-[0-9a-f]{10}$ is correctly enforced."""
    store = VoiceRecordStore(root=str(tmp_path))

    # Obviously non-matching case
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:not-a-valid-id", store)

    # Too short (only 6 hex chars instead of 10)
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-abc123", store)

    # Uppercase hex (should be lowercase only)
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-ABC1234567", store)

    # No dash (fails the v- prefix requirement)
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:vabc1234567", store)


def test_resolve_voice_dir_rejects_unknown_voice(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-abc1234567", store)  # never created


def test_resolve_voice_dir_rejects_voice_not_yet_ready(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "training")  # not "ready" yet
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-abc1234567", store)


def test_resolve_voice_dir_returns_checkpoint_path_when_ready(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "ready")
    result = resolve_voice_dir("custom-fast:v-abc1234567", store)
    assert result == store.checkpoint_dir("v-abc1234567")


def test_resolve_voice_dir_returns_independent_paths_for_different_voices(tmp_path):
    """Verify that resolve_voice_dir is a pure function across sequential calls.

    This test shows that resolving voice A's dir returns voice A's path, and
    resolving voice B's dir returns voice B's path, with no shared state or
    cross-contamination across two sequential calls.

    NOTE: This tests pure-function behavior across sequential calls only.
    Testing genuine concurrent-request safety on a warm container (could voice A's
    request ever get voice B's model/audio in parallel executions?) would require
    an actual multi-threaded/async test against a real or fake OrpheusCloneEngine
    instance, which is out of scope for this pure-Python resolve_voice_dir unit
    test. Concurrent safety should be verified in Task 6's Modal integration tests
    or a later integration test suite."""
    store = VoiceRecordStore(root=str(tmp_path))
    for vid in ("v-aaaaaaaaaa", "v-bbbbbbbbbb"):
        store.create(vid, {"speaker_name": vid})
        store.write_status(vid, "ready")

    dir_a = resolve_voice_dir("custom-fast:v-aaaaaaaaaa", store)
    dir_b = resolve_voice_dir("custom-fast:v-bbbbbbbbbb", store)
    assert dir_a != dir_b
    assert dir_a == store.checkpoint_dir("v-aaaaaaaaaa")
    assert dir_b == store.checkpoint_dir("v-bbbbbbbbbb")


def test_synthesize_raises_on_timeout_instead_of_hanging():
    from orpheus_clone_prod.serve import GenerationTimeoutError, _bounded_generate

    def runaway_generator():
        import time
        while True:
            time.sleep(0.05)
            yield b"\x00\x00"

    with pytest.raises(GenerationTimeoutError):
        list(_bounded_generate(runaway_generator(), timeout_s=0.2))
