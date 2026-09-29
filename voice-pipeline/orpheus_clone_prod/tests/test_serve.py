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


# --- Regression guards for the three verified serving fixes -----------------
# load_engine_for_checkpoint needs torch/vllm/orpheus-speech, none of which are
# installed in the unit-test env, so inject fake modules into sys.modules. The
# fakes are strict: they would have caught the original TypeError (extra
# tokenizer=/max_model_len= kwargs) and a wrong AsyncEngineArgs patch target.

import functools
import sys
import types


def _install_fake_serving_stack(monkeypatch):
    calls = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs

    fake_pkg = types.ModuleType("orpheus_tts")
    fake_pkg.__path__ = []  # mark as a package
    fake_engine_mod = types.ModuleType("orpheus_tts.engine_class")
    original_engine_args = object()  # sentinel: the unpatched library default
    fake_engine_mod.AsyncEngineArgs = original_engine_args

    class FakeOrpheusModel:
        # Mirrors orpheus-speech's real signature: only model_name and dtype
        # are accepted -- anything else raises TypeError, like the real one did.
        def __init__(self, model_name, dtype):
            calls["model_name"] = model_name
            calls["dtype"] = dtype
            # Snapshot the patch as seen at construction time (the real class
            # reads engine_class.AsyncEngineArgs inside __init__).
            calls["engine_args_at_construct"] = fake_engine_mod.AsyncEngineArgs

    fake_pkg.OrpheusModel = FakeOrpheusModel
    fake_pkg.engine_class = fake_engine_mod

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "orpheus_tts", fake_pkg)
    monkeypatch.setitem(sys.modules, "orpheus_tts.engine_class", fake_engine_mod)
    return calls, fake_torch, fake_vllm, original_engine_args


def test_load_engine_passes_only_model_name_and_dtype(monkeypatch):
    from orpheus_clone_prod.serve import load_engine_for_checkpoint

    calls, fake_torch, _, _ = _install_fake_serving_stack(monkeypatch)
    model = load_engine_for_checkpoint("/checkpoints/v-abc1234567/merged")  # TypeError if extra kwargs
    assert model is not None
    assert calls["model_name"] == "/checkpoints/v-abc1234567/merged"
    assert calls["dtype"] is fake_torch.bfloat16


def test_load_engine_patches_async_engine_args_on_engine_class_module_before_construct(monkeypatch):
    from orpheus_clone_prod.serve import load_engine_for_checkpoint

    calls, _, fake_vllm, original = _install_fake_serving_stack(monkeypatch)
    load_engine_for_checkpoint("/checkpoints/v-abc1234567/merged")

    patched = calls["engine_args_at_construct"]
    assert patched is not original, "AsyncEngineArgs on orpheus_tts.engine_class was not patched before OrpheusModel()"
    assert isinstance(patched, functools.partial)
    assert patched.func is fake_vllm.AsyncEngineArgs
    assert patched.keywords == {"max_model_len": 2048}
    # And it builds args with the capped context length.
    assert patched(model="x").kwargs["max_model_len"] == 2048


def test_synthesize_passes_explicit_stop_token_ids():
    from orpheus_clone_prod.serve import synthesize

    seen = {}

    class FakeModel:
        def generate_speech(self, **kwargs):
            seen.update(kwargs)
            yield b"\x01\x02"

    out = list(synthesize(FakeModel(), "hello", voice_tag="v-abc1234567"))
    assert out == [b"\x01\x02"]
    assert seen["stop_token_ids"] == [128258]
    assert seen["voice"] == "v-abc1234567"
    assert seen["prompt"] == "hello"
