"""
This is an import-and-wiring smoke test, not a Modal-execution test (Modal
apps are exercised by actually deploying/running them, matching this
repo's established practice of never mocking the training/serving GPU
path -- see the spec's Testing section). It only confirms main.py wires
Tasks 1-5 together without a typo/import error, which is cheap to check
and has caught real mistakes elsewhere in this codebase (e.g. the
"got an unexpected keyword argument" class of bug found by actually
running serve_benchmark.py).
"""
import importlib


def test_main_module_imports_without_error():
    module = importlib.import_module("orpheus_clone_prod.main")
    assert hasattr(module, "app")
    assert hasattr(module, "api")
    assert hasattr(module, "run_training_job_modal")
    assert hasattr(module, "OrpheusCloneEngine")


# --- OrpheusCloneEngine: one voice per container (modal.parameter) ---------------
# These check wiring and the container-local enter/method logic with the GPU
# engine stubbed out. They do NOT prove Modal's per-parameter container
# routing or GPU memory behavior -- that needs a real deploy (Task 10).

def _engine_user_cls():
    from orpheus_clone_prod import main
    return main.OrpheusCloneEngine._get_user_cls()


def _raw(name):
    return _engine_user_cls().__dict__[name]._get_raw_f()


def test_engine_is_parameterized_by_vid_only():
    from modal.cls import _get_class_constructor_signature

    sig = _get_class_constructor_signature(_engine_user_cls())
    assert list(sig.parameters) == ["vid"]
    assert sig.parameters["vid"].annotation is str


def test_engine_exposes_text_only_method_and_no_per_request_voice_swap():
    user_cls = _engine_user_cls()
    assert "synthesize_text" in user_cls.__dict__
    assert "synthesize_for_voice" not in user_cls.__dict__
    # Instantiating with the parameter is how api.py calls it.
    from orpheus_clone_prod import main
    obj = main.OrpheusCloneEngine(vid="v-abc1234567")
    assert hasattr(obj, "synthesize_text")


def _make_instance(vid):
    inst = object.__new__(_engine_user_cls())
    inst.vid = vid
    return inst


def test_engine_enter_loads_exactly_this_voices_checkpoint_once(tmp_path, monkeypatch):
    from orpheus_clone_prod import main
    from orpheus_clone_prod.storage import VoiceRecordStore

    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "ready")
    loaded = []
    monkeypatch.setattr(main, "CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr(main, "load_engine_for_checkpoint", lambda d: loaded.append(d) or "MODEL")
    monkeypatch.setattr(main, "synthesize", lambda model, text, voice_tag: iter([model.encode(), voice_tag.encode()]))

    inst = _make_instance("v-abc1234567")
    _raw("_init")(inst)
    assert loaded == [store.checkpoint_dir("v-abc1234567")]

    assert _raw("synthesize_text")(inst, "hi") == [b"MODEL", b"v-abc1234567"]
    assert _raw("synthesize_text")(inst, "again") == [b"MODEL", b"v-abc1234567"]
    assert len(loaded) == 1  # never reloads/swaps engines


def test_engine_enter_with_unready_voice_fails_calls_not_container(tmp_path, monkeypatch):
    import pytest
    from orpheus_clone_prod import main
    from orpheus_clone_prod.serve import UnknownVoiceError
    from orpheus_clone_prod.storage import VoiceRecordStore

    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})  # awaiting_dataset
    monkeypatch.setattr(main, "CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr(main, "load_engine_for_checkpoint", lambda d: pytest.fail("must not load"))

    for vid in ("v-abc1234567", "../etc"):
        inst = _make_instance(vid)
        _raw("_init")(inst)  # does not raise
        with pytest.raises(UnknownVoiceError):
            _raw("synthesize_text")(inst, "hi")
