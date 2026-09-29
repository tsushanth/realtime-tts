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
import os


def test_main_module_imports_without_error():
    module = importlib.import_module("orpheus_clone_prod.main")
    assert hasattr(module, "app")
    assert hasattr(module, "api")
    assert hasattr(module, "run_training_job_modal")
    assert hasattr(module, "OrpheusCloneEngine")


# --- run_training_job_modal: narrowed post-commit resurrection window ------
# A DELETE can arrive after train_job.py's own last delete-request check
# (which runs before save_checkpoint_dir/the trailing commit) but before
# that commit actually lands. run_training_job_modal now reloads and
# re-checks once more after its own commit to catch exactly that "late"
# delete. These tests call the underlying plain function via Modal's
# .local() (as opposed to .remote()/.spawn()), matching this file's existing
# pattern of exercising container-local logic without a real Modal deploy.

class _FakeVolume:
    """Stands in for the Modal Volume: reload()/commit() are no-ops here
    because the test uses a real on-disk VoiceRecordStore directly (a plain
    tmp_path root has no separate "committed" vs. "local" view), but calls
    are recorded so the test can assert the second reload+recheck actually
    happened."""

    def __init__(self):
        self.reloads = 0
        self.commits = 0

    def reload(self):
        self.reloads += 1

    def commit(self):
        self.commits += 1


def test_run_training_job_modal_completes_delete_requested_late_after_final_commit(tmp_path, monkeypatch):
    from orpheus_clone_prod import main
    from orpheus_clone_prod.storage import VoiceRecordStore

    vid = "v-abc1234567"
    store = VoiceRecordStore(root=str(tmp_path))
    store.create(vid, {"speaker_name": "Jane"})
    store.write_status(vid, "training")

    fake_volume = _FakeVolume()
    monkeypatch.setattr(main, "CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr(main, "checkpoint_volume", fake_volume)

    def fake_run_training_job(vid, root, reload_store=None):
        # Represents the state at the point of the second reload check: the
        # underlying job already ran to its own success exit point (its own
        # _deleted_mid_run check saw no delete yet) and wrote "ready" -- then
        # a delete request (marker file) arrived "late", i.e. is only visible on the
        # NEXT reload, which is exactly what run_training_job_modal's own
        # second reload+recheck exists to catch.
        store.write_status(vid, "ready", clip_count=20, trained_at=0.0)

        def _apply_late_delete_on_next_reload():
            store.request_delete(vid)

        # Piggyback on the fake volume's reload() so the "late" delete only
        # becomes visible starting with run_training_job_modal's own final
        # reload call, not any earlier one.
        fake_volume.reload = _apply_late_delete_on_next_reload

    monkeypatch.setattr(main, "run_training_job", fake_run_training_job)

    main.run_training_job_modal.local(vid)

    assert fake_volume.commits >= 2  # the trailing commit, plus the deferred-delete commit
    assert store.read_status(vid) is None
    assert not __import__("os").path.exists(store.checkpoint_dir(vid))


class _TwoViewVolume:
    """A fake Modal Volume with a real split between the COMMITTED state
    (what other containers see) and THIS container's LOCAL mount, modelling
    the semantics the resurrection race depends on:

    - reload(): pull the committed state into the local view (voice dirs
      gone from committed are removed locally).
    - commit(): push every file in the local view to committed, overwriting
      the committed copy of that same file (last-write-wins per file, as
      Modal documents for concurrent same-file writes). Files that exist only
      in committed -- i.e. written by another container that this container
      never wrote -- are left alone. A voice dir this container had seen and
      then removed locally is removed from committed.

    The API side of the test writes straight into the committed dir (an API
    write+commit in one step)."""

    def __init__(self, committed, local):
        self.committed = committed
        self.local = local
        self.reloads = 0
        self.commits = 0
        self.after_commit = []  # callbacks(volume) run after each commit
        self._seen = set()

    def reload(self):
        import shutil
        self.reloads += 1
        for vid in os.listdir(self.local):
            if not os.path.exists(os.path.join(self.committed, vid)):
                shutil.rmtree(os.path.join(self.local, vid))
        for dirpath, _dirs, files in os.walk(self.committed):
            rel = os.path.relpath(dirpath, self.committed)
            os.makedirs(os.path.join(self.local, rel), exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(dirpath, f), os.path.join(self.local, rel, f))
        self._seen = set(os.listdir(self.local))

    def commit(self):
        import shutil
        self.commits += 1
        for vid in self._seen - set(os.listdir(self.local)):
            shutil.rmtree(os.path.join(self.committed, vid), ignore_errors=True)
        for dirpath, _dirs, files in os.walk(self.local):
            rel = os.path.relpath(dirpath, self.local)
            os.makedirs(os.path.join(self.committed, rel), exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(dirpath, f), os.path.join(self.committed, rel, f))
        self._seen = set(os.listdir(self.local))
        for cb in self.after_commit:
            cb(self)


def test_delete_during_save_checkpoint_survives_jobs_ready_commit_and_deletes_voice(tmp_path, monkeypatch):
    """The scenario round 2 did not cover: a DELETE arrives while the job is
    inside save_checkpoint_dir -- AFTER train_job's own pre-save check saw no
    delete, BEFORE the job writes "ready" and commits. The job's "ready"
    manifest comes from its stale local snapshot and its commit lands after
    the API's, so under last-write-wins it overwrites the committed
    manifest.json. This test proves (a) that clobber really happens in this
    fake (a manifest-based delete_requested flag, written the way the
    previous round's API did, is gone after the job's commit), and (b) the
    marker file survives that same commit, and the job's post-commit check
    finds it and deletes the voice instead of leaving it "ready"."""
    import json

    from fastapi.testclient import TestClient

    from orpheus_clone_prod import main
    from orpheus_clone_prod.api import create_app
    from orpheus_clone_prod.storage import VoiceRecordStore

    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    committed = tmp_path / "committed"
    local = tmp_path / "local"
    committed.mkdir()
    local.mkdir()
    volume = _TwoViewVolume(str(committed), str(local))
    monkeypatch.setattr(main, "CHECKPOINT_ROOT", str(local))
    monkeypatch.setattr(main, "checkpoint_volume", volume)

    # API container: writes land directly in the committed state.
    auth = {"Authorization": "Bearer test-secret"}
    api = TestClient(create_app(root=str(committed), spawn_training=lambda vid, root: None))
    vid = api.post("/v1/orpheus-voices", headers=auth, json={
        "speaker_name": "Jane", "attested_by": "Jane", "consent": True,
        "consent_text_version": "v1", "consent_statement": "ok",
    }).json()["id"]
    api.put(f"/v1/orpheus-voices/{vid}/dataset", headers={**auth, "content-type": "application/zip"}, content=b"z")
    assert api.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=auth).status_code == 200
    committed_store = VoiceRecordStore(root=str(committed))
    manifest_path = os.path.join(str(committed), vid, "manifest.json")

    # Stub the GPU work; the merged checkpoint has a real file so its
    # (non-)arrival in committed state is observable.
    monkeypatch.setattr("orpheus_clone_prod.train_job._load_zip_bytes", lambda vid, root: b"zip")
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job.prepare_dataset",
        lambda zip_bytes, dataset_dir, voice_tag, min_clips=20: [{"text": "t", "audio": "a.wav"}] * 20,
    )

    def fake_finetune(rows, dataset_dir, voice_tag):
        out = os.path.join(dataset_dir, "fake_merged")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "model.safetensors"), "w") as f:
            f.write("weights")
        return out

    monkeypatch.setattr("orpheus_clone_prod.train_job._run_lora_finetune_and_merge", fake_finetune)

    reloads_before_save = []
    real_save = VoiceRecordStore.save_checkpoint_dir

    def save_with_delete_arriving_midway(self, v, local_dir):
        # By now train_job's pre-save check has already run (and passed).
        reloads_before_save.append(volume.reloads)
        real_save(self, v, local_dir)
        # The customer's DELETE lands mid-copy: the API reloads, sees
        # "training", and records the request.
        assert api.delete(f"/v1/orpheus-voices/{v}", headers=auth).status_code == 202
        # Also record the request the way the PREVIOUS round's API did (a
        # manifest field), to show the job's commit clobbers that.
        committed_store.write_status(v, "training", delete_requested=True)

    monkeypatch.setattr(VoiceRecordStore, "save_checkpoint_dir", save_with_delete_arriving_midway)

    after_job_commit = []

    def snapshot(vol):
        if not after_job_commit:
            with open(manifest_path) as f:
                after_job_commit.append((json.load(f), committed_store.is_delete_requested(vid)))

    volume.after_commit.append(snapshot)

    main.run_training_job_modal.local(vid)

    # The pre-save check really did run before the delete arrived (initial
    # reload + train_job's reload), so it could not have caught it.
    assert reloads_before_save == [2]
    # (a) The job's commit overwrote the manifest: status "ready", and the
    # manifest-based flag is gone -- the old mechanism would have lost it.
    manifest_after, marker_after = after_job_commit[0]
    assert manifest_after["status"] == "ready"
    assert "delete_requested" not in manifest_after
    # (b) ...but the marker file survived the same commit...
    assert marker_after is True
    # ...and the post-commit check completed the deletion in committed state.
    assert not os.path.exists(os.path.join(str(committed), vid))
    assert committed_store.read_status(vid) is None
    assert api.get(f"/v1/orpheus-voices/{vid}", headers=auth).status_code == 404


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
