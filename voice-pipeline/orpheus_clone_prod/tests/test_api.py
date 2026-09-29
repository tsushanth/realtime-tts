import re

import pytest
from fastapi.testclient import TestClient

from orpheus_clone_prod.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    spawned = []
    app = create_app(root=str(tmp_path), spawn_training=lambda vid, root: spawned.append(vid))
    client = TestClient(app)
    client.spawned = spawned
    return client


AUTH = {"Authorization": "Bearer test-secret"}
CONSENT_BODY = {
    "speaker_name": "Jane Doe",
    "attested_by": "Jane Doe",
    "consent": True,
    "consent_text_version": "2026-09-v1",
    "consent_statement": "I am authorized...",
}


def test_create_voice_requires_auth(client):
    resp = client.post("/v1/orpheus-voices", json=CONSENT_BODY)
    assert resp.status_code == 401


def test_create_voice_returns_id_matching_expected_shape(client):
    resp = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert re.match(r"^v-[0-9a-f]{10}$", body["id"])
    assert body["status"] == "awaiting_dataset"


def test_create_voice_rejects_missing_consent(client):
    bad = dict(CONSENT_BODY)
    del bad["consent"]
    resp = client.post("/v1/orpheus-voices", json=bad, headers=AUTH)
    assert resp.status_code == 400


def test_create_voice_rejects_empty_consent_statement(client):
    bad = dict(CONSENT_BODY)
    bad["consent_statement"] = ""
    resp = client.post("/v1/orpheus-voices", json=bad, headers=AUTH)
    assert resp.status_code == 400


def test_create_voice_rejects_explicit_consent_false(client):
    bad = dict(CONSENT_BODY)
    bad["consent"] = False
    resp = client.post("/v1/orpheus-voices", json=bad, headers=AUTH)
    assert resp.status_code == 400


def test_poll_unknown_voice_returns_404(client):
    resp = client.get("/v1/orpheus-voices/v-0000000000", headers=AUTH)
    assert resp.status_code == 404


def test_full_lifecycle_upload_commit_poll(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]

    upload_resp = client.put(
        f"/v1/orpheus-voices/{vid}/dataset",
        headers={**AUTH, "content-type": "application/zip"},
        content=b"fake-zip-bytes",
    )
    assert upload_resp.status_code == 200

    commit_resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert commit_resp.status_code == 200
    assert commit_resp.json()["status"] == "training"
    assert vid in client.spawned  # training was spawned, not run inline

    poll_resp = client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert poll_resp.json()["status"] == "training"


def test_commit_without_upload_returns_400(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 400


def test_delete_removes_the_voice(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    resp = client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert resp.status_code == 200
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).status_code == 404


def test_delete_rejected_with_409_while_training(client):
    vid = _create_and_upload(client)
    assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH).status_code == 200
    resp = client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert resp.status_code == 409
    assert "while it is training" in resp.json()["detail"]
    poll = client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert poll.status_code == 200
    assert poll.json()["status"] == "training"


@pytest.mark.parametrize("status", ["awaiting_dataset", "ready", "failed"])
def test_delete_allowed_for_non_training_statuses(tmp_path, monkeypatch, status):
    from orpheus_clone_prod.storage import VoiceRecordStore

    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    client = TestClient(create_app(root=str(tmp_path), spawn_training=lambda vid, root: None))
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    store = VoiceRecordStore(root=str(tmp_path))
    if status != "awaiting_dataset":
        store.write_status(vid, status)
    resp = client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert resp.status_code == 200
    assert store.read_status(vid) is None
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).status_code == 404


def test_spawn_failure_rolls_back_to_awaiting_dataset_and_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")

    def failing_spawn(vid, root):
        raise RuntimeError("modal spawn failed")

    app = create_app(root=str(tmp_path), spawn_training=failing_spawn)
    client = TestClient(app)
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    client.put(f"/v1/orpheus-voices/{vid}/dataset", headers={**AUTH, "content-type": "application/zip"}, content=b"z")

    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 503

    status = client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).json()
    assert status["status"] == "awaiting_dataset"  # rolled back, not stuck in "training"
    assert status["dataset_uploaded"] is True  # dataset_uploaded flag survives rollback


def test_create_voice_rejects_truthy_non_boolean_consent(client):
    for bad_value in ("no", "yes", 1, "true"):
        bad = dict(CONSENT_BODY)
        bad["consent"] = bad_value
        resp = client.post("/v1/orpheus-voices", json=bad, headers=AUTH)
        assert resp.status_code == 400, bad_value


def test_malformed_vid_returns_404(client):
    assert client.get("/v1/orpheus-voices/v-XYZ", headers=AUTH).status_code == 404
    assert client.delete("/v1/orpheus-voices/not-a-vid", headers=AUTH).status_code == 404


# --- Retrain guard (commit only from awaiting_dataset / failed) -------------

def _create_and_upload(client, headers=AUTH, body=CONSENT_BODY):
    vid = client.post("/v1/orpheus-voices", json=body, headers=headers).json()["id"]
    client.put(f"/v1/orpheus-voices/{vid}/dataset", headers={**headers, "content-type": "application/zip"}, content=b"z")
    return vid


def test_commit_rejected_while_training(client):
    vid = _create_and_upload(client)
    assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH).status_code == 200
    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 400
    assert "delete and recreate" in resp.json()["detail"]
    assert client.spawned.count(vid) == 1  # no duplicate training job


def test_commit_rejected_when_ready(tmp_path, monkeypatch):
    from orpheus_clone_prod.storage import VoiceRecordStore

    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    spawned = []
    client = TestClient(create_app(root=str(tmp_path), spawn_training=lambda vid, root: spawned.append(vid)))
    vid = _create_and_upload(client)
    VoiceRecordStore(root=str(tmp_path)).write_status(vid, "ready")
    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 400
    assert spawned == []
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).json()["status"] == "ready"  # untouched


def test_commit_allowed_after_failed_training(tmp_path, monkeypatch):
    from orpheus_clone_prod.storage import VoiceRecordStore

    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    spawned = []
    client = TestClient(create_app(root=str(tmp_path), spawn_training=lambda vid, root: spawned.append(vid)))
    vid = _create_and_upload(client)
    VoiceRecordStore(root=str(tmp_path)).write_status(vid, "failed", error="only 3 usable clips")
    assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH).status_code == 200
    assert spawned == [vid]


# --- Volume reload/commit hooks -----------------------------------------------

def test_store_hooks_reload_before_reads_and_commit_after_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    events = []

    def spawn(vid, root):
        events.append("spawn")

    app = create_app(
        root=str(tmp_path),
        spawn_training=spawn,
        store_reload=lambda: events.append("reload"),
        store_commit=lambda: events.append("commit"),
    )
    client = TestClient(app)

    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    assert events == ["commit"]

    events.clear()
    client.put(f"/v1/orpheus-voices/{vid}/dataset", headers={**AUTH, "content-type": "application/zip"}, content=b"z")
    assert events == ["reload", "commit"]

    events.clear()
    client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    # commit must land BEFORE spawn so the training container sees it
    assert events == ["reload", "commit", "spawn"]

    events.clear()
    client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert events == ["reload"]

    # Still training: delete is refused (409) and must not commit anything.
    events.clear()
    assert client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH).status_code == 409
    assert events == ["reload"]

    from orpheus_clone_prod.storage import VoiceRecordStore
    VoiceRecordStore(root=str(tmp_path)).write_status(vid, "ready")
    events.clear()
    assert client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH).status_code == 200
    assert events == ["reload", "commit"]


def test_spawn_failure_rollback_is_committed(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    events = []

    def failing_spawn(vid, root):
        events.append("spawn")
        raise RuntimeError("modal spawn failed")

    client = TestClient(create_app(
        root=str(tmp_path), spawn_training=failing_spawn,
        store_reload=lambda: events.append("reload"), store_commit=lambda: events.append("commit"),
    ))
    vid = _create_and_upload(client)
    events.clear()
    assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH).status_code == 503
    assert events == ["reload", "commit", "spawn", "commit"]


# --- /v1/orpheus-tts -----------------------------------------------------------

class _FakeRemoteMethod:
    def __init__(self, vid, behavior, calls):
        self._vid = vid
        self._behavior = behavior
        self._calls = calls

    def remote(self, text):
        self._calls.append((self._vid, text))
        return self._behavior(self._vid, text)


def _fake_engine_cls(behavior, calls):
    # Mirrors main.OrpheusCloneEngine's shape: parameterized by vid
    # (modal.parameter), one method taking only the text.
    class FakeEngine:
        def __init__(self, *, vid):
            self.synthesize_text = _FakeRemoteMethod(vid, behavior, calls)

    return FakeEngine


def _tts_client(tmp_path, monkeypatch, behavior):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    calls = []
    app = create_app(
        root=str(tmp_path),
        spawn_training=lambda vid, root: None,
        get_engine_cls=lambda: _fake_engine_cls(behavior, calls),
    )
    client = TestClient(app)
    client.engine_calls = calls
    return client


def _ready_voice(tmp_path, client):
    from orpheus_clone_prod.storage import VoiceRecordStore

    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    VoiceRecordStore(root=str(tmp_path)).write_status(vid, "ready")
    return vid


def test_tts_returns_audio_bytes_from_engine(tmp_path, monkeypatch):
    client = _tts_client(tmp_path, monkeypatch, lambda vid, text: [b"\x01\x02", b"\x03\x04"])
    vid = _ready_voice(tmp_path, client)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.content == b"\x01\x02\x03\x04"
    assert resp.headers["content-type"] == "audio/pcm"
    # The engine is instantiated with the bare vid as its class parameter.
    assert client.engine_calls == [(vid, "hi")]


def test_tts_requires_auth(tmp_path, monkeypatch):
    client = _tts_client(tmp_path, monkeypatch, lambda vid, text: [b"x"])
    vid = _ready_voice(tmp_path, client)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"})
    assert resp.status_code == 401
    assert client.engine_calls == []


def test_tts_requires_text(tmp_path, monkeypatch):
    client = _tts_client(tmp_path, monkeypatch, lambda vid, text: [b"x"])
    vid = _ready_voice(tmp_path, client)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": ""}, headers=AUTH)
    assert resp.status_code == 400
    assert client.engine_calls == []


@pytest.mark.parametrize(
    "voice",
    [
        "tara",  # not a custom-fast voice
        "custom-fast:",  # empty id
        "custom-fast:../etc",  # malformed id
        "custom-fast:v-ABC1234567",  # malformed (uppercase)
        "custom-fast:v-abc1234567",  # well-formed but unknown
    ],
)
def test_tts_invalid_or_unknown_voice_rejected_before_gpu(tmp_path, monkeypatch, voice):
    client = _tts_client(tmp_path, monkeypatch, lambda vid, text: [b"x"])
    resp = client.post("/v1/orpheus-tts", json={"voice": voice, "text": "hi"}, headers=AUTH)
    assert resp.status_code == 400
    assert client.engine_calls == []  # resolved in the web container; never reached the GPU


@pytest.mark.parametrize("status", ["awaiting_dataset", "training", "failed"])
def test_tts_not_ready_voice_rejected_before_gpu(tmp_path, monkeypatch, status):
    from orpheus_clone_prod.storage import VoiceRecordStore

    client = _tts_client(tmp_path, monkeypatch, lambda vid, text: [b"x"])
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    if status != "awaiting_dataset":
        VoiceRecordStore(root=str(tmp_path)).write_status(vid, status)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=AUTH)
    assert resp.status_code == 400
    assert client.engine_calls == []


def test_tts_engine_unknown_voice_error_still_maps_to_400(tmp_path, monkeypatch):
    # The engine container re-checks in @modal.enter() (race / direct caller)
    # and raises UnknownVoiceError from the method; the API maps it to 400.
    from orpheus_clone_prod.serve import UnknownVoiceError

    def raise_unknown(vid, text):
        raise UnknownVoiceError("voice not ready")

    client = _tts_client(tmp_path, monkeypatch, raise_unknown)
    vid = _ready_voice(tmp_path, client)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=AUTH)
    assert resp.status_code == 400


def test_tts_generation_timeout_maps_to_503(tmp_path, monkeypatch):
    from orpheus_clone_prod.serve import GenerationTimeoutError

    def raise_timeout(vid, text):
        raise GenerationTimeoutError("too slow")

    client = _tts_client(tmp_path, monkeypatch, raise_timeout)
    vid = _ready_voice(tmp_path, client)
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=AUTH)
    assert resp.status_code == 503


# --- Ownership isolation ----------------------------------------------------------

OWNER_A = {**AUTH, "X-Owner": "user-a"}
OWNER_B = {**AUTH, "X-Owner": "user-b"}
OWNED_BODY = {**CONSENT_BODY, "owner": "user-a"}


def _owned_ready_voice(tmp_path, monkeypatch):
    from orpheus_clone_prod.storage import VoiceRecordStore

    client = _tts_client(tmp_path, monkeypatch, lambda voice, text: [b"audio"])
    vid = client.post("/v1/orpheus-voices", json=OWNED_BODY, headers=AUTH).json()["id"]
    return client, vid, VoiceRecordStore(root=str(tmp_path))


def test_owner_is_stored_in_manifest(tmp_path, monkeypatch):
    client, vid, store = _owned_ready_voice(tmp_path, monkeypatch)
    assert store.read_status(vid)["owner"] == "user-a"


def test_wrong_owner_is_forbidden_on_every_voice_route(tmp_path, monkeypatch):
    client, vid, store = _owned_ready_voice(tmp_path, monkeypatch)
    for headers in (OWNER_B, AUTH):  # wrong owner, and missing X-Owner
        assert client.get(f"/v1/orpheus-voices/{vid}", headers=headers).status_code == 403
        assert client.put(
            f"/v1/orpheus-voices/{vid}/dataset", headers={**headers, "content-type": "application/zip"}, content=b"z"
        ).status_code == 403
        assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=headers).status_code == 403
        assert client.delete(f"/v1/orpheus-voices/{vid}", headers=headers).status_code == 403

    store.write_status(vid, "ready")
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=OWNER_B)
    assert resp.status_code == 403
    assert client.engine_calls == []  # never reached the GPU

    # Nothing was changed by the forbidden calls.
    assert store.read_status(vid)["status"] == "ready"
    assert not store.read_status(vid).get("dataset_uploaded")


def test_correct_owner_can_use_every_voice_route(tmp_path, monkeypatch):
    client, vid, store = _owned_ready_voice(tmp_path, monkeypatch)
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=OWNER_A).status_code == 200
    assert client.put(
        f"/v1/orpheus-voices/{vid}/dataset", headers={**OWNER_A, "content-type": "application/zip"}, content=b"z"
    ).status_code == 200
    assert client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=OWNER_A).status_code == 200

    store.write_status(vid, "ready")
    resp = client.post("/v1/orpheus-tts", json={"voice": f"custom-fast:{vid}", "text": "hi"}, headers=OWNER_A)
    assert resp.status_code == 200
    assert resp.content == b"audio"

    assert client.delete(f"/v1/orpheus-voices/{vid}", headers=OWNER_A).status_code == 200


def test_voice_without_stored_owner_skips_check(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=OWNER_B).status_code == 200


def test_create_rejects_non_string_owner(client):
    resp = client.post("/v1/orpheus-voices", json={**CONSENT_BODY, "owner": 123}, headers=AUTH)
    assert resp.status_code == 400


def test_create_falls_back_to_x_owner_header_when_body_has_no_owner(client):
    # This mirrors the real production caller (ReadAloudAI's backend), which
    # never puts "owner" in the create body -- it only ever sends X-Owner.
    headers = {**AUTH, "X-Owner": "user-42"}
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=headers).json()["id"]

    poll_resp = client.get(f"/v1/orpheus-voices/{vid}", headers=headers)
    assert poll_resp.status_code == 200
    assert poll_resp.json()["owner"] == "user-42"

    # Enforcement now actually works end-to-end for header-derived owners.
    wrong_owner_resp = client.get(f"/v1/orpheus-voices/{vid}", headers={**AUTH, "X-Owner": "someone-else"})
    assert wrong_owner_resp.status_code == 403


def test_create_prefers_explicit_body_owner_over_x_owner_header(client):
    headers = {**AUTH, "X-Owner": "header-owner"}
    body = {**CONSENT_BODY, "owner": "body-owner"}
    vid = client.post("/v1/orpheus-voices", json=body, headers=headers).json()["id"]

    poll_resp = client.get(f"/v1/orpheus-voices/{vid}", headers={**AUTH, "X-Owner": "body-owner"})
    assert poll_resp.status_code == 200
    assert poll_resp.json()["owner"] == "body-owner"
