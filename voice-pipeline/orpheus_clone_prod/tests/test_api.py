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
