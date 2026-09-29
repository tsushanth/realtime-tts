"""
Real end-to-end test against the deployed orpheus-clone-prod app: create a
voice, upload a real recording set, commit, poll to ready, synthesize, and
delete. Costs real GPU time (a real LoRA training run) and is not run in CI.

This is Task 10 from docs/superpowers/plans/2026-09-28-orpheus-streaming-voice-cloning.md,
finally written for real -- the plan named this file but it was never created;
Task 10 was deliberately left as a manual step given the GPU cost involved.

Run (after `cd voice-pipeline && modal deploy -m orpheus_clone_prod.main`):
    ORPHEUS_CLONE_BASE_URL=https://...modal.run \
    ORPHEUS_CLONE_SECRET=... \
    python3 -m pytest tests/test_e2e_real.py -v -s -m e2e_real
"""
import io
import os
import time
import zipfile

import pytest
import requests

pytestmark = pytest.mark.e2e_real

BASE = os.environ.get("ORPHEUS_CLONE_BASE_URL")
SECRET = os.environ.get("ORPHEUS_CLONE_SECRET")


@pytest.mark.skipif(not BASE or not SECRET, reason="set ORPHEUS_CLONE_BASE_URL and ORPHEUS_CLONE_SECRET to run")
def test_full_lifecycle_against_deployed_service():
    headers = {"Authorization": f"Bearer {SECRET}"}

    consent = {
        "speaker_name": "Test Speaker",
        "attested_by": "Test Speaker",
        "consent": True,
        "consent_text_version": "2026-09-v1",
        "consent_statement": "I am authorized to consent on behalf of the speaker named in this request.",
    }
    print("\n=== 1. Create voice ===")
    create_resp = requests.post(f"{BASE}/v1/orpheus-voices", json=consent, headers=headers, timeout=30)
    print(create_resp.status_code, create_resp.text)
    assert create_resp.status_code == 200, create_resp.text
    vid = create_resp.json()["id"]
    print(f"vid = {vid}")

    print("\n=== 2. Upload dataset (25 real clips from the pilot corpus) ===")
    sample_clips_dir = os.path.expanduser(
        "~/Documents/GitHub/worktrees/task-10-e2e-verification/training-data/audio"
    )
    if not os.path.isdir(sample_clips_dir):
        sample_clips_dir = os.path.expanduser("~/Documents/GitHub/realtime-tts/training-data/audio")
    all_clips = sorted(os.listdir(sample_clips_dir))
    chosen = all_clips[:25]
    print(f"using {len(chosen)} clips from {sample_clips_dir}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in chosen:
            zf.write(os.path.join(sample_clips_dir, name), name)
    upload_resp = requests.put(
        f"{BASE}/v1/orpheus-voices/{vid}/dataset",
        headers={**headers, "content-type": "application/zip"},
        data=buf.getvalue(),
        timeout=60,
    )
    print(upload_resp.status_code, upload_resp.text)
    assert upload_resp.status_code == 200, upload_resp.text

    print("\n=== 3. Commit (starts training) ===")
    t_commit = time.time()
    commit_resp = requests.post(f"{BASE}/v1/orpheus-voices/{vid}/dataset/commit", headers=headers, timeout=30)
    print(commit_resp.status_code, commit_resp.text)
    assert commit_resp.status_code == 200, commit_resp.text

    print("\n=== 4. Poll until ready (up to 90 min) ===")
    deadline = time.time() + 90 * 60
    status = None
    last_body = None
    while time.time() < deadline:
        try:
            poll_resp = requests.get(f"{BASE}/v1/orpheus-voices/{vid}", headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"  poll request failed transiently ({e}); retrying")
            time.sleep(10)
            continue
        last_body = poll_resp.json()
        status = last_body.get("status")
        elapsed = int(time.time() - t_commit)
        print(f"  [{elapsed}s] status={status} body={last_body}")
        if status in ("ready", "failed"):
            break
        time.sleep(30)
    training_seconds = time.time() - t_commit
    print(f"training took {training_seconds:.0f}s, final status={status}")
    assert status == "ready", f"training did not reach ready in time, last body: {last_body}"

    print("\n=== 5. Synthesize (measuring cold TTFB) ===")
    t0 = time.time()
    synth_resp = requests.post(
        f"{BASE}/v1/orpheus-tts",
        headers={**headers, "content-type": "application/json"},
        json={"text": "Hello, this is a test of my cloned voice.", "voice": f"custom-fast:{vid}"},
        timeout=120,
    )
    elapsed = time.time() - t0
    print(f"synth status={synth_resp.status_code}, elapsed={elapsed:.1f}s, bytes={len(synth_resp.content)}")
    assert synth_resp.status_code == 200, synth_resp.text
    assert len(synth_resp.content) > 1000, "response too small to be real audio"

    print("\n=== 6. Second synthesis (measuring warm TTFB) ===")
    t0 = time.time()
    synth_resp2 = requests.post(
        f"{BASE}/v1/orpheus-tts",
        headers={**headers, "content-type": "application/json"},
        json={"text": "This is a second, different sentence.", "voice": f"custom-fast:{vid}"},
        timeout=60,
    )
    elapsed2 = time.time() - t0
    print(f"synth2 status={synth_resp2.status_code}, elapsed={elapsed2:.1f}s, bytes={len(synth_resp2.content)}")
    assert synth_resp2.status_code == 200, synth_resp2.text

    with open("/tmp/e2e_real_sample_1.pcm", "wb") as f:
        f.write(synth_resp.content)
    with open("/tmp/e2e_real_sample_2.pcm", "wb") as f:
        f.write(synth_resp2.content)
    print("saved audio to /tmp/e2e_real_sample_{1,2}.pcm (PCM16 24kHz mono)")

    print("\n=== 7. Delete ===")
    del_resp = requests.delete(f"{BASE}/v1/orpheus-voices/{vid}", headers=headers, timeout=30)
    print(del_resp.status_code, del_resp.text)
    assert del_resp.status_code == 200, del_resp.text

    poll_after_delete = requests.get(f"{BASE}/v1/orpheus-voices/{vid}", headers=headers, timeout=30)
    print(f"poll after delete: {poll_after_delete.status_code}")
    assert poll_after_delete.status_code == 404

    print("\n=== SUMMARY ===")
    print(f"training_seconds={training_seconds:.0f}")
    print(f"cold_synth_seconds={elapsed:.1f}")
    print(f"warm_synth_seconds={elapsed2:.1f}")
