import os
import shutil
import tempfile
import pytest
from moto import mock_aws
import boto3
from voice_storage import VoiceStorage

BUCKET = "test-piper-voices"

@pytest.fixture
def storage():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        cache_dir = tempfile.mkdtemp()
        try:
            yield VoiceStorage(
                bucket=BUCKET,
                endpoint_url=None,  # moto intercepts real boto3 calls, no real endpoint needed
                cache_dir=cache_dir,
                max_cache_entries=3,
            )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


def test_put_then_get_round_trip(storage):
    files = {
        "model.onnx": b"fake-onnx-bytes",
        "model.onnx.json": b'{"espeak": {"voice": "en"}}',
        "owner.json": b'{"public": true}',
    }
    storage.put("test-voice-1", files)

    local_dir = storage.get("test-voice-1")
    assert local_dir is not None
    assert open(os.path.join(local_dir, "model.onnx"), "rb").read() == b"fake-onnx-bytes"
    assert open(os.path.join(local_dir, "owner.json"), "rb").read() == b'{"public": true}'


def test_get_missing_voice_returns_none(storage):
    assert storage.get("does-not-exist") is None


def test_lru_evicts_oldest_when_over_capacity(storage):
    # storage fixture has max_cache_entries=3
    files = {"model.onnx": b"x", "model.onnx.json": b"{}", "owner.json": b'{"public": true}'}
    for vid in ["v1", "v2", "v3", "v4"]:
        storage.put(vid, files)
        storage.get(vid)  # populate cache

    with storage._lock:
        cached_ids = set(storage._lru.keys())
    assert cached_ids == {"v2", "v3", "v4"}  # v1 evicted, it was fetched first
    assert not os.path.exists(os.path.join(storage.cache_dir, "v1"))


def test_put_invalidates_stale_cache_entry(storage):
    files_v1 = {"model.onnx": b"version-1", "model.onnx.json": b"{}", "owner.json": b'{"public": true}'}
    storage.put("versioned", files_v1)
    local_dir = storage.get("versioned")
    assert open(os.path.join(local_dir, "model.onnx"), "rb").read() == b"version-1"

    files_v2 = {"model.onnx": b"version-2", "model.onnx.json": b"{}", "owner.json": b'{"public": true}'}
    storage.put("versioned", files_v2)
    local_dir = storage.get("versioned")
    assert open(os.path.join(local_dir, "model.onnx"), "rb").read() == b"version-2"
