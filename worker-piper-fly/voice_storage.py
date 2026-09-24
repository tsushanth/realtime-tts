"""Tigris-backed voice storage with a local LRU disk cache. Tigris (Fly's S3-compatible
object storage) is the source of truth for every published voice - built-in catalog
voices and customer/cloned voices alike, no distinction at this layer (see
2026-09-22-android-realtime-tts-production-readiness-design.md, "Piece 1"). Each machine
keeps a bounded local cache of recently-used voices' raw files so a warm machine doesn't
re-fetch from Tigris on every request; a machine that's never seen a voice fetches it on
first use."""
import os
import shutil
import tempfile
from collections import OrderedDict
from threading import Lock, Condition

import boto3

VOICE_FILES = ("model.onnx", "model.onnx.json", "owner.json")


class VoiceStorage:
    def __init__(self, bucket: str, endpoint_url: str | None, cache_dir: str, max_cache_entries: int):
        self.bucket = bucket
        self.cache_dir = cache_dir
        self.max_cache_entries = max_cache_entries
        self._client = boto3.client("s3", endpoint_url=endpoint_url)
        self._lru: "OrderedDict[str, str]" = OrderedDict()  # vid -> local dir path
        self._lock = Lock()
        self._fetch_cv = Condition(self._lock)  # condition variable for single-flighting fetches
        self._fetching: dict[str, bool] = {}  # vid -> whether a fetch is in progress
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, vid: str) -> str | None:
        with self._fetch_cv:
            # Check if already cached
            cached = self._lru.get(vid)
            if cached is not None and os.path.isdir(cached):
                self._lru.move_to_end(vid)
                return cached

            # Wait for any in-flight fetch for this vid to complete (single-flighting)
            while self._fetching.get(vid, False):
                self._fetch_cv.wait()
                # After fetch completes, check cache again
                cached = self._lru.get(vid)
                if cached is not None and os.path.isdir(cached):
                    self._lru.move_to_end(vid)
                    return cached

            # Mark this vid as being fetched
            self._fetching[vid] = True

        # Fetch outside the lock to avoid blocking other operations
        local_dir = os.path.join(self.cache_dir, vid)
        tmp_dir = None
        voice_not_found = False
        try:
            # Use a unique temp directory per fetch attempt to avoid races
            tmp_dir = tempfile.mkdtemp(dir=self.cache_dir, prefix=f"{vid}.")

            for fname in VOICE_FILES:
                key = f"{vid}/{fname}"
                dest = os.path.join(tmp_dir, fname)
                try:
                    self._client.download_file(self.bucket, key, dest)
                except self._client.exceptions.ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    # Check both "NoSuchKey" and "404" (moto uses 404 as the error code)
                    is_not_found = error_code in ("NoSuchKey", "404")
                    if fname == "model.onnx" and is_not_found:
                        # Voice doesn't exist in storage
                        voice_not_found = True
                        break
                    elif fname == "model.onnx":
                        # model.onnx fetch failed for reasons other than not found (403, throttle, etc)
                        raise RuntimeError(
                            f"failed to fetch voice {vid!r} from storage (error {error_code}): {e}"
                        ) from e
                    else:
                        # metadata files are expected to exist if model.onnx exists
                        raise RuntimeError(
                            f"voice {vid!r} missing required file {fname!r} in storage (error {error_code}): {e}"
                        ) from e

            if voice_not_found:
                return None

            # Atomic rename: move temp to final location
            shutil.rmtree(local_dir, ignore_errors=True)
            os.rename(tmp_dir, local_dir)
            tmp_dir = None  # mark as successfully moved

        finally:
            # Clean up temp directory if fetch failed
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Update LRU and signal waiting threads
            with self._fetch_cv:
                self._fetching.pop(vid, None)

                if not voice_not_found and os.path.isdir(local_dir):
                    self._lru[vid] = local_dir
                    self._lru.move_to_end(vid)
                    while len(self._lru) > self.max_cache_entries:
                        _, evicted_dir = self._lru.popitem(last=False)
                        shutil.rmtree(evicted_dir, ignore_errors=True)

                self._fetch_cv.notify_all()

        return self._lru.get(vid) if self._lru.get(vid) and os.path.isdir(self._lru[vid]) else None

    def put(self, vid: str, files: dict[str, bytes]) -> None:
        for fname, data in files.items():
            self._client.put_object(Bucket=self.bucket, Key=f"{vid}/{fname}", Body=data)
        with self._lock:
            cached = self._lru.pop(vid, None)
        if cached:
            shutil.rmtree(cached, ignore_errors=True)  # force a re-fetch of the new version

    def delete(self, vid: str) -> bool:
        existed = False
        for fname in VOICE_FILES:
            key = f"{vid}/{fname}"
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                existed = True
            except self._client.exceptions.ClientError:
                continue
            self._client.delete_object(Bucket=self.bucket, Key=key)
        with self._lock:
            cached = self._lru.pop(vid, None)
        if cached:
            shutil.rmtree(cached, ignore_errors=True)
        return existed

    def list_ids(self) -> list[str]:
        seen = set()
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []):
                seen.add(prefix["Prefix"].rstrip("/"))
        return sorted(seen)
