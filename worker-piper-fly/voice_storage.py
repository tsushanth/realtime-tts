"""Tigris-backed voice storage with a local LRU disk cache. Tigris (Fly's S3-compatible
object storage) is the source of truth for every published voice - built-in catalog
voices and customer/cloned voices alike, no distinction at this layer (see
2026-09-22-android-realtime-tts-production-readiness-design.md, "Piece 1"). Each machine
keeps a bounded local cache of recently-used voices' raw files so a warm machine doesn't
re-fetch from Tigris on every request; a machine that's never seen a voice fetches it on
first use."""
import os
import shutil
from collections import OrderedDict
from threading import Lock

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
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, vid: str) -> str | None:
        with self._lock:
            cached = self._lru.get(vid)
            if cached is not None and os.path.isdir(cached):
                self._lru.move_to_end(vid)
                return cached

        local_dir = os.path.join(self.cache_dir, vid)
        tmp_dir = local_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        fetched_any = False
        for fname in VOICE_FILES:
            key = f"{vid}/{fname}"
            dest = os.path.join(tmp_dir, fname)
            try:
                self._client.download_file(self.bucket, key, dest)
                fetched_any = True
            except self._client.exceptions.ClientError as e:
                if fname == "model.onnx":
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return None  # the model itself is required; missing = voice doesn't exist
                # model.onnx.json/owner.json are expected to always exist alongside model.onnx
                # for any voice this class wrote via put(); a missing one here is a real error,
                # not a "voice doesn't exist" case, so re-raise.
                raise RuntimeError(f"voice {vid!r} missing required file {fname!r} in storage") from e
        if not fetched_any:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        shutil.rmtree(local_dir, ignore_errors=True)
        os.rename(tmp_dir, local_dir)

        with self._lock:
            self._lru[vid] = local_dir
            self._lru.move_to_end(vid)
            while len(self._lru) > self.max_cache_entries:
                _, evicted_dir = self._lru.popitem(last=False)
                shutil.rmtree(evicted_dir, ignore_errors=True)
        return local_dir

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
