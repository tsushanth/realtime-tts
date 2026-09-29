"""
Per-voice record storage for the Orpheus streaming-clone service.

Layout under `root` (a Modal Volume mount in production, a tmp dir in tests):
  {vid}/manifest.json   - consent fields + status + timestamps, the single
                           source of truth read_status() returns
  {vid}/dataset/         - uploaded audio, staged before training consumes it
  {vid}/merged/          - the final HF-format checkpoint, only ever visible
                           once fully written (see save_checkpoint_dir)

Modal Volumes are not a real object store with atomic per-key overwrite like
S3 -- a killed container mid-write must never leave a directory that
read_status() would treat as "ready". save_checkpoint_dir therefore stages
into a sibling temp dir and os.rename()s it into place in one step, the same
atomic-swap pattern intake.py and voice_design_dev.py already use on this
same class of storage.
"""
import json
import os
import shutil
import time


class VoiceRecordStore:
    def __init__(self, root: str):
        self.root = root

    def _dir(self, vid: str) -> str:
        return os.path.join(self.root, vid)

    def _manifest_path(self, vid: str) -> str:
        return os.path.join(self._dir(vid), "manifest.json")

    def create(self, vid: str, consent: dict) -> None:
        os.makedirs(self._dir(vid), exist_ok=True)
        manifest = dict(consent)
        manifest["status"] = "awaiting_dataset"
        manifest["created_at"] = time.time()
        self._write_manifest(vid, manifest)

    def _write_manifest(self, vid: str, manifest: dict) -> None:
        # Atomic write: write to a temp file in the same directory, then
        # rename -- rename is atomic on the same filesystem, a plain write
        # is not (a reader could see a half-written JSON file).
        path = self._manifest_path(vid)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f)
        os.rename(tmp_path, path)

    def mark_dataset_uploaded(self, vid: str) -> None:
        """Mark dataset as uploaded. Preserves existing status if already set."""
        manifest = self.read_status(vid)
        if manifest is None:
            # Voice record does not exist yet; default to awaiting_dataset
            manifest = {"status": "awaiting_dataset", "created_at": time.time()}
        # Preserve current status, only add dataset_uploaded flag
        manifest["dataset_uploaded"] = True
        manifest["updated_at"] = time.time()
        self._write_manifest(vid, manifest)

    def write_status(self, vid: str, status: str, **fields) -> None:
        """Write status to a voice record. Voice record must exist (created via create())."""
        manifest = self.read_status(vid)
        if manifest is None:
            raise ValueError(f"Voice record '{vid}' does not exist. Call create() first.")
        manifest["status"] = status
        manifest.update(fields)
        manifest["updated_at"] = time.time()
        self._write_manifest(vid, manifest)

    def read_status(self, vid: str) -> dict | None:
        path = self._manifest_path(vid)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def save_checkpoint_dir(self, vid: str, local_dir: str) -> None:
        dest = os.path.join(self._dir(vid), "merged")
        staging = dest + ".staging"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        shutil.copytree(local_dir, staging)
        os.rename(staging, dest)  # atomic: merged/ never appears half-written

    def dataset_dir(self, vid: str) -> str:
        path = os.path.join(self._dir(vid), "dataset")
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_dir(self, vid: str) -> str:
        return os.path.join(self._dir(vid), "merged")

    def delete(self, vid: str) -> None:
        shutil.rmtree(self._dir(vid), ignore_errors=True)
