"""
Training job for the Orpheus streaming-clone service: transcribe the
uploaded audio, SNAC-tokenize, LoRA fine-tune, merge, persist.

Reuses orpheus_clone_pilot/finetune_pilot.py's verified SNAC tokenization
and LoRA training logic rather than re-deriving it -- that module's
constants (AUDIO_TOKENS_START, CODEBOOK_SIZE, special tokens, dedup) are
load-bearing and were checked against canopylabs' own data-prep notebook.

Runs in the transformers==4.46.3-pinned training image (see Global
Constraints) -- a DIFFERENT image than serve.py's vllm==0.7.3/
transformers==4.48.2 image. Never share these images.

Delete-during-training: the API rejects DELETE while status=="training"
(409), but as defense in depth this job re-checks, right before writing its
final status, that the voice record still exists in the latest committed
Volume state. If it was deleted mid-run, the job removes its own local copy
of the voice dir (so main.py's trailing commit propagates the deletion
instead of resurrecting it) and returns without writing anything.

This job deliberately does NOT rewrite manifest.json at startup (the API
already committed status="training" before spawning). A startup rewrite
would be a local uncommitted change to manifest.json, and Modal's
Volume.reload() "may implicitly commit" such changes -- i.e. the re-check
itself could re-push the manifest and then see its own write.
"""
import logging
import os
import time

from orpheus_clone_prod.prepare_dataset import prepare_dataset
from orpheus_clone_prod.storage import VoiceRecordStore

logger = logging.getLogger(__name__)


def _load_zip_bytes(vid: str, root: str) -> bytes:
    store = VoiceRecordStore(root=root)
    zip_path = os.path.join(store.dataset_dir(vid), "upload.zip")
    with open(zip_path, "rb") as f:
        return f.read()


def _run_lora_finetune_and_merge(rows: list[dict], dataset_dir: str, voice_tag: str) -> str:
    """Thin wrapper around finetune_pilot.py's tokenize+train+merge logic,
    parameterized by voice_tag instead of the pilot's hardcoded "joanna".
    Returns the local path to the merged checkpoint directory."""
    from orpheus_clone_pilot.finetune_pilot import run_pilot_for_voice  # lazy: heavy deps

    return run_pilot_for_voice(rows=rows, dataset_dir=dataset_dir, voice_tag=voice_tag)


def _deleted_mid_run(vid: str, store: VoiceRecordStore, reload_store) -> bool:
    """Reload the latest committed Volume state and report whether the voice
    should be discarded rather than given its final status write -- either
    because the record is gone outright, or because a delete was requested
    while training was in flight (delete_requested, set by the API's soft
    delete-during-training path). Either way this is the safe point to
    actually complete that deletion: training has reached its own terminal
    point (success or failure), so there's no longer a race with an
    in-flight training write. A failed reload (e.g. open files) falls back
    to the local view."""
    if reload_store is not None:
        try:
            reload_store()
        except Exception as e:
            logger.warning(f"volume reload before final write failed for {vid}, using local view: {e}")
    record = store.read_status(vid)
    if record is None:
        logger.warning(f"voice {vid} was deleted while training; discarding training output")
        store.delete(vid)
        return True
    if record.get("delete_requested"):
        logger.warning(f"voice {vid} had a delete requested while training; completing deletion now")
        store.delete(vid)
        return True
    return False


def run_training_job(vid: str, root: str, reload_store=None) -> None:
    """reload_store: callable that makes the latest committed Volume state
    visible in this container (main.py passes checkpoint_volume.reload).
    None in unit tests / non-Volume roots."""
    store = VoiceRecordStore(root=root)
    try:
        dataset_dir = store.dataset_dir(vid)
        zip_bytes = _load_zip_bytes(vid, root)
        rows = prepare_dataset(zip_bytes, dataset_dir, voice_tag=vid, min_clips=20)
        merged_dir = _run_lora_finetune_and_merge(rows, dataset_dir, voice_tag=vid)
        if _deleted_mid_run(vid, store, reload_store):
            return
        store.save_checkpoint_dir(vid, merged_dir)
        store.write_status(vid, "ready", clip_count=len(rows), trained_at=time.time())
    except Exception as e:
        try:
            if _deleted_mid_run(vid, store, reload_store):
                # Not a training failure worth surfacing: the voice is gone.
                logger.warning(f"training for deleted voice {vid} also failed (ignored): {e}")
                return
            store.write_status(vid, "failed", error=str(e))
        except Exception:
            pass  # don't let a failure to record status mask the real error
        raise
