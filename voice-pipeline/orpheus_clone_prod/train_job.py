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
"""
import os
import time

from orpheus_clone_prod.prepare_dataset import prepare_dataset
from orpheus_clone_prod.storage import VoiceRecordStore


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


def run_training_job(vid: str, root: str) -> None:
    store = VoiceRecordStore(root=root)
    try:
        store.write_status(vid, "training")
        dataset_dir = store.dataset_dir(vid)
        zip_bytes = _load_zip_bytes(vid, root)
        rows = prepare_dataset(zip_bytes, dataset_dir, voice_tag=vid, min_clips=20)
        merged_dir = _run_lora_finetune_and_merge(rows, dataset_dir, voice_tag=vid)
        store.save_checkpoint_dir(vid, merged_dir)
        store.write_status(vid, "ready", clip_count=len(rows), trained_at=time.time())
    except Exception as e:
        try:
            store.write_status(vid, "failed", error=str(e))
        except Exception:
            pass  # don't let a failure to record status mask the real error
        raise
