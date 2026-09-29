"""
Deployable Modal app for the Orpheus streaming-clone service. Ties together
storage.py, prepare_dataset.py, train_job.py, api.py, and serve.py.

Deploy (module mode, from voice-pipeline/):
    cd voice-pipeline && modal deploy -m orpheus_clone_prod.main
Module mode matters: Modal only auto-includes the entrypoint's own source.
In module mode that's the whole orpheus_clone_prod package; in file mode
(`modal deploy .../main.py`) it is main.py alone, and every container would
fail on `from orpheus_clone_prod... import`. orpheus_clone_pilot (needed by
train_job.py's lazy finetune_pilot import) is added explicitly via
add_local_python_source below.

Secrets required: orpheus-clone-secret (ORPHEUS_CLONE_SECRET), hf-token
(HF_TOKEN, for the gated canopylabs/orpheus-tts-0.1-pretrained repo).
"""
import logging

import modal

from orpheus_clone_prod.serve import load_engine_for_checkpoint, resolve_voice_dir, synthesize
from orpheus_clone_prod.storage import VoiceRecordStore
from orpheus_clone_prod.train_job import run_training_job

logger = logging.getLogger(__name__)

app = modal.App("orpheus-clone-prod")

checkpoint_volume = modal.Volume.from_name("orpheus-clone-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
api_secret = modal.Secret.from_name("orpheus-clone-secret")

CHECKPOINT_ROOT = "/checkpoints"

# Training image: pinned transformers==4.46.3, matching finetune_pilot.py's
# already-verified training environment. NEVER share this image with the
# serving image below -- see Global Constraints.
# fastapi is here only because this module is imported in every container
# (api() imports create_app lazily now, but keep it installed so a future
# top-level import can't silently crash every training container again).
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.3", "datasets", "accelerate", "peft==0.13.2",
        "snac", "soundfile", "huggingface_hub", "numpy<2",
        "faster-whisper", "fastapi==0.109.0", "python-multipart",
    )
    .add_local_python_source("orpheus_clone_pilot")
)

# Serving image: pinned transformers==4.48.2 (the floor vllm==0.7.3
# declares) -- an unpinned resolve broke with a TokenizersBackend
# AttributeError, confirmed by running. NEVER share this image with the
# training image above.
serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.48.2", "huggingface_hub", "numpy<2", "soundfile",
        "snac", "vllm==0.7.3", "orpheus-speech",
    )
    .add_local_python_source("orpheus_clone_pilot")
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0", "python-multipart")


@app.function(image=train_image, gpu="A10G", timeout=7200, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
def run_training_job_modal(vid: str):
    # A warm container reused for a later spawn would otherwise see a stale
    # snapshot (missing this voice's manifest/upload.zip).
    checkpoint_volume.reload()
    try:
        # reload_store lets the job re-check, just before its final write,
        # that the voice wasn't deleted mid-run (see train_job.py).
        run_training_job(vid, root=CHECKPOINT_ROOT, reload_store=checkpoint_volume.reload)
    finally:
        # Commit on failure too, so the "failed" status (and its error) is
        # visible to the API instead of the voice sitting at "training".
        # If the voice was deleted mid-run, the job has already removed its
        # local copy, so this commit propagates the deletion.
        checkpoint_volume.commit()


@app.cls(image=serve_image, gpu="A10G", timeout=300, scaledown_window=300, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
class OrpheusCloneEngine:
    @modal.enter()
    def _init(self):
        # checkpoint_dir -> loaded OrpheusModel. Capped at ONE entry: vLLM
        # reserves ~90% of GPU memory per engine by default, so a second
        # engine can't fit alongside the first on an A10G. Consequence: only
        # one voice is "hot" per container; switching voices in a warm
        # container pays a full engine reload. Accepted tradeoff for now.
        # Engines are loaded lazily on first request for a voice (not in
        # this @modal.enter), since the voice isn't known until then.
        self._models = {}

    @modal.method()
    def synthesize_for_voice(self, voice: str, text: str):
        # See newly-ready / newly-deleted voices instead of this warm
        # container's startup snapshot. reload() can fail if files on the
        # volume are held open; don't fail synthesis over it.
        try:
            checkpoint_volume.reload()
        except Exception as e:
            logger.warning(f"checkpoint_volume.reload() failed, using cached view: {e}")
        store = VoiceRecordStore(root=CHECKPOINT_ROOT)
        checkpoint_dir = resolve_voice_dir(voice, store)  # raises UnknownVoiceError -> caller maps to 400
        if checkpoint_dir not in self._models:
            # Evict any other voice's engine first (see _init). Rely on GC to
            # release GPU memory; no manual CUDA management here.
            for other in list(self._models):
                del self._models[other]
            self._models[checkpoint_dir] = load_engine_for_checkpoint(checkpoint_dir)
        vid = voice.split(":", 1)[1]
        return list(synthesize(self._models[checkpoint_dir], text, voice_tag=vid))


@app.function(image=web_image, secrets=[api_secret], volumes={CHECKPOINT_ROOT: checkpoint_volume}, timeout=120)
@modal.asgi_app()
def api():
    from orpheus_clone_prod.api import create_app  # lazy: fastapi only needed in the web container

    def spawn_training(vid: str, root: str):
        run_training_job_modal.spawn(vid)

    return create_app(
        root=CHECKPOINT_ROOT,
        spawn_training=spawn_training,
        get_engine_cls=lambda: OrpheusCloneEngine,
        store_reload=checkpoint_volume.reload,
        store_commit=checkpoint_volume.commit,
    )
