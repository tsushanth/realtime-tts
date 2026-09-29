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

from orpheus_clone_prod.serve import UnknownVoiceError, load_engine_for_checkpoint, resolve_voice_dir, synthesize
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
    """One voice per container. `vid` is a Modal class parameter, so each
    distinct voice gets its own container pool: Modal routes
    OrpheusCloneEngine(vid=X) calls only to containers started for X, and
    starts a new container for a voice with no warm one.

    Why: vLLM reserves ~90% of GPU memory per engine, and deleting a vLLM
    0.7.3 engine does not reliably free it (background threads hold refs),
    so swapping voices inside one warm container risked OOM. The cost is a
    cold start whenever a voice has no warm container.

    A voice's checkpoint never changes once ready (the API only commits
    from awaiting_dataset/failed), so caching the engine for the life of
    the container is safe. Deleted voices are rejected in the web container
    before .remote(), so there is no per-request Volume reload here.

    Callers: resolve/validate the voice in the web container first (api.py
    does, via serve.resolve_voice_dir), then OrpheusCloneEngine(vid=vid).
    """

    vid: str = modal.parameter()

    @modal.enter()
    def _init(self):
        self._model = None
        self._init_error = None
        store = VoiceRecordStore(root=CHECKPOINT_ROOT)
        try:
            # Same single authority as the web container; guards a race or a
            # direct caller passing an unready/malformed vid.
            checkpoint_dir = resolve_voice_dir(f"custom-fast:{self.vid}", store)
        except UnknownVoiceError as e:
            # Don't crash the container (Modal would retry startup); fail
            # each call with UnknownVoiceError -> 400 at the API instead.
            logger.warning(f"engine container for {self.vid!r} has no usable checkpoint: {e}")
            self._init_error = e
            return
        self._model = load_engine_for_checkpoint(checkpoint_dir)

    @modal.method()
    def synthesize_text(self, text: str):
        if self._model is None:
            raise UnknownVoiceError(str(self._init_error or f"voice not ready: {self.vid!r}"))
        return list(synthesize(self._model, text, voice_tag=self.vid))


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
