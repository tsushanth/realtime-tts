"""
Deployable Modal app for the Orpheus streaming-clone service. Ties together
storage.py, prepare_dataset.py, train_job.py, api.py, and serve.py.

Deploy: modal deploy voice-pipeline/orpheus_clone_prod/main.py
Secrets required: orpheus-clone-secret (ORPHEUS_CLONE_SECRET), hf-token
(HF_TOKEN, for the gated canopylabs/orpheus-tts-0.1-pretrained repo).
"""
import modal

from orpheus_clone_prod.api import create_app
from orpheus_clone_prod.serve import load_engine_for_checkpoint, resolve_voice_dir, synthesize
from orpheus_clone_prod.storage import VoiceRecordStore
from orpheus_clone_prod.train_job import run_training_job

app = modal.App("orpheus-clone-prod")

checkpoint_volume = modal.Volume.from_name("orpheus-clone-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
api_secret = modal.Secret.from_name("orpheus-clone-secret")

CHECKPOINT_ROOT = "/checkpoints"

# Training image: pinned transformers==4.46.3, matching finetune_pilot.py's
# already-verified training environment. NEVER share this image with the
# serving image below -- see Global Constraints.
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
        "faster-whisper",
    )
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
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0", "python-multipart")


@app.function(image=train_image, gpu="A10G", timeout=3600, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
def run_training_job_modal(vid: str):
    run_training_job(vid, root=CHECKPOINT_ROOT)
    checkpoint_volume.commit()


@app.cls(image=serve_image, gpu="A10G", timeout=300, scaledown_window=300, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
class OrpheusCloneEngine:
    @modal.enter()
    def _init(self):
        self._models = {}  # checkpoint_dir -> loaded OrpheusModel, one per warm container

    @modal.method()
    def synthesize_for_voice(self, voice: str, text: str):
        store = VoiceRecordStore(root=CHECKPOINT_ROOT)
        checkpoint_dir = resolve_voice_dir(voice, store)  # raises UnknownVoiceError -> caller maps to 400
        if checkpoint_dir not in self._models:
            self._models[checkpoint_dir] = load_engine_for_checkpoint(checkpoint_dir)
        vid = voice.split(":", 1)[1]
        return list(synthesize(self._models[checkpoint_dir], text, voice_tag=vid))


@app.function(image=web_image, secrets=[api_secret], volumes={CHECKPOINT_ROOT: checkpoint_volume}, timeout=120)
@modal.asgi_app()
def api():
    def spawn_training(vid: str, root: str):
        run_training_job_modal.spawn(vid)

    return create_app(root=CHECKPOINT_ROOT, spawn_training=spawn_training)
