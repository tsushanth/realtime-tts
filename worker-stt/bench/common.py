"""Shared Modal image + volume for the STT benchmark and prototype."""
import modal

MODELS = ["tiny", "base", "small", "distil-large-v3", "large-v3-turbo"]

def _bake_models():
    from faster_whisper.utils import download_model
    for m in MODELS:
        download_model(m, cache_dir="/models")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "curl")
    .pip_install("faster-whisper==1.1.1", "jiwer", "soundfile", "scipy", "numpy<2", "fastapi[standard]", "python-multipart", "websockets", "requests")
    .run_function(_bake_models)
)
vol = modal.Volume.from_name("stt-bench-data", create_if_missing=True)
