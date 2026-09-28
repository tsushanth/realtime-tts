#!/usr/bin/env python3
"""Pre-download XTTS v2 weights during image build so cold starts are fast."""
import os

# Coqui TTS expects TOS agreement file before downloading the model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
model_dir = os.path.expanduser(f"~/.local/share/tts/{model_name.replace('/', '--')}")
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "tos_agreed.txt"), "w") as f:
    f.write("y\n")

from TTS.api import TTS
print("Downloading XTTS v2 weights (~2 GB)...", flush=True)
tts = TTS(model_name)
print("Download complete.", flush=True)
