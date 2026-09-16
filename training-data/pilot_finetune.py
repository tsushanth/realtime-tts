"""Phase 0 pilot, corrected: FINE-TUNE Matcha-TTS's official pretrained
LJSpeech checkpoint on our tiny Polly-Joanna slice, rather than training
from scratch. 279 samples/500 steps from random init would never produce
coherent speech regardless of architecture — that's not a fair test.
Starting from pretrained weights (already knows general speech structure)
and adapting to our voice/domain is both the correct pilot design AND,
if it works, a much cheaper path for the real production run too — real
measured cost from this pilot: ~$0.01 for 300 steps on a T4; even 50,000
steps extrapolates to ~$1.77. NOT the same cost class as the from-scratch
pretraining estimate ($62-195, or earlier $500-1,500) quoted elsewhere in
this repo's history — that number belonged to a different plan (training
a FastPitch-style acoustic model + vocoder from random init), and doesn't
apply once you're fine-tuning an already-converged checkpoint instead.

REFERENCE FOR NEXT TIME — repo-wide package version conflicts hit while
getting this working (8 rounds of debugging), so nobody has to rediscover
these:
  - matcha-tts's PyPI package (the "matcha-tts" wheel) ships a broken
    `configs/` — just an empty package marker, not the real Hydra YAML
    tree. MUST clone the GitHub repo and `pip install -e .` from source
    instead; pip-installing the package name alone silently gives you no
    usable configs.
  - matcha-tts declares many more runtime deps than its own setup.py
    installs automatically (rich, pandas, Unidecode, inflect, seaborn,
    gdown, wget, ipywidgets, notebook, pytest, pre-commit, torchvision,
    hydra-optuna-sweeper, gradio==3.43.2) — pip only warns about these as
    "not installed" rather than failing at install time, so the failures
    surface one at a time, at import, deep in the training run. Install
    them all up front (see below) rather than chasing ImportErrors one
    per debug cycle.
  - unpinned `torchvision` drags `torch` up to whatever's newest (2.14 as
    of writing), which breaks the pinned `torchaudio==2.1.2`. Pin
    torchvision to the version matching your torch (0.16.2 <-> 2.1.2).
  - `setuptools>=81` dropped `pkg_resources` entirely, which
    `lightning`'s fabric module still imports at load time. Pin
    `setuptools<81`.
  - `diffusers==0.25.0` (an old version Matcha's decoder needs) imports
    `cached_download` from `huggingface_hub`, removed in recent
    huggingface_hub releases. Pin `huggingface_hub==0.20.3`.
  - `matplotlib>=3.8` removed `FigureCanvasAgg.tostring_rgb()`, which
    Matcha's own validation-image plotting code (`matcha/utils/utils.py`)
    still calls. Pin `matplotlib==3.7.5`.
  - Lightning's `TensorBoardLogger` (not `CSVLogger`) is required — the
    model's `on_validation_end` hook calls `self.logger.experiment.
    add_image(...)`, a TensorBoard-specific API that other loggers don't
    implement, so validation crashes with any other logger.
  - Hydra's `compose()` API (used here instead of the full `@hydra.main`
    decorator, since we need to load pretrained weights into the model
    *before* calling `trainer.fit`, not just compose-and-run) does NOT
    populate `cfg.hydra` the way the real runtime does. Any config value
    using a `${hydra:...}` interpolation (the default trainer/callbacks
    configs both do, for their output-directory paths) throws
    "HydraConfig was not set" at instantiation time. Fix: don't
    `hydra.utils.instantiate(cfg.trainer)` or `cfg.callbacks` at all —
    construct `lightning.Trainer(...)` directly with plain kwargs, and
    pass `callbacks=none` in the compose overrides.
  - Modal function return values containing large payloads (here, base64
    audio) can silently fail to transfer if the `modal run` CLI
    disconnects first ("local client disconnected"). Use `@app.
    local_entrypoint()` to call `.remote()` and handle/save the result
    locally, not a bare `modal run module.py::function_name` invocation.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1", "git", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        # The PyPI "matcha-tts" package's configs/ dir is broken (empty
        # stub) — must install from source to get the real Hydra configs.
        "git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git /opt/matcha-src"
    )
    .workdir("/opt/matcha-src")
    .run_commands("pip install -e . --no-deps")
    .pip_install(
        # Matcha's own declared deps that its setup.py doesn't actually
        # install (see docstring above) — installing individually rather
        # than trusting `pip install -e .` to pull them all in.
        "lightning==2.1.4", "pytorch-lightning==2.1.4", "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0", "rootutils", "phonemizer==3.2.1", "einops",
        "conformer==0.3.2", "diffusers==0.25.0", "Cython", "numpy<2.0.0",
        "librosa", "matplotlib==3.7.5", "tensorboard", "rich", "pandas", "Unidecode",
        "inflect", "seaborn", "gdown", "wget", "ipywidgets", "notebook",
        "pytest", "pre-commit", "torchvision==0.16.2", "hydra-optuna-sweeper==1.2.0",
        # Pinned to avoid known breakages — see docstring above for why each one:
        "gradio==3.43.2", "setuptools<81", "huggingface_hub==0.20.3",
    )
    .run_commands(
        "cd /opt/matcha-src/matcha/utils/monotonic_align && python3 setup.py build_ext --inplace"
    )
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/matcha_ljspeech.ckpt "
        "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt"
    )
    .add_local_dir("pilot", remote_path="/pilot")
)

app = modal.App("tts-pilot-matcha-ft", image=image)

DATA_YAML = """\
_target_: matcha.data.text_mel_datamodule.TextMelDataModule
name: pilot
train_filelist_path: /pilot/train.txt
valid_filelist_path: /pilot/val.txt
batch_size: 8
num_workers: 4
pin_memory: True
cleaners: [english_cleaners2]
add_blank: True
n_spks: 1
n_fft: 1024
n_feats: 80
sample_rate: 22050
hop_length: 256
win_length: 1024
f_min: 0
f_max: 8000
data_statistics:
  mel_mean: -5.536622
  mel_std: 2.116101
seed: 1234
load_durations: false
"""


@app.function(gpu="T4", timeout=3600)
def run_pilot():
    import os
    import shutil
    import sys

    os.chdir("/opt/matcha-src")
    os.makedirs("configs/data", exist_ok=True)
    with open("configs/data/pilot.yaml", "w") as f:
        f.write(DATA_YAML)

    sys.path.insert(0, "/opt/matcha-src")

    import hydra
    from hydra import compose, initialize_config_dir
    import torch
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    with initialize_config_dir(config_dir="/opt/matcha-src/configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=pilot",
                "run_name=pilot_ft",
                "model.out_size=172",
                "+trainer.max_steps=300",
                "trainer.check_val_every_n_epoch=1000",  # skip mid-run val, our val set is tiny anyway
                "logger=csv",
                "callbacks=none",
            ],
        )

    L.seed_everything(cfg.seed, workers=True)

    print("=== Instantiating datamodule ===")
    datamodule = hydra.utils.instantiate(cfg.data)

    print("=== Instantiating model (fresh, then loading pretrained LJSpeech weights) ===")
    model = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load("/ckpt/matcha_ljspeech.ckpt", map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("First few missing keys:", missing[:10])
    if unexpected:
        print("First few unexpected keys:", unexpected[:10])

    print("=== Instantiating trainer directly (bypassing Hydra — compose() doesn't populate cfg.hydra, which trainer/default.yaml's ${hydra:...} interpolations need) ===")
    logger = TensorBoardLogger(save_dir="/tmp/pilot_logs", name="pilot_ft")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=300,
        precision="16-mixed",
        logger=logger,
        callbacks=[],
        enable_checkpointing=False,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
    )

    print("=== Fine-tuning (300 steps on 279 samples) ===")
    trainer.fit(model=model, datamodule=datamodule)  # no ckpt_path -> fresh optimizer state

    print("=== Synthesizing test sentences with the fine-tuned model, in-memory (no reload) ===")
    import base64
    import wget as wget_module
    import numpy as np
    import soundfile as sf
    from matcha.hifigan.config import v1
    from matcha.hifigan.denoiser import Denoiser
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.text import sequence_to_text, text_to_sequence
    from matcha.utils.utils import intersperse

    if not os.path.exists("/ckpt/hifigan_T2_v1.ckpt"):
        wget_module.download(
            "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
            "/ckpt/hifigan_T2_v1.ckpt",
        )

    def load_hifigan(checkpoint_path, device):
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    device = torch.device("cuda")
    model = model.to(device).eval()
    vocoder = load_hifigan("/ckpt/hifigan_T2_v1.ckpt", device)
    denoiser = Denoiser(vocoder, mode="zeros")

    test_sentences = [
        "Sure, I can help with that. Let me pull up your account details.",
        "Is there anything else I can help you with today?",
        "Thanks for calling, have a great day.",
    ]

    results = []
    with torch.inference_mode():
        for i, text in enumerate(test_sentences):
            x = torch.tensor(
                intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
                dtype=torch.long, device=device,
            )[None]
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
            output = model.synthesise(x, x_lengths, n_timesteps=10, temperature=0.667, spks=None, length_scale=0.95)
            audio = vocoder(output["mel"]).clamp(-1, 1)
            audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze().numpy()
            wav_bytes_path = f"/tmp/pilot_sample_{i}.wav"
            sf.write(wav_bytes_path, audio, 22050, "PCM_16")
            with open(wav_bytes_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            results.append({"text": text, "audio_b64": b64})
            print(f"Synthesized [{i}]: {text}")

    print("=== Saving final checkpoint ===")
    os.makedirs("/output", exist_ok=True)
    trainer.save_checkpoint("/output/pilot_finetuned.ckpt")
    print("Done. Checkpoint at /output/pilot_finetuned.ckpt")

    return results


@app.local_entrypoint()
def main():
    import base64
    import os

    results = run_pilot.remote()
    out_dir = os.path.join(os.path.dirname(__file__), "pilot_output")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["audio_b64"]))
        print(f"Saved: {path}  —  \"{r['text']}\"")
