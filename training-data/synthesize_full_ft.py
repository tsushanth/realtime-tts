"""Synthesize test sentences with the full-corpus fine-tuned Matcha-TTS
checkpoint (see full_finetune.py) so the result can actually be listened to,
same as pilot_finetune.py did for the 279-sample pilot.

Reuses the same image/environment as full_finetune.py (same gotchas already
solved there - see that file's docstring), just adds the HiFi-GAN vocoder
download and loads /checkpoints/full_ft/last.ckpt instead of training.
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
        "git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git /opt/matcha-src"
    )
    .workdir("/opt/matcha-src")
    .run_commands("pip install -e . --no-deps")
    .pip_install(
        "lightning==2.1.4", "pytorch-lightning==2.1.4", "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0", "rootutils", "phonemizer==3.2.1", "einops",
        "conformer==0.3.2", "diffusers==0.25.0", "Cython", "numpy<2.0.0",
        "librosa", "matplotlib==3.7.5", "tensorboard", "rich", "pandas", "Unidecode",
        "inflect", "seaborn", "gdown", "wget", "ipywidgets", "notebook",
        "pytest", "pre-commit", "torchvision==0.16.2", "hydra-optuna-sweeper==1.2.0",
        "gradio==3.43.2", "setuptools<81", "huggingface_hub==0.20.3",
    )
    .run_commands(
        "cd /opt/matcha-src/matcha/utils/monotonic_align && python3 setup.py build_ext --inplace"
    )
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/hifigan_T2_v1.ckpt "
        "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
    )
)

app = modal.App("tts-full-matcha-synth", image=image)
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]


@app.function(gpu="T4", timeout=600, volumes={"/checkpoints": checkpoint_volume})
def synthesize():
    import sys
    sys.path.insert(0, "/opt/matcha-src")

    import base64
    import torch
    import soundfile as sf
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.hifigan.config import v1
    from matcha.hifigan.denoiser import Denoiser
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.text import text_to_sequence
    from matcha.utils.utils import intersperse

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    def load_hifigan(checkpoint_path, device):
        h = AttrDict(v1)
        hifigan = HiFiGAN(h).to(device)
        hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan

    device = torch.device("cuda")

    print("Loading fine-tuned checkpoint from /checkpoints/full_ft/last.ckpt")
    # Same PyTorch 2.6+ weights_only issue hit on the Jetson - this checkpoint
    # is our own, produced by full_finetune.py in this same repo, so trusted.
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    model = MatchaTTS.load_from_checkpoint(
        "/checkpoints/full_ft/last.ckpt", map_location=device
    )
    model = model.to(device).eval()

    vocoder = load_hifigan("/ckpt/hifigan_T2_v1.ckpt", device)
    denoiser = Denoiser(vocoder, mode="zeros")

    results = []
    with torch.inference_mode():
        for i, text in enumerate(TEST_SENTENCES):
            x = torch.tensor(
                intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
                dtype=torch.long, device=device,
            )[None]
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
            output = model.synthesise(
                x, x_lengths, n_timesteps=10, temperature=0.667, spks=None, length_scale=0.95
            )
            audio = vocoder(output["mel"]).clamp(-1, 1)
            audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze().numpy()
            wav_path = f"/tmp/full_ft_sample_{i}.wav"
            sf.write(wav_path, audio, 22050, "PCM_16")
            with open(wav_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            results.append({"text": text, "audio_b64": b64})
            print(f"Synthesized [{i}]: {text}")

    return results


@app.local_entrypoint()
def main():
    import base64
    import os

    results = synthesize.remote()
    out_dir = os.path.join(os.path.dirname(__file__), "full_ft_output")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["audio_b64"]))
        print(f"Saved: {path}  -  \"{r['text']}\"")
