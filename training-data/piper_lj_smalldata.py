"""Small-data voice-cloning feasibility test for the customer-voice pipeline.

Base: Piper's public-domain LJ Speech medium checkpoint (licence-clean, see synthesize_ljspeech.py).
Data: ONE speaker from LibriTTS-R train-clean-100 (CC BY 4.0) - the speaker with the most audio
(LibriTTS caps speakers at ~25 min, so 2h single-speaker sets aren't available here; 10 and ~25 min
are also the realistic range for customer uploads). Two variants: small (~10 min) and full
(all that speaker's clips), each fine-tuned from the LJ base via --model.warmstart_ckpt, then
exported to ONNX and used to synthesize the same 5 test sentences. Everything lands in the
tts-checkpoints volume under lj_small/ (samples in lj_small/<variant>/samples/).
Hyperparameters mirror piper_full_finetune.py (batch 8, T4).
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
    .pip_install("torch==2.1.2", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2")
    .run_commands("git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src")
    .apt_install("python3-dev")
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    .pip_install("numpy<2")
    # dataset.py's prepare_data() logs "Processing utterances..." exactly
    # once at the start and "Processed N utterance(s)" exactly once at the
    # end - nothing in between, no matter how many files. At full-corpus
    # scale that step is silent for a long time by design, not because it's
    # stuck - patch in progress logging every 500 utterances (see
    # patch_piper_progress.py) so this is actually observable instead of
    # indistinguishable from a real hang.
    .add_local_file("patch_piper_progress.py", remote_path="/tmp/patch_piper_progress.py", copy=True)
    .run_commands("python3 /tmp/patch_piper_progress.py")
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/lj_medium.ckpt "
        "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
        "en/en_US/ljspeech/medium/lj-med_1000.ckpt'"
    )
    .pip_install("scipy", "soundfile")
)

app = modal.App("tts-piper-lj-smalldata", image=image)
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
VARIANTS = {"small": {"minutes": 10, "steps": 3000}, "full": {"minutes": 1000, "steps": 4000}}


@app.function(gpu="T4", timeout=6 * 3600, volumes={"/checkpoints": checkpoint_volume})
def run_all(only: str = "", gender: str = "", steps: int = 0, name: str = "", base: str = "", rank: int = 0, minutes: int = 1000):
    import csv, glob, os, runpy, shutil, subprocess, sys, time
    from collections import defaultdict

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    base_ckpt = "/ckpt/lj_medium.ckpt"
    if base:  # e.g. en/en_US/john/medium/john-2599.ckpt from rhasspy/piper-checkpoints
        base_ckpt = "/tmp/base.ckpt"
        subprocess.run(f"wget -q -O {base_ckpt} https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{base}", shell=True, check=True)
    root = "/tmp/libri"
    os.makedirs(root, exist_ok=True)
    t0 = time.time()
    print("=== downloading + extracting LibriTTS-R train-clean-100 ===", flush=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    dur = defaultdict(float)
    utts = defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        info = sf.info(wav)
        d = info.frames / info.samplerate
        txt = wav[:-4] + ".normalized.txt"
        if not os.path.exists(txt) or not (1.0 <= d <= 12.0):
            continue
        dur[spk] += d
        utts[spk].append((wav, txt, d))
    if gender:
        # LibriTTS-R ships speakers.tsv: READER<TAB>GENDER<TAB>SUBSET<TAB>NAME
        spk_gender = {}
        for tsv in glob.glob(f"{root}/**/speakers.tsv", recursive=True):
            for line in open(tsv):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    spk_gender[parts[0]] = parts[1]
        dur = {k: v for k, v in dur.items() if spk_gender.get(k) == gender}
        assert dur, f"no speakers with gender {gender}"
    spk = sorted(dur, key=dur.get, reverse=True)[rank]  # rank 0 = speaker with the most audio
    print(f"chosen speaker {spk}: {dur[spk]/60:.1f} min in {len(utts[spk])} clips (of {len(dur)} speakers)", flush=True)
    clips = sorted(utts[spk])

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    import piper.espeakbridge  # noqa: F401
    import piper.train.__main__ as piper_main
    from piper import PiperVoice

    variants = {name: {"minutes": minutes, "steps": steps}} if name else VARIANTS
    for name, cfg in variants.items():
        if only and name != only:
            continue
        out = f"/checkpoints/lj_small/{name}"
        wavs = f"/tmp/{name}_wavs"
        os.makedirs(wavs, exist_ok=True)
        os.makedirs(out, exist_ok=True)
        rows, total = [], 0.0
        for i, (wav, txt, d) in enumerate(clips):
            if total >= cfg["minutes"] * 60:
                break
            audio, sr = sf.read(wav)
            audio = resample_poly(audio, 147, 160) if sr == 24000 else audio
            fn = f"{i:05d}.wav"
            sf.write(f"{wavs}/{fn}", audio, 22050, "PCM_16")
            rows.append((fn, open(txt).read().strip()))
            total += d
        with open(f"/tmp/{name}.csv", "w", newline="") as f:
            csv.writer(f, delimiter="|").writerows(rows)
        print(f"=== variant {name}: {len(rows)} clips, {total/60:.1f} min, {cfg['steps']} steps ===", flush=True)

        os.chdir("/opt/piper-src")
        sys.argv = [
            "piper.train", "fit", "--data.voice_name", f"lj_{name}",
            "--data.csv_path", f"/tmp/{name}.csv", "--data.audio_dir", wavs,
            "--data.espeak_voice", "en-us", "--data.cache_dir", f"/tmp/cache_{name}",
            "--data.config_path", f"{out}/config.json", "--data.batch_size", "8",
            "--model.sample_rate", "22050", "--model.warmstart_ckpt", base_ckpt,
            "--trainer.max_steps", str(cfg["steps"]), "--trainer.default_root_dir", out,
        ]
        piper_main._DEFAULT_CALLBACKS = piper_main._DEFAULT_CALLBACKS[:1]
        try:
            piper_main.main()
        except SystemExit as e:
            print(f"piper.train exit {e.code}", flush=True)
        checkpoint_volume.commit()

        ckpt = sorted(glob.glob(f"{out}/lightning_logs/version_*/checkpoints/*.ckpt"), key=os.path.getmtime)[-1]
        onnx = f"{out}/model.onnx"
        sys.argv = ["piper.train.export_onnx", "--checkpoint", ckpt, "--output-file", onnx]
        runpy.run_module("piper.train.export_onnx", run_name="__main__")
        shutil.copy(f"{out}/config.json", onnx + ".json")
        voice = PiperVoice.load(onnx)
        os.makedirs(f"{out}/samples", exist_ok=True)
        for i, text in enumerate(TEST_SENTENCES):
            pcm = np.concatenate([c.audio_int16_array for c in voice.synthesize(text)])
            sf.write(f"{out}/samples/sample_{i}.wav", pcm, voice.config.sample_rate, "PCM_16")
        checkpoint_volume.commit()
        print(f"=== variant {name} done, samples in {out}/samples ===", flush=True)


@app.local_entrypoint()
def main(only: str = "", gender: str = "", steps: int = 0, name: str = "", base: str = "", rank: int = 0, minutes: int = 1000):
    print("Spawned:", run_all.spawn(only, gender, steps, name, base, rank, minutes).object_id)
