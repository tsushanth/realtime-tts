"""Full-corpus Piper fine-tune: scales piper_pilot_finetune.py's working
approach (see that file's docstring for every environment gotcha already
solved there) from the 279-sample pilot to the full 22,011-sample
Polly-Joanna corpus, matching what full_finetune.py did for Matcha-TTS.

Data comes from the same `tts-corpus-full` Modal Volume already used for
Matcha-TTS's full run - the WAVs there are already flat-named
(000000.wav, 000001.wav, ...) matching Piper's required audio_dir layout
with no reshuffling needed, just a filename|text CSV built from the
already-committed full/train.txt and full/val.txt (Matcha-style
path|text, reformatted the same way the pilot's `convert()` helper does).

Checkpoints persist to the tts-checkpoints Volume (same one Matcha-TTS's
full run uses, different subdirectory) so they outlive the container -
the pilot never needed this since it was a 5-minute smoke test, but a
~4.6 hour run needs real persistence and the ability to resume/inspect
without holding a live connection the whole time.

Cost/time basis: pilot measured ~1.14-1.39 it/s (avg ~1.2) at batch_size=8
on a T4. Keeping batch_size identical to the pilot so this rate is a valid
basis for the estimate rather than an untested guess: 20,000 steps / 1.2
it/s ~= 4.6 hours, ~$2.50-3 at T4 pricing - same order of magnitude as
Matcha-TTS's full run, just slower per-step so longer wall-clock for the
same step count.

**Real gotcha found running this**: reading `piper/train/vits/dataset.py`'s
`prepare_data()` shows it's a fully single-threaded loop over every CSV row
- espeak phonemization, a real Silero VAD neural-net forward pass, and
spectrogram computation, one utterance at a time - with **no progress
logging at all** between a single "Processing utterances..." line at the
very start and "Processed N utterance(s)" at the very end. That's already
a real gap (no way to tell "working" from "stuck" during this phase), but
it doesn't fully explain what actually happened: one attempt (using a
blocking `.remote()` call, since retried with `.spawn()` for an unrelated
reason - see below) got through this exact step AND 1435 training steps in
18 minutes; a later attempt with identical code stalled on the same step
for 8+ hours before being killed. Same code, wildly different outcomes -
more consistent with intermittent Modal Volume I/O flakiness (the WAVs are
read one at a time from a Volume mount) than a deterministic compute
bottleneck. Two real fixes applied regardless of exact root cause: (1) copy
the referenced WAVs from the Volume to local container disk before calling
`piper.train`, removing repeated network-volume I/O as a variable entirely;
(2) patch the installed `dataset.py` to log progress every 500 utterances
instead of never, so "stuck vs. slow" is observable instead of guessed at.
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
        "mkdir -p /ckpt && wget -q -O /ckpt/lessac_medium.ckpt "
        "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
        "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt'"
    )
    .add_local_dir("full", remote_path="/filelists_src")
)

app = modal.App("tts-piper-full-ft", image=image)

corpus_volume = modal.Volume.from_name("tts-corpus-full")
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")

MAX_STEPS = 20000  # same order of magnitude as Matcha-TTS's full run; see docstring for basis


@app.function(
    gpu="T4",
    timeout=8 * 3600,
    volumes={"/data": corpus_volume, "/checkpoints": checkpoint_volume},
)
def run_full_piper_finetune():
    import os
    import sys

    import piper.espeakbridge  # noqa: F401  sanity check, see piper_pilot_finetune.py

    print("=== Building Piper CSV filelists from the full-corpus filelists ===")
    def convert(src_filelist, dst_csv):
        import csv
        rows = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)  # /data/wavs_full/000000.wav -> 000000.wav
                rows.append((fname, text))
        with open(dst_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return len(rows)

    n_train = convert("/filelists_src/train.txt", "/tmp/train.csv")
    n_val = convert("/filelists_src/val.txt", "/tmp/val.csv")
    print(f"{n_train} train / {n_val} val rows")

    # Copy the referenced WAVs from the Volume to local container disk before
    # piper's own (single-threaded, per-file) prepare_data() reads them -
    # removes repeated network-volume I/O latency as a contributing factor
    # to the earlier 8+ hour stall. See module docstring for the incident.
    print("=== Copying referenced WAVs from the volume to local disk ===")
    import shutil
    import time
    os.makedirs("/tmp/wavs_local", exist_ok=True)
    needed = set()
    for csv_path in ("/tmp/train.csv", "/tmp/val.csv"):
        with open(csv_path) as f:
            for line in f:
                fname = line.split("|", 1)[0]
                needed.add(fname)
    t0 = time.time()
    for i, fname in enumerate(sorted(needed)):
        shutil.copy(f"/data/wavs_full/{fname}", f"/tmp/wavs_local/{fname}")
        if (i + 1) % 2000 == 0:
            print(f"Copied {i + 1}/{len(needed)} files...")
    print(f"Copied {len(needed)} files in {time.time() - t0:.1f}s")

    import torch
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.chdir("/opt/piper-src")
    sys.argv = [
        "piper.train", "fit",
        "--data.voice_name", "full_ft",
        "--data.csv_path", "/tmp/train.csv",
        "--data.audio_dir", "/tmp/wavs_local",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/tmp/piper_cache",
        "--data.config_path", "/checkpoints/piper_full_ft/config.json",
        "--data.batch_size", "8",  # matches the pilot - keeps the measured it/s rate valid
        "--model.sample_rate", "22050",
        "--model.warmstart_ckpt", "/ckpt/lessac_medium.ckpt",
        "--trainer.max_steps", str(MAX_STEPS),
        "--trainer.default_root_dir", "/checkpoints/piper_full_ft",
    ]
    print("Running:", " ".join(sys.argv))

    import piper.train.__main__ as piper_main
    piper_main._DEFAULT_CALLBACKS = [piper_main._DEFAULT_CALLBACKS[0]]  # drop val_mos, see pilot script
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exited with code {e.code}")

    checkpoint_volume.commit()
    print("=== Contents of /checkpoints/piper_full_ft ===")
    for root, dirs, files in os.walk("/checkpoints/piper_full_ft"):
        for f in files:
            print(os.path.join(root, f))


@app.local_entrypoint()
def main():
    # .spawn() instead of .remote(): two prior attempts both got cancelled
    # mid-training ("Function call was cancelled by user or a failure") at
    # different points (once almost immediately, once ~18 minutes/1435 steps
    # in) despite --detach. That pattern is consistent with something in
    # this environment enforcing a duration cap on the local `modal run`
    # client process itself, which cancels the still-blocking .remote() call
    # when the local process dies - unrelated to --detach, which only
    # protects against a clean disconnect, not the local process being
    # killed outright. .spawn() returns immediately with a call handle
    # instead of blocking for hours, so the local process doesn't need to
    # survive the whole training run at all.
    call = run_full_piper_finetune.spawn()
    print(f"Spawned function call: {call.object_id}")
    print("Check progress via `modal app list` / `modal container list`, or "
          "retrieve the result later with FunctionCall.from_id(...).get()")
