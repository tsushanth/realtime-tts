"""Modal training dispatch for TTSCoreRecipe. This is
training-data/piper_full_finetune.py's exact image/volume/CLI-invocation
approach (see that file's docstring for the hard-won gotchas already
solved there: WAV-copy-before-training, progress-patching,
.spawn()-not-.remote() for long runs), parameterized by candidate config
instead of hardcoded to one checkpoint/step-count. Read that file in full
before changing this one - the gotchas documented there apply here
unchanged."""
import os

import modal

# Modal resolves add_local_file/add_local_dir paths against the CLIENT PROCESS's
# cwd, not this module's directory. The documented invocation is
# `cd <repo root> && python3 -m harness.run_cycle ...`, but relying on that is
# fragile, so anchor to the repo root absolutely via __file__ instead.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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
    .add_local_file(os.path.join(_REPO_ROOT, "training-data/patch_piper_progress.py"), remote_path="/tmp/patch_piper_progress.py", copy=True)
    .run_commands("python3 /tmp/patch_piper_progress.py")
    .add_local_dir(os.path.join(_REPO_ROOT, "training-data/full"), remote_path="/filelists_src")
)

app = modal.App("harness-tts-core-train", image=image)

corpus_volume = modal.Volume.from_name("tts-corpus-full")
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")


def build_piper_train_args(warmstart_url: str, max_steps: int, size_label: str, warmstart_local_path: str) -> list[str]:
    """Pure function so the CLI-argument construction is unit-testable without
    a real Modal container. warmstart_url is accepted (and unused here) so the
    signature mirrors what the caller has available - it's downloaded to
    warmstart_local_path by the Modal function itself, not by this function."""
    return [
        "piper.train", "fit",
        "--data.voice_name", f"harness_tts_core_{size_label}",
        "--data.csv_path", "/tmp/train.csv",
        "--data.audio_dir", "/tmp/wavs_local",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/tmp/piper_cache",
        "--data.config_path", f"/checkpoints/harness_tts_core_{size_label}/config.json",
        "--data.batch_size", "8",
        "--model.sample_rate", "22050",
        "--model.warmstart_ckpt", warmstart_local_path,
        "--trainer.max_steps", str(max_steps),
        "--trainer.default_root_dir", f"/checkpoints/harness_tts_core_{size_label}",
    ]


@app.function(gpu="T4", timeout=8 * 3600, volumes={"/data": corpus_volume, "/checkpoints": checkpoint_volume})
def run_piper_finetune(warmstart_url: str, max_steps: int, size_label: str) -> str:
    import os
    import subprocess
    import sys

    import piper.espeakbridge  # noqa: F401  sanity check, see piper_pilot_finetune.py

    subprocess.run(["wget", "-q", "-O", "/tmp/warmstart.ckpt", warmstart_url], check=True)

    def convert(src_filelist, dst_csv):
        import csv
        rows = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)
                rows.append((fname, text))
        with open(dst_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return len(rows)

    n_train = convert("/filelists_src/train.txt", "/tmp/train.csv")
    n_val = convert("/filelists_src/val.txt", "/tmp/val.csv")
    print(f"{n_train} train / {n_val} val rows")

    import shutil
    os.makedirs("/tmp/wavs_local", exist_ok=True)
    needed = set()
    for csv_path in ("/tmp/train.csv", "/tmp/val.csv"):
        with open(csv_path) as f:
            for line in f:
                fname = line.split("|", 1)[0]
                needed.add(fname)
    for fname in sorted(needed):
        shutil.copy(f"/data/wavs_full/{fname}", f"/tmp/wavs_local/{fname}")

    import torch
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.chdir("/opt/piper-src")
    sys.argv = build_piper_train_args(warmstart_url, max_steps, size_label, "/tmp/warmstart.ckpt")
    print("Running:", " ".join(sys.argv))

    import piper.train.__main__ as piper_main
    piper_main._DEFAULT_CALLBACKS = [piper_main._DEFAULT_CALLBACKS[0]]
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exited with code {e.code}")

    checkpoint_volume.commit()
    out_dir = f"/checkpoints/harness_tts_core_{size_label}"
    print(f"=== Contents of {out_dir} ===")
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            print(os.path.join(root, f))
    return out_dir
