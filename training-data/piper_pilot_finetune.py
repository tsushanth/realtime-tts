"""Phase 0 pilot for Piper fine-tuning (CPU-serving candidate), same
cheap-first discipline as pilot_finetune.py for Matcha-TTS: small data slice,
short run, verify it actually works before scaling to the full corpus.

Reuses the same 279-sample / 20-validation Polly-Joanna slice already
prepared for the Matcha-TTS pilot (`pilot/train.txt`, `pilot/wavs/*.wav`) -
just reformatted into Piper's expected flat `filename|text` CSV instead of
Matcha's `path|text` filelist.

Piper specifics (researched from OHF-Voice/piper1-gpl's actual TRAINING.md,
setup.py, and open GitHub issues before writing any of this - see
training-data/README.md for the full research writeup):

- Training entrypoint is a LightningCLI (`python3 -m piper.train fit`), not a
  plain argparse script. Data format is a flat, no-header `filename|text` CSV
  (filenames only, resolved against one `--data.audio_dir`) - different from
  Matcha's `path|text` with full paths, and different from Matcha in that
  there's no separate preprocessing step; phonemization/caching happens
  inline during `fit`.
- Base checkpoint: `en_US-lessac-medium`
  (rhasspy/piper-checkpoints on HF, file
  `en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt`) - the canonical
  example base in Piper's own docs, and its 22050Hz "medium" quality matches
  our existing WAVs with no resampling needed.
- Known live issue (OHF-Voice/piper1-gpl#226, open as of the research date):
  loading older checkpoints under newer PyTorch throws because
  `torch.load`'s `weights_only=True` default (PyTorch 2.6+) rejects the
  checkpoint's non-tensor globals - the *exact* failure mode already hit and
  fixed for Matcha-TTS's checkpoint (see pilot_finetune.py / Jetson notes in
  DECISIONS.md). Reusing that same fix here pre-emptively (monkeypatch
  `torch.load` to force `weights_only=False`) rather than waiting to hit the
  error - this checkpoint is official/trusted (rhasspy's own HF dataset).
  **Real gotcha found running this**: the monkeypatch only works applied
  in-process - the first version of this script launched training via
  `subprocess.run`, a fresh Python process that never saw the parent's
  patched `torch.load` at all. Fixed by importing `piper.train.__main__`
  directly and calling its `main()` in-process instead of a subprocess (an
  earlier intermediate version used `runpy.run_module`, which works for this
  but re-executes the module's top-level code from scratch on every call -
  incompatible with the later `_DEFAULT_CALLBACKS` patch below, which needs
  the module imported normally so the mutation sticks).
- **Second real gotcha, unrelated to the above**: `--ckpt_path` (Lightning's
  built-in "resume this exact run" mechanism, which strictly validates the
  checkpoint's saved hyperparameters against the current model class) failed
  with "Subcommand 'fit' does not accept option 'model.sample_bytes'" - the
  lessac/medium checkpoint's saved config includes a `sample_bytes`
  hyperparameter the current repo's model class no longer accepts, a
  version-drift issue between when that checkpoint was trained and the
  current piper1-gpl code. The CLI exposes a separate `--model.warmstart_ckpt`
  flag that loads weights only, without that strict resume-hyperparameter
  validation - the correct mechanism for fine-tuning from a different/older
  checkpoint, and what this script actually uses.
- No documented step/epoch target for fine-tuning - `max_epochs=-1` is
  hardcoded as the default (issue #191 reports a fine-tune left running for
  12 days because nothing stops it automatically). `--trainer.max_steps`
  (standard PyTorch Lightning CLI flag) works fine to cap this pilot -
  confirmed by actually running it, not assumed; it wasn't documented
  anywhere in piper1-gpl's own docs.
- **Third real gotcha**: `piper.train.__main__` hardcodes a second
  `ModelCheckpoint` monitoring `val_mos` (a UTMOS quality-predictor score we
  never set up). Its own source comment says a missing `val_mos` should just
  log a warning and skip - true on whatever Lightning version that comment
  was written against, but our installed `lightning==2.6.6` hard-raises
  `MisconfigurationException` instead (a real behavior change between
  Lightning versions). Fixed by mutating `_DEFAULT_CALLBACKS` to drop that
  one callback before calling `main()` - see the runtime code below.
- No exact dependency version pins exist upstream (setup.py uses ranges like
  `torch>=2,<3`) - pinning torch here to a version known to predate the
  weights_only default change (2.1.2, matching what already works for
  Matcha-TTS in this repo) rather than trusting an unpinned range to resolve
  to something untested.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
    .pip_install(
        "torch==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # torch==2.1.2 was compiled against NumPy 1.x's C API; letting later
    # installs (librosa etc.) resolve NumPy 2.x breaks torch's NumPy
    # interop silently at import time (just a warning) but hard-fails later
    # ("RuntimeError: Numpy is not available") the first time anything
    # actually needs the two to interoperate - same root cause already hit
    # and fixed for Matcha-TTS's environment. Pin before anything else can
    # pull in NumPy 2.
    .pip_install("numpy<2")
    .run_commands(
        "git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src"
    )
    .apt_install("python3-dev")
    # --no-build-isolation (below) means pip will NOT create an isolated env
    # from pyproject.toml's [build-system].requires - it uses whatever's
    # already installed, so those have to be installed into the main
    # environment explicitly first, or setup.py's `from skbuild import
    # setup` fails with a plain ModuleNotFoundError.
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    # Two SEPARATE native extensions, built two different ways, and both
    # need to land inside the SOURCE TREE (not site-packages) because this
    # has to stay an editable install - `build_monotonic_align.sh` (below)
    # only knows how to write into `src/piper/train/vits/monotonic_align/`,
    # which is only on the import path for an editable install:
    #   1. `espeakbridge` - a CMake/scikit-build extension (see
    #      pyproject.toml's [build-system], CMakeLists.txt). Plain
    #      `pip install -e '.[train]'` silently produced NO espeakbridge .so
    #      at all (ImportError at runtime) - the classic pip pitfall where an
    #      editable install's build step runs in an isolated temp env whose
    #      artifacts never make it back into the source tree.
    #      `--no-build-isolation` fixes this by building in-place instead.
    #   2. `monotonic_align` - an older, separate Cython extension NOT
    #      wired into setup.py's build_ext or CMakeLists.txt at all (grepped
    #      both - it's absent from each) - the upstream repo's own dev
    #      workflow (script/dev_build) only ever builds this one via the
    #      standalone `build_monotonic_align.sh`, so it has to be run as its
    #      own separate step regardless of how the main package was built.
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    # The missing piece: pip's PEP-660 "editable install" path never invokes
    # scikit-build's CMake build at all for a classic `scikit-build` (not
    # scikit-build-core) setup.py - confirmed by an in-container diagnostic
    # rerun with `-v`, which showed zero cmake/CMake/ninja/configure output
    # anywhere in the install. Calling `setup.py build_ext --inplace`
    # DIRECTLY does trigger it, because skbuild.setup() overrides the
    # build_ext distutils command - this is exactly upstream's own
    # `script/dev_build`, just spelled out instead of relying on their
    # wrapper script. Two separate steps are both required: `pip install -e`
    # for deps/metadata (above), and this for the actual native build.
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    # The earlier `numpy<2` pin didn't survive the `.[train]` install -
    # something in that dependency set (onnx/jsonargparse/etc., not
    # explicitly pinned by piper's own setup.py) re-resolved and pulled
    # NumPy 2.x back in despite being installed first. Force it back down
    # as the last word after everything else is installed, rather than
    # trying to chase down which transitive dependency did it.
    .pip_install("numpy<2")
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/lessac_medium.ckpt "
        "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
        "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt'"
    )
    .add_local_dir("pilot", remote_path="/pilot_src")
)

app = modal.App("tts-piper-pilot-ft", image=image)

MAX_STEPS = 300  # same order of magnitude as the Matcha pilot, for a first smoke test


@app.function(gpu="T4", timeout=1800)
def run_piper_pilot():
    import csv
    import os
    import shutil
    import sys

    # Sanity check: confirms the image's explicit `setup.py build_ext
    # --inplace` step (see image definition above) actually produced a
    # working espeakbridge - fails fast with a clear message instead of a
    # confusing failure three steps later if a future image change breaks it.
    import piper.espeakbridge  # noqa: F401

    # Reformat the Matcha-style pilot filelist (path|text, full container
    # paths like /pilot/wavs/000123.wav) into Piper's flat filename|text CSV
    # pointing at one audio_dir - the actual required transform identified
    # by the research above.
    os.makedirs("/pilot/audio", exist_ok=True)
    os.makedirs("/pilot/cache", exist_ok=True)

    def convert(src_filelist, dst_csv):
        rows = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)
                src_wav = os.path.join("/pilot_src/wavs", fname)
                dst_wav = os.path.join("/pilot/audio", fname)
                if not os.path.exists(dst_wav):
                    shutil.copy(src_wav, dst_wav)
                rows.append((fname, text))
        with open(dst_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return len(rows)

    n_train = convert("/pilot_src/train.txt", "/pilot/train.csv")
    n_val = convert("/pilot_src/val.txt", "/pilot/val.csv")
    print(f"Converted {n_train} train / {n_val} val rows to Piper CSV format")

    # Same PyTorch weights_only issue already hit fine-tuning Matcha-TTS, but
    # a different root cause here: Lightning's own CLI (_parse_ckpt_path in
    # lightning/pytorch/cli.py) hardcodes `weights_only=True` when peeking at
    # --ckpt_path, independent of torch's own default. A subprocess-based
    # invocation (the first version of this script) does NOT work, because
    # the monkeypatch only exists in this parent process's torch module, not
    # a freshly spawned `python -m piper.train` process's. Fix: call the
    # training entrypoint's main() directly, in-process, via argv +
    # runpy - so the patch below actually applies to the torch.load call
    # that matters.
    import torch
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.chdir("/opt/piper-src")
    sys.argv = [
        "piper.train", "fit",
        "--data.voice_name", "pilot_ft",
        "--data.csv_path", "/pilot/train.csv",
        "--data.audio_dir", "/pilot/audio",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/pilot/cache",
        "--data.config_path", "/pilot/config.json",
        "--data.batch_size", "8",  # matches the Matcha pilot's batch size, small dataset
        "--model.sample_rate", "22050",
        "--model.warmstart_ckpt", "/ckpt/lessac_medium.ckpt",
        "--trainer.max_steps", str(MAX_STEPS),
        "--trainer.default_root_dir", "/output",
    ]
    print("Running in-process:", " ".join(sys.argv))

    # piper.train.__main__ hardcodes a second ModelCheckpoint monitoring
    # "val_mos" (a UTMOS quality score we haven't set up a predictor for).
    # Its own comment says a missing val_mos metric should just log a
    # warning and skip - true on whatever Lightning version that comment was
    # written against, but our installed lightning==2.6.6 hard-raises
    # MisconfigurationException instead (a real behavior change between
    # Lightning versions, not something wrong with our setup). Dropping that
    # one callback rather than chasing down which older Lightning version
    # still has the soft-skip behavior. Import as a real module (not via
    # runpy, which re-executes top-level code from scratch on each call and
    # would silently undo this patch) so the mutated callback list is what
    # main() actually uses.
    import piper.train.__main__ as piper_main
    piper_main._DEFAULT_CALLBACKS = [piper_main._DEFAULT_CALLBACKS[0]]
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exited with code {e.code}")

    print("=== Contents of /output after training ===")
    for root, dirs, files in os.walk("/output"):
        for f in files:
            print(os.path.join(root, f))


@app.local_entrypoint()
def main():
    run_piper_pilot.remote()
