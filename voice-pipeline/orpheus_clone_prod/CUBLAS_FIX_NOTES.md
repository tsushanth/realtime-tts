# libcublas.so.12 fix — faster-whisper in train_image

## Root cause

`train_image` (in `main.py`) installs `torch==2.5.1`/`torchaudio==2.5.1` from
the `cu121` extra index, and separately installs `faster-whisper`. Torch
bundles its own copy of the CUDA runtime libraries inside its wheel, but
those are private to torch's own `.so` loading path — they are not placed
somewhere `ctranslate2` (faster-whisper's inference backend) will find them.

`ctranslate2` dynamically loads `libcublas.so.12` / cuDNN itself via the
standard dynamic linker search path (`LD_LIBRARY_PATH` / `ldconfig`), and the
`debian_slim` base image has no system CUDA install and no
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip packages providing those
`.so` files on that path. Result: every `WhisperModel(..., device="cuda")`
load in `prepare_dataset.py` failed with:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

This is a well-documented, common failure mode for faster-whisper/ctranslate2
GPU deployments in minimal containers — confirmed against current upstream
issues (e.g. SYSTRAN/faster-whisper#717, #783, #1320; the wyoming-faster-whisper
port of the same issue). The documented fixes are either (a) `pip install
nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` and export `LD_LIBRARY_PATH`
pointing at those packages' installed `lib/` dirs before the interpreter
starts, or (b) run Whisper on CPU, which ctranslate2 supports natively
without any CUDA runtime dependency.

Because `load_transcription_model()` wraps the model load in a
`try/except` that raises `TranscriptionUnavailableError`, this particular
failure was already being surfaced as a systemic-failure exception rather
than silently producing "0 usable clips" — that per-clip-swallowing concern
described in the task does not apply to the current state of this file
(see `prepare_dataset.py`: `load_transcription_model()` is called eagerly,
outside the per-clip loop, specifically so a load failure propagates instead
of being counted as N bad clips). Fixing the CUDA loading problem itself is
what matters here.

## Fix chosen: option 2 — CPU-mode Whisper

`prepare_dataset.py`'s `load_transcription_model()` now instantiates:

```python
WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
```

instead of `device="cuda", compute_type="float16"`. `int8` is the
compute_type faster-whisper's own docs recommend for CPU inference.

Rationale for CPU over the `nvidia-cublas-cu12`/`LD_LIBRARY_PATH` route:

- The A10G GPU reserved for this container is needed for the LoRA
  fine-tuning step, not for the one-time dataset-transcription pass that
  happens before training starts. Transcription speed only affects wall
  clock of that pre-training pass, not correctness.
- The GPU-lib fix requires computing `LD_LIBRARY_PATH` dynamically inside the
  Modal image build (the nvidia pip packages install path depends on the
  Python version/venv layout) and getting it exported *before* the Python
  process that imports ctranslate2 starts. That's an extra moving part that
  is easy to get subtly wrong in a Modal `Image` definition and hard to
  verify without an actual GPU deploy — exactly the risk this task called
  out as unacceptable to get wrong twice.
- CPU mode sidesteps the CUDA-runtime-library problem for faster-whisper
  entirely and is trivially verifiable without any GPU (see below), which
  makes it the safer fix to ship without another real Modal deploy cycle.

No change was needed to `train_image` in `main.py` for this fix — torch's
`cu121` install stays as-is (still needed for the LoRA training step), and
`faster-whisper` is unaffected by device selection at install time.

## Verification

- **Local repro attempt**: this sandbox's outbound network is restricted
  (`pip install faster-whisper` fails with a TLS/SSL error), so I could not
  actually run `faster_whisper.WhisperModel(device="cpu", ...)` against a
  real audio file end-to-end in this environment. I could not independently
  reproduce the exact `libcublas.so.12` error or prove the CPU path runs,
  by execution, in this sandbox.
- **Documentation verification**: confirmed via web search against current
  upstream faster-whisper/ctranslate2 GitHub issues (SYSTRAN/faster-whisper
  #717, #783, #1320, and the same issue filed against
  wyoming-faster-whisper) that (a) this exact error message is the known
  symptom of ctranslate2 not finding cuBLAS/cuDNN `.so` files, (b) the
  documented GPU-side fix is installing `nvidia-cublas-cu12` +
  `nvidia-cudnn-cu12` and setting `LD_LIBRARY_PATH`, and (c) CPU mode
  (`device="cpu"`) is a supported, standard ctranslate2 execution path that
  does not touch CUDA at all, so it is not subject to this failure class by
  construction.
- **Test suite**: `pytest tests/` — 88 passed, 1 skipped, 0 failed. None of
  the tests instantiate a real `WhisperModel`; `transcribe_clip` /
  `load_transcription_model` are monkeypatched in tests, so this change is
  not exercised by the suite (expected, per the task).

## Confidence

**MEDIUM** — the fix is grounded in the actual documented failure mode and a
supported ctranslate2 execution path (not a guess), but it is unverified by
real execution in this environment (no network access to install
faster-whisper/ctranslate2 here, and no GPU available to reproduce the
original failure either way). The next real Modal deploy is the first true
end-to-end confirmation.

## Lower-priority item: error surfacing

Already handled in the existing code and not touched further: a whisper
load failure raises `TranscriptionUnavailableError` (distinct from
`DatasetTooSmallError`) before any per-clip loop runs, so this exact
failure mode was never going to be misreported to the customer as "0 usable
clips" once the model load itself throws. That existing design is sound and
this fix (making the model load succeed on CPU) makes the distinction moot
for this specific failure in practice, per the task's guidance.
