"""Phase 0 pilot for Kokoro fine-tuning via StyleTTS2, same cheap-first
discipline as the Matcha-TTS and Piper pilots. This is the highest-risk of
the three candidate architectures (see training-data/README.md's Kokoro
research writeup), so the goal here is narrower than the other two pilots:
answer "does the direct-from-Kokoro-weights shortcut work at all for a
same-language fine-tune" as cheaply as possible, BEFORE considering the
full tens-of-GPU-hours Stage 1 + 2 run the community repos actually needed
for their (cross-lingual) use case.

Forking semidark/kikiri-tts directly rather than reimplementing its
checkpoint-conversion and StyleTTS2 patches from scratch, per the research's
explicit "lowest-risk approach" recommendation - its patched StyleTTS2
submodule already has the real fixes (symbol-table remap, weight_norm API
migration, restored loss tensors, torch.load weights_only fix, etc.) baked
in, and its Utils/ (JDC pitch extractor, ASR aligner, PLBERT) are bundled
directly in the submodule as real files, not placeholders - and, useful
for us specifically, all three were pretrained on ENGLISH corpora, so for
an English fine-tune (not kikiri's German target) they're a native match,
not a cross-lingual compromise.

**The actual experiment**: kikiri-tts's own config (config_german_ft.yml)
fine-tunes from `first_stage.pth` - the output of a full Stage 1 training
run they needed because German is a new language for Kokoro (new phoneme
distribution, new speaker). Our case is different: we want a different
ENGLISH voice, the same language Kokoro already knows. `train_second.py`
itself already supports skipping Stage 1 entirely and loading a
`pretrained_model` directly (`second_stage_load_pretrained: true`) - a real,
existing code path, just not the one kikiri's own config exercises. This
pilot tests whether that shortcut - fine-tuning directly from the converted
Kokoro checkpoint, no Stage 1 - produces something coherent for a
same-language voice adaptation. Nobody has verified this combination before
(the two community repos are both cross-lingual), so this is a genuine
unknown, not a known-good recipe - that's exactly why it's being tried at
minimal cost before considering the expensive alternative.

Data: reuses the same 279-sample Polly-Joanna pilot slice as Matcha-TTS and
Piper's pilots, resampled to 24kHz (StyleTTS2's expected rate, vs. the
22050Hz used elsewhere) and phonemized to IPA via misaki's English G2P
(`misaki[en]`, the same G2P Kokoro itself uses at inference time - not the
generic espeak wrapper the German recipe needed).

**Result: the shortcut works.** A full epoch (139 steps at batch_size=2,
matching the 279-sample dataset) completed cleanly - loss decreased
sensibly across the whole epoch (0.79 -> 0.55), validation ran without
error, checkpoint saved. This confirms the direct-from-Kokoro-weights path
is viable for a same-language fine-tune, not just a theory. Not yet
ear-verified - `train_second.py`'s own TensorBoard voicepack-preview step
failed silently (`extract_voicepack: no WAV files found in` - a real gap in
this minimal config, `root_path` was left empty and needs pointing at the
audio directory), so there's a real checkpoint but no synthesized audio
from it yet. Building that inference step is the natural next task before
deciding whether to scale this to the full corpus, matching the same
by-ear-verify-before-scaling discipline used for Matcha-TTS and Piper.
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
    .pip_install("numpy<2")  # same NumPy2/torch2.1.2 ABI break as Matcha-TTS and Piper
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/semidark/kikiri-tts.git /opt/kikiri-src"
    )
    .workdir("/opt/kikiri-src")
    .pip_install(
        "accelerate",
        # Unpinned `transformers` resolved to a release new enough to no
        # longer recognize torch==2.1.2 as "available" at all (lazy backend
        # detection raised "requires the PyTorch library but it was not
        # found" from two independent code paths - misaki's BART fallback
        # and PLBERT's CustomAlbert - ruling out a misaki-specific bug).
        # Pinning to a version contemporaneous with torch 2.1.2 (~Nov 2023)
        # instead of chasing this further.
        "transformers==4.36.2",
        "librosa", "soundfile", "pyyaml",
        "tensorboard", "munch", "phonemizer", "huggingface_hub", "Cython",
        "einops", "einops-exts",
        # NOT the PyPI package "monotonic_align" (a real gotcha hit here):
        # its import name collides with the LOCAL resemble-ai/monotonic_align
        # clone TRAINING_GUIDE.md calls for below, and the PyPI one doesn't
        # have `mask_from_lens` - StyleTTS2's own utils.py needs the specific
        # local clone, not a same-named generic package.
        #
        # Found by actually grepping every import across StyleTTS2/Utils/
        # scripts rather than trusting TRAINING_GUIDE.md's pip-install list,
        # which is incomplete (same class of gotcha as Matcha-TTS/Piper's
        # own docs both being incomplete) - pandas, click, matplotlib, tqdm
        # are all real runtime deps of files this pilot actually imports.
        "pandas", "click", "matplotlib", "tqdm",
    )
    .run_commands(
        "pip install 'misaki[en,de] @ git+https://github.com/semidark/misaki.git@main'"
    )
    # TRAINING_GUIDE.md's literal steps (clone, then `build_ext --inplace`
    # inside it) leave the package importable only via a path hack, and a
    # broken one at that: the clone nests as
    # StyleTTS2/monotonic_align/monotonic_align/__init__.py (outer = repo
    # root, inner = the actual package). Adding StyleTTS2/ to sys.path (as
    # this pilot does, to reach train_second.py) makes Python resolve
    # `import monotonic_align` to the OUTER directory instead - which has no
    # __init__.py of its own, so Python treats it as an empty PEP 420
    # namespace package ("cannot import name X from Y (unknown location)").
    # Fix: actually `pip install` the clone so it registers properly in
    # site-packages instead of relying on directory nesting + sys.path luck.
    .run_commands(
        "cd StyleTTS2 && git clone --depth 1 "
        "https://github.com/resemble-ai/monotonic_align.git "
        "&& pip install ./monotonic_align"
    )
    .run_commands("python3 scripts/prepare_training.py convert-weights")
    .run_commands("python3 scripts/prepare_training.py patch-styletts2")
    .add_local_dir("pilot", remote_path="/pilot_src")
)

app = modal.App("tts-kokoro-pilot-ft", image=image)


@app.function(gpu="T4", timeout=1800)
def run_kokoro_pilot():
    import os
    import shutil
    import sys

    sys.path.insert(0, "/opt/kikiri-src")
    sys.path.insert(0, "/opt/kikiri-src/StyleTTS2")
    os.chdir("/opt/kikiri-src/StyleTTS2")

    import torch
    import torchaudio
    from misaki import en as misaki_en

    print("=== Resampling pilot audio to 24kHz and phonemizing (English) ===")
    # misaki's G2P constructs a real BART-based FallbackNetwork (for
    # out-of-vocabulary words) unless `fallback` is truthy - passing False
    # doesn't skip it (falsy, so the ternary still builds the real network).
    # That network's construction hit "BartForConditionalGeneration requires
    # the PyTorch library but it was not found" despite torch being
    # installed - a transformers/torch interop issue not worth debugging for
    # a fallback path our simple call-center sentences shouldn't need. A
    # no-op callable satisfies the truthy check and skips it entirely; if it
    # is ever actually invoked (an OOV word), returning an empty phoneme
    # string is an acceptable degradation for this smoke test.
    g2p = misaki_en.G2P(trf=False, british=False, fallback=lambda tk: ("", 1))

    os.makedirs("/pilot/audio24k", exist_ok=True)

    def process(src_filelist, dst_filelist):
        lines_out = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)
                src_wav = os.path.join("/pilot_src/wavs", fname)
                dst_wav = os.path.join("/pilot/audio24k", fname)
                if not os.path.exists(dst_wav):
                    wav, sr = torchaudio.load(src_wav)
                    if sr != 24000:
                        wav = torchaudio.functional.resample(wav, sr, 24000)
                    torchaudio.save(dst_wav, wav, 24000)
                phonemes, _ = g2p(text)
                lines_out.append(f"{dst_wav}|{phonemes}|0")
        with open(dst_filelist, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return len(lines_out)

    n_train = process("/pilot_src/train.txt", "/pilot/train_list.txt")
    n_val = process("/pilot_src/val.txt", "/pilot/val_list.txt")
    print(f"Prepared {n_train} train / {n_val} val StyleTTS2-format rows")

    print("=== Writing fine-tune config (direct-from-Kokoro-weights shortcut) ===")
    config_yaml = f"""\
log_dir: "/output/kokoro_pilot"
first_stage_path: "first_stage.pth"
pretrained_model: "/opt/kikiri-src/training/kokoro_base.pth"
load_only_params: true
second_stage_load_pretrained: true
# batch_size=4 (what the research said should fit in 10GB+) hit a real
# CUDA OOM on this T4: Stage 2 loads generator + WavLM SLM discriminator +
# ASR + JDC + PLBERT simultaneously, and this T4 instance's 14.56GB was
# fully consumed. batch_size=1 then hit a WavLM conv1d shape mismatch
# ([1, 40000, 1] vs expected 1-channel input) - looks like a bare
# `.squeeze()` somewhere in the SLM loss path collapsing the batch dim
# itself when it happens to equal 1, not a real architecture bug. 2 is
# small enough to likely dodge the OOM while avoiding that ambiguity.
batch_size: 2
epochs: 1
epochs_1st: 1
epochs_2nd: 1
save_freq: 1
data_params:
  train_data: "/pilot/train_list.txt"
  val_data: "/pilot/val_list.txt"
  root_path: ""
  OOD_data: "/pilot/train_list.txt"
  min_length: 10
  num_workers: 2
preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300
    n_mels: 80
    fmin: 0
    fmax: 8000
model_params:
  dim_in: 64
  n_token: 178
  hidden_dim: 512
  style_dim: 128
  max_dur: 50
  multispeaker: false
  n_mels: 80
  dropout: 0.2
  n_layer: 3
  text_encoder_kernel_size: 5
  decoder:
    type: istftnet
    upsample_rates: [10, 6]
    upsample_kernel_sizes: [20, 12]
    upsample_initial_channel: 512
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    gen_istft_n_fft: 20
    gen_istft_hop_size: 5
  diffusion:
    embedding_mask_proba: 0.1
    transformer:
      num_layers: 3
      num_heads: 8
      head_features: 64
      multiplier: 2
    dist:
      sigma_data: 0.2
      estimate_sigma_data: true
      mean: -3.0
      std: 1.0
  plbert:
    hidden_size: 768
    num_attention_heads: 12
    intermediate_size: 2048
    max_position_embeddings: 512
    num_hidden_layers: 12
    dropout: 0.1
  slm:
    model: "microsoft/wavlm-base-plus"
    sr: 16000
    hidden: 768
    nlayers: 13
    initial_channel: 64
loss_params:
  lambda_gen: 1.0
  lambda_mel: 5.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_s2s: 1.0
  lambda_mono: 1.0
  lambda_slm: 1.0
  lambda_diff: 0.0
  lambda_sty: 0.0
  TMA_epoch: 0
  diff_epoch: 999
  joint_epoch: 999
optimizer_params:
  lr: 0.0001
  bert_lr: 0.00001
  ft_lr: 0.0001
F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: "Utils/PLBERT/"
slmadv_params:
  min_len: 100
  max_len: 500
  batch_percentage: 0.5
  iter: 10
  thresh: 5
  scale: 0.01
  sig: 1.5
"""
    os.makedirs("/output/kokoro_pilot", exist_ok=True)
    config_path = "/pilot/config_pilot.yml"
    with open(config_path, "w") as f:
        f.write(config_yaml)

    # Same weights_only fix already needed for Matcha-TTS and Piper's
    # checkpoints - kikiri-tts's own load_checkpoint util calls torch.load
    # without weights_only=False, and our converted kokoro_base.pth pickles
    # non-tensor structure (nested dict of state_dicts under "net").
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    print("=== Running train_second.py (1 epoch smoke test) ===")
    import train_second
    # main() is a @click.command(), confirmed by reading the source directly
    # (not guessed) - must be invoked via Click's calling convention with an
    # explicit argv-style list and standalone_mode=False, not as a plain
    # Python function call (which would fail: Click wraps it, and a bare
    # `main(config_path)` call doesn't go through Click's option parsing at
    # all) or left to call sys.exit() on completion (standalone_mode=True,
    # the default, would kill this whole Modal function via SystemExit).
    train_second.main(["--config_path", config_path], standalone_mode=False)

    print("=== Contents of /output after pilot ===")
    for root, dirs, files in os.walk("/output"):
        for f in files:
            print(os.path.join(root, f))


@app.local_entrypoint()
def main():
    run_kokoro_pilot.remote()
