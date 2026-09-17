"""Synthesize test sentences with the Kokoro pilot fine-tune (see
kokoro_pilot_finetune.py), same purpose as the Matcha-TTS/Piper synthesis
scripts: get real audio to actually listen to, not just trust a decreasing
loss number.

Retrains the same 1-epoch pilot fresh in the same container (its checkpoint
lived only in the ephemeral training container, never persisted - same
situation Piper's synthesis script was in) then, in the same process,
reloads the saved checkpoint through StyleTTS2's own model-construction
path (ASR/F0/PLBERT + build_model + load_checkpoint - the same sequence
train_second.py's own main() uses, just without re-training) and runs
StyleTTS2's own Kokoro-faithful inference helpers
(kokoro_tb_utils.extract_voicepack + run_kokoro_inference) - the same
functions train_second.py's own TensorBoard preview uses, which silently
produced nothing in the training run because that run's minimal config
left root_path empty.

English test-token preparation is new: kokoro_tb_utils.prepare_test_tokens
only exists for German (misaki.espeak.EspeakG2P). Writing the English
equivalent here using misaki.en.G2P (the same G2P used for training data
prep in kokoro_pilot_finetune.py) + kokoro_symbols.TextCleaner (generated
by the patch-styletts2 step, already in the image).
"""
import modal

# Same image as kokoro_pilot_finetune.py, duplicated rather than imported
# cross-file - Modal re-imports the entrypoint module inside the remote
# container to locate the decorated function, and that container never had
# kokoro_pilot_finetune.py mounted (same lesson learned the hard way with
# Piper's synthesis script).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1", "git", "wget")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("numpy<2")
    .run_commands(
        "git clone --recurse-submodules --depth 1 "
        "https://github.com/semidark/kikiri-tts.git /opt/kikiri-src"
    )
    .workdir("/opt/kikiri-src")
    .pip_install(
        "accelerate",
        "transformers==4.36.2",
        "librosa", "soundfile", "pyyaml",
        "tensorboard", "munch", "phonemizer", "huggingface_hub", "Cython",
        "einops", "einops-exts",
        "pandas", "click", "matplotlib", "tqdm",
    )
    .run_commands(
        "pip install 'misaki[en,de] @ git+https://github.com/semidark/misaki.git@main'"
    )
    .run_commands(
        "cd StyleTTS2 && git clone --depth 1 "
        "https://github.com/resemble-ai/monotonic_align.git "
        "&& pip install ./monotonic_align"
    )
    .run_commands("python3 scripts/prepare_training.py convert-weights")
    .run_commands("python3 scripts/prepare_training.py patch-styletts2")
    .add_local_dir("pilot", remote_path="/pilot_src")
)

app = modal.App("tts-kokoro-pilot-synth", image=image)

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]

CONFIG_YAML = """\
log_dir: "/output/kokoro_pilot"
first_stage_path: "first_stage.pth"
pretrained_model: "/opt/kikiri-src/training/kokoro_base.pth"
load_only_params: true
second_stage_load_pretrained: true
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


@app.function(gpu="T4", timeout=1800)
def train_and_synthesize():
    import base64
    import os
    import sys

    sys.path.insert(0, "/opt/kikiri-src")
    sys.path.insert(0, "/opt/kikiri-src/StyleTTS2")
    os.chdir("/opt/kikiri-src/StyleTTS2")

    import torch
    import torchaudio
    from misaki import en as misaki_en

    print("=== Resampling pilot audio to 24kHz and phonemizing (English) ===")
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

    process("/pilot_src/train.txt", "/pilot/train_list.txt")
    process("/pilot_src/val.txt", "/pilot/val_list.txt")

    os.makedirs("/output/kokoro_pilot", exist_ok=True)
    config_path = "/pilot/config_pilot.yml"
    with open(config_path, "w") as f:
        f.write(CONFIG_YAML)

    print("=== Training (same 1-epoch pilot as kokoro_pilot_finetune.py) ===")
    import train_second  # applies its own torch.load(weights_only=False) monkeypatch on import
    train_second.main(["--config_path", config_path], standalone_mode=False)

    print("=== Reloading the trained checkpoint for inference ===")
    import yaml
    from munch import Munch
    from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
    from Utils.PLBERT.util import load_plbert
    from kokoro_symbols import TextCleaner
    from kokoro_tb_utils import extract_voicepack, run_kokoro_inference
    from utils import recursive_munch

    config = yaml.safe_load(open(config_path))
    device = "cuda"

    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    _ = [model[key].eval() for key in model]

    ckpt_path = "/output/kokoro_pilot/epoch_2nd_00000.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    model, _, _, _ = load_checkpoint(model, None, ckpt_path, load_only_params=True)
    _ = [model[key].eval() for key in model]

    print("=== Extracting voicepack from our fine-tuning audio ===")
    voicepack, acoustic_norm, prosodic_norm = extract_voicepack(
        model, "/pilot/audio24k", device, n_samples=200
    )
    print(f"acoustic_norm={acoustic_norm:.4f}  prosodic_norm={prosodic_norm:.4f}")
    if voicepack is None:
        raise RuntimeError("extract_voicepack found no WAV files - check root_path")

    print("=== Preparing English test tokens (misaki.en.G2P + TextCleaner) ===")
    text_cleaner = TextCleaner()
    test_tokens = []
    for text in TEST_SENTENCES:
        phonemes, _ = g2p(text)
        token_ids = text_cleaner(phonemes)
        if not token_ids or len(token_ids) > 510:
            print(f"Skipping (bad token length): {text[:40]}")
            continue
        test_tokens.append((text, token_ids))
    print(f"Prepared {len(test_tokens)} test sentences")

    print("=== Running Kokoro-faithful inference ===")
    results_raw = run_kokoro_inference(model, test_tokens, voicepack, device, text_cleaner)

    import soundfile as sf
    results = []
    for i, (text, audio) in enumerate(results_raw):
        wav_path = f"/tmp/kokoro_pilot_sample_{i}.wav"
        sf.write(wav_path, audio, 24000, "PCM_16")
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        results.append({"text": text, "audio_b64": b64})
        print(f"Synthesized [{i}]: {text}")

    return results


@app.local_entrypoint()
def main():
    import base64
    import os

    results = train_and_synthesize.remote()
    out_dir = os.path.join(os.path.dirname(__file__), "kokoro_pilot_output")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["audio_b64"]))
        print(f"Saved: {path}  -  \"{r['text']}\"")
