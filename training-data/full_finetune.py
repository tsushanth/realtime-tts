"""Full-corpus Matcha-TTS fine-tune: same approach as pilot_finetune.py (see
that file's docstring for every environment gotcha already worked around here),
scaled from the 279-sample pilot to the full 22,011-sample Polly-Joanna corpus.

Three real differences from the pilot, both because this is a multi-hour run
instead of a 70-second smoke test:

1. Data comes from a Modal Volume (`tts-corpus-full`), not `add_local_dir` —
   baking multiple GB of audio into every image build/rebuild would be slow
   and wasteful. Volumes mount at runtime instead.
2. The trained checkpoint is saved to a **second** volume (`tts-checkpoints`)
   that outlives the container. The pilot saved to `/output` inside the
   container and only survived because it also returned synthesized audio
   directly in the function's response — fine for a 70-second run, not
   something to rely on for a run that might take hours and could be
   interrupted. Also checkpoints periodically during training (not just at the
   end) so a crash partway through doesn't lose everything.
3. Getting the ~4.2GB of WAV audio (tarred) onto the corpus volume in the
   first place turned out to need its own workaround: uploading it directly
   from this Mac (via `modal volume put` or `aws s3 cp` on the single 4.2GB
   file) died silently and non-deterministically partway through every single
   time (at 7%, 17%, then 1% - no error, no consistent cutoff point),
   regardless of destination or whether the sandbox was disabled. That
   pattern - clean transfer, then a silent unrecoverable stop with no error
   text at all - pointed to a flaky home network link, not a bug in either
   CLI (confirmed by one attempt surfacing a real `SSLV3_ALERT_BAD_RECORD_MAC`
   TLS error). Fix: split the tar into 100MB chunks with `split -b 100m` and
   `aws s3 sync` them individually - each chunk is small enough to reliably
   finish even on a flaky link, and `sync` skips already-uploaded chunks on
   rerun, so a mid-job failure only costs the chunks in flight, not the whole
   transfer. `setup_corpus_volume()` below then reassembles them **inside
   Modal** (cloud-to-cloud S3 read, not dependent on this Mac's connection at
   all) and extracts onto the volume. Run this once before the real training
   function; it's idempotent (skips work if the volume already has the data).

Cost check before running: pilot measured ~70s/300 steps on a T4 (~$0.01).
Per-step time is roughly constant regardless of total dataset size (each step
trains on one batch, not the whole dataset), so N steps here costs about the
same as N steps in the pilot. Real cost scales with --max-steps, not corpus
size. See training-data/README.md for the current estimate at the step count
actually run.
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
        "mkdir -p /ckpt && wget -q -O /ckpt/matcha_ljspeech.ckpt "
        "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt "
        "&& wget -q -O /ckpt/hifigan_T2_v1.ckpt "
        "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1"
    )
    .add_local_dir("full", remote_path="/filelists")
)

app = modal.App("tts-full-matcha-ft", image=image)

corpus_volume = modal.Volume.from_name("tts-corpus-full")
checkpoint_volume = modal.Volume.from_name("tts-checkpoints", create_if_missing=True)

setup_image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")

S3_BUCKET = "tts-training-corpus-451115460668"
S3_PARTS_PREFIX = "parts/"


@app.function(
    image=setup_image,
    volumes={"/data": corpus_volume},
    secrets=[modal.Secret.from_name("tts-corpus-s3-reader")],
    timeout=1800,
)
def setup_corpus_volume():
    """Idempotent: downloads the chunked tar from S3, reassembles it, and
    extracts onto the corpus volume, skipping entirely if already done. Run
    this once (`modal run full_finetune.py::setup_corpus_volume`) before the
    real training run - the credentials here are a dedicated, read-only-on-
    this-one-bucket IAM user (tts-corpus-modal-reader), not the main AWS
    account's broader shared key, so this container never sees more S3 access
    than it needs.
    """
    import os
    import subprocess
    import tarfile

    marker = "/data/.corpus_extracted"
    if os.path.exists(marker):
        print("Corpus already extracted on the volume, skipping.")
        return

    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PARTS_PREFIX):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    keys.sort()
    print(f"Found {len(keys)} parts in s3://{S3_BUCKET}/{S3_PARTS_PREFIX}")

    tar_path = "/tmp/wavs_full.tar"
    with open(tar_path, "wb") as out:
        for i, key in enumerate(keys):
            print(f"Downloading part {i+1}/{len(keys)}: {key}")
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            out.write(resp["Body"].read())

    size_gb = os.path.getsize(tar_path) / 1e9
    print(f"Reassembled tar: {size_gb:.2f} GB")

    print("Extracting onto /data ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall("/data")
    os.remove(tar_path)

    with open(marker, "w") as f:
        f.write("done")
    corpus_volume.commit()
    print("Done. /data/wavs_full is ready on the tts-corpus-full volume.")


@app.local_entrypoint()
def setup():
    setup_corpus_volume.remote()

DATA_YAML = """\
_target_: matcha.data.text_mel_datamodule.TextMelDataModule
name: full
train_filelist_path: /filelists/train.txt
valid_filelist_path: /filelists/val.txt
batch_size: 8
num_workers: 8
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

MAX_STEPS = 20000  # same batch_size as the pilot, so its measured $0.01/300-steps rate
                    # extrapolates cleanly: 20000/300 * $0.01 ~= $0.67, ~1.3h wall time
CHECKPOINT_EVERY_N_STEPS = 2000


@app.function(
    gpu="T4",  # same tier the pilot's real cost numbers came from - don't guess at a different tier's cost
    timeout=4 * 3600,
    volumes={"/data": corpus_volume, "/checkpoints": checkpoint_volume},
)
def run_full_finetune():
    import os
    import sys

    os.chdir("/opt/matcha-src")
    os.makedirs("configs/data", exist_ok=True)
    with open("configs/data/full.yaml", "w") as f:
        f.write(DATA_YAML)

    sys.path.insert(0, "/opt/matcha-src")

    import hydra
    from hydra import compose, initialize_config_dir
    import torch
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint

    with initialize_config_dir(config_dir="/opt/matcha-src/configs", version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=full",
                "run_name=full_ft",
                "model.out_size=172",
                f"+trainer.max_steps={MAX_STEPS}",
                "trainer.check_val_every_n_epoch=1",
                "logger=csv",
                "callbacks=none",
            ],
        )

    L.seed_everything(cfg.seed, workers=True)

    print("=== Instantiating datamodule ===")
    datamodule = hydra.utils.instantiate(cfg.data)

    print("=== Instantiating model, loading pretrained LJSpeech weights ===")
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load("/ckpt/matcha_ljspeech.ckpt", map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    os.makedirs("/checkpoints/full_ft", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="/checkpoints/full_ft",
        filename="step{step}",
        every_n_train_steps=CHECKPOINT_EVERY_N_STEPS,
        save_top_k=-1,  # keep every periodic checkpoint, not just the "best" one
        save_last=True,
    )

    logger = TensorBoardLogger(save_dir="/checkpoints/full_ft_logs", name="full_ft")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=MAX_STEPS,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=50,
        gradient_clip_val=5.0,
    )

    print(f"=== Fine-tuning ({MAX_STEPS} steps on {21791} train / 220 val samples) ===")
    trainer.fit(model=model, datamodule=datamodule)

    checkpoint_volume.commit()  # flush volume writes before the container exits
    print("Done. Checkpoints in /checkpoints/full_ft on the tts-checkpoints volume.")


@app.local_entrypoint()
def main():
    run_full_finetune.remote()
