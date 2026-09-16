"""Build train/val filelists for the full-corpus Matcha-TTS fine-tune.

Reads manifest.jsonl (idx -> text) and pairs each entry with its converted WAV
in wavs_full/, in the Matcha-TTS TextMelDataModule format: `path|text` per line.
Paths are written as they'll appear once the wavs are uploaded to a Modal Volume
mounted at /data (see full_finetune.py) - not local Mac paths.

Skips any index whose WAV is missing or zero-byte (should be none, given the
bulk-generation cleanup already done, but cheap to double check here rather than
let a bad file crash a training run hours in).
"""
import json
import os
import random

MANIFEST = "manifest.jsonl"
WAV_DIR = "wavs_full"
REMOTE_WAV_DIR = "/data/wavs_full"  # mount path inside the Modal container
VAL_FRACTION = 0.01  # ~220 held-out samples, plenty for spot-checking a 22k-sample run
SEED = 42

records = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records[rec["idx"]] = rec["text"]

pairs = []
missing = 0
for idx, text in records.items():
    wav_name = f"{idx:06d}.wav"
    local_path = os.path.join(WAV_DIR, wav_name)
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        missing += 1
        continue
    pairs.append((f"{REMOTE_WAV_DIR}/{wav_name}", text))

print(f"{len(pairs)} usable pairs, {missing} missing/empty skipped")

random.seed(SEED)
random.shuffle(pairs)

n_val = max(1, int(len(pairs) * VAL_FRACTION))
val_pairs = pairs[:n_val]
train_pairs = pairs[n_val:]

os.makedirs("full", exist_ok=True)
with open("full/train.txt", "w") as f:
    for path, text in train_pairs:
        f.write(f"{path}|{text}\n")
with open("full/val.txt", "w") as f:
    for path, text in val_pairs:
        f.write(f"{path}|{text}\n")

print(f"train: {len(train_pairs)}  val: {len(val_pairs)}")
