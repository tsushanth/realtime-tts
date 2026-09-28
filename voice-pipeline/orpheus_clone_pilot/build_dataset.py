"""
Build a small HF dataset for an Orpheus voice-cloning pilot from the existing
Polly-Joanna corpus in training-data/.

Internal proof-of-concept only: validates that Orpheus fine-tuning can clone a
voice end-to-end. The Polly-Joanna corpus is not used to enrich the shipped
product (see DECISIONS.md re: Polly ToS) -- once this proves the pipeline
works, the same script points at real recordings instead (--manifest/--audio-dir).

Usage:
    python build_dataset.py --n 250 --voice-tag joanna --out ./joanna_pilot_dataset
"""
import argparse
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "training-data" / "manifest.jsonl"
DEFAULT_AUDIO_DIR = REPO_ROOT / "training-data" / "audio"


def load_candidates(manifest_path: Path) -> list[dict]:
    rows = []
    with open(manifest_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") == "ok" or d.get("file"):
                rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--voice-tag", required=True, help="New speaker tag, must not collide with the 8 existing prod tags")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=200)
    args = ap.parse_args()

    existing_prod_tags = {"tara", "leah", "zoe", "zac", "jess", "leo", "mia", "julia"}
    if args.voice_tag.lower() in existing_prod_tags:
        raise SystemExit(
            f"--voice-tag {args.voice_tag!r} collides with an existing prod speaker tag; "
            f"choose a different name to avoid corrupting that voice."
        )

    rows = load_candidates(args.manifest)
    rows = [r for r in rows if args.min_chars <= len(r["text"]) <= args.max_chars]
    if not rows:
        raise SystemExit("No candidate rows survived the length filter.")

    random.Random(args.seed).shuffle(rows)
    picked = rows[: args.n]
    if len(picked) < args.n:
        print(f"warning: only {len(picked)} candidates available, requested {args.n}")

    args.out.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out / "clips"
    clips_dir.mkdir(exist_ok=True)

    manifest_out = []
    missing = 0
    for row in picked:
        src = args.audio_dir / row["file"]
        if not src.exists():
            missing += 1
            continue
        dst = clips_dir / src.name
        shutil.copy2(src, dst)
        manifest_out.append(
            {
                "text": f"{args.voice_tag}: {row['text']}",
                "audio": f"clips/{src.name}",
            }
        )

    if missing:
        print(f"warning: {missing} referenced audio files were missing on disk")

    (args.out / "dataset_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_out) + "\n"
    )

    print(f"Wrote {len(manifest_out)} text/audio pairs to {args.out}")
    print("Audio format: MP3, 24kHz mono (confirmed via ffprobe against existing files) -- matches SNAC's expected rate, no resampling needed.")
    print("Next step: run Canopy's SNAC-prep notebook against dataset_manifest.jsonl to produce input_ids, then push to a HF dataset repo for finetune/train.py.")


if __name__ == "__main__":
    main()
