"""Builds a pipeline-format test dataset (consent.json, metadata.csv, wavs/) in the
`voice-datasets` Volume from one LibriTTS-R speaker (CC BY 4.0), so train_job.py can be run end to
end without a real customer. The consent file states plainly that this is public data.

modal run make_test_dataset.py --voice-id test-male-2 --gender M --rank 2
"""
import modal

app = modal.App("voice-test-dataset", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget").pip_install("soundfile", "numpy"))
datasets = modal.Volume.from_name("voice-datasets", create_if_missing=True)
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"


@app.function(timeout=3600, volumes={"/datasets": datasets})
def build(voice_id: str, gender: str, rank: int):
    import glob, json, os, shutil, subprocess
    from collections import defaultdict
    import soundfile as sf

    root = "/tmp/libri"; os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    spk_gender = {}
    for tsv in glob.glob(f"{root}/**/speakers.tsv", recursive=True):
        for line in open(tsv):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: spk_gender[p[0]] = p[1]
    dur, utts = defaultdict(float), defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        info = sf.info(wav); d = info.frames / info.samplerate
        txt = wav[:-4] + ".normalized.txt"
        if os.path.exists(txt) and spk_gender.get(spk) == gender:
            dur[spk] += d; utts[spk].append((wav, txt))
    spk = sorted(dur, key=dur.get, reverse=True)[rank]
    out = f"/datasets/{voice_id}"
    shutil.rmtree(out, ignore_errors=True); os.makedirs(f"{out}/wavs")
    with open(f"{out}/metadata.csv", "w") as f:
        for wav, txt in sorted(utts[spk]):
            shutil.copy(wav, f"{out}/wavs/{os.path.basename(wav)}")
            f.write(f"{os.path.basename(wav)}|{open(txt).read().strip()}\n")
    json.dump({"consent": True, "speaker_name": f"LibriTTS-R speaker {spk}", "attested_by": "test fixture",
               "note": "public dataset (CC BY 4.0), not a real customer voice"}, open(f"{out}/consent.json", "w"))
    datasets.commit()
    return spk, len(utts[spk]), dur[spk] / 60


@app.local_entrypoint()
def main(voice_id: str, gender: str = "M", rank: int = 2):
    print("Spawned:", build.spawn(voice_id, gender, rank).object_id)
