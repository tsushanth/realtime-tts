"""Export licence-audited public Piper checkpoints to ONNX and store them in the `house-voices`
Modal Volume as <voice_id>/{model.onnx, model.onnx.json, owner.json}; smoke-test each on CPU.

Usage:
    modal run voices/export_voices.py                       # all voices in VOICES
    modal run voices/export_voices.py --only en-us-kristin,de-de-mls
Outputs: voices/samples/<voice_id>/sample_{1,4}.wav (+ en: sample_2,3,5) and voices/results.json.

Licence verdicts live in voices/LICENSES.md. Model files are NOT committed; they live in the Volume.
Image approach copied from training-data/synthesize_ljspeech.py (piper1-gpl from source,
torch.load weights_only patch, piper.train.export_onnx).
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
    .pip_install("torch==2.1.2")
    .pip_install("numpy<2")
    .run_commands("git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src")
    .apt_install("python3-dev")
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    .pip_install("numpy<2")
)

app = modal.App("house-voices-export", image=image)
volume = modal.Volume.from_name("house-voices", create_if_missing=True)

HF = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"

EN = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]

def _pair(a, b):  # non-English: natural equivalents of sentence 1 and 4
    return {0: a, 3: b}

TEXTS = {
    "en": EN,
    "es": _pair("Gracias por llamar, puedo ayudarle con eso. Permítame consultar los datos de su cuenta ahora mismo.",
                "¿Hay algo más en lo que pueda ayudarle hoy?"),
    "fr": _pair("Merci de votre appel, je peux vous aider avec cela. Je consulte les détails de votre compte tout de suite.",
                "Y a-t-il autre chose que je puisse faire pour vous aujourd'hui ?"),
    "de": _pair("Vielen Dank für Ihren Anruf, dabei kann ich Ihnen helfen. Ich rufe jetzt gleich Ihre Kontodaten auf.",
                "Kann ich heute noch etwas für Sie tun?"),
    "pt": _pair("Obrigado por ligar, posso ajudá-lo com isso. Deixe-me abrir os dados da sua conta agora mesmo.",
                "Há mais alguma coisa em que eu possa ajudar hoje?"),
    "it": _pair("Grazie per aver chiamato, posso aiutarla con questo. Apro subito i dettagli del suo account.",
                "Posso aiutarla in qualcos'altro oggi?"),
    "nl": _pair("Bedankt voor uw telefoontje, daarbij kan ik u helpen. Ik haal uw accountgegevens er meteen bij.",
                "Kan ik u vandaag nog ergens anders mee helpen?"),
    "ru": _pair("Спасибо за звонок, я могу вам с этим помочь. Сейчас я открою данные вашей учётной записи.",
                "Могу ли я вам ещё чем-нибудь помочь сегодня?"),
    "pl": _pair("Dziękuję za telefon, chętnie w tym pomogę. Zaraz sprawdzę dane Państwa konta.",
                "Czy mogę dziś jeszcze w czymś pomóc?"),
}

PD = "Public domain"
def cc0(ds): return ("CC0-1.0", f"{ds} (CC0 dataset; via rhasspy/piper-checkpoints)")

# id -> (hf dir, ckpt file, language code for TEXTS, licence, attribution, speaker_id or None)
VOICES = {
    # --- clean provenance: trained from scratch / from clean base, clean data ---
    "en-us-ljspeech": ("en/en_US/ljspeech/medium", "lj-med_1000.ckpt", "en", PD, "LJ Speech dataset (Keith Ito), public domain; Piper voice by Bryce Beattie / rhasspy.", None),
    "en-us-kristin": ("en/en_US/kristin/medium", "kristin-2000.ckpt", "en", PD, "Voice trained on LibriVox recordings (public domain) by Bryce Beattie via rhasspy/piper-checkpoints.", None),
    "en-us-john": ("en/en_US/john/medium", "john-2599.ckpt", "en", PD, "Voice trained on LibriVox recordings (public domain), finetuned from Kristin, by Bryce Beattie via rhasspy/piper-checkpoints.", None),
    "fr-fr-mls": ("fr/fr_FR/mls/medium", "epoch=317-step=3124032.ckpt", "fr", "CC-BY-4.0", "Trained on Multilingual LibriSpeech (openslr.org/94), CC BY 4.0, via rhasspy/piper-checkpoints.", 0),
    "de-de-mls": ("de/de_DE/mls/medium", "epoch=128-step=5149698.ckpt", "de", "CC-BY-4.0", "Trained on Multilingual LibriSpeech (openslr.org/94), CC BY 4.0, via rhasspy/piper-checkpoints.", 0),
    "nl-nl-mls": ("nl/nl_NL/mls/medium", "epoch=242-step=9245178.ckpt", "nl", "CC-BY-4.0", "Trained on Multilingual LibriSpeech (openslr.org/94), CC BY 4.0, via rhasspy/piper-checkpoints.", 0),
    "it-it-serena": ("it/it_IT/serena/medium", "epoch=14-step=83250.ckpt", "it", "CC-BY-4.0", "Trained on committa/serena-synthetic-it-27h (CC BY 4.0, synthetic data from Qwen3-TTS) via rhasspy/piper-checkpoints.", None),
    # --- dataset licence OK but finetuned from the Lessac (Blizzard research-only) checkpoint: FLAGGED ---
    "en-us-joe": ("en/en_US/joe/medium", "epoch=7889-step=1221224.ckpt", "en", "CC0-1.0", "Joe dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "en-us-mike": ("en/en_US/mike/medium", "epoch=5460-val_mos=4.2686.ckpt", "en", "CC0-1.0", "Mike dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "en-us-sam": ("en/en_US/sam/medium", "epoch=4688-step=106008.ckpt", "en", "Apache-2.0", "Sam Accenture non-binary voice dataset, Apache-2.0; via rhasspy/piper-checkpoints.", None),
    "es-es-davefx": ("es/es_ES/davefx/medium", "epoch=5629-step=1605020.ckpt", "es", "CC0-1.0", "DaveFX dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "es-mx-ald": ("es/es_MX/ald/medium", "epoch=9999-step=1753600.ckpt", "es", "Unlicense", "Ald Mexican Spanish speech dataset (rmcpantoja), Unlicense; via rhasspy/piper-checkpoints.", None),
    "fr-fr-siwis": ("fr/fr_FR/siwis/medium", "epoch=3304-step=2050940.ckpt", "fr", "CC-BY-4.0", "SIWIS French Speech Synthesis Database, University of Edinburgh, CC BY 4.0; via rhasspy/piper-checkpoints.", None),
    "de-de-thorsten": ("de/de_DE/thorsten/medium", "epoch=3135-step=2702056.ckpt", "de", "CC0-1.0", "Thorsten-Voice (Thorsten Mueller), CC0; via rhasspy/piper-checkpoints.", None),
    "pt-br-faber": ("pt/pt_BR/faber/medium", "epoch=6159-step=1230728.ckpt", "pt", "CC0-1.0", "Faber dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "nl-nl-pim": ("nl/nl_NL/pim/medium", "epoch=5120-step=100504.ckpt", "nl", "CC0-1.0", "Pim dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "ru-ru-denis": ("ru/ru_RU/denis/medium", "epoch=4474-step=1521860.ckpt", "ru", "CC0-1.0", "Denis dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
    "pl-pl-gosia": ("pl/pl_PL/gosia/medium", "epoch=5001-step=1457672.ckpt", "pl", "CC0-1.0", "Gosia dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", None),
}


@app.function(cpu=4, timeout=1800, volumes={"/vol": volume})
def export_one(voice_id: str):
    try:
        return _export_one(voice_id)
    except Exception as e:  # keep the id so the caller can report which voice failed
        return {"id": voice_id, "error": repr(e)[:800]}


def _export_one(voice_id: str):
    import base64, json, os, runpy, shutil, sys, time, urllib.parse, urllib.request

    import numpy as np
    import soundfile as sf
    import torch

    d, ckpt_name, lang, licence, attribution, speaker = VOICES[voice_id]
    work = f"/tmp/{voice_id}"
    os.makedirs(work, exist_ok=True)
    ckpt = f"{work}/m.ckpt"
    for name, dst in ((ckpt_name, ckpt), ("config.json", f"{work}/config.json")):
        urllib.request.urlretrieve(HF + d + "/" + urllib.parse.quote(name), dst)

    orig = torch.load
    torch.load = lambda *a, **kw: orig(*a, **{**kw, "weights_only": False})
    onnx = f"{work}/model.onnx"
    t0 = time.time()
    sys.argv = ["piper.train.export_onnx", "--checkpoint", ckpt, "--output-file", onnx]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    export_s = time.time() - t0
    shutil.copy(f"{work}/config.json", onnx + ".json")
    cfg = json.load(open(onnx + ".json"))
    espeak = cfg.get("espeak", {}).get("voice", "")
    owner = {"public": True, "license": licence, "attribution": attribution,
             "language": d.split("/")[1], "espeak_voice": espeak}
    json.dump(owner, open(f"{work}/owner.json", "w"), indent=2)

    out = f"/vol/{voice_id}"
    os.makedirs(out, exist_ok=True)
    for f in ("model.onnx", "model.onnx.json", "owner.json"):
        shutil.copy(f"{work}/{f}", f"{out}/{f}")
    volume.commit()

    from piper import PiperVoice, SynthesisConfig
    t0 = time.time()
    voice = PiperVoice.load(onnx)
    load_s = time.time() - t0
    sr = voice.config.sample_rate
    kw = {"syn_config": SynthesisConfig(speaker_id=speaker)} if speaker is not None else {}
    texts = TEXTS[lang]
    items = list(enumerate(texts)) if isinstance(texts, list) else list(texts.items())
    list(voice.synthesize("Warm up.", **kw))  # first-call overhead excluded from RTF
    samples = []
    for i, text in items:
        t0 = time.time()
        pcm = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, **kw)])
        st = time.time() - t0
        sec = len(pcm) / sr
        wav = f"{work}/s{i+1}.wav"
        sf.write(wav, pcm, sr, "PCM_16")
        samples.append({"n": i + 1, "text": text, "synth_s": st, "audio_s": sec, "rtf": st / sec,
                        "b64": base64.b64encode(open(wav, "rb").read()).decode()})
    return {"id": voice_id, "sample_rate": sr, "num_speakers": cfg.get("num_speakers", 1), "speaker_id": speaker,
            "espeak_voice": espeak, "export_s": export_s, "load_s": load_s,
            "onnx_mb": os.path.getsize(onnx) / 1e6, "samples": samples}


@app.local_entrypoint()
def main(only: str = ""):
    import base64, json, os

    ids = [v for v in only.split(",") if v] or list(VOICES)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "results.json")
    state = json.load(open(path)) if os.path.exists(path) else {"voices": {}, "failed": {}}
    # results are saved as each voice completes so a dropped connection loses nothing
    for r in export_one.map(ids, order_outputs=False):
        vid = r["id"]
        if "error" in r:
            state["failed"][vid] = r["error"]
            print(f"FAILED {vid}: {r['error']}", flush=True)
        else:
            d = os.path.join(here, "samples", vid)
            os.makedirs(d, exist_ok=True)
            for s_ in r["samples"]:
                open(os.path.join(d, f"sample_{s_['n']}.wav"), "wb").write(base64.b64decode(s_.pop("b64")))
            r["mean_rtf"] = sum(s_["rtf"] for s_ in r["samples"]) / len(r["samples"])
            state["voices"][vid] = r
            state["failed"].pop(vid, None)
            print(f"OK {vid}: sr={r['sample_rate']} spk={r['num_speakers']} onnx={r['onnx_mb']:.0f}MB mean RTF={r['mean_rtf']:.3f}", flush=True)
        json.dump(state, open(path, "w"), indent=1, ensure_ascii=False)
