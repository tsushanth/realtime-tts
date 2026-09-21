"""Export the licence-tiered voice set into the `house-voices` Modal Volume and build voices/catalog.json.

    modal run voices/export_v2.py                      # everything in SPEC
    modal run voices/export_v2.py --only en-gb-cori,fr-fr-mls-f
    python3 voices/export_v2.py --listen               # (local) rebuild voices/listen_<lang>.wav from voices/samples

For every SPEC entry: fetch the ONNX (from the Volume if already exported, else straight from rhasspy/piper-voices),
write /vol/<id>/{model.onnx, model.onnx.json, owner.json}, synthesize five call-centre sentences on CPU, measure
median F0 (pyin) and CPU real-time factor, then merge everything into voices/catalog.json. Multi-speaker models are
exported once per pinned speaker (owner.json speaker_id). Model files are NOT committed; they live in the Volume.

Tiers (details and evidence in voices/LICENSES.md):
  A  clean data licence AND scratch / clean-base lineage
  B  clean data licence but weights descend from Piper's research-only Lessac base (or unknown base): owner risk decision
  C  non-commercial / share-alike / unknown data: never exported here
Nothing here touches production; publishing is voices/publish_voice.py (explicit, tier-gated).
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PV = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

# ----------------------------------------------------------------------------- texts
TEXTS = {
    "en": ["Thanks for calling, I can help you with that. Let me pull up your account details right now.",
           "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
           "I understand your frustration, let me see what I can do to make this right.",
           "Is there anything else I can help you with today?",
           "Your extension is six six three five."],
    "es": ["Gracias por llamar, puedo ayudarle con eso. Permítame consultar los datos de su cuenta ahora mismo.",
           "Su pedido debería llegar en un plazo de tres a cinco días hábiles, y le enviaré un correo de confirmación en breve.",
           "Entiendo su frustración, déjeme ver qué puedo hacer para resolverlo.",
           "¿Hay algo más en lo que pueda ayudarle hoy?",
           "Su extensión es el seis seis tres cinco."],
    "fr": ["Merci de votre appel, je peux vous aider avec cela. Je consulte les détails de votre compte tout de suite.",
           "Votre commande devrait arriver dans un délai de trois à cinq jours ouvrés, et je vous enverrai un e-mail de confirmation sous peu.",
           "Je comprends votre frustration, laissez-moi voir ce que je peux faire pour arranger cela.",
           "Y a-t-il autre chose que je puisse faire pour vous aujourd'hui ?",
           "Votre numéro de poste est le six six trois cinq."],
    "de": ["Vielen Dank für Ihren Anruf, dabei kann ich Ihnen helfen. Ich rufe jetzt gleich Ihre Kontodaten auf.",
           "Ihre Bestellung sollte innerhalb von drei bis fünf Werktagen ankommen, und ich schicke Ihnen in Kürze eine Bestätigung per E-Mail.",
           "Ich verstehe Ihren Ärger, lassen Sie mich sehen, was ich tun kann, um das in Ordnung zu bringen.",
           "Kann ich heute noch etwas für Sie tun?",
           "Ihre Durchwahl lautet sechs sechs drei fünf."],
    "nl": ["Bedankt voor uw telefoontje, daarbij kan ik u helpen. Ik haal uw accountgegevens er meteen bij.",
           "Uw bestelling zou binnen drie tot vijf werkdagen moeten aankomen, en ik stuur u zo een bevestiging per e-mail.",
           "Ik begrijp uw frustratie, laat me kijken wat ik kan doen om dit op te lossen.",
           "Kan ik u vandaag nog ergens anders mee helpen?",
           "Uw toestelnummer is zes zes drie vijf."],
    "it": ["Grazie per aver chiamato, posso aiutarla con questo. Apro subito i dettagli del suo account.",
           "Il suo ordine dovrebbe arrivare entro tre o cinque giorni lavorativi, e le invierò a breve un'e-mail di conferma.",
           "Capisco la sua frustrazione, vediamo cosa posso fare per risolvere la situazione.",
           "Posso aiutarla in qualcos'altro oggi?",
           "Il suo interno è sei sei tre cinque."],
    "pl": ["Dziękuję za telefon, chętnie w tym pomogę. Zaraz sprawdzę dane Państwa konta.",
           "Państwa zamówienie powinno dotrzeć w ciągu trzech do pięciu dni roboczych, a wkrótce wyślę wiadomość z potwierdzeniem.",
           "Rozumiem Państwa frustrację, sprawdzę, co mogę zrobić, aby to naprawić.",
           "Czy mogę dziś jeszcze w czymś pomóc?",
           "Państwa numer wewnętrzny to sześć sześć trzy pięć."],
    "pt": ["Obrigado por ligar, posso ajudá-lo com isso. Deixe-me abrir os dados da sua conta agora mesmo.",
            "Seu pedido deve chegar em três a cinco dias úteis, e enviarei um e-mail de confirmação em breve.",
            "Entendo a sua frustração, deixe-me ver o que posso fazer para resolver isso.",
            "Há mais alguma coisa em que eu possa ajudar hoje?",
            "O seu ramal é seis seis três cinco."],
    "ru": ["Спасибо за звонок, я могу вам с этим помочь. Сейчас я открою данные вашей учётной записи.",
           "Ваш заказ должен прийти в течение трёх-пяти рабочих дней, и я скоро отправлю подтверждение по электронной почте.",
           "Я понимаю ваше недовольство, позвольте посмотреть, что я могу сделать.",
           "Могу ли я вам ещё чем-нибудь помочь сегодня?",
           "Ваш добавочный номер шесть шесть три пять."],
}

# ----------------------------------------------------------------------------- licences / attribution
PD = "Public domain"
MLS_ATTR = ("Multilingual LibriSpeech (Pratap et al., openslr.org/94), CC BY 4.0. Piper voice via rhasspy/piper-checkpoints.")
LIBRIVOX = "Voice trained on LibriVox recordings (public domain) by Bryce Beattie, via rhasspy/piper-voices."
VCTK_ATTR = ("CSTR VCTK Corpus 0.92 (Yamagishi, Veaux, MacDonald; University of Edinburgh, doi:10.7488/ds/2645), CC BY 4.0. "
             "Piper voice via rhasspy/piper-voices.")
ES_ATTR = ("CML-TTS Spanish (Oliveira et al. 2023, openslr.org/146, derived from Multilingual LibriSpeech / LibriVox), CC BY 4.0; base model LibriTTS-R "
           "(openslr.org/141, CC BY 4.0) via rhasspy/piper-checkpoints; finetuned by realtime-tts.")
LESSAC_NOTE = "weights finetuned from Piper's Lessac base (Blizzard 2013 research-only licence): owner risk decision"


def cc0(ds):
    return "CC0-1.0", f"{ds} (CC0 dataset), via rhasspy/piper-voices."


def E(id, src, locale, accent, gender, tier, lic, attr, speaker=None, note="", quality="medium"):
    return dict(id=id, src=src, locale=locale, accent=accent, gender=gender, tier=tier, license=lic,
                attribution=attr, speaker=speaker, note=note, quality=quality)


def vctk(pid, sid, locale, accent, gender, note=""):
    return E(f"{locale.lower().replace('_', '-')}-vctk-{pid}", "pv:en/en_GB/vctk/medium/en_GB-vctk-medium", locale, accent,
             gender, "B", "CC-BY-4.0", VCTK_ATTR, sid,
             f"VCTK speaker {pid} ({accent}); {LESSAC_NOTE}. Phonemes are en-gb-x-rp for every speaker: only the timbre/prosody "
             f"carries the accent. {note}".strip())


# src forms: "vol:<dir in volume>" (already exported earlier) or "pv:<path without extension>" from rhasspy/piper-voices.
SPEC = [
    # ---------------- tier A ----------------
    E("en-us-ljspeech", "vol:en-us-ljspeech", "en_US", "US", "F", "A", PD, "LJ Speech dataset (Keith Ito), public domain; Piper voice by Bryce Beattie / rhasspy.", note="scratch, LJ Speech"),
    E("en-us-kristin", "vol:en-us-kristin", "en_US", "US", "F", "A", PD, "Voice trained on LibriVox recordings (public domain) by Bryce Beattie via rhasspy/piper-checkpoints.", note="scratch"),
    E("en-us-john", "vol:en-us-john", "en_US", "US", "M", "A", PD, "Voice trained on LibriVox recordings (public domain), finetuned from Kristin, by Bryce Beattie via rhasspy/piper-checkpoints.", note="kristin (scratch) -> john"),
    E("en-us-norman", "pv:en/en_US/norman/medium/en_US-norman-medium", "en_US", "US", "M", "A", PD, LIBRIVOX, note="scratch (card); onnx only, no checkpoint published"),
    E("en-gb-cori", "pv:en/en_GB/cori/medium/en_GB-cori-medium", "en_GB", "British (LibriVox reader)", "F", "A", PD, LIBRIVOX, note="scratch 640 epochs (card); espeak voice 'en'"),
    E("fr-fr-mls-f", "vol:fr-fr-mls", "fr_FR", "France", "F", "A", "CC-BY-4.0", MLS_ATTR, 55, "MLS speaker index 55; scratch. Slow audiobook pacing"),
    E("fr-fr-mls-m", "vol:fr-fr-mls", "fr_FR", "France", "M", "A", "CC-BY-4.0", MLS_ATTR, 105, "MLS speaker index 105; scratch. Slow audiobook pacing"),
    E("de-de-mls-f", "vol:de-de-mls", "de_DE", "Germany", "F", "A", "CC-BY-4.0", MLS_ATTR, 36, "MLS speaker index 36; scratch. Very slow audiobook pacing"),
    E("de-de-mls-m", "vol:de-de-mls", "de_DE", "Germany", "M", "A", "CC-BY-4.0", MLS_ATTR, 44, "MLS speaker index 44; scratch. Very slow audiobook pacing"),
    E("nl-nl-mls-f", "vol:nl-nl-mls", "nl_NL", "Netherlands", "F", "A", "CC-BY-4.0", MLS_ATTR, 23, "MLS speaker index 23; scratch"),
    E("nl-nl-mls-m", "vol:nl-nl-mls", "nl_NL", "Netherlands", "M", "A", "CC-BY-4.0", MLS_ATTR, 7, "MLS speaker index 7; scratch"),
    E("nl-be-rdh", "pv:nl/nl_BE/rdh/medium/nl_BE-rdh-medium", "nl_BE", "Flemish", "M", "A", *cc0("r-dh/dutch-vl-tts (github.com/r-dh/dutch-vl-tts)"), note="scratch (card); onnx only"),
    E("nl-nl-alex", "pv:nl/nl_NL/alex/medium/nl_NL-alex-medium", "nl_NL", "Netherlands", "M", "A", *cc0("OHF-Voice/voice-datasets nl_NL alex"), note="finetuned from nl_BE rdh (scratch, CC0) per card only; unverified beyond card"),
    E("it-it-serena", "vol:it-it-serena", "it_IT", "Italy", "F", "A", "CC-BY-4.0", "Trained on committa/serena-synthetic-it-27h (CC BY 4.0, synthetic data from Qwen3-TTS voice cloning; Tatoeba sentences CC BY 2.0 FR) via rhasspy/piper-checkpoints.", note="scratch, SYNTHETIC training data: reference-clip provenance unknown"),
    E("es-es-carlfm-xlow", "pv:es/es_ES/carlfm/x_low/es_ES-carlfm-x_low", "es_ES", "Spain", "M", "A", PD, "carlfm01/my-speech-datasets (public domain per repo README; audio origin not stated), via rhasspy/piper-voices.", note="WEAK: x_low 16 kHz, uploader-asserted public-domain, audio source unstated", quality="x_low"),
    E("es-pilot-f", "vol:_work/es_pilot", "es_ES", "Spanish (accent unlabelled; CML-TTS reader 10246, LibriVox)", "F", "A", "CC-BY-4.0", ES_ATTR, 0, "PILOT, 15.4k steps. 2-speaker finetune from the scratch-trained LibriTTS-R base on CML-TTS speakers 10246 (12 h). Accent (Spain vs Latin America) unverified: audition"),
    E("es-pilot-m", "vol:_work/es_pilot", "es_ES", "Spanish (accent unlabelled; CML-TTS reader 3946, LibriVox)", "M", "A", "CC-BY-4.0", ES_ATTR, 1, "PILOT, 15.4k steps. Same model, CML-TTS speaker 3946 (12 h). Accent unverified: audition"),
    # ---------------- tier B: clean data licence, Lessac lineage (or unknown base) ----------------
    E("en-us-joe", "vol:en-us-joe", "en_US", "US", "M", "B", "CC0-1.0", "Joe dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("en-us-mike", "vol:en-us-mike", "en_US", "US", "M", "B", "CC0-1.0", "Mike dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE + "; CC0 claim from card only"),
    E("en-us-sam", "vol:en-us-sam", "en_US", "US", "NB", "B", "Apache-2.0", "Sam Accenture non-binary voice dataset, Apache-2.0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE + "; source repo 404, licence unverifiable"),
    E("en-gb-alba", "pv:en/en_GB/alba/medium/en_GB-alba-medium", "en_GB", "Scottish", "F", "B", "CC-BY-4.0", "Alba speech corpus (CSTR, University of Edinburgh, datashare.ed.ac.uk/handle/10283/3270), CC BY 4.0; via rhasspy/piper-voices.", note=LESSAC_NOTE + "; dataset adds a moral-rights clause: no use derogatory to the voice talent"),
    E("en-gb-jenny_dioco", "pv:en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium", "en_GB", "Irish (Dioco 'Jenny')", "F", "B", "Custom (Dioco Jenny licence, commercial use allowed with attribution)", "Voice 'Jenny (Dioco)': attribution REQUIRED in software/websites generating audio with it (github.com/dioco-group/jenny-tts-dataset).", note=LESSAC_NOTE + " (card says Lessac; author page says scratch: conflicting); mandatory attribution as 'Jenny (Dioco)'"),
    vctk("p229", 85, "en_GB", "English (Southern England)", "F"),
    vctk("p232", 60, "en_GB", "English (Southern England)", "M"),
    vctk("p285", 70, "en_GB", "Scottish (Edinburgh)", "M"),
    vctk("p288", 58, "en_IE", "Irish (Dublin)", "F"),
    vctk("p245", 97, "en_IE", "Irish (Dublin)", "M"),
    vctk("p376", 26, "en_IN", "Indian", "M"),
    vctk("p251", 92, "en_IN", "Indian", "M"),
    vctk("p248", 87, "en_IN", "Indian", "F", "Only Indian female in VCTK; low naturalness proxy (UTMOS 2.7 vs ~3.7 median): audition before use."),
    vctk("p326", 71, "en_AU", "Australian (Sydney)", "M", "Very low pitch (F0 ~80 Hz)."),
    vctk("p374", 25, "en_AU", "Australian", "M"),
    vctk("p307", 49, "en_CA", "Canadian (Ontario)", "F"),
    vctk("p312", 53, "en_CA", "Canadian (Hamilton)", "F"),
    vctk("p316", 47, "en_CA", "Canadian (Alberta)", "M"),
    vctk("p335", 42, "en_NZ", "New Zealand", "F"),
    vctk("p314", 35, "en_ZA", "South African (Cape Town)", "F"),
    vctk("p347", 32, "en_ZA", "South African (Johannesburg)", "M"),
    E("es-es-davefx", "vol:es-es-davefx", "es_ES", "Spain", "M", "B", "CC0-1.0", "DaveFX dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("es-mx-ald", "vol:es-mx-ald", "es_MX", "Mexico", "F", "B", "Unlicense", "Ald Mexican Spanish speech dataset (rmcpantoja), Unlicense; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE + " via davefx"),
    E("es-es-sharvard-f", "pv:es/es_ES/sharvard/medium/es_ES-sharvard-medium", "es_ES", "Spain", "F", "B", "CC-BY-3.0", "Sharvard corpus (Aubanel et al., University of Edinburgh, datashare.ed.ac.uk/handle/10283/574), CC BY 3.0; via rhasspy/piper-voices.", "F", LESSAC_NOTE + "; read speech of Harvard-style sentences"),
    E("es-es-sharvard-m", "pv:es/es_ES/sharvard/medium/es_ES-sharvard-medium", "es_ES", "Spain", "M", "B", "CC-BY-3.0", "Sharvard corpus (Aubanel et al., University of Edinburgh), CC BY 3.0; via rhasspy/piper-voices.", "M", LESSAC_NOTE),
    E("fr-fr-siwis", "vol:fr-fr-siwis", "fr_FR", "France", "F", "B", "CC-BY-4.0", "SIWIS French Speech Synthesis Database, University of Edinburgh, CC BY 4.0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("pt-br-faber", "vol:pt-br-faber", "pt_BR", "Brazil", "M", "B", "CC0-1.0", "Faber dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("pt-br-cadu", "pv:pt/pt_BR/cadu/medium/pt_BR-cadu-medium", "pt_BR", "Brazil", "M", "B", *cc0("OHF-Voice/voice-datasets pt_BR cadu"), note=LESSAC_NOTE),
    E("pt-br-jeff", "pv:pt/pt_BR/jeff/medium/pt_BR-jeff-medium", "pt_BR", "Brazil", "M", "B", *cc0("OHF-Voice/voice-datasets pt_BR jeff"), note=LESSAC_NOTE),
    E("pl-pl-gosia", "vol:pl-pl-gosia", "pl_PL", "Poland", "F", "B", "CC0-1.0", "Gosia dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("pl-pl-darkman", "pv:pl/pl_PL/darkman/medium/pl_PL-darkman-medium", "pl_PL", "Poland", "M", "B", *cc0("OHF-Voice/voice-datasets pl_PL darkman"), note=LESSAC_NOTE),
    E("pl-pl-mc_speech", "pv:pl/pl_PL/mc_speech/medium/pl_PL-mc_speech-medium", "pl_PL", "Poland", "M", "B", "CC0-1.0", "The MC Speech Dataset (Kaggle, czyzi0), CC0 per card; via rhasspy/piper-voices.", note=LESSAC_NOTE + "; Kaggle page not verifiable"),
    E("nl-nl-pim", "vol:nl-nl-pim", "nl_NL", "Netherlands", "M", "B", "CC0-1.0", "Pim dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE),
    E("nl-nl-ronnie", "pv:nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium", "nl_NL", "Netherlands", "M", "B", *cc0("OHF-Voice/voice-datasets nl_NL ronnie"), note=LESSAC_NOTE),
    E("it-it-paola", "pv:it/it_IT/paola/medium/it_IT-paola-medium", "it_IT", "Italy", "F", "B", *cc0("paolapersico1/Voice-Dataset-Italian"), note=LESSAC_NOTE),
    E("de-de-thorsten", "vol:de-de-thorsten", "de_DE", "Germany", "M", "B", "CC0-1.0", "Thorsten-Voice (Thorsten Mueller), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE + "; best-sounding German voice"),
    E("ru-ru-denis", "vol:ru-ru-denis", "ru_RU", "Russia", "M", "B", "CC0-1.0", "Denis dataset (OHF-Voice/voice-datasets), CC0; via rhasspy/piper-checkpoints.", note=LESSAC_NOTE + " (Russian is outside the current priority list)"),
]
# libritts_r speakers are appended by voices/libritts_picks.json (written after the survey)
_p = os.path.join(HERE, "libritts_picks.json")
if os.path.exists(_p):
    for pk in json.load(open(_p)):
        SPEC.append(E(pk["id"], "vol:_work/libritts_r", "en_US", "US (LibriTTS-R reader, accent unlabelled)", pk["gender"], "A", "CC-BY-4.0",
                      "LibriTTS-R (Koizumi et al., openslr.org/141), CC BY 4.0. Piper voice via rhasspy/piper-checkpoints (best.ckpt, scratch).",
                      pk["speaker_id"], f"LibriTTS-R multi-speaker scratch model, speaker index {pk['speaker_id']}"))

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("piper-tts==1.8.0", "numpy<2", "librosa", "soundfile"))
app = modal.App("house-voices-export-v2", image=image)
volume = modal.Volume.from_name("house-voices", create_if_missing=True)


@app.function(cpu=4, timeout=3000, volumes={"/vol": volume})
def export_group(src: str, entries: list):
    """One source model, several catalog entries (pinned speakers)."""
    try:
        return _export_group(src, entries)
    except Exception as e:  # noqa: BLE001
        return [{"id": x["id"], "error": repr(e)[:600]} for x in entries]


def _export_group(src, entries):
    import base64, shutil, time, urllib.request
    import librosa, numpy as np, soundfile as sf
    from piper import PiperVoice, SynthesisConfig

    kind, path = src.split(":", 1)
    work = "/tmp/src"
    os.makedirs(work, exist_ok=True)
    if kind == "vol":
        shutil.copy(f"/vol/{path}/model.onnx", f"{work}/model.onnx")
        shutil.copy(f"/vol/{path}/model.onnx.json", f"{work}/model.onnx.json")
    else:
        for ext in (".onnx", ".onnx.json"):
            urllib.request.urlretrieve(PV + path + ext, f"{work}/model{ext}")
    cfg = json.load(open(f"{work}/model.onnx.json"))
    espeak = cfg.get("espeak", {}).get("voice", "")
    smap = cfg.get("speaker_id_map", {})
    voice = PiperVoice.load(f"{work}/model.onnx")
    sr = voice.config.sample_rate
    out = []
    for e in entries:
        spk = e["speaker"]
        if isinstance(spk, str):  # speaker given by name (sharvard "M"/"F")
            spk = smap[spk]
        lang = e["locale"].split("_")[0]
        d = f"/vol/{e['id']}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(f"{work}/model.onnx", f"{d}/model.onnx")
        # Known upstream bug in rhasspy/piper-checkpoints: some shipped config.json files have the wrong
        # espeak.voice/language (e.g. de/de_DE/mls/medium says "nl"/Dutch despite being German MLS data
        # and a correctly-trained model) - found via this repo's eval framework (garbled output, high WER
        # on de-de-mls-f/m) and confirmed empirically: overriding espeak.voice to match the voice's own
        # intended locale gives correct output, so only the shipped metadata was wrong. Apply the same
        # per-voice override here (mirrors export_voices.py's _export_one fix) since this script copies
        # model.onnx.json independently rather than reading the already-fixed base from the volume.
        voice_espeak = espeak
        if espeak != lang:
            print(f"WARNING: {e['id']}: source config espeak.voice={espeak!r} != expected {lang!r} - "
                  f"overriding to {lang!r}. Re-verify by ear/WER before trusting this for a NEW voice.")
            voice_espeak = lang
        cfg_out = json.load(open(f"{work}/model.onnx.json"))
        cfg_out["espeak"]["voice"] = voice_espeak
        json.dump(cfg_out, open(f"{d}/model.onnx.json", "w"))
        owner = {"public": True, "license": e["license"], "attribution": e["attribution"], "language": e["locale"],
                 "espeak_voice": voice_espeak, "tier": e["tier"]}
        if spk is not None:
            owner["speaker_id"] = int(spk)
        json.dump(owner, open(f"{d}/owner.json", "w"), indent=2, ensure_ascii=False)
        kw = {"syn_config": SynthesisConfig(speaker_id=int(spk))} if spk is not None else {}
        list(voice.synthesize("Warm up.", **kw))
        samples, f0s, tot_syn, tot_aud = [], [], 0.0, 0.0
        for i, text in enumerate(TEXTS[lang]):
            t0 = time.time()
            pcm = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, **kw)])
            st = time.time() - t0
            tot_syn += st
            tot_aud += len(pcm) / sr
            if i < 2:
                f0, _, _ = librosa.pyin(pcm.astype(np.float32) / 32768, fmin=60, fmax=450, sr=sr)
                f0s.extend(f0[~np.isnan(f0)].tolist())
            wav = f"/tmp/{e['id']}_{i + 1}.wav"
            sf.write(wav, pcm, sr, "PCM_16")
            keep = (1, 2, 3, 4, 5) if e["tier"] == "A" else (1, 4)  # keep results (and the repo) small for tier B
            if i + 1 in keep:
                samples.append({"n": i + 1, "text": text, "b64": base64.b64encode(open(wav, "rb").read()).decode()})
        out.append({"id": e["id"], "sample_rate": sr, "num_speakers": cfg.get("num_speakers", 1), "espeak_voice": espeak,
                    "speaker_id": None if spk is None else int(spk), "onnx_mb": os.path.getsize(f"{work}/model.onnx") / 1e6,
                    "cpu_rtf": tot_syn / tot_aud, "median_f0": float(np.median(f0s)) if f0s else None, "samples": samples})
    volume.commit()
    return out


@app.local_entrypoint()
def main(only: str = ""):
    import base64
    ids = {v for v in only.split(",") if v}
    todo = [e for e in SPEC if not ids or e["id"] in ids]
    groups = {}
    for e in todo:
        groups.setdefault(e["src"], []).append(e)
    path = os.path.join(HERE, "catalog.json")
    cat = {v["id"]: v for v in json.load(open(path))["voices"]} if os.path.exists(path) else {}
    by_id = {e["id"]: e for e in SPEC}
    failed = {}
    chunks = [(s, es[i:i + 3]) for s, es in groups.items() for i in range(0, len(es), 3)]  # short calls: long ones hit gRPC deadlines
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def call(c):
        for attempt in range(3):
            try:
                return export_group.remote(*c)
            except Exception as e:  # noqa: BLE001
                print("retry", c[1][0]["id"], type(e).__name__, flush=True)
        return [{"id": x["id"], "error": "client errors"} for x in c[1]]
    def results():
        with ThreadPoolExecutor(6) as ex:
            for f in as_completed([ex.submit(call, c) for c in chunks]):
                yield f.result()
    for res in results():
        for r in res:
            if "error" in r:
                failed[r["id"]] = r["error"]
                print("FAILED", r["id"], r["error"], flush=True)
                continue
            e = by_id[r["id"]]
            d = os.path.join(HERE, "samples", e["id"])
            os.makedirs(d, exist_ok=True)
            for s in r["samples"]:
                if True:
                    open(os.path.join(d, f"sample_{s['n']}.wav"), "wb").write(base64.b64decode(s["b64"]))
            cat[e["id"]] = {
                "id": e["id"], "language": e["locale"], "accent": e["accent"], "gender": e["gender"],
                "speaker_id": r["speaker_id"], "tier": e["tier"], "quality": e["quality"], "license": e["license"],
                "attribution": e["attribution"], "sample": f"voices/samples/{e['id']}/sample_1.wav",
                "median_f0_hz": round(r["median_f0"], 1) if r["median_f0"] else None,
                "cpu_rtf": round(r["cpu_rtf"], 3), "sample_rate": r["sample_rate"], "espeak_voice": r["espeak_voice"],
                "onnx_mb": round(r["onnx_mb"], 1), "volume_path": f"house-voices:/{e['id']}", "notes": e["note"]}
            print("OK", e["id"], f"rtf={r['cpu_rtf']:.3f} f0={r['median_f0']}", flush=True)
            json.dump({"note": "cpu_rtf measured on a Modal 4-vCPU container (piper-tts 1.8.0, default ORT threads); "
                       "median_f0 via librosa pyin on the first two sample sentences; see LICENSES.md for tier meanings.",
                       "voices": sorted(cat.values(), key=lambda v: (v["tier"], v["language"], v["id"]))},
                      open(path, "w"), indent=1, ensure_ascii=False)
    if failed:
        print("FAILED:", failed)


def build_listen():
    """Local: concatenate sample_1 of each voice per language (tier order) into listen_<lang>.wav + listen_<lang>.txt index."""
    import wave
    import numpy as np
    cat = json.load(open(os.path.join(HERE, "catalog.json")))["voices"]
    langs = sorted({v["language"].split("_")[0] for v in cat})
    for lang in langs:
        vs = sorted([v for v in cat if v["language"].split("_")[0] == lang], key=lambda v: (v["tier"], v["language"], v["id"]))
        chunks, idx, t, sr0 = [], [], 0.0, 22050
        for v in vs:
            p = os.path.join(HERE, "samples", v["id"], "sample_1.wav")
            if not os.path.exists(p):
                continue
            w = wave.open(p)
            sr, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if sr != sr0:  # x_low / 16k voices: linear resample to the common rate
                x = np.arange(int(len(pcm) * sr0 / sr)) * sr / sr0
                pcm = np.interp(x, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)
            idx.append(f"{t:7.1f}s  tier {v['tier']}  {v['id']}  ({v['language']}, {v['accent']}, {v['gender']}"
                       f"{'' if v['speaker_id'] is None else ', spk ' + str(v['speaker_id'])})")
            chunks += [pcm, np.zeros(int(sr0 * 1.2), np.int16)]
            t += (len(pcm) + int(sr0 * 1.2)) / sr0
        if not chunks:
            continue
        with wave.open(os.path.join(HERE, f"listen_{lang}.wav"), "wb") as o:
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr0)
            o.writeframes(np.concatenate(chunks).tobytes())
        open(os.path.join(HERE, f"listen_{lang}.txt"), "w").write("\n".join(idx) + "\n")
        print("listen", lang, len(idx), "voices", f"{t:.0f}s")


if __name__ == "__main__":
    import sys
    if "--listen" in sys.argv:
        build_listen()
