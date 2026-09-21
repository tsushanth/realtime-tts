"""Compare speakers of a multi-speaker Piper voice on a fixed sentence (CPU, cheap).

    modal run voices/speaker_survey.py --voice vctk            # 109 speakers
    modal run voices/speaker_survey.py --voice de-de-mls       # all speakers of a house voice
Per speaker: median F0 (librosa pyin), UTMOS22 naturalness proxy (English-trained MOS predictor: only a rough
cross-language proxy!), clipping, duration, CPU RTF. WAVs go to the house-voices Volume under _survey/<voice>/spkN.wav
and the metrics to voices/survey_<voice>.json. Listening proxies only; final picks should be auditioned by ear.
"""
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("piper-tts==1.8.0", "numpy<2", "librosa", "soundfile", "requests")
         .pip_install("torch==2.1.2", "torchaudio==2.1.2", extra_index_url="https://download.pytorch.org/whl/cpu"))
app = modal.App("house-voices-survey", image=image)
vol = modal.Volume.from_name("house-voices")

PV = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
SOURCES = {  # name -> (onnx source, sentence)
    "vctk": (PV + "en/en_GB/vctk/medium/en_GB-vctk-medium.onnx", "Thanks for calling, I can help you with that. Let me pull up your account details right now. Is there anything else I can help you with today?"),
    "de-de-mls": ("vol", "Vielen Dank für Ihren Anruf, dabei kann ich Ihnen helfen. Ich rufe jetzt gleich Ihre Kontodaten auf. Kann ich heute noch etwas für Sie tun?"),
    "fr-fr-mls": ("vol", "Merci de votre appel, je peux vous aider avec cela. Je consulte les détails de votre compte tout de suite. Y a-t-il autre chose que je puisse faire pour vous aujourd'hui ?"),
    "en-us-libritts_r": ("vol:_work/libritts_r", "Thanks for calling, I can help you with that. Let me pull up your account details right now. Is there anything else I can help you with today?"),
    "nl-nl-mls": ("vol", "Bedankt voor uw telefoontje, daarbij kan ik u helpen. Ik haal uw accountgegevens er meteen bij. Kan ik u vandaag nog ergens anders mee helpen?"),
}


@app.function(cpu=4, timeout=7200, volumes={"/vol": vol})
def survey(voice: str, speakers: list = None):
    import json, os, time, urllib.request
    import librosa, numpy as np, soundfile as sf, torch
    from piper import PiperVoice, SynthesisConfig
    src, text = SOURCES[voice]
    os.makedirs("/tmp/m", exist_ok=True)
    if src == "vol" or src.startswith("vol:"):
        onnx = f"/vol/{src[4:] if src.startswith('vol:') else voice}/model.onnx"
    else:
        onnx = "/tmp/m/model.onnx"
        urllib.request.urlretrieve(src, onnx); urllib.request.urlretrieve(src + ".json", onnx + ".json")
    v = PiperVoice.load(onnx)
    n = v.config.num_speakers
    sr = v.config.sample_rate
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    out = f"/vol/_survey/{voice}"; os.makedirs(out, exist_ok=True)
    res = []
    for s in (speakers if speakers is not None else range(n)):
        cfg = SynthesisConfig(speaker_id=int(s), noise_scale=0.667, noise_w_scale=0.8)
        t0 = time.time()
        pcm = np.concatenate([c.audio_float_array for c in v.synthesize(text, cfg)])
        dt = time.time() - t0
        dur = len(pcm) / sr
        f0, vf, _ = librosa.pyin(pcm, fmin=60, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        m = float(utmos(torch.from_numpy(pcm).unsqueeze(0).float(), sr).item())
        sf.write(f"{out}/spk{s}.wav", pcm, sr)
        res.append(dict(speaker_id=int(s), median_f0=float(np.median(f0)) if len(f0) else None, utmos=m,
                        peak=float(np.abs(pcm).max()), dur_s=dur, rtf=dt / dur))
    vol.commit()
    return dict(voice=voice, num_speakers=n, sample_rate=sr, text=text, speakers=res)


@app.local_entrypoint()
def main(voice: str, speakers: str = ""):
    import json, os
    sp = [int(x) for x in speakers.split(",") if x] or None
    r = survey.remote(voice, sp)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"survey_{voice}.json")
    json.dump(r, open(p, "w"), indent=1, ensure_ascii=False)
    top = sorted(r["speakers"], key=lambda x: -x["utmos"])[:8]
    print("wrote", p, "n=", len(r["speakers"]))
    for t in top: print(t)
