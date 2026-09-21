"""Intelligibility proxy for the Spanish pilot: transcribe generated samples with faster-whisper (small, CPU, Spanish) and
report word error rate vs the intended text. A proxy only: it says the speech is intelligible Spanish, not that it sounds natural.

    modal run voices/es_pilot/asr_check.py --dir runs/es2spk/samples_4160
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("faster-whisper==1.0.3", "requests", "huggingface_hub", "soundfile", "numpy<2")
app = modal.App("es-pilot-asr", image=image)
vol = modal.Volume.from_name("es-pilot")
TEXTS = [
    "Gracias por llamar, puedo ayudarle con eso. Permítame consultar los datos de su cuenta ahora mismo.",
    "Su pedido debería llegar en un plazo de tres a cinco días hábiles, y le enviaré un correo de confirmación en breve.",
    "Entiendo su frustración, déjeme ver qué puedo hacer para resolverlo.",
    "¿Hay algo más en lo que pueda ayudarle hoy?",
    "Su extensión es el seis seis tres cinco.",
]


@app.function(cpu=4, memory=8192, timeout=1800, volumes={"/data": vol})
def check(d: str):
    import glob, os, re, unicodedata
    from faster_whisper import WhisperModel
    vol.reload()
    m = WhisperModel("small", device="cpu", compute_type="int8")
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()).split()
    def wer(r, h):
        d = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            p, d[0] = d[0], i
            for j, hw in enumerate(h, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (rw != hw))
        return d[-1] / max(len(r), 1)
    out = []
    for f in sorted(glob.glob(f"/data/{d}/*.wav")):
        i = int(os.path.basename(f).split("_")[-1].split(".")[0]) - 1
        segs, _ = m.transcribe(f, language="es", beam_size=5)
        hyp = " ".join(s.text for s in segs)
        out.append({"file": os.path.basename(f), "wer": wer(norm(TEXTS[i]), norm(hyp)), "hyp": hyp.strip()})
    return out


@app.local_entrypoint()
def main(dir: str):
    r = check.remote(dir)
    for x in r:
        print(f"{x['wer']:.2f}  {x['file']}  {x['hyp'][:110]}")
    print("mean WER", sum(x["wer"] for x in r) / len(r))
