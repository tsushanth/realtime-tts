"""Seam-fix listening test for Kokoro's word-boundary first chunk.
Per sentence renders: old (previous chunking), new (current), T (trim silence + crossfade),
C (T + a comma ending chunk 1), E (T + an ellipsis ending chunk 1). Each variant synthesizes the chunks
separately (as a client hears consecutive frames) and joins them.
modal run listen_seam_fix.py -> kokoro_seam_samples/ + kokoro_seam_compare.wav"""
import modal

import re

from app import chunk_text, image as _base_image  # same image and chunker the production worker uses

image = _base_image.add_local_python_source("app")  # the container re-imports this file, which imports app

SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
    "I can see your account and the recent order you mentioned, and it shipped on Tuesday.",
    "Unfortunately the item you ordered is currently out of stock at that location.",
]
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def old_chunks(text, max_chars=90, first_chunk_max_chars=35):
    """chunk_text as it was before the word-boundary change (comma split only)."""
    text = text.strip()
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()] or [text]
    chunks, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf); buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf)
    if chunks and len(chunks[0]) > first_chunk_max_chars:
        sub = [p.strip() for p in re.split(r"(?<=[,])\s+", chunks[0]) if p.strip()]
        if len(sub) > 1:
            chunks = [sub[0], " ".join(sub[1:])] + chunks[1:]
    return chunks

app = modal.App("kokoro-seam-listen", image=image)


@app.function(gpu="T4", timeout=900)
def render(jobs):
    import numpy as np
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    next(pipe("Warm up.", voice="af_heart"))

    def synth(text):
        a = next(pipe(text, voice="af_heart", speed=1.0))[2]
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        return a.astype("float32").reshape(-1)

    def trim(a, thr=0.01, keep=int(24000 * 0.01)):
        idx = np.where(np.abs(a) > thr)[0]
        if len(idx) == 0:
            return a
        return a[max(0, idx[0] - keep): idx[-1] + 1 + keep]

    def join(parts, cross_ms=20):
        n = int(24000 * cross_ms / 1000)
        out = parts[0]
        for p in parts[1:]:
            if len(out) < n or len(p) < n:
                out = np.concatenate([out, p]); continue
            fade = np.linspace(0, 1, n, dtype="float32")
            mix = out[-n:] * np.sqrt(1 - fade) + p[:n] * np.sqrt(fade)
            out = np.concatenate([out[:-n], mix, p[n:]])
        return out

    res = []
    for name, mode, chunks in jobs:
        if mode == "plain":
            audio = np.concatenate([synth(c) for c in chunks])
        else:
            cs = list(chunks)
            if len(cs) > 1:
                if mode == "comma":
                    cs[0] = cs[0].rstrip(".,;:!? ") + ","
                elif mode == "ellipsis":
                    cs[0] = cs[0].rstrip(".,;:!? ") + "..."
            audio = join([trim(synth(c)) for c in cs])
        res.append((name, (np.clip(audio, -1, 1) * 32767).astype("int16").tobytes()))
    return res


@app.local_entrypoint()
def main():
    import os, wave
    jobs, keep = [], []
    for i, s in enumerate(SENTENCES):
        o, n = old_chunks(s), chunk_text(s)
        if o == n:
            continue
        keep.append(i)
        jobs += [(f"old_{i}", "plain", o), (f"new_{i}", "plain", n), (f"T_{i}", "trim", n), (f"C_{i}", "comma", n), (f"E_{i}", "ellipsis", n)]
    res = dict(render.remote(jobs))
    os.makedirs("kokoro_seam_samples", exist_ok=True)
    def wav(path, pcm):
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    gap, combo = b"\x00\x00" * 20000, b""
    for i in keep:
        for v in ("old", "new", "T", "C", "E"):
            wav(f"kokoro_seam_samples/{v}_{i}.wav", res[f"{v}_{i}"]); combo += res[f"{v}_{i}"] + gap
        combo += gap * 2
    wav("kokoro_seam_compare.wav", combo)
    print("sentences:", keep, " order per sentence: old, new, T(trim+crossfade), C(+comma), E(+ellipsis)")
