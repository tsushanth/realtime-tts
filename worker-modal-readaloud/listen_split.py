"""Listening test for the Kokoro first-chunk change: renders each sentence with the OLD chunking
(comma split only) and the NEW chunking (plus word-boundary first chunk), each chunk synthesized
separately and concatenated exactly as a client hears consecutive frames.
modal run listen_split.py  ->  kokoro_split_samples/{old,new}_N.wav + kokoro_split_compare.wav"""
import re

import modal

from app import chunk_text, image as _base_image  # same image and chunker the production worker uses

image = _base_image.add_local_python_source("app")  # the container re-imports this file, which imports app

app = modal.App("kokoro-split-listen", image=image)

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


@app.function(gpu="T4", timeout=900)
def render(jobs):
    import numpy as np
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    next(pipe("Warm up.", voice="af_heart"))
    out = []
    for name, chunks in jobs:
        parts = []
        for c in chunks:
            a = next(pipe(c, voice="af_heart", speed=1.0))[2]
            a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            parts.append(a.astype("float32").reshape(-1))
        audio = np.concatenate(parts)
        out.append((name, (np.clip(audio, -1, 1) * 32767).astype("int16").tobytes()))
    return out


@app.local_entrypoint()
def main():
    import os, wave
    jobs, plan = [], []
    for i, s in enumerate(SENTENCES):
        o, n = old_chunks(s), chunk_text(s)
        print(i, "old", o, "| new", n, "(same)" if o == n else "(DIFFERENT)")
        jobs += [(f"old_{i}", o), (f"new_{i}", n)]
    res = dict(render.remote(jobs))
    os.makedirs("kokoro_split_samples", exist_ok=True)
    def wav(path, pcm):
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
    gap = b"\x00\x00" * 24000
    combo = b""
    for i in range(len(SENTENCES)):
        wav(f"kokoro_split_samples/old_{i}.wav", res[f"old_{i}"]); wav(f"kokoro_split_samples/new_{i}.wav", res[f"new_{i}"])
        combo += res[f"old_{i}"] + gap + res[f"new_{i}"] + gap * 2
    wav("kokoro_split_compare.wav", combo)
