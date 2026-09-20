"""Model-only time-to-first-audio, with and without FIRST_CHUNK_SPLIT, plus listening samples.

Run inside the project's image so onnxruntime/piper match production:
  docker run --rm -e SESSION_SECRET=x -e ORT_INTRA_THREADS=4 \
      -v $PWD/bench_first_chunk.py:/app/bench_first_chunk.py -v $PWD/split_samples:/out \
      piper-tts-feat python /app/bench_first_chunk.py --out /out --runs 20

Per sentence and mode (off/on) it reports the median/p95 wall time of everything the server does
before the first audio bytes exist: phonemize + first-chunk inference + resample/encode
(`first_ms`), and the inference alone (`model_ms`). It also writes <out>/{off,on}/NN.wav: the
whole sentence exactly as a client would hear it (24 kHz mono, chunks concatenated as sent, the
seam included) so a human can listen for audible splices. The model is stochastic (VITS noise),
so off/on WAVs differ a little even where no split happens.
NOTE: absolute numbers depend on the host CPU; compare off vs on, not against Fly.
"""
import argparse
import os
import statistics
import time
import wave

import numpy as np

import server

SENTENCES = [
    "Thank you for calling Brightside Insurance, my name is Sarah and I'll be happy to help you with your claim today.",
    "I understand you're having trouble with your last bill, so let me pull up your account and take a look at the charges.",
    "Before we go any further, could you please confirm the phone number and the email address on your account?",
    "Good news, your replacement card has already shipped and it should arrive within three to five business days.",
    "Yes, of course, I can reschedule your appointment for Thursday, and I'll send a confirmation text right away.",
    "I'm sorry about the wait, we're seeing higher call volumes than usual this afternoon; a technician will call you back within the hour.",
]


def timed_first_chunk(eng, text, split):
    t0 = time.perf_counter()
    chunks = eng.sentences(text, split_first=split)
    t1 = time.perf_counter()
    eng.synth(chunks[0], 1.0)
    t2 = time.perf_counter()
    return (t2 - t0) * 1000, len(chunks)


def model_only(eng, text, split):
    ids = eng.sentences(text, split_first=split)[0]
    cfg = server.SynthesisConfig(length_scale=eng.voice.config.length_scale)
    t0 = time.perf_counter()
    eng.voice.phoneme_ids_to_audio(ids, cfg)
    return (time.perf_counter() - t0) * 1000


def write_wav(path, pcm_chunks):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"".join(pcm_chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default=None, help="write off/ and on/ WAVs here")
    a = ap.parse_args()
    eng = server.engine
    print(f"{'#':>2} {'words':>5} {'mode':>4} {'chunks':>6} {'first_ms med':>12} {'p95':>7} {'model_ms med':>12}  first-chunk words")
    summary = {"off": [], "on": []}
    for n, text in enumerate(SENTENCES, 1):
        for mode, split in (("off", False), ("on", True)):
            firsts, models = [], []
            for _ in range(a.runs):
                f, nch = timed_first_chunk(eng, text, split)
                firsts.append(f)
                models.append(model_only(eng, text, split))
            ids = eng.sentences(text, split_first=split)
            # words in the first chunk = spaces in its phoneme ids (id of " ")
            space = eng.voice.config.phoneme_id_map[" "][0]
            words = 1 + sum(1 for i in ids[0] if i == space)
            med, p95 = statistics.median(firsts), sorted(firsts)[int(0.95 * (len(firsts) - 1))]
            summary[mode].append(med)
            print(f"{n:>2} {len(text.split()):>5} {mode:>4} {nch:>6} {med:>12.1f} {p95:>7.1f} {statistics.median(models):>12.1f}  {words}")
            if a.out:
                os.makedirs(os.path.join(a.out, mode), exist_ok=True)
                pcm = [eng.synth(c, 1.0)[0] for c in ids]
                if len(pcm) > 1:  # seam diagnostics: loudness step and edge levels around the splice
                    x = [np.frombuffer(p, dtype=np.int16).astype(np.float64) for p in pcm]
                    db = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-9)
                    e = 1200  # 50 ms
                    print(f"     seam: chunk RMS {db(x[0]):.1f} / {db(x[1]):.1f} dBFS-ish (step {db(x[1]) - db(x[0]):+.1f} dB); "
                          f"end of chunk1 {db(x[0][-e:]):.1f}, start of chunk2 {db(x[1][:e]):.1f}; "
                          f"chunk1 {len(x[0]) / 24000:.2f}s + chunk2 {len(x[1]) / 24000:.2f}s")
                write_wav(os.path.join(a.out, mode, f"{n:02d}.wav"), pcm)
    for m in ("off", "on"):
        print(f"mean of per-sentence median first_ms [{m}]: {statistics.mean(summary[m]):.1f}")


if __name__ == "__main__":
    main()
