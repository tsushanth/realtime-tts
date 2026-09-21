"""Call-center style turns via macOS `say` (stand-in for the Piper endpoint, which needs a gateway session token we do not
hold). References are the spoken form. Half the turns contain a deliberate mid-utterance hesitation pause (700-900 ms)
where a real caller reads out an order number / spells something: that is what tests premature endpointing.
Writes data/call/*.wav (16k clean), data/calltel/*.wav (8k mu-law round trip -> 16k) + manifest.json in each."""
import json, os, random, subprocess, numpy as np, soundfile as sf
from scipy.signal import resample_poly
random.seed(1)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
W = "zero one two three four five six seven eight nine".split()
names = ["Priya Raman", "John Smith", "Maria Gonzalez", "Wei Chen", "Olivia Brown", "Ahmed Khan", "Sarah O'Connor", "Michael Johnson"]
def digs(n): return [random.randrange(10) for _ in range(n)]
items = []
for i in range(48):
    kind = i % 6; pause = i % 2 == 0
    ds = digs(random.choice([4, 6, 7])); sp = " ".join(W[d] for d in ds)
    nm = random.choice(names)
    if kind == 0: parts = ["my order number is", sp]
    elif kind == 1: parts = ["yes this is", nm, "calling about my account"]
    elif kind == 2: parts = ["the confirmation code is", " ".join(W[d] for d in digs(5)), "thank you"]
    elif kind == 3: parts = ["i would like to change my delivery address to", str(random.randrange(10, 999)) + " Oak Street"]
    elif kind == 4: parts = ["yes", ]
    else: parts = ["no thanks that is all i needed today"]
    if kind == 3: parts[1] = " ".join(W[int(c)] for c in str(random.randrange(10, 999))) + " oak street"
    ref = " ".join(parts)
    say = (" [[slnc 800]] ".join(parts)) if (pause and len(parts) > 1) else " ".join(parts)
    items.append((say, ref, pause and len(parts) > 1))
os.makedirs(f"{D}/call", exist_ok=True); os.makedirs(f"{D}/calltel", exist_ok=True)
def mulaw_rt(x):
    mu = 255.0; y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
man = []
for i, (say, ref, p) in enumerate(items):
    aiff = f"/private/tmp/call_{i}.wav"
    subprocess.run(["say", "-v", "Samantha", "-r", "170", "-o", aiff, "--data-format=LEI16@16000", say], check=True)
    x, sr = sf.read(aiff, dtype="float32"); assert sr == 16000
    x = x / max(1e-6, np.abs(x).max()) * 0.5
    x8 = mulaw_rt(np.clip(resample_poly(x, 1, 2), -1, 1)); t = resample_poly(x8, 2, 1).astype("float32")
    sf.write(f"{D}/call/call{i:02d}.wav", x, 16000, subtype="PCM_16"); sf.write(f"{D}/calltel/call{i:02d}.wav", t, 16000, subtype="PCM_16")
    man.append({"id": f"call{i:02d}", "dur": len(x) / 16000, "ref": ref, "pause": p})
for d in ("call", "calltel"): json.dump(man, open(f"{D}/{d}/manifest.json", "w"))
print(len(man), "utts; with pause:", sum(m["pause"] for m in man))
