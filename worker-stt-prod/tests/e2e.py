"""E2E checks against a deployed TEST app. usage: STT_SESSION_SECRET_FILE=... python e2e.py https://<app>.modal.run [clip dir]
Expects c10.wav c10.m4a c10.mulaw c10.pcm c60.mp3 in the dir (see README)."""
import os, sys, uuid
from client import mint, post
base, d = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "/tmp/sttprod")
secret = open(os.environ["STT_SESSION_SECRET_FILE"]).read().strip()
tok = mint(secret, "e2e-key", 3600000)
rd = lambda n: open(os.path.join(d, n), "rb").read()
fails = []
def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    if not cond: fails.append(name)

st, b, _ = post(base, None, rd("c10.wav"));                       check("no token -> 401", st == 401, (st, b))
st, b, _ = post(base, "x.y", rd("c10.wav"));                      check("garbage token -> 401", st == 401, (st, b))
st, b, _ = post(base, mint(secret, "k", -1000), rd("c10.wav"));   check("expired token -> 401", st == 401, (st, b))
st, b, _ = post(base, mint("wrong-secret"), rd("c10.wav"));       check("wrong-secret token -> 401", st == 401, (st, b))
st, b, _ = post(base, tok, rd("c10.wav"), "?format=mp5");         check("bad format -> 400", st == 400, (st, b))
st, b, _ = post(base, tok, rd("c10.wav"), "?language=xx");        check("bad language -> 400", st == 400, (st, b))
st, b, _ = post(base, tok, b"");                                  check("empty -> 400", st == 400, (st, b))
st, b, _ = post(base, tok, os.urandom(5000));                     check("garbage bytes -> 400", st == 400, (st, b))
st, b, _ = post(base, tok, b"#EXTM3U\n#EXTINF:1,\nhttp://example.com/x.mp3\n"); check("playlist file -> 400", st == 400, (st, b))
st, b, _ = post(base, tok, rd("c10.wav"), "?format=wav&language=en&word_timestamps=false")
check("wav ok, no words", st == 200 and b["words"] == [] and b["language"] == "en" and abs(b["duration"] - 10) < 0.1, (st, str(b)[:200]))
st, b, _ = post(base, tok, rd("c10.wav")); check("wav auto: words+segments+lang", st == 200 and b["words"] and b["segments"] and b["language"] == "en" and "committee" in b["text"])
st, b, _ = post(base, tok, rd("c60.mp3"), "?format=mp3"); check("mp3", st == 200 and "committee" in b["text"].lower())
st, b, _ = post(base, tok, rd("c10.m4a")); check("m4a auto", st == 200 and "committee" in b["text"].lower())
st, b, _ = post(base, tok, rd("c10.mulaw"), "?format=mulaw_8000"); check("mulaw_8000", st == 200 and "committee" in b["text"].lower() and abs(b["duration"] - 10) < 0.1)
st, b, _ = post(base, tok, rd("c10.pcm"), "?format=pcm_16000"); check("pcm_16000", st == 200 and "committee" in b["text"].lower())
bd = uuid.uuid4().hex
mp = (f'--{bd}\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n--{bd}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
      f'Content-Type: audio/wav\r\n\r\n').encode() + rd("c10.wav") + f"\r\n--{bd}--\r\n".encode()
st, b, _ = post(base, tok, mp, "", f"multipart/form-data; boundary={bd}"); check("multipart", st == 200 and b["language"] == "en", (st, str(b)[:200]))
st, b, _ = post(base, tok, b"--x--", "", "multipart/form-data; boundary=x"); check("multipart without file -> 400", st == 400, (st, b))
# limits
st, b, _ = post(base, tok, b"\xff" * (8000 * (3 * 3600 + 60)), "?format=mulaw_8000"); check("3h+1min mulaw -> 413", st == 413, (st, b))
if os.environ.get("BIG"):
    st, b, _ = post(base, tok, b"\0" * (201 * 1024 * 1024), "?format=pcm_16000"); check("201 MB -> 413", st == 413, (st, b))
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
