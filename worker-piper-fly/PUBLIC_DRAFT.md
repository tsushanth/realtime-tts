# DRAFT - not published. For wording approval before anything goes on readaloudai.org.

Items marked **[DECIDE]** need an owner call. Every number below was measured in this
repo's investigation (see worker-modal-piper/README.md and this dir's README); nothing is
extrapolated except where marked.

---

## 1. Developers page: engine section (proposed copy)

### Two engines, one API

| | `kokoro` (default) | `piper` (new) |
|---|---|---|
| Best for | Highest naturalness | Live voice agents and phone calls: lowest latency, lowest price |
| Time to first audio (warm) | see current docs | ~170 ms measured server-side region-local |
| Price | $0.01 / 1K chars | **[DECIDE]** proposed $0.003-0.004 / 1K chars |
| Output | PCM16 mono 24 kHz | PCM16 mono 24 kHz |
| Voices | current set | one voice (a single American English voice) |

Choose the engine when you authorize:

```js
const { token, url } = await fetch("https://api.readaloudai.org/tts/authorize", {
  method: "POST",
  body: JSON.stringify({ key: "YOUR_KEY", engine: "piper" }), // omit for kokoro
}).then(r => r.json());
const ws = new WebSocket(`${url}?token=${token}`);
```

The WebSocket protocol is identical for both engines. Limits for `piper` today: up to 5,000
characters per request, and a shared pool with a small concurrency cap (see
"Capacity"); at capacity you receive `{"type":"error","message":"at capacity, retry shortly"}`
and close code 1013 - retry with backoff.

Free tier characters are shared across engines. **[DECIDE]** whether Piper characters should
burn free-tier at a discounted rate.

---

## 2. Benchmark page (proposed copy)

**Time to first audio: ReadAloud Piper vs ElevenLabs**

Measured from the same machine (Fly.io, San Jose), interleaved, same hour, warm connection,
short call-center sentences, streaming WebSocket/HTTP, first audio byte.

| Service | Warm connection | New connection |
|---|---|---|
| ReadAloud Piper (San Jose) | 168-171 ms | ~156-228 ms |
| ElevenLabs Flash v2.5 | 173-174 ms | ~195-220 ms |
| ElevenLabs Multilingual v2 | ~1.0-1.1 s | not measured separately |

**Read this before quoting it.** Piper and Flash are a statistical tie on latency; we are
not claiming faster. The claims we can support are: comparable latency to ElevenLabs' fastest
model, at a fraction of the price, and roughly 6x lower latency than Multilingual v2, the
model people pick for quality. Quality is subjective: we do not claim parity with ElevenLabs
voices; **[DECIDE]** whether to publish audio samples side by side so buyers can judge.

**Price per 1M characters (list prices, as of 2026-09):** ElevenLabs Flash $50; ElevenLabs
Multilingual v2 and v3 $100; ReadAloud Kokoro $10; ReadAloud Piper **[DECIDE]** ~$3-4.

**Capacity (honest limits)**

- One shared-CPU machine sustains roughly 2-4 simultaneous live calls at the numbers above.
  Beyond that latency rises sharply. The service caps connections at 4 and rejects the rest
  cleanly rather than degrading everyone.
- A dedicated-CPU machine measured 8-16 concurrent calls at ~325-345 ms median before
  degrading, at about 10x the machine cost. We will scale on demand; **[DECIDE]** the
  spend trigger.

**Methodology.** Client and server both in the same region; times are first-audio-byte after
sending the request; interleaved runs to share network conditions; ElevenLabs measured with
our own account, default settings, `pcm_24000` output. Results vary by hour and vantage;
we publish the script alongside. **[DECIDE]** whether to publish the probe scripts
(currently /tmp only; would move into the repo).

---

## 3. Things that must be true before this is public

1. **Price decided** and the billing job updated: `/admin/usage/drain` now returns
   `piperChars` (a subset of `chars`). Until the job prices it, Piper bills at the Kokoro
   rate (never free). Backend change: `charge = (chars - piperChars) * kokoro + piperChars * piper`.
2. **Licensing sign-off**: Piper is GPL-3.0 (we serve it over a network, but confirm the
   obligation), the model was trained on Polly-generated speech (check Amazon's terms for
   training use), and the base checkpoint was Lessac (check its licence).
3. **Capacity plan** for a launch spike beyond ~4 concurrent streams.
4. Second voice or explicit "single voice" wording on the page.
