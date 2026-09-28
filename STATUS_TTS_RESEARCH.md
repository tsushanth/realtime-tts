# TTS Streaming + Voice Cloning Research — ReadAloud AI

## Goal
Find an open-source TTS model that supports **both** real-time streaming (~sub-500ms first audio) **and** zero-shot voice cloning from arbitrary reference audio.

## Models Evaluated

### 1. Fish Speech v1.5 (S2-Pro) — ALREADY TESTED
- **Architecture**: LLM text-to-semantic + VQGAN decoder
- **Streaming**: ✅ Emits audio chunks incrementally
- **Voice cloning**: ✅ Zero-shot from reference audio (base64 inline)
- **Latency on A10G**: 2–4s first audio (hot container)
- **Root cause**: LLM forward pass is the bottleneck (~2s for short text)
- **Reference caching experiment**: No improvement — VQGAN encode is fast, LLM generation dominates

**Verdict**: Too slow for telephony. Good for articles/audiobooks.

---

### 2. Orpheus TTS (~3B, vLLM + SNAC)
- **Architecture**: Llama-3B backbone + SNAC codec decoder
- **Streaming**: ✅ vLLM streaming generation + per-chunk SNAC decode
- **Claimed latency**: ~200ms streaming (with vLLM), reducible to ~100ms
- **Voice cloning**: ❌ NOT zero-shot from arbitrary audio
  - Uses hardcoded speaker tags ("tara", "leah", "zoe", etc.)
  - Fine-tuning requires ~50–300 examples per speaker
  - Pretrained model supports few-shot conditioning (text+speech pairs in prompt)
  - README lists "Fix voice cloning Colab notebook" as TODO — feature is not production-ready

**Verdict**: Fast enough for telephony, but cannot clone arbitrary voices from short audio clips. Would require fine-tuning per voice.

---

### 3. F5-TTS (~1B, Flow Matching)
- **Architecture**: DiT (Diffusion Transformer) flow matching + Vocos/BigVGAN vocoder
- **Streaming**: ❌ NOT streaming — generates full audio in one diffusion pass
- **Total synthesis speed**: ~250ms for short text (RTF ~0.04)
- **Voice cloning**: ✅ Zero-shot from reference audio (up to 12s)
- **Real-time factor**: 0.04 means 5s of audio takes 0.2s to generate

**Verdict**: Very fast total synthesis, but NOT streaming. You must wait for the full audio before playback starts. Not suitable for telephony.

---

### 4. XTTS v2 (Coqui) — ALREADY DEPLOYED
- **Architecture**: Tortoise-style autoregressive LLM + HiFi-GAN
- **Streaming**: ❌ Batch generation only
- **Voice cloning**: ✅ Instant, 6-second reference
- **Latency**: ~1–2s for short text

**Verdict**: Good quality cloning, but not streaming. Too slow for telephony.

---

### 5. ElevenLabs (Commercial)
- **Architecture**: Proprietary
- **Streaming**: ✅ ~200ms first audio
- **Voice cloning**: ✅ Instant, 30-second reference
- **Cost**: ~$0.18/minute

**Verdict**: The only proven option that meets both requirements.

---

## Summary Matrix

| Model | Streaming | Zero-Shot Clone | First Audio Latency | Open Source |
|-------|-----------|-----------------|---------------------|-------------|
| **Fish Speech** | ✅ | ✅ | ~2–4s | ✅ |
| **Orpheus TTS** | ✅ | ❌ | ~200ms | ✅ |
| **F5-TTS** | ❌ | ✅ | N/A (batch) | ✅ |
| **XTTS v2** | ❌ | ✅ | ~1–2s | ✅ |
| **ElevenLabs** | ✅ | ✅ | ~200ms | ❌ |

---

## Honest Assessment

**No open-source model currently supports BOTH real-time streaming AND zero-shot voice cloning from arbitrary audio.**

The fundamental tension:
- **Streaming TTS** requires an autoregressive or token-streaming architecture that emits audio tokens incrementally
- **Zero-shot voice cloning** from short audio requires a model that can extract speaker embeddings and condition generation on them in a single forward pass
- The intersection of these two capabilities is extremely rare in open-source

Orpheus TTS comes closest on latency but lacks zero-shot cloning. F5-TTS has cloning but no streaming. Fish Speech has both but the LLM is too large/slow for sub-500ms on A10G.

---

## Recommended Paths Forward

### Path A: ElevenLabs for Telephony (Immediate)
Use ElevenLabs for the phone agent where streaming + cloning is non-negotiable.
- Keep your existing Piper infrastructure for API/billing
- Use ElevenLabs API directly from the call-loop
- Cost: ~$0.18/min

### Path B: Hybrid Architecture (Recommended)
- **Telephony**: Use ElevenLabs with cloned voices
- **Articles/Audiobooks**: Use Fish Speech (streaming chunks, zero-shot clone) — 2s latency is fine
- **Batch processing**: Use XTTS v2 or F5-TTS

### Path C: Fine-Tune Orpheus (Research)
If you want to own the full stack:
1. Fine-tune Orpheus TTS on ~300 examples per voice you want to support
2. Deploy on Modal with vLLM
3. Use speaker tags to select voices at inference time
4. Achieve ~200ms streaming with your own model

Downside: Requires significant data collection and training per voice. Not "instant cloning."

### Path D: Wait for Smaller Models
- Orpheus plans to release 1B, 400M, and 150M variants
- Smaller LLMs may achieve sub-500ms on A10G with vLLM
- Zero-shot cloning from arbitrary audio may improve in future versions
- Timeline: Unknown

---

## Files in Repo
- `voice-pipeline/fish_speech_dev.py` — Fish Speech v1.5 prototype (working, deployed, stopped)
- `voice-pipeline/xtts_clone.py` — XTTS v2 instant cloning (working, deployed)
