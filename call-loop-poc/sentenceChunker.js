// Buffers streamed LLM tokens and emits text as soon as a clause/sentence
// boundary is seen, so TTS can start on sentence 1 while the LLM is still
// generating sentence 2. Mirrors the chunking rationale in ../DECISIONS.md
// ("Chunking: sentence/clause boundaries, ~90 chars") but chunks at the
// LLM-token layer rather than the TTS-input layer.
const BOUNDARY_RE = /[.!?]+[\s"')\]]*$/;
// Was 8 — too low a bar let a short sentence ("Got it.", "please.", "what
// works best for you.") fire its own isolated, separately-fetched TTS call
// instead of merging into a neighboring chunk. Each isolated short chunk is
// a real audible gap (its own synth+network round trip) before or after it,
// long enough for the other party's own turn-detection (VAD) to read the
// silence as "they're done talking" and barge in mid-reply — reproduced and
// root-caused on a real call (MYSTERY_SHOPPER_PATTERNS.md, cycle 17 & 21:
// the dangling "please." and the dropped "what works best for you." are the
// same mechanism). Raising the threshold means a short sentence accumulates
// into the next one (or the final end-of-stream flush) instead of becoming
// its own fragment, cutting the number of gaps for exactly the pattern that
// reproduced this. Deliberate latency/reliability tradeoff, not a full fix
// — a short FIRST sentence now starts TTS slightly later than before. The
// doc flags a more complete fix (e.g. gap-aware flushing, not just a longer
// static threshold) as its own dedicated investigation, not a one-line patch.
const MIN_CHUNK_CHARS = 30;

export class SentenceChunker {
  constructor(onChunk) {
    this.buffer = '';
    this.onChunk = onChunk;
  }

  push(token) {
    this.buffer += token;
    if (this.buffer.length >= MIN_CHUNK_CHARS && BOUNDARY_RE.test(this.buffer)) {
      this.flush();
    }
  }

  flush() {
    const text = this.buffer.trim();
    this.buffer = '';
    if (text) this.onChunk(text);
  }
}
