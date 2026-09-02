// Buffers streamed LLM tokens and emits text as soon as a clause/sentence
// boundary is seen, so TTS can start on sentence 1 while the LLM is still
// generating sentence 2. Mirrors the chunking rationale in ../DECISIONS.md
// ("Chunking: sentence/clause boundaries, ~90 chars") but chunks at the
// LLM-token layer rather than the TTS-input layer.
const BOUNDARY_RE = /[.!?]+[\s"')\]]*$/;
const MIN_CHUNK_CHARS = 8;

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
