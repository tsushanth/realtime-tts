// One-off benchmark: drives real turns through the deployed call-loop-poc
// over its WS `/call` endpoint, once per TTS backend, using the sanctioned
// `user_text` debug path (bypasses mic/Deepgram, doesn't touch the audio
// pipeline — see server.js's onClientMessage comment). Measures wall-clock
// from sending the turn to the first binary TTS byte arriving back — i.e.
// LLM-time + TTS-time combined, NOT including STT (there's no real audio
// input in this test). Not committed as a permanent script dependency of
// the app — a scratch tool for a specific benchmark request.
import WebSocket from 'ws';

const URL = process.env.CALL_LOOP_WS_URL || 'wss://call-loop-poc.fly.dev/call';
const BACKENDS = (process.env.BACKENDS || 'kokoro,elevenlabs,cartesia,minimax').split(',');
const TRIALS = Number(process.env.TRIALS || 5);
const PROMPTS = [
  'What are your business hours?',
  'Do you have any openings this Thursday afternoon?',
  'Can you tell me a bit about what services you offer?',
  'I need to speak to a real person, can you transfer me?',
  'Great, thanks so much for your help today.',
];

function percentile(sorted, p) {
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

// Resolves on the first binary audio byte — that IS the metric we want
// (turn latency: text-in -> first audio out). Deliberately doesn't wait for
// a 'done' event: the HTTP-based backends (elevenlabs/cartesia/minimax)
// never emit one at all (only the kokoro gateway path forwards chunk_meta/
// done/error as tts_event — see server.js's _onTtsMessage vs _speakHttpTts),
// so waiting for it would hang every non-kokoro trial until the timeout.
function runOneTurn(ws, text) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    let resolved = false;
    const onMessage = (data, isBinary) => {
      if (isBinary && !resolved) {
        resolved = true;
        ws.off('message', onMessage);
        resolve({ firstByteMs: Date.now() - t0 });
      }
    };
    ws.on('message', onMessage);
    ws.send(JSON.stringify({ type: 'user_text', text }));
    // Safety timeout — a backend with no key/balance (minimax right now)
    // never produces a binary frame at all, so this is the only way that
    // trial ends instead of hanging forever.
    setTimeout(() => {
      if (!resolved) {
        resolved = true;
        ws.off('message', onMessage);
        resolve({ firstByteMs: null, timedOut: true });
      }
    }, 10000);
  });
}

async function benchBackend(backend) {
  const ws = new WebSocket(URL);
  await new Promise((resolve, reject) => {
    ws.once('open', resolve);
    ws.once('error', reject);
  });
  ws.send(JSON.stringify({
    type: 'context',
    systemPrompt: 'You are a friendly, concise phone receptionist. Keep replies to one short sentence.',
    voice: 'af_heart',
    ttsBackend: backend,
  }));
  await new Promise((r) => setTimeout(r, 500)); // let context settle before the first turn

  const results = [];
  for (let i = 0; i < TRIALS; i++) {
    const text = PROMPTS[i % PROMPTS.length];
    const r = await runOneTurn(ws, text);
    results.push(r);
    console.log(`  [${backend}] trial ${i + 1}/${TRIALS}: firstByte=${r.firstByteMs === null ? 'N/A' : r.firstByteMs + 'ms'}${r.timedOut ? ' TIMED OUT (no audio)' : ''}`);
    // Generous gap so a still-streaming kokoro response (multiple chunks)
    // fully drains before the next trial's first-byte timer starts — a
    // straggling frame from trial N would otherwise get misattributed as
    // trial N+1's "first byte".
    await new Promise((r2) => setTimeout(r2, 2500));
  }
  ws.close();
  return results;
}

async function main() {
  const report = {};
  for (const backend of BACKENDS) {
    console.log(`\n=== ${backend} ===`);
    try {
      const results = await benchBackend(backend);
      const ttfbs = results.filter((r) => r.firstByteMs !== null).map((r) => r.firstByteMs).sort((a, b) => a - b);
      const failures = results.filter((r) => r.firstByteMs === null).length;
      report[backend] = {
        trials: results.length,
        failures,
        ttfb_ms: ttfbs.length
          ? { min: ttfbs[0], p50: percentile(ttfbs, 50), p95: percentile(ttfbs, 95), max: ttfbs[ttfbs.length - 1] }
          : null,
      };
    } catch (err) {
      report[backend] = { error: err.message };
    }
  }
  console.log('\n\n=== SUMMARY (turn latency: text-in -> first TTS audio byte; excludes STT) ===');
  console.table(
    Object.fromEntries(
      Object.entries(report).map(([k, v]) => [
        k,
        v.ttfb_ms
          ? { trials: v.trials, failures: v.failures, min_ms: v.ttfb_ms.min, p50_ms: v.ttfb_ms.p50, p95_ms: v.ttfb_ms.p95, max_ms: v.ttfb_ms.max }
          : { trials: v.trials, failures: v.failures, min_ms: 'N/A', p50_ms: 'N/A', p95_ms: 'N/A', max_ms: 'N/A' },
      ])
    )
  );
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
