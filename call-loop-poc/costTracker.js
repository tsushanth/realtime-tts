// Per-call cost accounting — Stage 2 of the rollout plan. Real, current
// per-provider rates (verified this session, not recalled from training
// data — re-verify before trusting these for actual pricing decisions,
// providers change rates without notice).
import { buildRegistry, ratesFor } from './llmProviders.js';

// Per-token rates for models outside this table come from the model registry (llmProviders.js).
const LLM_REGISTRY = buildRegistry();

export const RATES = {
  deepgramFluxPerMin: 0.0065,
  claude: {
    'claude-haiku-4-5-20251001': { input: 1 / 1_000_000, output: 5 / 1_000_000 },
    'claude-sonnet-4-6': { input: 3 / 1_000_000, output: 15 / 1_000_000 },
  },
  elevenlabsPerChar: 0.05 / 1000, // eleven_flash_v2_5
  kokoroPerChar: 0, // self-hosted — marginal cost ~0; GPU rental (if on) is billed separately by the hour, not per-call
  // Cartesia's Sonic pay-as-you-go rate ($50/M chars, docs.cartesia.ai/pricing)
  // — same as ElevenLabs Flash. No account exists yet to confirm an actual
  // committed-tier rate; this is the public pay-as-you-go price.
  cartesiaPerChar: 0.05 / 1000,
  // MiniMax speech-2.8-hd pay-as-you-go rate ($100/M chars,
  // platform.minimax.io/docs/guides/pricing-paygo) — 2x Cartesia/ElevenLabs.
  // Matches the model server.js defaults to (MINIMAX_MODEL); the cheaper
  // speech-2.8-turbo tier is $60/M chars if that's used instead.
  minimaxPerChar: 0.1 / 1000,
  // Carrier-leg telephony, per minute. Twilio is not priced here (unchanged behaviour: no telephony line).
  // Telnyx: inbound $0.0032 + Voice API $0.002 + media streaming $0.0035, 60 s minimum on the carrier leg.
  telephony: {
    telnyx: { perMin: 0.0032 + 0.002 + 0.0035, minSeconds: 60 },
  },
  openaiRealtimeMini: {
    // 1 token per 100ms of user speech, 1 token per 50ms of assistant speech
    inputTokPerSec: 10,
    outputTokPerSec: 20,
    inputPrice: 10 / 1_000_000,
    outputPrice: 20 / 1_000_000,
  },
};

export class CallCostTracker {
  constructor({ ttsBackend = 'kokoro', voiceEngine = 'cascaded' } = {}) {
    this.ttsBackend = ttsBackend;
    this.voiceEngine = voiceEngine;
    this.sttSeconds = 0;
    this.llmUsage = []; // { model, inputTokens, outputTokens }
    this.ttsChars = 0;
    this.s2sUserSeconds = 0;
    this.s2sAssistantSeconds = 0;
    // Billable flow event counts. Only counted and logged here; the engine
    // reports nothing to Stripe (billing is the web app's daily job, see
    // TIERED-BILLING.md). Distinct from the dollar-cost breakdown below.
    this.bookingEvents = 0;
    this.transferEvents = 0;
    this.messageEvents = 0;
    this.carrier = 'twilio'; // see setCarrier()
    this.telephonySeconds = 0;
  }

  // Opt-in: only a non-default carrier adds a telephony line to breakdown(); 'twilio' leaves it exactly as before.
  setCarrier(carrier) {
    this.carrier = carrier || 'twilio';
  }

  addTelephonySeconds(sec) {
    this.telephonySeconds += sec;
  }

  addBillableEvent(kind) {
    if (kind === 'booking') this.bookingEvents += 1;
    else if (kind === 'transfer') this.transferEvents += 1;
    else if (kind === 'message') this.messageEvents += 1;
  }

  addSttSeconds(sec) {
    this.sttSeconds += sec;
  }

  addLlmUsage(model, inputTokens, outputTokens) {
    this.llmUsage.push({ model, inputTokens, outputTokens });
  }

  addTtsChars(chars) {
    this.ttsChars += chars;
  }

  addS2sAudioSeconds(userSec, assistantSec) {
    this.s2sUserSeconds += userSec;
    this.s2sAssistantSeconds += assistantSec;
  }

  breakdown() {
    if (this.voiceEngine === 's2s') {
      const inTok = this.s2sUserSeconds * RATES.openaiRealtimeMini.inputTokPerSec;
      const outTok = this.s2sAssistantSeconds * RATES.openaiRealtimeMini.outputTokPerSec;
      const cost = inTok * RATES.openaiRealtimeMini.inputPrice + outTok * RATES.openaiRealtimeMini.outputPrice;
      return { engine: 's2s', openaiRealtime: cost, total: cost };
    }

    const sttCost = (this.sttSeconds / 60) * RATES.deepgramFluxPerMin;
    let llmCost = 0;
    for (const { model, inputTokens, outputTokens } of this.llmUsage) {
      const rate = RATES.claude[model] ?? ratesFor(LLM_REGISTRY, model);
      if (!rate) continue; // unknown model — skip rather than guess a wrong rate
      llmCost += inputTokens * rate.input + outputTokens * rate.output;
    }
    const TTS_RATE_BY_BACKEND = {
      elevenlabs: RATES.elevenlabsPerChar,
      cartesia: RATES.cartesiaPerChar,
      minimax: RATES.minimaxPerChar,
      kokoro: RATES.kokoroPerChar,
    };
    const ttsRate = TTS_RATE_BY_BACKEND[this.ttsBackend] ?? RATES.kokoroPerChar;
    const ttsCost = this.ttsChars * ttsRate;

    const telephonyRate = RATES.telephony[this.carrier];
    if (telephonyRate) {
      const billable = Math.max(this.telephonySeconds, telephonyRate.minSeconds);
      const telephonyCost = this.telephonySeconds > 0 ? (billable / 60) * telephonyRate.perMin : 0;
      return {
        engine: 'cascaded',
        ttsBackend: this.ttsBackend,
        deepgram: sttCost,
        claude: llmCost,
        tts: ttsCost,
        telephony: telephonyCost,
        total: sttCost + llmCost + ttsCost + telephonyCost,
      };
    }

    return {
      engine: 'cascaded',
      ttsBackend: this.ttsBackend,
      deepgram: sttCost,
      claude: llmCost,
      tts: ttsCost,
      total: sttCost + llmCost + ttsCost,
    };
  }

  logSummary(label = 'call') {
    const b = this.breakdown();
    const parts = Object.entries(b)
      .filter(([k]) => k !== 'engine' && k !== 'ttsBackend' && k !== 'total')
      .map(([k, v]) => `${k}=$${v.toFixed(5)}`)
      .join(' ');
    console.log(`[cost] ${label} (${b.engine}${b.ttsBackend ? '/' + b.ttsBackend : ''}): ${parts} total=$${b.total.toFixed(5)}`);
    return b;
  }
}
