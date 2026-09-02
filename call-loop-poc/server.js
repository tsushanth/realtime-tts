// Retell-style low-latency call loop POC, built on top of this repo's own
// realtime-tts gateway (see ../README.md for that leg's protocol/latency
// numbers). New pieces added here: streaming STT (Deepgram Flux — native
// semantic turn detection, not acoustic-silence VAD), streaming LLM
// (Anthropic), sentence-boundary chunking to pipeline LLM->TTS, and
// barge-in (kills in-flight TTS the moment the caller starts talking again).
//
// Browser <-WS-> this server <-WS-> Deepgram Flux (STT + turn detection)
//                            <-HTTP streaming-> Anthropic (LLM)
//                            <-WS-> realtime-tts gateway (TTS)
//
// Run: DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... npm start
import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import Anthropic from '@anthropic-ai/sdk';
import { SentenceChunker } from './sentenceChunker.js';
import { TwilioCallAdapter } from './twilioAdapter.js';
import { CallCostTracker } from './costTracker.js';
import { reportCallUsage } from './stripeMeter.js';
import { resolveInboundCall, fetchKnowledgeItems } from './tenantLookup.js';

const PORT = process.env.PORT || 8090;
// The Deepgram+Claude+TTS pipeline — the only engine this repo runs now.
// (A speech-to-speech alternative on OpenAI's Realtime API used to live
// here behind VOICE_ENGINE=s2s; removed 2026-08-28 — no flow/billing
// support, and the OpenAI key was pulled after disabling that billing
// account, so it could no longer run anyway. Not worth maintaining a dead
// code path.)
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const TTS_GATEWAY_WS_URL = process.env.TTS_GATEWAY_WS_URL || 'ws://127.0.0.1:8080/tts';
const TTS_GATEWAY_API_KEY = process.env.TTS_GATEWAY_API_KEY;
const TTS_VOICE = process.env.TTS_VOICE || 'af_heart';
// 'kokoro' (default) is our own realtime-tts gateway (CPU or GPU backend,
// unaffected by this flag). 'elevenlabs' is a third backend for the cascaded
// engine's TTS leg only — higher quality voice, real per-character cost,
// see Stage 1 of the rollout plan. This is only the process-wide default —
// a per-session {"type":"context"} message with a `ttsBackend` field
// overrides it per call (see CallSession.onClientMessage), so a multi-tenant
// caller can pick elevenlabs for one tenant and kokoro for another without a
// restart.
const TTS_BACKEND = process.env.TTS_BACKEND === 'elevenlabs' ? 'elevenlabs' : 'kokoro';
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';
if (TTS_BACKEND === 'elevenlabs' && !ELEVENLABS_API_KEY) {
  console.warn('[call-loop] TTS_BACKEND=elevenlabs but ELEVENLABS_API_KEY not set — TTS will fail');
}
// Haiku over Sonnet for the voice path specifically — a phone reply doesn't
// need Sonnet's depth of reasoning, and LLM TTFB was the single biggest
// latency line item measured in real calls (1.3-1.8s), bigger than
// everything else in the pipeline combined.
const LLM_MODEL = process.env.LLM_MODEL || 'claude-haiku-4-5-20251001';
// Only needed for a flow's 'transfer' node type on a real (Twilio) phone
// call — redirects the live call via Twilio's REST API. Not needed for
// browser calls (there's nothing to redirect) or flows with no transfer node.
const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID;
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN;
const SYSTEM_PROMPT =
  'You are a concise, friendly voice assistant on a phone call. Keep replies to 1-2 short ' +
  'sentences unless asked for more detail. Never use markdown, bullet points, or emoji — ' +
  'this is spoken audio.';

// Flux decides "they're done talking" from the words themselves, not just
// silence — eot_threshold is the confidence bar for a real EndOfTurn, higher
// = waits for more certainty before handing off to the LLM. StartOfTurn
// (below) is what drives barge-in and fires independently of eot_threshold.
const DEEPGRAM_WS_URL =
  'wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000' +
  '&eot_threshold=0.7&eot_timeout_ms=5000';

if (!DEEPGRAM_API_KEY) console.warn('[call-loop] DEEPGRAM_API_KEY not set — STT will fail');
if (!ANTHROPIC_API_KEY) console.warn('[call-loop] ANTHROPIC_API_KEY not set — LLM will fail');

const anthropic = ANTHROPIC_API_KEY ? new Anthropic({ apiKey: ANTHROPIC_API_KEY }) : null;

const app = express();
app.use(express.static('public'));

app.use(express.urlencoded({ extended: false })); // Twilio POSTs form-encoded fields (To, From, CallSid)

// Per-call context resolved here (before the WS even connects) and handed
// off to the Media Stream once it starts — keyed by CallSid, since that's
// the only identifier both this webhook and the later 'start' event share.
// Swept on a timer so a call that gets a TwiML response but never actually
// opens the stream (e.g. caller hangs up mid-ring) doesn't leak forever.
const pendingCallContext = new Map();
setInterval(() => {
  const cutoff = Date.now() - 60_000;
  for (const [callSid, entry] of pendingCallContext) {
    if (entry.createdAt < cutoff) pendingCallContext.delete(callSid);
  }
}, 30_000).unref();

// Twilio POSTs here when the call is answered — the TwiML tells it to open
// a Media Streams WebSocket back to us at /twilio-stream, on this same host
// (works automatically whether that's localhost through a tunnel or a real
// deploy, since it's derived from the request itself, not hardcoded). Also
// resolves which tenant/flow owns the dialed number (see tenantLookup.js) —
// unrouted numbers (unregistered, or a 'retell'-engine version) fall back to
// this process's static single-tenant config exactly as before.
app.post('/twilio/voice', async (req, res) => {
  const callSid = req.body.CallSid;
  // Routing normally keys off the dialed number (req.body.To) — correct for
  // a real inbound call, but wrong for an outbound call WE place to a real
  // phone (e.g. a live demo call): there, To is the callee's own number, not
  // any tenant's routed number. ?routeAs=<number> lets an outbound call
  // explicitly say which tenant's flow it should run, without touching the
  // real inbound-routing path at all.
  const toNumber = req.query.routeAs || req.body.To;
  if (callSid) {
    const resolved = await resolveInboundCall(toNumber).catch((err) => {
      console.error('[call-loop] tenant lookup failed', err);
      return null;
    });
    if (resolved) pendingCallContext.set(callSid, { ...resolved, createdAt: Date.now() });
  }
  const twiml =
    `<?xml version="1.0" encoding="UTF-8"?>` +
    `<Response><Connect><Stream url="wss://${req.headers.host}/twilio-stream" /></Connect></Response>`;
  res.type('text/xml').send(twiml);
});

const server = http.createServer(app);
// Two WebSocketServer instances each bound with {server, path} don't reliably
// coexist on one shared http.Server in this ws version — the second path
// registered gets a 400 on every handshake instead of falling through to it.
// ws's own docs' fix for multiple paths on one server: noServer mode on both,
// with manual routing on the server's single 'upgrade' event instead.
const wss = new WebSocketServer({ noServer: true });
const twilioWss = new WebSocketServer({ noServer: true });

server.on('upgrade', (req, socket, head) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  if (pathname === '/call') {
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  } else if (pathname === '/twilio-stream') {
    twilioWss.handleUpgrade(req, socket, head, (ws) => twilioWss.emit('connection', ws, req));
  } else {
    socket.destroy();
  }
});

wss.on('connection', (clientWs) => {
  console.log('[call-loop] client connected');
  const session = new CallSession(clientWs);
  clientWs.on('message', (data, isBinary) => session.onClientMessage(data, isBinary));
  clientWs.on('close', () => session.close());
  clientWs.on('error', (err) => console.error('[call-loop] client ws error', err));
});

twilioWss.on('connection', (twilioWs) => {
  console.log('[call-loop] Twilio call connected');
  // The adapter presents the exact same interface a browser WebSocket would
  // — the session class has no idea this is a phone call, not a browser tab.
  const adapter = new TwilioCallAdapter(twilioWs);
  const session = new CallSession(adapter);
  adapter.on('message', (data, isBinary) => session.onClientMessage(data, isBinary));
  adapter.on('close', () => session.close());
  adapter.on('error', (err) => console.error('[call-loop] twilio adapter error', err));
  // Per-tenant routing resolved back in /twilio/voice (see tenantLookup.js)
  // — handed to the session the same way a browser client does it, via a
  // synthesized 'context' message, rather than duplicating that parsing
  // logic here.
  adapter.on('start', (callSid) => {
    const resolved = pendingCallContext.get(callSid);
    if (!resolved) return;
    pendingCallContext.delete(callSid);
    session.onClientMessage(JSON.stringify({
      type: 'context',
      flow: resolved.flow,
      ...(resolved.ttsBackend ? { ttsBackend: resolved.ttsBackend } : {}),
      ...(resolved.stripeCustomerId ? { stripeCustomerId: resolved.stripeCustomerId } : {}),
    }), false);
  });
});

class CallSession {
  constructor(clientWs) {
    this.clientWs = clientWs;
    this.history = [];
    this.turnSeq = 0;
    this.activeTurn = 0; // turn id currently allowed to speak; 0 once nothing is in flight
    this.turnState = null; // { id, llmDone, pendingTts } — tracks when activeTurn can go back to 0
    this.ttsWs = null;
    this.dgConnection = null;
    // Per-call context — overridable by a {"type":"context"} message sent
    // right after connect (see onClientMessage), so a caller (e.g. a
    // multi-tenant product embedding this) can hand each session its own
    // business prompt/voice instead of every call getting the same
    // hardcoded assistant. Falls back to the module defaults untouched.
    this.systemPrompt = SYSTEM_PROMPT;
    this.voice = TTS_VOICE;
    this.greeting = null;
    this.ttsBackend = TTS_BACKEND;
    // Conversation flow (optional) — a real node-based state machine, set via
    // the {"type":"context"} message's `flow` field. When absent, the session
    // behaves exactly as before (this.systemPrompt used verbatim every turn).
    // See _buildNodeSystemPrompt/_buildTransitionTool/_applyTransition for how
    // a node's own instructions + edges become a per-turn prompt + a real
    // tool call the LLM uses to advance state, instead of flattening the
    // whole graph into one static prompt upfront.
    this.flow = null;
    this.flowNodesById = null;
    this.currentNodeId = null;
    this.collectedData = {};
    // Set via the {"type":"context"} message's `stripeCustomerId` field —
    // present only when this call belongs to a real billed tenant (browser
    // demo calls and flow-MCP test calls have none, and simply don't get
    // metered). See stripeMeter.js.
    this.stripeCustomerId = null;
    this.cost = new CallCostTracker({ ttsBackend: TTS_BACKEND, voiceEngine: 'cascaded' });
    this._callStartedAt = Date.now();
    this._connectDeepgram();
  }

  send(obj) {
    if (this.clientWs.readyState === WebSocket.OPEN) this.clientWs.send(JSON.stringify(obj));
  }

  _connectDeepgram() {
    if (!DEEPGRAM_API_KEY) return;
    const dg = new WebSocket(DEEPGRAM_WS_URL, { headers: { Authorization: `Token ${DEEPGRAM_API_KEY}` } });
    this.dgConnection = dg;

    dg.on('open', () => console.log('[call-loop] deepgram (flux) connected'));

    dg.on('message', (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        return;
      }
      if (msg.type !== 'TurnInfo') return;

      if (msg.event === 'StartOfTurn') {
        // Semantic + acoustic turn-start — caller began speaking, cut the
        // assistant off now rather than waiting for a full transcript.
        this._bargeIn();
      } else if (msg.event === 'Update') {
        const text = msg.transcript?.trim();
        if (text) this.send({ type: 'transcript', text, isFinal: false });
      } else if (msg.event === 'EndOfTurn') {
        // High-confidence, semantically-aware turn end (eot_threshold) — this
        // is what nova-2 + acoustic VAD couldn't do: it waits for a complete
        // thought, not just a gap in the audio.
        const text = msg.transcript?.trim();
        if (text) this._onUserTurnComplete(text);
      }
    });

    dg.on('error', (err) => console.error('[call-loop] deepgram error', err));
    dg.on('close', () => console.log('[call-loop] deepgram closed'));
  }

  onClientMessage(data, isBinary) {
    if (isBinary) {
      // Raw PCM16 mono 16kHz audio frame from the browser mic.
      if (this.dgConnection?.readyState === WebSocket.OPEN) {
        this.dgConnection.send(data);
      }
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === 'hangup') this.close();
    // Text-input debug path — lets a turn be driven without a mic/Deepgram
    // (e.g. scripting a flow test), bypassing STT entirely. Real calls never
    // send this; it's additive and doesn't change the audio path at all.
    if (msg.type === 'user_text' && typeof msg.text === 'string' && msg.text.trim()) {
      this._onUserTurnComplete(msg.text.trim());
    }
    if (msg.type === 'context') {
      // Must arrive before the first user turn to take effect — it's read
      // fresh at the start of each turn, not snapshotted at connect time.
      if (typeof msg.systemPrompt === 'string' && msg.systemPrompt.trim()) {
        this.systemPrompt = msg.systemPrompt;
      }
      if (typeof msg.voice === 'string' && msg.voice.trim()) {
        this.voice = msg.voice;
      }
      if (typeof msg.stripeCustomerId === 'string' && msg.stripeCustomerId.trim()) {
        this.stripeCustomerId = msg.stripeCustomerId.trim();
      }
      if (msg.ttsBackend === 'elevenlabs' || msg.ttsBackend === 'kokoro') {
        if (msg.ttsBackend === 'elevenlabs' && !ELEVENLABS_API_KEY) {
          console.warn('[call-loop] context requested ttsBackend=elevenlabs but ELEVENLABS_API_KEY not set — falling back to kokoro');
        } else {
          this.ttsBackend = msg.ttsBackend;
          this.cost.ttsBackend = msg.ttsBackend;
        }
      }
      if (Array.isArray(msg.flow?.nodes) && msg.flow.nodes.length > 0) {
        this.flow = msg.flow;
        this.flowNodesById = new Map(this.flow.nodes.map((n) => [n.id, n]));
        this.currentNodeId = msg.flow.startNodeId || this.flow.nodes[0].id;
        console.log(`[call-loop] flow set — ${this.flow.nodes.length} nodes, starting at "${this.currentNodeId}"`);
        this.send({ type: 'flow_state', currentNodeId: this.currentNodeId, nodeType: this.flowNodesById.get(this.currentNodeId)?.type, collectedData: this.collectedData });
      }
      if (typeof msg.greeting === 'string' && msg.greeting.trim()) {
        this.greeting = msg.greeting;
        this.history.push({ role: 'assistant', content: msg.greeting });
        // Speak it as a real turn (not a special-cased id) so barge-in works
        // on the greeting exactly like it does on every other response.
        const turnId = ++this.turnSeq;
        this.activeTurn = turnId;
        this.turnState = { id: turnId, llmDone: true, pendingTts: 0 };
        this._speak(msg.greeting, turnId, Date.now());
      } else if (this.flow) {
        // No explicit greeting text — let the flow's own start node (usually
        // a 'greeting'-type node) generate the opening line itself, same as
        // every other node's turn.
        this._runNodeTurn(this.currentNodeId);
      }
      console.log(`[call-loop] context set — prompt: ${this.systemPrompt.length} chars, voice: ${this.voice}, ttsBackend: ${this.ttsBackend}`);
    }
  }

  async _onUserTurnComplete(userText) {
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0 };
    const turnStartedAt = Date.now();
    console.log(`[call-loop] turn ${turnId} user: "${userText}"`);

    this.history.push({ role: 'user', content: userText });
    this.send({ type: 'user_turn', turnId, text: userText });

    await this._generateTurn(turnId, turnStartedAt);
  }

  // Generates an assistant turn for the *current flow node* with no new
  // caller utterance — used for the flow's opening line, and for node types
  // (function/knowledge_base/goodbye/transfer) that act as soon as the flow
  // enters them rather than waiting on the caller to say something first.
  async _runNodeTurn(nodeId) {
    this.currentNodeId = nodeId;
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0 };
    // Anthropic's API requires at least one message — an auto-advance turn
    // (the flow's opening line, or a node the flow enters without the caller
    // saying anything) has nothing real to put there yet on a brand-new call.
    const isCallOpening = this.history.length === 0;
    if (isCallOpening) {
      this.history.push({ role: 'user', content: '[Call connected — begin the flow.]' });
    }
    await this._generateTurn(turnId, Date.now(), { isNodeEntry: true, suppressTransitionTool: isCallOpening });
  }

  // Shared by both a real caller turn and a flow auto-advance turn — the
  // only difference is whether a user message was already pushed to
  // this.history before calling in. When this.flow is set, the system
  // prompt/tools are scoped to just the current node (see
  // _buildNodeSystemPrompt/_buildTransitionTool) instead of the flat
  // this.systemPrompt every other turn uses, and the LLM signals when to
  // advance by calling the transition_flow tool rather than us guessing from
  // the model's prose.
  async _generateTurn(turnId, turnStartedAt, { isNodeEntry = false, suppressTransitionTool = false } = {}) {
    if (!anthropic) {
      this.send({ type: 'error', message: 'ANTHROPIC_API_KEY not configured' });
      if (this.turnState?.id === turnId) this.turnState.llmDone = true;
      return;
    }

    const node = this.flow ? this.flowNodesById.get(this.currentNodeId) : null;
    if (this.turnState?.id === turnId) {
      this.turnState.nodeType = node?.type || null;
      this.turnState.nodeParams = node?.params || null;
    }

    // A function node's side effect happens before it says anything, and
    // only once — on the turn that actually enters the node (via
    // _runNodeTurn). Without the isNodeEntry guard, every later turn where
    // the conversation just happens to still be sitting in this node (e.g.
    // the caller asking a follow-up before transitioning away) would re-run
    // the webhook, which is wrong — it's a one-time side effect of arriving
    // at the node, not a per-turn one.
    if (node?.type === 'function' && isNodeEntry) {
      await this._executeFunctionNode(node);
    }
    if (node?.type === 'knowledge_base' && isNodeEntry) {
      await this._executeKnowledgeBaseNode(node);
    }

    const systemPrompt = node ? this._buildNodeSystemPrompt(node) : this.systemPrompt;
    // The call's very opening turn has no real caller utterance to justify
    // any edge yet — only the synthetic "[Call connected]" seed message —
    // so the transition tool is withheld for that one turn specifically.
    // Without this, Haiku would sometimes call it anyway (observed jumping
    // straight from greeting to the next node before the caller had said a
    // word), reading "begin the flow" too literally. Every other node-entry
    // turn (function/knowledge_base/goodbye/transfer auto-advance, or the
    // greeting reached via a real transition) keeps the tool as normal.
    const tools = node && node.edges.length > 0 && !suppressTransitionTool
      ? [this._buildTransitionTool(node)]
      : undefined;

    let firstTokenAt = null;
    let assistantText = '';
    const chunker = new SentenceChunker((sentence) => {
      if (this.activeTurn !== turnId) {
        console.log(`[call-loop] turn ${turnId} sentence chunk dropped — activeTurn is now ${this.activeTurn}: "${sentence}"`);
        return;
      }
      this._speak(sentence, turnId, turnStartedAt);
    });

    try {
      const stream = anthropic.messages.stream({
        model: LLM_MODEL,
        system: systemPrompt,
        max_tokens: 300,
        messages: this.history,
        ...(tools ? { tools } : {}),
      });

      stream.on('text', (delta) => {
        if (this.activeTurn !== turnId) return;
        if (!firstTokenAt) {
          firstTokenAt = Date.now();
          console.log(`[call-loop] turn ${turnId} LLM TTFB: ${firstTokenAt - turnStartedAt}ms`);
        }
        assistantText += delta;
        chunker.push(delta);
      });

      const final = await stream.finalMessage();
      if (final.usage) this.cost.addLlmUsage(LLM_MODEL, final.usage.input_tokens, final.usage.output_tokens);
      if (this.activeTurn === turnId) {
        // Safety net for a terminal node (goodbye/transfer): there's no
        // transition tool to distract the model there, but Haiku
        // occasionally still returns empty text for a short instruction like
        // "thank them and say goodbye." Dead air is a much worse failure
        // there than anywhere else in the flow — it's the caller's very last
        // impression, or happens right as they're being handed off — so fall
        // back to speaking the node's own instruction text verbatim rather
        // than silently hanging up/transferring with nothing said.
        if (!assistantText && (node?.type === 'goodbye' || node?.type === 'transfer')) {
          console.warn(`[call-loop] node "${node.id}" (${node.type}) produced no speech — falling back to its prompt text`);
          assistantText = node.prompt;
          this._speak(assistantText, turnId, turnStartedAt);
        }
        chunker.flush();
        if (assistantText) this.history.push({ role: 'assistant', content: assistantText });
        // TTS-backend-independent observability — chunk_meta only exists on
        // the kokoro path, so a client (browser UI, or a headless test
        // harness like the flow MCP server) that wants "what did the
        // assistant actually say" needs a signal that doesn't depend on
        // which TTS backend this call happens to be using.
        this.send({ type: 'assistant_turn', turnId, text: assistantText, nodeId: this.currentNodeId });

        const toolUse = final.content.find((b) => b.type === 'tool_use' && b.name === 'transition_flow');
        if (toolUse && this.turnState?.id === turnId) {
          this.turnState.transition = toolUse.input;
        }
      }
    } catch (err) {
      console.error('[call-loop] LLM error', err);
      this.send({ type: 'error', message: 'LLM request failed' });
    } finally {
      if (this.turnState?.id === turnId) {
        this.turnState.llmDone = true;
        this._maybeRetireTurn(turnId);
      }
    }
  }

  // Builds this turn's system prompt from just the current flow node's own
  // instructions — not the whole flow flattened into one wall of text — so
  // the model only ever has to reason about "what am I doing right now,"
  // with collected-so-far data and the node's real edges (via the
  // transition_flow tool) as its only additional context.
  _buildNodeSystemPrompt(node) {
    const gs = this.flow?.globalSettings || {};
    let prompt =
      `You are a concise, friendly voice assistant on a phone call, currently in the ` +
      `"${node.id}" step of a structured conversation flow.\n\n` +
      `Step instructions: ${node.prompt}\n`;
    if (node.extract) {
      prompt += `Collect these fields before moving on, asking for whichever are still missing: ${Object.keys(node.extract).join(', ')}.\n`;
    }
    if (Object.keys(this.collectedData).length > 0) {
      prompt += `Already collected this call: ${JSON.stringify(this.collectedData)}\n`;
    }
    if (node.edges.length > 0) {
      prompt +=
        `\nWhen this step's goal has been met, call the transition_flow tool to move to the ` +
        `next step. If it hasn't been met yet, keep talking and don't call the tool.\n`;
    }
    prompt += gs.allowInterruptions === false
      ? 'Complete your sentences before listening.\n'
      : 'Allow the caller to interrupt you.\n';
    prompt +=
      'Keep replies to 1-2 short sentences unless asked for more detail. Never use markdown, ' +
      'bullet points, or emoji — this is spoken audio. Always say at least one sentence out ' +
      'loud on every turn, even if you are also calling a tool — never respond with nothing.';
    return prompt;
  }

  // The current node's real edges become the tool's actual enum of valid
  // targets (and their natural-language conditions become the tool's
  // description) — the model picks one instead of us parsing free text to
  // guess where the conversation should go next.
  _buildTransitionTool(node) {
    const properties = {
      next_node_id: {
        type: 'string',
        enum: node.edges.map((e) => e.target),
        description: node.edges.map((e) => `${e.target}: ${e.condition}`).join('; '),
      },
    };
    if (node.extract) {
      properties.extracted = {
        type: 'object',
        properties: Object.fromEntries(Object.keys(node.extract).map((k) => [k, { type: 'string' }])),
        description: 'Fields the caller has actually provided during this step so far.',
      };
    }
    return {
      name: 'transition_flow',
      description: "Call this once this step's goal has been met and it's time to move to the next step in the flow.",
      input_schema: { type: 'object', properties, required: ['next_node_id'] },
    };
  }

  // Applied once the turn that produced this transition has fully finished
  // speaking (see _maybeRetireTurn) — not the instant the tool call arrives —
  // so the caller always hears the current node's full response before the
  // flow moves on.
  _applyTransition({ next_node_id, extracted }) {
    if (extracted && typeof extracted === 'object') {
      Object.assign(this.collectedData, extracted);
    }
    const nextNode = this.flowNodesById.get(next_node_id);
    if (!nextNode) {
      console.warn(`[call-loop] flow transition to unknown node "${next_node_id}" — ignoring`);
      return;
    }
    // Billable event = actually completing the node's job (leaving it via a
    // real transition), not just visiting it — matches flowBuilder.ts's
    // fixed node ids ('booking'/'take_message') for the wizard-built flows.
    // A custom flow that reuses these ids for something else would also bill
    // here; that's an acceptable trade-off for not needing a separate
    // "billable" flag on the node schema yet.
    if (this.currentNodeId === 'booking') this.cost.addBillableEvent('booking');
    else if (this.currentNodeId === 'take_message') this.cost.addBillableEvent('message');
    console.log(`[call-loop] flow transition -> "${next_node_id}" (${nextNode.type})`);
    this.send({ type: 'flow_state', currentNodeId: next_node_id, nodeType: nextNode.type, collectedData: this.collectedData });
    // These node types act as soon as the flow enters them (run a webhook,
    // look something up, say goodbye and hang up, transfer the call) rather
    // than waiting for the caller to speak first.
    const AUTO_ADVANCE_TYPES = new Set(['function', 'knowledge_base', 'goodbye', 'transfer']);
    if (AUTO_ADVANCE_TYPES.has(nextNode.type)) {
      this._runNodeTurn(next_node_id);
    } else {
      this.currentNodeId = next_node_id;
    }
  }

  // Runs a flow 'function' node's configured webhook and folds the result
  // into history as a system-style note before the node's own turn is
  // generated, so the model's dialogue and next transition are actually
  // informed by what the webhook returned instead of the model guessing.
  // Was previously a disclosed gap: a knowledge_base node ran as a plain
  // prompt with nothing behind it — no real Supabase lookup. node.params.
  // knowledgeBaseId is stamped on by tenantLookup.js (resolved from the
  // agent's actual calldesk_knowledge_bases row, since the node itself has
  // no column for it); folded into history the same way a function node's
  // webhook result is, so the model answers from real content instead of
  // guessing.
  async _executeKnowledgeBaseNode(node) {
    const knowledgeBaseId = node.params?.knowledgeBaseId;
    if (!knowledgeBaseId) {
      console.warn(`[call-loop] knowledge_base node "${node.id}" has no knowledgeBaseId — running with no KB content`);
      return;
    }
    const items = await fetchKnowledgeItems(knowledgeBaseId);
    if (items.length === 0) {
      console.warn(`[call-loop] knowledge_base node "${node.id}" — KB ${knowledgeBaseId} has no items`);
      return;
    }
    const qa = items.map((it, i) => `${i + 1}. Q: ${it.question}\n   A: ${it.answer}`).join('\n');
    this.history.push({
      role: 'user',
      content: `[System note: knowledge base content for this step — answer using these facts when relevant, otherwise say you're not sure and offer to have someone follow up:\n${qa}]`,
    });
  }

  async _executeFunctionNode(node) {
    const url = node.params?.webhookUrl;
    if (!url) {
      console.warn(`[call-loop] function node "${node.id}" has no params.webhookUrl — skipping call`);
      return;
    }
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ function: node.function, collectedData: this.collectedData }),
        signal: AbortSignal.timeout(8000),
      });
      const result = await res.json().catch(() => ({}));
      console.log(`[call-loop] function node "${node.id}" (${node.function}) -> HTTP ${res.status}`);
      this.history.push({
        role: 'user',
        content: `[System note: function "${node.function}" returned ${JSON.stringify(result)}]`,
      });
    } catch (err) {
      console.error(`[call-loop] function node "${node.id}" webhook failed`, err);
      this.history.push({
        role: 'user',
        content: `[System note: function "${node.function}" failed — let the caller know something went wrong and offer to have someone follow up]`,
      });
    }
  }

  // Redirects the live call via Twilio's REST API — only possible on a real
  // phone call (TwilioCallAdapter exposes .callSid; a browser call has
  // nothing to redirect) and only with Twilio credentials configured.
  // Otherwise falls back to hanging up rather than silently doing nothing.
  async _executeTransfer(params) {
    const to = params?.transferTo;
    const callSid = this.clientWs?.callSid;
    if (!to || !callSid || !TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
      console.warn(
        `[call-loop] transfer requested but cannot complete it (to=${to || 'unset'}, ` +
        `callSid=${callSid ? 'present' : 'missing — not a Twilio call?'}, ` +
        `twilioCredsConfigured=${Boolean(TWILIO_ACCOUNT_SID && TWILIO_AUTH_TOKEN)}) — hanging up instead`
      );
      this.close();
      return;
    }
    try {
      const twiml = `<?xml version="1.0" encoding="UTF-8"?><Response><Dial>${to}</Dial></Response>`;
      const auth = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
      const res = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Calls/${callSid}.json`,
        {
          method: 'POST',
          headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ Twiml: twiml }).toString(),
        }
      );
      console.log(`[call-loop] transfer -> ${to} (Twilio responded HTTP ${res.status})`);
      if (res.ok) this.cost.addBillableEvent('transfer');
      // Twilio's own <Dial> now owns the call — our media-stream WS leg will
      // get a 'stop' event and close() normally once that dial ends.
    } catch (err) {
      console.error('[call-loop] transfer failed', err);
      this.close();
    }
  }

  // A turn only stops being "active" (eligible for barge-in) once the LLM
  // has finished generating AND every TTS chunk it dispatched has finished
  // playing/erroring/being cancelled — otherwise a stray VAD blip after the
  // conversation has gone quiet reads as a spurious barge-in on nothing.
  _maybeRetireTurn(turnId) {
    if (this.turnState?.id !== turnId) return;
    if (this.turnState.llmDone && this.turnState.pendingTts <= 0 && this.activeTurn === turnId) {
      this.activeTurn = 0;
      // Flow side effects run only after the caller has fully heard this
      // node's response — a transition, hangup, or transfer landing mid-TTS
      // would cut the assistant off on its own words.
      if (this.turnState.nodeType === 'goodbye') {
        console.log('[call-loop] flow reached goodbye node — hanging up');
        this.close();
        return;
      }
      if (this.turnState.nodeType === 'transfer') {
        this._executeTransfer(this.turnState.nodeParams);
        return;
      }
      if (this.turnState.transition) this._applyTransition(this.turnState.transition);
    }
  }

  _ensureTtsSocket() {
    if (this.ttsWs && this.ttsWs.readyState === WebSocket.OPEN) return this.ttsWs;
    // Only the cloud gateway (realtime-tts-gateway.fly.dev) requires this —
    // a local dev worker (ws://127.0.0.1:8080/tts) has no auth at all, so
    // this is a no-op there when the env var is unset.
    const ws = new WebSocket(
      TTS_GATEWAY_WS_URL,
      TTS_GATEWAY_API_KEY ? { headers: { Authorization: `Bearer ${TTS_GATEWAY_API_KEY}` } } : undefined
    );
    ws.binaryType = 'arraybuffer';
    ws.on('open', () => console.log('[call-loop] tts gateway connected'));
    ws.on('error', (err) => console.error('[call-loop] tts gateway error', err));
    ws.on('close', () => {
      console.log('[call-loop] tts gateway closed');
      if (this.ttsWs === ws) this.ttsWs = null;
    });
    ws.on('message', (data, isBinary) => this._onTtsMessage(data, isBinary));
    this.ttsWs = ws;
    return ws;
  }

  _speak(text, turnId, turnStartedAt) {
    this.cost.addTtsChars(text.length);
    // Reserve this turn's "still speaking" slot immediately, synchronously —
    // not inside a possibly-deferred dispatch. On a session's first turn,
    // the TTS connection has to open/request fresh, and if the counter only
    // incremented once that finished, a fast LLM response could see
    // pendingTts still at 0 in between, conclude the turn was already fully
    // done, and reset activeTurn — causing dispatch to then silently drop
    // audio that was actually still in flight.
    if (this.turnState?.id === turnId) this.turnState.pendingTts++;

    if (this.ttsBackend === 'elevenlabs') {
      this._speakElevenLabs(text, turnId, turnStartedAt);
      return;
    }

    const ws = this._ensureTtsSocket();
    const dispatch = () => {
      if (this.activeTurn !== turnId) {
        console.log(`[call-loop] turn ${turnId} dropped before dispatch — activeTurn is now ${this.activeTurn} (superseded while waiting for TTS socket)`);
        if (this.turnState?.id === turnId) this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        return;
      }
      this._pendingTurnStart = this._pendingTurnStart || turnStartedAt;
      ws.send(JSON.stringify({ type: 'synthesize', text, voice: this.voice, speed: 1.0 }));
    };
    if (ws.readyState === WebSocket.OPEN) dispatch();
    else ws.once('open', dispatch);
  }

  // ElevenLabs streaming, requested as pcm_24000 — the exact same wire
  // format our own TTS gateway already emits, so it drops straight into the
  // same downstream forwarding path (_onTtsMessage's binary branch does the
  // same job for the gateway; this does its own forwarding since there's no
  // gateway WS in this path, just an HTTP stream).
  async _speakElevenLabs(text, turnId, turnStartedAt) {
    if (!this._elevenAborts) this._elevenAborts = new Set();
    const controller = new AbortController();
    this._elevenAborts.add(controller);
    let loggedTtfb = false;

    try {
      const res = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000`,
        {
          method: 'POST',
          headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, model_id: 'eleven_flash_v2_5' }),
          signal: controller.signal,
        }
      );
      if (!res.ok || !res.body) {
        console.error(`[call-loop] ElevenLabs request failed: ${res.status}`);
        return;
      }
      // Buffer the full response before handing it to the resample/mulaw
      // pipeline downstream — resampling is a *stateful* decimation
      // (take every Nth sample), and running it separately on each small,
      // arbitrarily-sized HTTP chunk resets that state at every chunk
      // boundary instead of continuing the same phase across the whole
      // utterance. That's a structural mismatch, not a framing bug, and no
      // per-chunk fix (like the earlier odd-byte-alignment attempt) can
      // patch around it — it corrupts the underlying sample sequence, not
      // just the edges. Costs a little TTFB (no longer forwarding the very
      // first bytes immediately) in exchange for actually correct audio.
      const parts = [];
      for await (const chunk of res.body) {
        if (this.activeTurn !== turnId) break; // barge-in mid-stream
        parts.push(Buffer.from(chunk));
      }
      if (this.activeTurn === turnId && parts.length > 0) {
        loggedTtfb = true;
        console.log(`[call-loop] TTS TTFB: ${Date.now() - turnStartedAt}ms (turn latency end-to-end, elevenlabs)`);
        let full = Buffer.concat(parts);
        if (full.length % 2 !== 0) full = full.subarray(0, full.length - 1); // drop any trailing odd byte
        if (this.clientWs.readyState === WebSocket.OPEN) {
          this.clientWs.send(full, { binary: true });
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') console.error('[call-loop] ElevenLabs stream error', err);
    } finally {
      this._elevenAborts.delete(controller);
      if (this.turnState?.id === turnId) {
        this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        this._maybeRetireTurn(turnId);
      }
    }
  }

  _onTtsMessage(data, isBinary) {
    if (isBinary) {
      // PCM16LE mono 24kHz chunk immediately following a chunk_meta message.
      if (this.clientWs.readyState === WebSocket.OPEN) {
        this.clientWs.send(data, { binary: true });
      }
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === 'chunk_meta' && this._pendingTurnStart) {
      console.log(`[call-loop] TTS TTFB: ${Date.now() - this._pendingTurnStart}ms (turn latency end-to-end)`);
      this._pendingTurnStart = null;
    }
    if (msg.type === 'done' || msg.type === 'cancelled' || msg.type === 'error') {
      if (this.turnState) {
        this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        this._maybeRetireTurn(this.turnState.id);
      }
    }
    // Forward control messages (chunk_meta/done/error) so the browser can
    // sequence playback and know when the assistant has finished speaking.
    this.send({ type: 'tts_event', event: msg });
  }

  _bargeIn() {
    if (this.activeTurn === 0) return;
    console.log(`[call-loop] barge-in — cancelling turn ${this.activeTurn}`);
    this.activeTurn = 0; // no active turn is allowed to speak until the next final transcript
    if (this._elevenAborts) {
      for (const controller of this._elevenAborts) controller.abort();
      this._elevenAborts.clear();
    }
    if (this.ttsWs?.readyState === WebSocket.OPEN) {
      this.ttsWs.send(JSON.stringify({ type: 'stop' }));
    }
    if (typeof this.clientWs.clearQueue === 'function') this.clientWs.clearQueue();
    this.send({ type: 'barge_in' });
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    console.log('[call-loop] client disconnected');
    this.dgConnection?.close();
    this.ttsWs?.close();
    // A flow's goodbye node calls this proactively to end the call — unlike
    // every other close() caller (browser tab closed, Twilio's own 'stop'
    // event), which is already reacting to the transport having closed
    // itself. Without this, a real phone call would just sit connected but
    // silent forever once the flow decided it was over — STT/TTS torn down,
    // but the actual call never hung up. Safe to call even when close() was
    // itself triggered BY the transport closing — both TwilioCallAdapter and
    // a plain ws.WebSocket treat a second close() as a no-op.
    this.clientWs.close?.();
    // Deepgram bills for the whole connected duration, not per-turn audio —
    // total call wall-clock is the right proxy, not summed turn lengths.
    const voiceSeconds = (Date.now() - this._callStartedAt) / 1000;
    this.cost.addSttSeconds(voiceSeconds);
    this.cost.logSummary();
    // Fire-and-forget — close() must not block hangup on a Stripe round
    // trip, and a metering failure shouldn't surface as a call failure.
    reportCallUsage(this.stripeCustomerId, {
      voiceSeconds,
      bookingEvents: this.cost.bookingEvents,
      transferEvents: this.cost.transferEvents,
      messageEvents: this.cost.messageEvents,
    }).catch((err) => console.error('[call-loop] usage reporting failed', err));
  }
}

server.listen(PORT, () => {
  console.log(`[call-loop] listening on http://localhost:${PORT}`);
  console.log(`[call-loop] TTS gateway: ${TTS_GATEWAY_WS_URL}`);
});
