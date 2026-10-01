// Regression test for a real mystery-shopper production bug (2026-09-28):
// _buildTurnSystemPrompt used to append SLOT_SAFETY_INSTRUCTION to every
// turn's system prompt unconditionally, including the shopper's own turn.
// That instruction is written from the business agent's point of view
// ("the caller has not told you", "ask ONE soft clarification") — on a
// shopper session it told the shopper's LLM to act like the one collecting
// fields from "the caller", i.e. like the receptionist instead of the
// customer. Two real calls reproduced the shopper asking the business agent
// for name/time/callback number instead of answering. Fix: skip
// SLOT_SAFETY_INSTRUCTION when this.isShopper is true.
//
// Tested via CallSession.prototype._buildTurnSystemPrompt.call(stub, ...)
// on a minimal stub, same approach as test/turnTaking.test.js — this method
// is pure string assembly over a handful of fields, no network/timers.
import { describe, it, expect } from 'vitest';
import { CallSession } from '../server.js';

function makeStub(overrides = {}) {
  return {
    flow: null,
    systemPrompt: 'You are Alex Morgan, calling a medical clinic to book an appointment...',
    lang: null,
    expressiveDelivery: false,
    isShopper: false,
    ...overrides,
  };
}

describe('_buildTurnSystemPrompt (system prompt assembly)', () => {
  it('a normal (non-shopper) session gets the slot-safety instruction appended', () => {
    const stub = makeStub({ isShopper: false });
    const prompt = CallSession.prototype._buildTurnSystemPrompt.call(stub, null, false);
    expect(prompt).toContain('Slot accuracy');
    expect(prompt).toContain('the caller has not told you');
  });

  it('a shopper session does NOT get the slot-safety instruction appended (real bug: it told the shopper to act like the receptionist)', () => {
    const stub = makeStub({ isShopper: true });
    const prompt = CallSession.prototype._buildTurnSystemPrompt.call(stub, null, false);
    expect(prompt).not.toContain('Slot accuracy');
    expect(prompt).not.toContain('the caller has not told you');
    // The shopper's own persona/system prompt must still be present, untouched.
    expect(prompt).toContain('You are Alex Morgan');
  });

  it('a shopper session with a flow node still skips slot-safety even though a normal node-entry turn would get it', () => {
    const node = { id: 'n1', type: 'conversation', params: {}, edges: [] };
    const stub = makeStub({
      isShopper: true,
      flow: { globalSettings: {} },
      _buildNodeSystemPrompt: () => 'NODE PROMPT',
    });
    const prompt = CallSession.prototype._buildTurnSystemPrompt.call(stub, node, true);
    expect(prompt).toContain('NODE PROMPT');
    expect(prompt).not.toContain('Slot accuracy');
  });
});
