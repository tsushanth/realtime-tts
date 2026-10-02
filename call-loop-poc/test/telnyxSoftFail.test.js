// Features not implemented for Telnyx in v1 must log, soft-fail and never call Twilio with a Telnyx call id.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

process.env.NODE_ENV = 'test';
process.env.TWILIO_ACCOUNT_SID = 'ACfake';
process.env.TWILIO_AUTH_TOKEN = 'fake-token';
const { CallSession } = await import('../server.js');
const { CallCostTracker, RATES } = await import('../costTracker.js');
const P = CallSession.prototype;

describe('unsupported-on-telnyx guards', () => {
  let warn, fetchSpy;
  const realFetch = globalThis.fetch;
  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    fetchSpy = globalThis.fetch = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }));
  });
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });
  const stub = (extra = {}) => ({ clientWs: { carrier: 'telnyx', callSid: 'CS-T' }, callSid: 'CS-T', id: 'sid', collectedData: {}, close: vi.fn(), _applyTransition: vi.fn(), _runNodeTurn: vi.fn(), currentNodeId: 'n', ...extra });
  const said = () => warn.mock.calls.map((c) => c[0]).join('\n');

  it('transfer: logs, hangs up (same as the existing cannot-transfer path), no Twilio call', async () => {
    const s = stub();
    await P._executeTransfer.call(s, { transferTo: '+15550001111' });
    expect(said()).toContain('transfer is not supported on telnyx yet');
    expect(s.close).toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('detour redirect returns false without calling Twilio', async () => {
    expect(await P._redirectForDetour.call(stub(), '<Response/>')).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('payment: reports payment_status=failed to the flow and re-runs the node turn, no Twilio call', async () => {
    const s = stub();
    await P._executePayment.call(s, { amount: 5 });
    expect(s.collectedData.payment_status).toBe('failed');
    expect(s.collectedData.payment_status_detail).toBe('unsupported_on_telnyx');
    expect(s._paymentAwaitingResume).toBe(true);
    expect(s._runNodeTurn).toHaveBeenCalledWith('n');
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(said()).toContain('not supported on telnyx yet');
  });

  it('press_digit: skips the tones and advances along the first edge', async () => {
    const s = stub();
    await P._executePressDigit.call(s, { id: 'pd', params: { digits: '1' }, edges: [{ target: 'next' }] });
    expect(s._applyTransition).toHaveBeenCalledWith({ next_node_id: 'next' });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('sms: skipped, no Twilio call', async () => {
    const s = stub({ phoneNumber: '+1555', tenantNumber: '+1666' });
    await P._executeSmsNode.call(s, { id: 'sms1', params: { body: 'hi' } });
    expect(said()).toContain('not supported on telnyx yet');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('a Twilio call (no carrier on the adapter) still takes the original path', async () => {
    const s = { clientWs: { callSid: 'CA1' }, collectedData: {}, close: vi.fn(), _stashForResume: vi.fn() };
    await P._executeTransfer.call(s, { transferTo: '+15550001111' });
    expect(said()).not.toContain('telnyx');
    expect(fetchSpy).toHaveBeenCalled(); // reached the Twilio REST call
  });
});

describe('carrier-aware telephony cost', () => {
  it('default tracker: breakdown has no telephony key (unchanged)', () => {
    const t = new CallCostTracker();
    t.addSttSeconds(30);
    t.addTelephonySeconds(30);
    expect(Object.keys(t.breakdown())).toEqual(['engine', 'ttsBackend', 'deepgram', 'claude', 'tts', 'total']);
  });
  it('telnyx: per-minute rate with 60 s minimum on the carrier leg', () => {
    const per = 0.0032 + 0.002 + 0.0035;
    const t = new CallCostTracker();
    t.setCarrier('telnyx');
    t.addTelephonySeconds(20);
    expect(t.breakdown().telephony).toBeCloseTo(per, 6); // billed as 60 s
    const u = new CallCostTracker();
    u.setCarrier('telnyx');
    u.addTelephonySeconds(150);
    expect(u.breakdown().telephony).toBeCloseTo(2.5 * per, 6);
    expect(u.breakdown().total).toBeCloseTo(u.breakdown().telephony + u.breakdown().deepgram, 8);
    const z = new CallCostTracker();
    z.setCarrier('telnyx');
    expect(z.breakdown().telephony).toBe(0); // no call leg, no charge
    expect(RATES.telephony.telnyx.minSeconds).toBe(60);
  });
});
