import { describe, it, expect } from 'vitest';
import { createDemoGuard, clientIpFromRequest, parseAllowlist, normalizeIp } from '../demoGuard.js';

const OK_ORIGIN = 'https://calldesk.tech';

function clock(start = 1_000_000) {
  let t = start;
  return { now: () => t, advance: (ms) => { t += ms; } };
}

describe('clientIpFromRequest', () => {
  it('prefers Fly-Client-IP, then X-Forwarded-For, then the socket', () => {
    expect(clientIpFromRequest({ headers: { 'fly-client-ip': '1.1.1.1', 'x-forwarded-for': '2.2.2.2' } })).toBe('1.1.1.1');
    expect(clientIpFromRequest({ headers: { 'x-forwarded-for': '2.2.2.2, 3.3.3.3' } })).toBe('2.2.2.2');
    expect(clientIpFromRequest({ headers: {}, socket: { remoteAddress: '::ffff:4.4.4.4' } })).toBe('4.4.4.4');
  });
  it('normalizes IPv4-mapped addresses', () => {
    expect(normalizeIp('::FFFF:1.2.3.4')).toBe('1.2.3.4');
  });
});

describe('allowlist', () => {
  it('lets an allowlisted IP bypass every limit, including origin and concurrency', () => {
    const g = createDemoGuard({ allowlist: '9.9.9.9', perIpConcurrent: 1, perIpPerHour: 1 });
    for (let i = 0; i < 50; i++) {
      const a = g.admit({ ip: '9.9.9.9', origin: 'https://evil.example' });
      expect(a.ok).toBe(true);
      expect(a.unlimited).toBe(true);
      expect(a.maxSessionMs).toBeNull();
    }
  });
  it('matches an IPv6 network with a trailing *', () => {
    const g = createDemoGuard({ allowlist: '2601:647:6801:7e70:*', perIpPerHour: 1, perIpConcurrent: 1 });
    expect(g.admit({ ip: '2601:647:6801:7e70:ad9f:d240:11dd:b422' }).unlimited).toBe(true);
    expect(g.admit({ ip: '2601:647:6801:7e70:1:2:3:4' }).unlimited).toBe(true);
    expect(g.admit({ ip: '2601:647:6801:9999::1' }).unlimited).toBe(false);
  });
  it('ignores blanks and does not treat a bare * entry as match-everything', () => {
    const { exact, prefixes } = parseAllowlist(' , 1.2.3.4 ,');
    expect([...exact]).toEqual(['1.2.3.4']);
    expect(prefixes).toEqual([]);
    const g = createDemoGuard({ allowlist: '*', perIpConcurrent: 1 });
    g.admit({ ip: '5.5.5.5' });
    expect(g.admit({ ip: '5.5.5.5' }).ok).toBe(false);
  });
});

describe('origin check', () => {
  it('rejects a browser origin we do not own, but allows requests with no Origin', () => {
    const g = createDemoGuard();
    expect(g.admit({ ip: '1.1.1.1', origin: 'https://evil.example' })).toMatchObject({ ok: false, code: 'origin_not_allowed' });
    expect(g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN }).ok).toBe(true);
    expect(g.admit({ ip: '1.1.1.2' }).ok).toBe(true);
  });
});

describe('limits', () => {
  it('limits concurrent sessions per IP, and release frees a slot (only once)', () => {
    const g = createDemoGuard({ perIpConcurrent: 2 });
    const a = g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN });
    g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN });
    expect(g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN })).toMatchObject({ ok: false, code: 'too_many_sessions' });
    a.release();
    a.release(); // a second release must not free a second slot
    expect(g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN }).ok).toBe(true);
    expect(g.admit({ ip: '1.1.1.1', origin: OK_ORIGIN }).ok).toBe(false);
  });
  it('limits sessions per hour and per day using a sliding window', () => {
    const c = clock();
    const g = createDemoGuard({ perIpPerHour: 2, perIpPerDay: 3, perIpConcurrent: 99, now: c.now });
    g.admit({ ip: '1.1.1.1' }).release();
    g.admit({ ip: '1.1.1.1' }).release();
    expect(g.admit({ ip: '1.1.1.1' })).toMatchObject({ ok: false, code: 'rate_limited' });
    c.advance(61 * 60 * 1000);
    g.admit({ ip: '1.1.1.1' }).release(); // third of the day
    c.advance(61 * 60 * 1000);
    expect(g.admit({ ip: '1.1.1.1' })).toMatchObject({ ok: false, code: 'rate_limited' }); // daily cap
    c.advance(24 * 60 * 60 * 1000);
    expect(g.admit({ ip: '1.1.1.1' }).ok).toBe(true);
  });
  it('limits total concurrent sessions across IPs', () => {
    const g = createDemoGuard({ globalConcurrent: 2 });
    g.admit({ ip: '1.1.1.1' });
    g.admit({ ip: '2.2.2.2' });
    expect(g.admit({ ip: '3.3.3.3' })).toMatchObject({ ok: false, code: 'busy' });
  });
  it('reports the session length cap for limited sessions', () => {
    const g = createDemoGuard({ maxSessionMs: 123_000 });
    expect(g.admit({ ip: '1.1.1.1' }).maxSessionMs).toBe(123_000);
  });
  it('can be switched off entirely', () => {
    const g = createDemoGuard({ disabled: true, perIpConcurrent: 1 });
    for (let i = 0; i < 5; i++) expect(g.admit({ ip: '1.1.1.1', origin: 'https://evil.example' }).ok).toBe(true);
  });
});
