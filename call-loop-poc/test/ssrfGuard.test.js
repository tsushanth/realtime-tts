import { describe, it, expect, afterEach } from 'vitest';
import http from 'node:http';
import { isPrivateOrLocalIp, validateUrlShape, guardedLookup, safeFetch, SsrfBlockedError } from '../ssrfGuard.js';

describe('isPrivateOrLocalIp', () => {
  const blocked = [
    '0.0.0.0', '127.0.0.1', '127.255.255.254', '10.0.0.1', '10.255.255.255', '172.16.0.1', '172.31.255.255', '192.168.1.1',
    '169.254.169.254', '100.64.0.1', '100.127.255.255', '192.0.0.1', '192.0.2.5', '198.18.0.1', '198.19.255.255', '198.51.100.7',
    '203.0.113.9', '224.0.0.1', '239.255.255.250', '240.0.0.1', '255.255.255.255',
    '::', '::1', '::ffff:127.0.0.1', '::ffff:7f00:1', '::ffff:a00:1', '::ffff:169.254.169.254', '::7f00:1', 'fc00::1', 'fd00::1',
    'fdaa:0:1::3', 'fe80::1', 'fe80::1%eth0', 'febf::1', 'fec0::1', 'ff02::1', '64:ff9b::7f00:1', '100::1', '2001:db8::1', '2001::1', '2002:7f00:1::',
  ];
  const allowed = ['8.8.8.8', '1.1.1.1', '93.184.216.34', '100.63.255.255', '100.128.0.1', '172.15.0.1', '172.32.0.1', '192.0.1.1', '198.17.0.1', '223.255.255.255',
    '2606:4700:4700::1111', '2001:4860:4860::8888', '::ffff:8.8.8.8', '2a00:1450:4001:81b::200e'];
  for (const ip of blocked) it(`blocks ${ip}`, () => expect(isPrivateOrLocalIp(ip)).toBe(true));
  for (const ip of allowed) it(`allows ${ip}`, () => expect(isPrivateOrLocalIp(ip)).toBe(false));
  it('refuses things that are not IP addresses', () => {
    for (const x of ['', 'localhost', 'example.com', '999.1.1.1', '1.2.3', null, undefined, 42]) expect(isPrivateOrLocalIp(x)).toBe(true);
  });
});

describe('validateUrlShape', () => {
  const bad = [
    'file:///etc/passwd', 'ftp://example.com/', 'gopher://example.com/', 'javascript:alert(1)', 'data:text/plain,hi', 'not a url', '',
    'http://user:pass@example.com/', 'http://localhost/', 'http://LOCALHOST:8090/', 'http://foo.localhost/', 'http://svc.internal/', 'http://printer.local/',
    'http://metadata.google.internal/computeMetadata/v1/', 'http://127.0.0.1/', 'http://127.1/', 'http://0x7f.0.0.1/', 'http://2130706433/', 'http://0177.0.0.1/',
    'http://[::1]/', 'http://[::ffff:127.0.0.1]/', 'http://[fdaa:0:1::3]/', 'http://169.254.169.254/latest/meta-data/', 'http://10.0.0.5:8080/', 'http://192.168.0.1/',
    'http://localhost./', 'http://example.com@127.0.0.1/',
  ];
  const good = ['https://example.com/hook', 'http://example.com:8080/x?y=1', 'https://api.partner.io/v1/mcp', 'https://8.8.8.8/', 'https://[2606:4700:4700::1111]/'];
  for (const u of bad) it(`rejects ${JSON.stringify(u)}`, () => expect(() => validateUrlShape(u)).toThrow(SsrfBlockedError));
  for (const u of good) it(`accepts ${u}`, () => expect(() => validateUrlShape(u)).not.toThrow());
});

describe('guardedLookup (connect-time check, the DNS-rebinding defence)', () => {
  const run = (lookup, opts = { all: true }) => new Promise((resolve) => guardedLookup({ lookup })('h.test', opts, (err, a, f) => resolve({ err, a, f })));
  it('rejects when ANY resolved address is private, even if others are public', async () => {
    const r = await run(async () => [{ address: '8.8.8.8', family: 4 }, { address: '10.0.0.7', family: 4 }]);
    expect(r.err).toBeInstanceOf(SsrfBlockedError);
  });
  it('rejects a hostname that resolves to loopback', async () => {
    expect((await run(async () => [{ address: '127.0.0.1', family: 4 }])).err).toBeInstanceOf(SsrfBlockedError);
  });
  it('rejects an empty answer and passes lookup errors through', async () => {
    expect((await run(async () => [])).err).toBeInstanceOf(SsrfBlockedError);
    expect((await run(async () => { throw new Error('ENOTFOUND'); })).err.message).toBe('ENOTFOUND');
  });
  it('passes public addresses through in both all and single modes', async () => {
    const pub = async () => [{ address: '93.184.216.34', family: 4 }];
    expect((await run(pub)).a).toEqual([{ address: '93.184.216.34', family: 4 }]);
    const single = await run(pub, {});
    expect(single.a).toBe('93.184.216.34'); expect(single.f).toBe(4);
  });
  it('asks again on every connection, so a rebinding answer is caught on the second lookup', async () => {
    let n = 0;
    const lookup = guardedLookup({ lookup: async () => [{ address: n++ === 0 ? '93.184.216.34' : '127.0.0.1', family: 4 }] });
    const call = () => new Promise((resolve) => lookup('rebind.test', { all: true }, (err) => resolve(err)));
    expect(await call()).toBeNull();
    expect(await call()).toBeInstanceOf(SsrfBlockedError);
  });
});

// Real sockets: a local server stands in for "a public host". The hostname decides what the test guard allows.
describe('safeFetch against a local server', () => {
  const servers = [];
  const start = (handler) => new Promise((resolve) => {
    const s = http.createServer(handler); servers.push(s);
    s.listen(0, '127.0.0.1', () => resolve(s.address().port));
  });
  afterEach(async () => { await Promise.all(servers.splice(0).map((s) => new Promise((r) => { s.closeAllConnections?.(); s.close(r); }))); });
  // All test hostnames resolve to loopback; only *.ok.test is "allowed".
  const deps = { lookup: async () => [{ address: '127.0.0.1', family: 4 }], isAllowedIp: (ip, host) => /\.ok\.test$/.test(host) };

  it('blocks loopback by default, with no test hooks', async () => {
    const port = await start((req, res) => res.end('secret'));
    await expect(safeFetch(`http://127.0.0.1:${port}/`)).rejects.toBeInstanceOf(SsrfBlockedError);
    await expect(safeFetch(`http://localhost:${port}/`)).rejects.toBeInstanceOf(SsrfBlockedError);
  });
  it('blocks a name that resolves to a private address even though the URL looks innocent', async () => {
    const port = await start((req, res) => res.end('secret'));
    await expect(safeFetch(`http://evil.bad.test:${port}/`, {}, deps)).rejects.toBeInstanceOf(SsrfBlockedError);
  });
  it('POSTs a body and returns a normal Response', async () => {
    const port = await start((req, res) => { let b = ''; req.on('data', (c) => (b += c)); req.on('end', () => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ got: b, method: req.method, ct: req.headers['content-type'] })); }); });
    const res = await safeFetch(`http://a.ok.test:${port}/hook`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Test': '1' }, body: '{"hello":"world"}' }, deps);
    expect(res.ok).toBe(true); expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/json');
    expect(await res.json()).toEqual({ got: '{"hello":"world"}', method: 'POST', ct: 'application/json' });
  });
  it('returns non-2xx responses without throwing', async () => {
    const port = await start((req, res) => { res.statusCode = 503; res.end('down'); });
    const res = await safeFetch(`http://a.ok.test:${port}/`, {}, deps);
    expect(res.ok).toBe(false); expect(res.status).toBe(503); expect(await res.text()).toBe('down');
  });
  it('handles an empty 204', async () => {
    const port = await start((req, res) => { res.statusCode = 204; res.end(); });
    const res = await safeFetch(`http://a.ok.test:${port}/`, {}, deps);
    expect(res.status).toBe(204);
  });
  it('follows a redirect between allowed hosts and drops Authorization when the origin changes', async () => {
    const seen = {};
    const p2 = await start((req, res) => { seen.auth2 = req.headers.authorization; seen.custom2 = req.headers['x-secret']; res.end('arrived'); });
    const p1 = await start((req, res) => { seen.auth1 = req.headers.authorization; res.statusCode = 302; res.setHeader('location', `http://b.ok.test:${p2}/final`); res.end(); });
    const res = await safeFetch(`http://a.ok.test:${p1}/start`, { headers: { Authorization: 'Bearer top-secret', 'X-Secret': 'x' } }, deps);
    expect(await res.text()).toBe('arrived');
    expect(seen.auth1).toBe('Bearer top-secret');
    expect(seen.auth2).toBeUndefined();
    expect(seen.custom2).toBeUndefined();
  });
  it('keeps headers on a same-origin redirect', async () => {
    let auth;
    const port = await start((req, res) => {
      if (req.url === '/a') { res.statusCode = 307; res.setHeader('location', '/b'); return res.end(); }
      auth = req.headers.authorization; res.end('ok');
    });
    await safeFetch(`http://a.ok.test:${port}/a`, { headers: { Authorization: 'Bearer same' } }, deps);
    expect(auth).toBe('Bearer same');
  });
  it('refuses a redirect into a private destination (the classic SSRF bypass)', async () => {
    let hits = 0;
    const target = await start((req, res) => { hits++; res.end('metadata'); });
    const port = await start((req, res) => { res.statusCode = 302; res.setHeader('location', `http://internal.bad.test:${target}/`); res.end(); });
    await expect(safeFetch(`http://a.ok.test:${port}/`, {}, deps)).rejects.toBeInstanceOf(SsrfBlockedError);
    expect(hits).toBe(0);
  });
  it('refuses a redirect to a literal metadata address and to a non-http scheme', async () => {
    const p1 = await start((req, res) => { res.statusCode = 301; res.setHeader('location', 'http://169.254.169.254/latest/meta-data/'); res.end(); });
    await expect(safeFetch(`http://a.ok.test:${p1}/`, {}, deps)).rejects.toBeInstanceOf(SsrfBlockedError);
    const p2 = await start((req, res) => { res.statusCode = 302; res.setHeader('location', 'file:///etc/passwd'); res.end(); });
    await expect(safeFetch(`http://a.ok.test:${p2}/`, {}, deps)).rejects.toBeInstanceOf(SsrfBlockedError);
  });
  it('stops redirect loops', async () => {
    const port = await start((req, res) => { res.statusCode = 302; res.setHeader('location', '/again'); res.end(); });
    await expect(safeFetch(`http://a.ok.test:${port}/`, {}, deps)).rejects.toThrow(/Too many redirects/);
  });
  it('turns a POST into a GET on 303', async () => {
    const methods = [];
    const port = await start((req, res) => { methods.push(req.method); if (req.url === '/') { res.statusCode = 303; res.setHeader('location', '/done'); return res.end(); } res.end('ok'); });
    await safeFetch(`http://a.ok.test:${port}/`, { method: 'POST', body: 'x' }, deps);
    expect(methods).toEqual(['POST', 'GET']);
  });
  it('caps the response size', async () => {
    const port = await start((req, res) => res.end('x'.repeat(5000)));
    await expect(safeFetch(`http://a.ok.test:${port}/`, { maxBytes: 1000 }, deps)).rejects.toThrow(/too large/i);
  });
  it('rejects early when the server declares a body that is too large, and when a body is streamed past the cap', async () => {
    const declared = await start((req, res) => { res.setHeader('content-length', '999999'); res.write('x'); setTimeout(() => res.end(), 50); });
    await expect(safeFetch(`http://a.ok.test:${declared}/`, { maxBytes: 1000 }, deps)).rejects.toThrow(/too large/i);
    const streamed = await start((req, res) => { res.write('x'.repeat(600)); setTimeout(() => { res.write('y'.repeat(600)); res.end(); }, 30); });
    await expect(safeFetch(`http://a.ok.test:${streamed}/`, { maxBytes: 1000 }, deps)).rejects.toThrow(/too large/i);
  });
  it('treats a connection cut mid-body as a failure, not a short success', async () => {
    const port = await start((req, res) => { res.setHeader('content-length', '100'); res.write('partial'); setTimeout(() => req.socket.destroy(), 30); });
    await expect(safeFetch(`http://a.ok.test:${port}/`, {}, deps)).rejects.toThrow();
  });
  it('times out a server that never answers', async () => {
    const port = await start(() => { /* never respond */ });
    await expect(safeFetch(`http://a.ok.test:${port}/`, { timeoutMs: 250 }, deps)).rejects.toThrow(/timed out/i);
  });
});
