// SSRF guard for every fetch whose URL a tenant (or anyone but us) controls: function-node webhooks, MCP servers,
// the code node's sandboxed fetch and tenant post-call webhooks.
//
// What it blocks: non-http(s) schemes, URLs with credentials, localhost / .internal / .local style names, any address in
// a private, loopback, link-local, CGNAT, multicast, documentation or reserved range (IPv4 and IPv6, including
// IPv4-mapped IPv6 and Fly's fdaa:: private network), and redirects into any of those.
//
// How: the address is validated at CONNECT time, inside the `lookup` the socket itself uses, so a hostname that
// resolves to a public address when we check and a private one when we connect (DNS rebinding) cannot slip through.
// Redirects are followed manually (max 3), each hop re-validated, and headers other than content-type are dropped when
// a redirect leaves the origin. Responses are capped in size and time.
import dns from 'node:dns/promises';
import net from 'node:net';
import http from 'node:http';
import https from 'node:https';

export class SsrfBlockedError extends Error {
  constructor(message) { super(message); this.name = 'SsrfBlockedError'; }
}

const BLOCKED_HOSTNAME_SUFFIXES = ['.localhost', '.local', '.internal', '.localdomain', '.home.arpa', '.lan', '.intranet', '.corp'];
const BLOCKED_HOSTNAMES = new Set(['localhost', 'metadata', 'metadata.google.internal', 'instance-data']);

function ipv4Bytes(ip) {
  const parts = ip.split('.');
  if (parts.length !== 4) return null;
  const nums = parts.map((p) => (/^\d{1,3}$/.test(p) ? Number(p) : NaN));
  return nums.some((n) => !(n >= 0 && n <= 255)) ? null : nums;
}

function blockedIpv4([a, b, c]) {
  if (a === 0 || a === 10 || a === 127) return true;           // this-network, private, loopback
  if (a === 100 && b >= 64 && b <= 127) return true;            // CGNAT 100.64/10
  if (a === 169 && b === 254) return true;                       // link-local, cloud metadata
  if (a === 172 && b >= 16 && b <= 31) return true;              // private
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return true; // 192.0.0/24 protocol assignments, 192.0.2/24 TEST-NET-1
  if (a === 192 && b === 168) return true;                       // private
  if (a === 198 && (b === 18 || b === 19)) return true;          // benchmarking 198.18/15
  if (a === 198 && b === 51 && c === 100) return true;           // TEST-NET-2
  if (a === 203 && b === 0 && c === 113) return true;            // TEST-NET-3
  if (a >= 224) return true;                                     // multicast, reserved, broadcast
  return false;
}

// Expand an IPv6 literal to 16 bytes. Handles "::" compression and a trailing dotted IPv4.
function ipv6Bytes(ip) {
  let s = ip.split('%')[0].toLowerCase();
  const dotted = /^(.*:)(\d+\.\d+\.\d+\.\d+)$/.exec(s);
  if (dotted) {
    const v4 = ipv4Bytes(dotted[2]);
    if (!v4) return null;
    s = `${dotted[1]}${((v4[0] << 8) | v4[1]).toString(16)}:${((v4[2] << 8) | v4[3]).toString(16)}`;
  }
  const halves = s.split('::');
  if (halves.length > 2) return null;
  const head = halves[0] ? halves[0].split(':') : [];
  const tail = halves.length === 2 && halves[1] ? halves[1].split(':') : [];
  const missing = 8 - head.length - tail.length;
  if (halves.length === 1 ? head.length !== 8 : missing < 1) return null;
  const groups = [...head, ...Array(halves.length === 2 ? missing : 0).fill('0'), ...tail];
  if (groups.length !== 8) return null;
  const out = [];
  for (const g of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(g)) return null;
    const n = parseInt(g, 16);
    out.push(n >> 8, n & 255);
  }
  return out;
}

export function isPrivateOrLocalIp(ip) {
  if (typeof ip !== 'string') return true;
  const addr = ip.replace(/^\[|\]$/g, '');
  if (net.isIPv4(addr)) {
    const b = ipv4Bytes(addr);
    return b ? blockedIpv4(b) : true;
  }
  if (net.isIPv6(addr)) {
    const b = ipv6Bytes(addr);
    if (!b) return true;
    const first96Zero = b.slice(0, 12).every((x) => x === 0);
    const mapped = b.slice(0, 10).every((x) => x === 0) && b[10] === 255 && b[11] === 255; // ::ffff:a.b.c.d
    if (mapped) return blockedIpv4(b.slice(12));
    if (first96Zero) return true;                                    // ::, ::1, deprecated IPv4-compatible ::a.b.c.d
    if (b[0] === 0x00 && b[1] === 0x64 && b[2] === 0xff && b[3] === 0x9b && b.slice(4, 12).every((x) => x === 0)) return true; // NAT64 64:ff9b::/96
    if (b[0] === 0x01 && b[1] === 0x00 && b.slice(2, 8).every((x) => x === 0)) return true; // discard 100::/64
    if ((b[0] & 0xfe) === 0xfc) return true;                         // unique local fc00::/7 (includes Fly fdaa::)
    if (b[0] === 0xfe && (b[1] & 0xc0) === 0x80) return true;        // link-local fe80::/10
    if (b[0] === 0xfe && (b[1] & 0xc0) === 0xc0) return true;        // site-local fec0::/10
    if (b[0] === 0xff) return true;                                  // multicast
    if (b[0] === 0x20 && b[1] === 0x01 && b[2] === 0x0d && b[3] === 0xb8) return true; // documentation 2001:db8::/32
    if (b[0] === 0x20 && b[1] === 0x01 && b[2] === 0x00 && b[3] === 0x00) return true; // Teredo 2001::/32
    if (b[0] === 0x20 && b[1] === 0x02) return true;                 // 6to4 2002::/16 (embeds an IPv4 we would have to vet)
    return false;
  }
  return true; // not an IP at all: refuse rather than guess
}

/** Throws SsrfBlockedError for anything that is not a plain public http(s) URL shape. Returns the parsed URL. */
export function validateUrlShape(urlString) {
  let url;
  try { url = new URL(String(urlString)); } catch { throw new SsrfBlockedError('Invalid URL'); }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new SsrfBlockedError('Only http:// and https:// URLs are allowed');
  if (url.username || url.password) throw new SsrfBlockedError('URLs with embedded credentials are not allowed');
  const host = url.hostname.replace(/^\[|\]$/g, '').replace(/\.$/, '').toLowerCase();
  if (!host) throw new SsrfBlockedError('Invalid URL');
  if (BLOCKED_HOSTNAMES.has(host) || BLOCKED_HOSTNAME_SUFFIXES.some((s) => host.endsWith(s))) throw new SsrfBlockedError('Requests to internal host names are not allowed');
  if (net.isIP(host) && isPrivateOrLocalIp(host)) throw new SsrfBlockedError('Requests to private/local network addresses are not allowed');
  return url;
}

// A `lookup` for http(s).request that refuses to hand the socket a non-public address.
export function guardedLookup({ isAllowedIp = (ip) => !isPrivateOrLocalIp(ip), lookup = (h, o) => dns.lookup(h, o) } = {}) {
  return (hostname, options, callback) => {
    if (typeof options === 'function') { callback = options; options = {}; }
    lookup(hostname, { all: true, family: options?.family || 0, hints: options?.hints }).then((addrs) => {
      if (!addrs.length) return callback(new SsrfBlockedError(`DNS lookup for "${hostname}" returned nothing`));
      if (addrs.some((a) => !isAllowedIp(a.address, hostname))) return callback(new SsrfBlockedError('Requests to private/local network addresses are not allowed'));
      return options?.all ? callback(null, addrs) : callback(null, addrs[0].address, addrs[0].family);
    }, (err) => callback(err));
  };
}

function requestOnce(url, { method, headers, body, timeoutMs, maxBytes, lookup }) {
  return new Promise((resolve, reject) => {
    const client = url.protocol === 'https:' ? https : http;
    const payload = body === undefined || body === null ? null : Buffer.isBuffer(body) ? body : Buffer.from(String(body));
    const h = { 'accept-encoding': 'identity', ...Object.fromEntries(Object.entries(headers || {}).map(([k, v]) => [k.toLowerCase(), String(v)])) };
    if (payload) h['content-length'] = String(payload.length);
    const tooLarge = () => new Error(`Response too large (over ${maxBytes} bytes)`);
    const req = client.request(url, { method, headers: h, lookup, signal: AbortSignal.timeout(timeoutMs) }, (res) => {
      const declared = Number(res.headers['content-length']);
      if (declared > maxBytes) { reject(tooLarge()); req.destroy(); return; }
      const chunks = []; let size = 0;
      res.on('data', (c) => {
        size += c.length;
        if (size > maxBytes) { reject(tooLarge()); req.destroy(); return; }
        chunks.push(c);
      });
      res.on('end', () => resolve({ status: res.statusCode, statusText: res.statusMessage || '', headers: res.headers, body: Buffer.concat(chunks) }));
      res.on('error', reject);
      // A connection that drops mid-body must not look like a complete (shorter) response.
      res.on('close', () => { if (!res.complete) reject(new Error('Connection closed before the response completed')); });
    });
    req.on('error', (err) => reject(err.name === 'TimeoutError' || err.code === 'ABORT_ERR' ? new Error(`Request timed out after ${timeoutMs} ms`) : err));
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * fetch() for untrusted URLs. Resolves to a standard Response (so call sites keep using .ok/.status/.text()/.json()/.headers).
 * Rejects with SsrfBlockedError when the URL or any redirect hop is not a public http(s) destination.
 * `deps` exists for tests only: nothing a tenant controls can reach it.
 */
export async function safeFetch(urlString, options = {}, deps = {}) {
  const { method = 'GET', headers, body, timeoutMs = 8000, maxBytes = 1_000_000, maxRedirects = 3 } = options;
  const lookup = guardedLookup(deps);
  let url = validateUrlShape(urlString);
  let curMethod = String(method).toUpperCase(); let curBody = body; let curHeaders = headers || {};
  for (let hop = 0; ; hop++) {
    const r = await requestOnce(url, { method: curMethod, headers: curHeaders, body: curBody, timeoutMs, maxBytes, lookup });
    const location = r.headers.location;
    if ([301, 302, 303, 307, 308].includes(r.status) && location) {
      if (hop >= maxRedirects) throw new Error('Too many redirects');
      const next = validateUrlShape(new URL(location, url).toString());
      if (next.origin !== url.origin) {
        // Leaving the origin: never forward Authorization, cookies or custom headers to someone else.
        curHeaders = Object.fromEntries(Object.entries(curHeaders).filter(([k]) => k.toLowerCase() === 'content-type'));
      }
      if (r.status === 303 || ((r.status === 301 || r.status === 302) && curMethod === 'POST')) { curMethod = 'GET'; curBody = undefined; curHeaders = Object.fromEntries(Object.entries(curHeaders).filter(([k]) => k.toLowerCase() !== 'content-type')); }
      url = next;
      continue;
    }
    const responseHeaders = new Headers();
    for (const [k, v] of Object.entries(r.headers)) for (const item of Array.isArray(v) ? v : [v]) if (item !== undefined) responseHeaders.append(k, item);
    const noBody = [101, 204, 205, 304].includes(r.status);
    return new Response(noBody ? null : r.body, { status: r.status, statusText: r.statusText, headers: responseHeaders });
  }
}
