// Admission control for browser sessions on the /call WebSocket.
//
// /call is reachable from any web page, and every session spends money (speech
// recognition, the model, and voice synthesis). This decides who may open one:
//   - allowlisted IPs skip every limit (for testing),
//   - a browser page must come from one of our own origins,
//   - each IP is limited in concurrent sessions and in sessions per hour / day,
//   - the whole server is limited in concurrent sessions.
// Sessions are also capped in length by the caller (see maxSessionMs).
// State is in memory, so limits reset when the server restarts and are per
// machine; that is fine for a single-machine deployment.

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;

export const DEFAULT_ALLOWED_ORIGINS = [
  'https://calldesk.tech',
  'https://www.calldesk.tech',
  'https://calldesk-tech.fly.dev',
  'https://call-loop-poc.fly.dev',
  'http://localhost:3000',
  'http://localhost:3001',
  'http://localhost:3100',
  'http://localhost:8090',
];

export const MESSAGES = {
  origin_not_allowed: 'This demo can only be used from calldesk.tech.',
  busy: 'The demo is busy right now. Try again in a minute.',
  too_many_sessions: 'You already have a demo open. Close it to start another.',
  rate_limited: "You've reached the demo limit for now. Set up your own agent to keep going, or try again later.",
  session_limit: 'The demo ended after its time limit.',
};

// Strips the IPv4-mapped IPv6 prefix so "::ffff:1.2.3.4" and "1.2.3.4" match.
export function normalizeIp(ip) {
  if (!ip) return '';
  const s = String(ip).trim().toLowerCase();
  return s.startsWith('::ffff:') ? s.slice(7) : s;
}

// Fly's proxy sets Fly-Client-IP to the real client address. X-Forwarded-For is
// the fallback for other proxies and local runs.
export function clientIpFromRequest(req) {
  const h = req.headers || {};
  const fly = h['fly-client-ip'];
  if (fly) return normalizeIp(Array.isArray(fly) ? fly[0] : fly);
  const xff = h['x-forwarded-for'];
  if (xff) return normalizeIp(String(Array.isArray(xff) ? xff[0] : xff).split(',')[0]);
  return normalizeIp(req.socket?.remoteAddress);
}

// "1.2.3.4, 2601:647:6801:7e70:*" -> exact addresses plus trailing-* prefixes.
// A prefix entry covers a whole IPv6 network, since home connections rotate the
// last half of their address.
export function parseAllowlist(value) {
  const exact = new Set();
  const prefixes = [];
  for (const raw of String(value || '').split(',')) {
    const entry = normalizeIp(raw);
    if (!entry) continue;
    if (entry.endsWith('*')) prefixes.push(entry.slice(0, -1));
    else exact.add(entry);
  }
  return { exact, prefixes };
}

function isAllowlisted(allow, ip) {
  if (!ip) return false;
  return allow.exact.has(ip) || allow.prefixes.some((p) => p && ip.startsWith(p));
}

export function createDemoGuard({
  perIpPerHour = 10,
  perIpPerDay = 30,
  perIpConcurrent = 2,
  globalConcurrent = 20,
  maxSessionMs = 210_000,
  allowlist = '',
  allowedOrigins = DEFAULT_ALLOWED_ORIGINS,
  disabled = false,
  now = () => Date.now(),
} = {}) {
  const allow = parseAllowlist(allowlist);
  const origins = new Set(allowedOrigins.map((o) => o.toLowerCase()));
  const starts = new Map(); // ip -> start timestamps within the last day
  const active = new Map(); // ip -> concurrent count
  let activeTotal = 0;

  function recent(ip, t) {
    const list = (starts.get(ip) || []).filter((s) => t - s < DAY_MS);
    if (list.length) starts.set(ip, list);
    else starts.delete(ip);
    return list;
  }

  function reject(code) {
    return { ok: false, code, message: MESSAGES[code] };
  }

  function admit({ ip, origin }) {
    const addr = normalizeIp(ip);
    if (disabled || isAllowlisted(allow, addr)) {
      return { ok: true, unlimited: true, maxSessionMs: null, release() {} };
    }
    if (origin && !origins.has(String(origin).toLowerCase())) return reject('origin_not_allowed');
    if (activeTotal >= globalConcurrent) return reject('busy');
    if ((active.get(addr) || 0) >= perIpConcurrent) return reject('too_many_sessions');
    const t = now();
    const list = recent(addr, t);
    if (list.length >= perIpPerDay || list.filter((s) => t - s < HOUR_MS).length >= perIpPerHour) {
      return reject('rate_limited');
    }
    list.push(t);
    starts.set(addr, list);
    active.set(addr, (active.get(addr) || 0) + 1);
    activeTotal += 1;
    let released = false;
    return {
      ok: true,
      unlimited: false,
      maxSessionMs,
      release() {
        if (released) return;
        released = true;
        const n = (active.get(addr) || 1) - 1;
        if (n > 0) active.set(addr, n);
        else active.delete(addr);
        activeTotal = Math.max(0, activeTotal - 1);
      },
    };
  }

  // Drops idle bookkeeping so the maps do not grow without bound.
  function prune() {
    const t = now();
    for (const ip of [...starts.keys()]) recent(ip, t);
  }

  return { admit, prune, stats: () => ({ activeTotal, trackedIps: starts.size }) };
}

// The guard the server uses, configured from the environment.
export function createDemoGuardFromEnv(env = process.env) {
  const num = (v, d) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Number(v) : d);
  const extraOrigins = String(env.DEMO_ALLOWED_ORIGINS || '').split(',').map((s) => s.trim()).filter(Boolean);
  return createDemoGuard({
    perIpPerHour: num(env.DEMO_PER_IP_PER_HOUR, 10),
    perIpPerDay: num(env.DEMO_PER_IP_PER_DAY, 30),
    perIpConcurrent: num(env.DEMO_PER_IP_CONCURRENT, 2),
    globalConcurrent: num(env.DEMO_GLOBAL_CONCURRENT, 20),
    maxSessionMs: num(env.DEMO_MAX_SESSION_SEC, 210) * 1000,
    allowlist: env.DEMO_ALLOWLIST_IPS || '',
    allowedOrigins: [...DEFAULT_ALLOWED_ORIGINS, ...extraOrigins],
    disabled: env.DEMO_GUARD_DISABLED === '1',
  });
}
