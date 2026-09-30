// tenantLookup.fetchCallAudioAssets: Supabase rows + Storage download -> context payload.
// Mocks global fetch (same approach as twilioVoiceIdempotency.test.js).
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-key';

const { fetchCallAudioAssets } = await import('../tenantLookup.js');

const ROWS = [
  { id: 'a1', asset_type: 'jingle', name: 'intro', description: '', mulaw8k_storage_path: 't1/a1.raw', enabled: true },
  { id: 'a2', asset_type: 'sound_effect', name: 'chime', description: 'after booking', mulaw8k_storage_path: 't1/a2.raw', enabled: true },
];
const BYTES = { 't1/a1.raw': [1, 2, 3], 't1/a2.raw': [9, 9] };

let calls;
function mockFetch({ rows = ROWS, failPaths = [] } = {}) {
  calls = [];
  globalThis.fetch = vi.fn(async (url) => {
    calls.push(String(url));
    if (String(url).includes('/rest/v1/tenant_call_audio_assets')) {
      return { ok: true, json: async () => rows };
    }
    const path = String(url).split('/storage/v1/object/call-audio-assets/')[1];
    if (failPaths.includes(path)) return { ok: false, status: 404 };
    return { ok: true, arrayBuffer: async () => new Uint8Array(BYTES[path]).buffer };
  });
}

beforeEach(() => {
  delete process.env.CALL_AUDIO_ASSETS_ENABLED;
  vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => vi.restoreAllMocks());

describe('fetchCallAudioAssets', () => {
  it('returns the encoded jingle + effects for a tenant with enabled assets', async () => {
    mockFetch();
    const out = await fetchCallAudioAssets('tenant-fresh-1');
    expect(out.jingle.name).toBe('intro');
    expect(Buffer.from(out.jingle.audio, 'base64')).toEqual(Buffer.from([1, 2, 3]));
    expect(out.effects).toHaveLength(1);
    expect(out.effects[0]).toMatchObject({ name: 'chime', description: 'after booking' });
    // Only enabled rows are requested at the DB level.
    expect(calls[0]).toContain('enabled=eq.true');
    expect(calls[0]).toContain('tenant_id=eq.tenant-fresh-1');
  });

  it('caches downloaded bytes per asset id: a second call re-queries rows but not Storage', async () => {
    mockFetch();
    await fetchCallAudioAssets('tenant-cache');
    const storageCallsFirst = calls.filter((u) => u.includes('/storage/')).length;
    mockFetch();
    await fetchCallAudioAssets('tenant-cache');
    expect(storageCallsFirst).toBe(0); // (both ids were already cached by the previous test)
    expect(calls.filter((u) => u.includes('/storage/'))).toHaveLength(0);
    expect(calls.filter((u) => u.includes('/rest/v1/'))).toHaveLength(1);
  });

  it('a failed download drops just that asset; the rest still play', async () => {
    // Fresh ids so nothing is cached.
    const rows = [
      { ...ROWS[0], id: 'b1', mulaw8k_storage_path: 't1/a1.raw' },
      { ...ROWS[1], id: 'b2', mulaw8k_storage_path: 't1/a2.raw' },
    ];
    mockFetch({ rows, failPaths: ['t1/a1.raw'] });
    const out = await fetchCallAudioAssets('tenant-partial');
    expect(out.jingle).toBeNull();
    expect(out.effects.map((e) => e.name)).toEqual(['chime']);
  });

  it('returns undefined when every download fails, or the tenant has no rows', async () => {
    const rows = [{ ...ROWS[1], id: 'c1' }];
    mockFetch({ rows, failPaths: ['t1/a2.raw'] });
    expect(await fetchCallAudioAssets('tenant-allfail')).toBeUndefined();
    mockFetch({ rows: [] });
    expect(await fetchCallAudioAssets('tenant-none')).toBeUndefined();
  });

  it('never throws: a network error yields undefined', async () => {
    globalThis.fetch = vi.fn(async () => { throw new Error('boom'); });
    expect(await fetchCallAudioAssets('tenant-neterr')).toBeUndefined();
  });

  it('kill switch off -> undefined WITHOUT touching the network', async () => {
    process.env.CALL_AUDIO_ASSETS_ENABLED = 'false';
    mockFetch();
    expect(await fetchCallAudioAssets('tenant-killed')).toBeUndefined();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});
