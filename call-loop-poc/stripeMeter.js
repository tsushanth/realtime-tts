// Reports real usage to Stripe as legacy metered usage records on the
// customer's subscription items. Switched from the newer billing/meter_events
// API (2026-09-26) because the calldesktech checkout creates subscriptions
// with legacy metered prices; usage_records on subscription_items is the API
// that actually feeds those prices.
//
// Fire-and-forget: a metering failure must never take down or delay a live
// call, so every call here only logs on error.
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;

// Price IDs from calldesktech/src/lib/constants.ts — the checkout creates
// subscriptions with these. Voice has multiple possibilities (backend switch).
const VOICE_PRICE_IDS = new Set([
  process.env.STRIPE_VOICE_KOKORO_PRICE || 'price_1UHXOqKFBTQTkmztIPlG72Wy',
  process.env.STRIPE_VOICE_ELEVENLABS_PRICE || 'price_1UHXOrKFBTQTkmzt8nJK5u1b',
  process.env.STRIPE_VOICE_CARTESIA_PRICE || 'price_1UHXOrKFBTQTkmztD1ehUQFd',
  process.env.STRIPE_VOICE_MINIMAX_PRICE || 'price_1UCszEKFBTQTkmztXKqkApW7',
]);
const BOOKING_PRICE_ID = process.env.STRIPE_BOOKING_PRICE_ID || 'price_1U8tJtKFBTQTkmzt8CqFVIDs';
const TRANSFER_PRICE_ID = process.env.STRIPE_TRANSFER_PRICE_ID || 'price_1U8tJtKFBTQTkmztdgtdAu8n';
const MESSAGE_PRICE_ID = process.env.STRIPE_MESSAGE_PRICE_ID || 'price_1U8tJuKFBTQTkmztzhIqUrg3';

// Cache subscription-item lookups per customer (they change rarely — only when
// syncVoicePriceForTenant swaps the voice price).
const itemCache = new Map(); // customerId -> { voiceItemId, bookingItemId, transferItemId, messageItemId, expiresAt }
const CACHE_TTL_MS = 5 * 60 * 1000;

async function resolveSubscriptionItems(stripeCustomerId) {
  const cached = itemCache.get(stripeCustomerId);
  if (cached && Date.now() < cached.expiresAt) return cached;

  if (!STRIPE_SECRET_KEY) return null;

  // Fetch the customer's active subscriptions.
  const subsRes = await fetch(
    `https://api.stripe.com/v1/customers/${encodeURIComponent(stripeCustomerId)}/subscriptions?status=active&limit=1`,
    { headers: { Authorization: `Bearer ${STRIPE_SECRET_KEY}` }, signal: AbortSignal.timeout(8000) }
  );
  if (!subsRes.ok) {
    console.error(`[stripe-meter] failed to fetch subscriptions for ${stripeCustomerId}: HTTP ${subsRes.status}`);
    return null;
  }
  const subs = await subsRes.json();
  const subscription = subs?.data?.[0];
  if (!subscription) return null;

  const result = { voiceItemId: null, bookingItemId: null, transferItemId: null, messageItemId: null, expiresAt: Date.now() + CACHE_TTL_MS };

  for (const item of subscription.items?.data || []) {
    const priceId = item.price?.id;
    if (!priceId) continue;
    if (VOICE_PRICE_IDS.has(priceId)) result.voiceItemId = item.id;
    else if (priceId === BOOKING_PRICE_ID) result.bookingItemId = item.id;
    else if (priceId === TRANSFER_PRICE_ID) result.transferItemId = item.id;
    else if (priceId === MESSAGE_PRICE_ID) result.messageItemId = item.id;
  }

  itemCache.set(stripeCustomerId, result);
  return result;
}

async function reportUsageRecord(itemId, value) {
  if (!STRIPE_SECRET_KEY || !itemId || !value) return;
  try {
    const res = await fetch(`https://api.stripe.com/v1/subscription_items/${encodeURIComponent(itemId)}/usage_records`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${STRIPE_SECRET_KEY}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ quantity: String(Math.max(0, Math.round(value))) }),
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      // A 4xx on a usage record usually means the item was just swapped (e.g.
      // syncVoicePriceForTenant changed the voice backend). Clear the cache
      // so the next call re-resolves the correct item.
      if (res.status >= 400 && res.status < 500) {
        for (const [customerId, cached] of itemCache) {
          if (cached.voiceItemId === itemId || cached.bookingItemId === itemId || cached.transferItemId === itemId || cached.messageItemId === itemId) {
            itemCache.delete(customerId);
            console.log(`[stripe-meter] cleared cache for ${customerId} after ${res.status} on usage record`);
            break;
          }
        }
      }
      console.error(`[stripe-meter] usage record -> HTTP ${res.status}: ${text}`);
    }
  } catch (err) {
    console.error('[stripe-meter] usage record failed', err);
  }
}

// One call's worth of usage, reported once at hangup — see CallSession.close().
export async function reportCallUsage(stripeCustomerId, { voiceSeconds, bookingEvents, transferEvents, messageEvents }) {
  if (!stripeCustomerId || !STRIPE_SECRET_KEY) return;
  const items = await resolveSubscriptionItems(stripeCustomerId);
  if (!items) return;

  await Promise.all([
    reportUsageRecord(items.voiceItemId, voiceSeconds),
    reportUsageRecord(items.bookingItemId, bookingEvents),
    reportUsageRecord(items.transferItemId, transferEvents),
    reportUsageRecord(items.messageItemId, messageEvents),
  ]);
}
