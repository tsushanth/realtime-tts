// Reports real usage to the Stripe meters created for calldesktech's
// usage-based pricing (calldesktech_voice_seconds/booking_events/
// transfer_events/message_events — see calldesktech's Stripe product
// "CallDeskTech Usage"). Fire-and-forget: a metering failure must never
// take down or delay a live call, so every call here only logs on error.
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;

export async function reportMeterEvent(eventName, stripeCustomerId, value) {
  if (!STRIPE_SECRET_KEY || !stripeCustomerId || !value) return;
  try {
    const res = await fetch('https://api.stripe.com/v1/billing/meter_events', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${STRIPE_SECRET_KEY}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        event_name: eventName,
        'payload[stripe_customer_id]': stripeCustomerId,
        'payload[value]': String(value),
      }).toString(),
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) {
      console.error(`[stripe-meter] ${eventName} -> HTTP ${res.status}: ${await res.text().catch(() => '')}`);
    }
  } catch (err) {
    console.error(`[stripe-meter] ${eventName} failed`, err);
  }
}

// One call's worth of usage, reported once at hangup — see CallSession.close().
export async function reportCallUsage(stripeCustomerId, { voiceSeconds, bookingEvents, transferEvents, messageEvents }) {
  if (!stripeCustomerId) return;
  await Promise.all([
    reportMeterEvent('calldesktech_voice_seconds', stripeCustomerId, Math.round(voiceSeconds)),
    reportMeterEvent('calldesktech_booking_events', stripeCustomerId, bookingEvents),
    reportMeterEvent('calldesktech_transfer_events', stripeCustomerId, transferEvents),
    reportMeterEvent('calldesktech_message_events', stripeCustomerId, messageEvents),
  ]);
}
