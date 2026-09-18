const https = require('https');
const KEY = process.env.ELEVENLABS_API_KEY, VOICE = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';
const TEXTS = [
  "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
  "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
  "I understand your frustration, let me see what I can do to make this right.",
  "Is there anything else I can help you with today?",
  "Your extension is six six three five.",
];
const MODELS = { flash: 'eleven_flash_v2_5', multilingual: 'eleven_multilingual_v2' };
function get(text, model) {
  return new Promise((res, rej) => {
    const body = JSON.stringify({ text, model_id: model });
    const req = https.request({ host: 'api.elevenlabs.io', path: `/v1/text-to-speech/${VOICE}/stream?output_format=mp3_44100_128`, method: 'POST',
      headers: { 'xi-api-key': KEY, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (r) => {
      const chunks = []; r.on('data', (d) => chunks.push(d));
      r.on('end', () => r.statusCode === 200 ? res(Buffer.concat(chunks)) : rej(new Error('status ' + r.statusCode)));
    });
    req.on('error', rej); req.write(body); req.end();
  });
}
(async () => {
  for (const [name, model] of Object.entries(MODELS))
    for (let i = 0; i < TEXTS.length; i++)
      console.log(`FILE ${name}_${i} ${(await get(TEXTS[i], model)).toString('base64')}`);
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });
