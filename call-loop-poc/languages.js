// Per-agent language support (globalSettings.language). English ('en', or unset) never
// touches anything here except resolveLanguage() returning null — every English call keeps its
// pre-existing code path byte for byte.
//
// Three things vary per language, each verified against the provider (see git log / report):
//   stt   — Deepgram Flux multilingual (v2, semantic end-of-turn) for the 8 languages it covers,
//           Nova-3 (v1, endpointing + UtteranceEnd) for the rest.
//   tts   — ElevenLabs eleven_multilingual_v2 with a premade voice. Kokoro (the $0.10/min default)
//           is English-only on the production Modal worker (KPipeline(lang_code='a') is hard-coded),
//           so `kokoro` is null everywhere until that worker is extended.
//   say   — the handful of fixed phrases the engine speaks itself (fillers, warmup, fallbacks).

// ElevenLabs premade voice "Sarah" — works across languages with eleven_multilingual_v2.
const EL_VOICE = 'EXAVITQu4vr4xnSDxMzb';

const LANGS = {
  es: {
    name: 'Spanish', dg: { kind: 'flux', hint: 'es' },
    say: {
      backchannel: ['Ajá.', 'Entendido.', 'Un segundo.', 'Claro.'],
      warmup: 'Un momento mientras me preparo.',
      calendar: 'Un momento, voy a revisar el calendario.',
      goodbye: '¡Muchas gracias por llamar! ¡Que tenga un excelente día!',
      transfer: 'Le conecto ahora, un momento por favor.',
    },
    phoneAskRe: /\b(tel[eé]fono|celular|m[oó]vil)\b[^.?!]{0,40}\bn[uú]mero\b|\bn[uú]mero\b[^.?!]{0,40}\b(tel[eé]fono|celular|llamar|contactar)\b/i,
    closingRe: /\b(adi[oó]s|hasta luego|hasta pronto|que tenga (un )?(buen|excelente) d[ií]a|cu[ií]dese)\b/i,
  },
  fr: {
    name: 'French', dg: { kind: 'flux', hint: 'fr' },
    say: {
      backchannel: ['Mm-hmm.', 'Très bien.', 'Un instant.', 'Bien sûr.'],
      warmup: 'Un instant, je me prépare.',
      calendar: 'Un instant, je vérifie le calendrier.',
      goodbye: 'Merci beaucoup de votre appel ! Excellente journée !',
      transfer: 'Je vous mets en relation, un instant s\'il vous plaît.',
    },
    phoneAskRe: /\bnum[eé]ro\b[^.?!]{0,40}\b(t[eé]l[eé]phone|portable|mobile|joindre|rappeler)\b|\b(t[eé]l[eé]phone|portable|mobile)\b[^.?!]{0,40}\bnum[eé]ro\b/i,
    closingRe: /\b(au revoir|[aà] bient[oô]t|bonne journ[eé]e|bonne soir[eé]e|prenez soin)\b/i,
  },
  'pt-BR': {
    name: 'Portuguese (Brazil)', dg: { kind: 'flux', hint: 'pt' },
    say: {
      backchannel: ['Aham.', 'Entendi.', 'Só um instante.', 'Claro.'],
      warmup: 'Só um instante enquanto eu me preparo.',
      calendar: 'Só um instante, vou verificar a agenda.',
      goodbye: 'Muito obrigado por ligar! Tenha um ótimo dia!',
      transfer: 'Vou transferir você agora, só um instante, por favor.',
    },
    phoneAskRe: /\b(telefone|celular)\b[^.?!]{0,40}\bn[uú]mero\b|\bn[uú]mero\b[^.?!]{0,40}\b(telefone|celular|ligar|contato)\b/i,
    closingRe: /\b(tchau|at[eé] logo|at[eé] mais|tenha um [oó]timo dia|bom dia para voc[eê])\b/i,
  },
  it: {
    name: 'Italian', dg: { kind: 'flux', hint: 'it' },
    say: {
      backchannel: ['Mm-hmm.', 'Capito.', 'Un attimo.', 'Certo.'],
      warmup: 'Un attimo mentre mi preparo.',
      calendar: 'Un attimo, controllo il calendario.',
      goodbye: 'Grazie mille per la chiamata! Buona giornata!',
      transfer: 'La metto in contatto ora, un attimo per favore.',
    },
    phoneAskRe: /\bnumero\b[^.?!]{0,40}\b(telefono|cellulare|chiamare|contattare)\b|\b(telefono|cellulare)\b[^.?!]{0,40}\bnumero\b/i,
    closingRe: /\b(arrivederci|ciao|a presto|buona giornata|buona serata)\b/i,
  },
  nl: {
    name: 'Dutch', dg: { kind: 'flux', hint: 'nl' },
    say: {
      backchannel: ['Mm-hmm.', 'Begrepen.', 'Een momentje.', 'Zeker.'],
      warmup: 'Een moment, ik maak me even klaar.',
      calendar: 'Een moment, ik kijk even in de agenda.',
      goodbye: 'Hartelijk dank voor uw telefoontje! Nog een fijne dag!',
      transfer: 'Ik verbind u nu door, een moment alstublieft.',
    },
    phoneAskRe: /\b(telefoon|mobiel)?nummer\b[^.?!]{0,40}\b(bereiken|bellen|telefoon|mobiel)\b|\b(telefoonnummer|mobiele nummer)\b/i,
    closingRe: /\b(tot ziens|doei|dag|fijne dag|prettige dag)\b/i,
  },
  de: {
    name: 'German', dg: { kind: 'flux', hint: 'de' },
    say: {
      backchannel: ['Mm-hmm.', 'Verstanden.', 'Einen Moment.', 'Gerne.'],
      warmup: 'Einen Moment, ich bereite mich kurz vor.',
      calendar: 'Einen Moment, ich schaue kurz in den Kalender.',
      goodbye: 'Vielen Dank für Ihren Anruf! Noch einen schönen Tag!',
      transfer: 'Ich verbinde Sie jetzt, einen Moment bitte.',
    },
    phoneAskRe: /\b(telefonnummer|handynummer|rufnummer)\b|\bnummer\b[^.?!]{0,40}\b(erreichen|anrufen)\b/i,
    closingRe: /\b(auf wiedersehen|tsch[uü]ss|sch[oö]nen tag|bis bald)\b/i,
  },
  hi: {
    name: 'Hindi', dg: { kind: 'flux', hint: 'hi' },
    say: {
      backchannel: ['हूँ।', 'ठीक है।', 'एक सेकंड।', 'जी बिल्कुल।'],
      warmup: 'एक क्षण, मैं तैयार हो रहा हूँ।',
      calendar: 'एक क्षण, मैं कैलेंडर देख लेता हूँ।',
      goodbye: 'कॉल करने के लिए बहुत धन्यवाद! आपका दिन शुभ हो!',
      transfer: 'मैं आपको अभी जोड़ रहा हूँ, कृपया एक क्षण रुकें।',
    },
    phoneAskRe: /(फ़?ोन|मोबाइल)\s*नंबर|नंबर[^.?!]{0,30}(फ़?ोन|संपर्क)/,
    closingRe: /(अलविदा|नमस्ते|धन्यवाद, फिर मिलेंगे|आपका दिन शुभ हो)/,
  },
  pl: {
    name: 'Polish', dg: { kind: 'nova3', code: 'pl' },
    say: {
      backchannel: ['Mhm.', 'Rozumiem.', 'Chwileczkę.', 'Oczywiście.'],
      warmup: 'Chwileczkę, już się przygotowuję.',
      calendar: 'Chwileczkę, sprawdzę kalendarz.',
      goodbye: 'Bardzo dziękuję za telefon! Miłego dnia!',
      transfer: 'Już łączę, proszę chwilę poczekać.',
    },
    phoneAskRe: /\bnumer\b[^.?!]{0,40}\b(telefonu|kontaktowy|komórki)\b|\b(telefon|komórk)\w*[^.?!]{0,40}\bnumer\b/i,
    closingRe: /\b(do widzenia|do us[lł]yszenia|mi[lł]ego dnia|cze[sś][cć])\b/i,
  },
  id: {
    name: 'Indonesian', dg: { kind: 'nova3', code: 'id' },
    say: {
      backchannel: ['Hmm.', 'Baik.', 'Sebentar ya.', 'Tentu.'],
      warmup: 'Sebentar, saya bersiap dulu.',
      calendar: 'Sebentar, saya cek kalendernya.',
      goodbye: 'Terima kasih banyak sudah menelepon! Semoga hari Anda menyenangkan!',
      transfer: 'Saya sambungkan sekarang, mohon tunggu sebentar.',
    },
    phoneAskRe: /\bnomor\b[^.?!]{0,40}\b(telepon|ponsel|hp|dihubungi)\b|\b(telepon|ponsel|hp)\b[^.?!]{0,40}\bnomor\b/i,
    closingRe: /\b(sampai jumpa|selamat tinggal|selamat siang|semoga hari Anda)\b/i,
  },
  ar: {
    name: 'Arabic', dg: { kind: 'nova3', code: 'ar' },
    say: {
      backchannel: ['أها.', 'حسنًا.', 'لحظة من فضلك.', 'بالتأكيد.'],
      warmup: 'لحظة من فضلك، أستعد للمحادثة.',
      calendar: 'لحظة، سأتحقق من التقويم.',
      goodbye: 'شكرًا جزيلًا لاتصالك! أتمنى لك يومًا سعيدًا!',
      transfer: 'سأقوم بتحويلك الآن، لحظة من فضلك.',
    },
    phoneAskRe: /رقم[^.?!؟]{0,30}(هاتف|جوال|موبايل)|(هاتف|جوال|موبايل)[^.?!؟]{0,30}رقم/,
    closingRe: /(مع السلامة|إلى اللقاء|يومًا سعيدًا)/,
  },
};

// Codes an operator or API user might type -> canonical key. 'pt' is treated as Brazilian since
// that's the only Portuguese variant offered.
const ALIASES = { pt: 'pt-BR', 'pt-br': 'pt-BR', pt_br: 'pt-BR' };

function isEnglish(code) {
  return !code || /^en([-_].*)?$/i.test(String(code).trim()) || String(code).trim() === '';
}

// Returns the language record (with .code) for a non-English supported language, or null for
// English/unset/unsupported (unsupported -> engine falls back to English behavior).
export function resolveLanguage(code) {
  if (typeof code !== 'string' || isEnglish(code)) return null;
  const c = code.trim();
  const key = LANGS[c] ? c : ALIASES[c.toLowerCase()] || (LANGS[c.split(/[-_]/)[0].toLowerCase()] ? c.split(/[-_]/)[0].toLowerCase() : null);
  if (!key || !LANGS[key]) return null;
  return { code: key, ...LANGS[key], tts: { backend: 'elevenlabs', elevenVoiceId: EL_VOICE } };
}

// Appended to the system prompt for every turn of a non-English agent.
export function languageInstruction(lang) {
  return (
    `\n\nLANGUAGE: This phone call is conducted in ${lang.name}. Speak and reply ONLY in ${lang.name}, ` +
    `in natural, colloquial spoken ${lang.name} as a native speaker would on the phone. The step instructions and ` +
    `any example lines above may be written in English — treat them as instructions and translate any wording you ` +
    `are told to say into ${lang.name}. Do not mix in English, except brand names. Read numbers, dates, times, and phone ` +
    `numbers aloud the way a native ${lang.name} speaker would, and keep sentences short. If the caller clearly and ` +
    `persistently speaks a different language, politely say so and continue in ${lang.name} unless they ask to switch.`
  );
}

export { LANGS };
