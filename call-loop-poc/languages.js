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

// ElevenLabs premade voice "George" (the account default; legacy premades such as Sarah are NOT available on this ElevenLabs account) — works across languages with eleven_multilingual_v2.
const EL_VOICE = 'JBFqnCBsd6RMkjVDRZzb';

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
  ja: {
    name: 'Japanese', dg: { kind: 'flux', hint: 'ja' },
    say: {
      backchannel: ['はい。', 'かしこまりました。', '少々お待ちください。', 'もちろんです。'],
      warmup: '少々お待ちください、準備いたします。',
      calendar: '少々お待ちください、カレンダーを確認します。',
      goodbye: 'お電話ありがとうございました。良い一日をお過ごしください。',
      transfer: 'ただいまおつなぎいたします。少々お待ちください。',
    },
    phoneAskRe: /電話番号|携帯番号|お電話番号/,
    closingRe: /(さようなら|失礼いたします|良い一日を|それでは)/,
  },
  ru: {
    name: 'Russian', dg: { kind: 'flux', hint: 'ru' },
    say: {
      backchannel: ['Угу.', 'Понятно.', 'Секунду.', 'Конечно.'],
      warmup: 'Секунду, я подготовлюсь.',
      calendar: 'Секунду, проверю календарь.',
      goodbye: 'Большое спасибо за звонок! Хорошего дня!',
      transfer: 'Сейчас соединю вас, одну секунду, пожалуйста.',
    },
    phoneAskRe: /\bномер\b[^.?!]{0,40}\b(телефона|мобильный|связаться)\b|\b(телефон|мобильный)\b[^.?!]{0,40}\bномер\b/i,
    closingRe: /\b(до свидания|всего доброго|хорошего дня|до встречи)\b/i,
  },
  zh: {
    name: 'Chinese (Mandarin)', dg: { kind: 'nova3', code: 'zh' },
    say: {
      backchannel: ['嗯。', '好的。', '稍等。', '当然。'],
      warmup: '请稍等，我准备一下。',
      calendar: '请稍等，我查一下日历。',
      goodbye: '非常感谢您的来电！祝您有美好的一天！',
      transfer: '现在为您转接，请稍等。',
    },
    phoneAskRe: /(电话|手机)号码|号码[^。？！.?!]{0,20}(联系|电话|手机)/,
    closingRe: /(再见|拜拜|祝你有美好的一天|回头见)/,
  },
  ko: {
    name: 'Korean', dg: { kind: 'nova3', code: 'ko' },
    say: {
      backchannel: ['네.', '알겠습니다.', '잠시만요.', '물론입니다.'],
      warmup: '잠시만 기다려 주세요, 준비하겠습니다.',
      calendar: '잠시만요, 일정을 확인해 보겠습니다.',
      goodbye: '전화해 주셔서 감사합니다! 좋은 하루 되세요!',
      transfer: '지금 연결해 드리겠습니다, 잠시만 기다려 주세요.',
    },
    phoneAskRe: /(전화|휴대폰|핸드폰)\s*번호|번호[^.?!]{0,20}(전화|연락)/,
    closingRe: /(안녕히 계세요|좋은 하루 되세요|다음에 또|안녕히 가세요)/,
  },
  tr: {
    name: 'Turkish', dg: { kind: 'nova3', code: 'tr' },
    say: {
      backchannel: ['Hı hı.', 'Anladım.', 'Bir saniye.', 'Tabii.'],
      warmup: 'Bir saniye, hazırlanıyorum.',
      calendar: 'Bir saniye, takvime bakıyorum.',
      goodbye: 'Aradığınız için çok teşekkürler! İyi günler dilerim!',
      transfer: 'Sizi şimdi bağlıyorum, bir saniye lütfen.',
    },
    phoneAskRe: /\b(telefon|cep)\b[^.?!]{0,40}\bnumaras[ıi]\b|\bnumara(n[ıi]z)?\b[^.?!]{0,40}\b(telefon|cep|ulaşmak|aramak)\b/i,
    closingRe: /\b(hoşça kal[ıi]n|görüşürüz|iyi günler|kendinize iyi bak[ıi]n)\b/i,
  },
  el: {
    name: 'Greek', dg: { kind: 'nova3', code: 'el' },
    say: {
      backchannel: ['Μάλιστα.', 'Κατάλαβα.', 'Μια στιγμή.', 'Βεβαίως.'],
      warmup: 'Μια στιγμή, ετοιμάζομαι.',
      calendar: 'Μια στιγμή, θα ελέγξω το ημερολόγιο.',
      goodbye: 'Ευχαριστούμε πολύ για το τηλεφώνημα! Να έχετε μια υπέροχη μέρα!',
      transfer: 'Σας συνδέω τώρα, μια στιγμή παρακαλώ.',
    },
    phoneAskRe: /\bαριθμ[όο][ςν]?\b[^.?!;]{0,40}\b(τηλεφώνου|κινητού|επικοινωνίας)\b|\b(τηλέφωνο|κινητό)\b[^.?!;]{0,40}\bαριθμ[όο][ςν]?\b/i,
    closingRe: /\b(αντίο|γεια σας|καλή σας μέρα|τα λέμε)\b/i,
  },
  bg: {
    name: 'Bulgarian', dg: { kind: 'nova3', code: 'bg' },
    say: {
      backchannel: ['Ммм.', 'Разбирам.', 'Един момент.', 'Разбира се.'],
      warmup: 'Един момент, подготвям се.',
      calendar: 'Един момент, ще проверя календара.',
      goodbye: 'Много благодаря за обаждането! Приятен ден!',
      transfer: 'Свързвам ви сега, един момент моля.',
    },
    phoneAskRe: /\bномер\b[^.?!]{0,40}\b(телефон|мобилен|контакт)\b|\b(телефон|мобилен)\b[^.?!]{0,40}\bномер\b/i,
    closingRe: /\b(довиждане|приятен ден|до скоро|чао)\b/i,
  },
  hr: {
    name: 'Croatian', dg: { kind: 'nova3', code: 'hr' },
    say: {
      backchannel: ['Mhm.', 'Razumijem.', 'Trenutak.', 'Naravno.'],
      warmup: 'Trenutak, pripremam se.',
      calendar: 'Trenutak, provjerit ću kalendar.',
      goodbye: 'Hvala vam puno na pozivu! Želim vam ugodan dan!',
      transfer: 'Sada vas spajam, samo trenutak molim.',
    },
    phoneAskRe: /\bbroj\b[^.?!]{0,40}\b(telefona|mobitela|kontakt)\b|\b(telefon|mobitel)\b[^.?!]{0,40}\bbroj\b/i,
    closingRe: /\b(doviđenja|ugodan dan|vidimo se|bok)\b/i,
  },
  cs: {
    name: 'Czech', dg: { kind: 'nova3', code: 'cs' },
    say: {
      backchannel: ['Hmm.', 'Rozumím.', 'Moment.', 'Jistě.'],
      warmup: 'Moment, připravím se.',
      calendar: 'Moment, podívám se do kalendáře.',
      goodbye: 'Děkuji moc za zavolání! Přeji hezký den!',
      transfer: 'Nyní vás přepojím, moment prosím.',
    },
    phoneAskRe: /\bčíslo\b[^.?!]{0,40}\b(telefonu|mobilu|kontakt)\b|\b(telefon|mobil)\b[^.?!]{0,40}\bčíslo\b/i,
    closingRe: /\b(na shledanou|hezký den|mějte se|ahoj)\b/i,
  },
  da: {
    name: 'Danish', dg: { kind: 'nova3', code: 'da' },
    say: {
      backchannel: ['Mm-hm.', 'Forstået.', 'Et øjeblik.', 'Selvfølgelig.'],
      warmup: 'Et øjeblik, jeg gør mig klar.',
      calendar: 'Et øjeblik, jeg tjekker kalenderen.',
      goodbye: 'Mange tak for opkaldet! Hav en rigtig god dag!',
      transfer: 'Jeg stiller dig igennem nu, et øjeblik.',
    },
    phoneAskRe: /\b(telefon|mobil)?nummer\b[^.?!]{0,40}\b(kontakte|ringe)\b|\b(telefonnummer|mobilnummer)\b/i,
    closingRe: /\b(farvel|god dag|vi tales ved|ha' det godt)\b/i,
  },
  tl: {
    name: 'Filipino', dg: { kind: 'nova3', code: 'tl' },
    say: {
      backchannel: ['Sige.', 'Naiintindihan ko.', 'Sandali lang.', 'Oo naman.'],
      warmup: 'Sandali lang, maghahanda ako.',
      calendar: 'Sandali lang, titingnan ko ang kalendaryo.',
      goodbye: 'Maraming salamat sa pagtawag! Magandang araw po!',
      transfer: 'Ikokonekta na kita ngayon, sandali lang po.',
    },
    phoneAskRe: /\bnumero\b[^.?!]{0,40}\b(telepono|cellphone|makontak)\b|\b(telepono|cellphone)\b[^.?!]{0,40}\bnumero\b/i,
    closingRe: /\b(paalam|magandang araw|ingat|hanggang sa muli)\b/i,
  },
  fi: {
    name: 'Finnish', dg: { kind: 'nova3', code: 'fi' },
    say: {
      backchannel: ['Joo.', 'Selvä.', 'Hetkinen.', 'Toki.'],
      warmup: 'Hetkinen, valmistaudun.',
      calendar: 'Hetkinen, tarkistan kalenterin.',
      goodbye: 'Kiitos paljon soitosta! Mukavaa päivänjatkoa!',
      transfer: 'Yhdistän sinut nyt, hetkinen vain.',
    },
    phoneAskRe: /\bnumero\b[^.?!]{0,40}\b(puhelin|matkapuhelin|tavoittaa)\b|\b(puhelin|matkapuhelin)\b[^.?!]{0,40}\bnumero\b/i,
    closingRe: /\b(näkemiin|hyvää päivänjatkoa|moikka|mukavaa päivää)\b/i,
  },
  ms: {
    name: 'Malay', dg: { kind: 'nova3', code: 'ms' },
    say: {
      backchannel: ['Baik.', 'Faham.', 'Sekejap ya.', 'Sudah tentu.'],
      warmup: 'Sekejap ya, saya bersedia dahulu.',
      calendar: 'Sekejap, saya semak kalendar.',
      goodbye: 'Terima kasih banyak kerana menelefon! Semoga hari anda ceria!',
      transfer: 'Saya sambungkan sekarang, sila tunggu sekejap.',
    },
    phoneAskRe: /\bnombor\b[^.?!]{0,40}\b(telefon|bimbit|hubungi)\b|\b(telefon|bimbit)\b[^.?!]{0,40}\bnombor\b/i,
    closingRe: /\b(selamat tinggal|jumpa lagi|semoga hari anda|selamat sejahtera)\b/i,
  },
  ro: {
    name: 'Romanian', dg: { kind: 'nova3', code: 'ro' },
    say: {
      backchannel: ['Mhm.', 'Am înțeles.', 'O clipă.', 'Sigur.'],
      warmup: 'O clipă, mă pregătesc.',
      calendar: 'O clipă, verific calendarul.',
      goodbye: 'Vă mulțumim mult pentru apel! O zi excelentă!',
      transfer: 'Vă conectez acum, o clipă vă rog.',
    },
    phoneAskRe: /\bnum[aă]r\b[^.?!]{0,40}\b(telefon|mobil|contact)\b|\b(telefon|mobil)\b[^.?!]{0,40}\bnum[aă]r\b/i,
    closingRe: /\b(la revedere|o zi bun[aă]|pe cur[aâ]nd|numai bine)\b/i,
  },
  sk: {
    name: 'Slovak', dg: { kind: 'nova3', code: 'sk' },
    say: {
      backchannel: ['Mhm.', 'Rozumiem.', 'Moment.', 'Iste.'],
      warmup: 'Moment, pripravím sa.',
      calendar: 'Moment, pozriem sa do kalendára.',
      goodbye: 'Ďakujeme veľmi pekne za zavolanie! Prajem pekný deň!',
      transfer: 'Teraz vás prepojím, moment prosím.',
    },
    phoneAskRe: /\bčíslo\b[^.?!]{0,40}\b(telefónu|mobilu|kontakt)\b|\b(telefón|mobil)\b[^.?!]{0,40}\bčíslo\b/i,
    closingRe: /\b(dovidenia|pekný deň|majte sa|ahoj)\b/i,
  },
  sv: {
    name: 'Swedish', dg: { kind: 'nova3', code: 'sv' },
    say: {
      backchannel: ['Mm.', 'Uppfattat.', 'Ett ögonblick.', 'Absolut.'],
      warmup: 'Ett ögonblick, jag förbereder mig.',
      calendar: 'Ett ögonblick, jag kollar kalendern.',
      goodbye: 'Tack så mycket för att du ringde! Ha en fin dag!',
      transfer: 'Jag kopplar dig nu, ett ögonblick.',
    },
    phoneAskRe: /\b(telefon|mobil)?nummer\b[^.?!]{0,40}\b(n[åa]\b|kontakta|ringa)\b|\b(telefonnummer|mobilnummer)\b/i,
    closingRe: /\b(hej då|ha en fin dag|vi hörs|på återseende)\b/i,
  },
  ta: {
    name: 'Tamil', dg: { kind: 'nova3', code: 'ta' },
    say: {
      backchannel: ['சரி.', 'புரிந்தது.', 'ஒரு நிமிடம்.', 'நிச்சயமாக.'],
      warmup: 'ஒரு கணம், நான் தயாராகிறேன்.',
      calendar: 'ஒரு கணம், நான் காலண்டரைப் பார்க்கிறேன்.',
      goodbye: 'அழைத்ததற்கு மிக்க நன்றி! இனிய நாளாக அமையட்டும்!',
      transfer: 'இப்போது உங்களை இணைக்கிறேன், ஒரு நிமிடம் பொறுங்கள்.',
    },
    phoneAskRe: /(தொலைபேசி|கைபேசி|மொபைல்)\s*எண்|எண்[^.?!]{0,20}(தொலைபேசி|தொடர்பு)/,
    closingRe: /(வணக்கம்|இனிய நாள்|பிறகு பேசலாம்|நன்றி)/,
  },
  uk: {
    name: 'Ukrainian', dg: { kind: 'nova3', code: 'uk' },
    say: {
      backchannel: ['Угу.', 'Зрозуміло.', 'Секунду.', 'Звісно.'],
      warmup: 'Секунду, я підготуюся.',
      calendar: 'Секунду, перевірю календар.',
      goodbye: 'Дуже дякую за дзвінок! Гарного дня!',
      transfer: 'Зараз з\'єдную вас, секунду, будь ласка.',
    },
    phoneAskRe: /\bномер\b[^.?!]{0,40}\b(телефону|мобільний|зв'язатися)\b|\b(телефон|мобільний)\b[^.?!]{0,40}\bномер\b/i,
    closingRe: /\b(до побачення|гарного дня|на все добре|бувайте)\b/i,
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

// --- Mid-call language switching (additive only; nothing above this point is touched) ---
//
// Deepgram research (2026-09-21, docs.deepgram.com): Flux (our 8-language STT path above) has no
// `language=multi`/auto-detect mode as of this writing — `language` is a fixed hint per connection,
// same as Nova-3. Nova-3's own multi-language auto-detect (`language=multi`) only covers a fixed
// English+Spanish(+9 more) code-switching set and does not report which language a given
// utterance was in via a clean per-turn field usable here. So there is no supported way to keep one
// STT socket open and have Deepgram itself tell us the caller switched languages — the only real
// mechanism is what the engine already does for Transcription Mode and per-agent language: close
// the socket and reopen it with a new hint. That's what _maybeSwitchLanguage in server.js does.
//
// Trigger: a small set of near-unambiguous function words/diacritics per language, checked against
// the caller's own transcript text (not the LLM's reply — the LLM can echo a language back without
// the caller actually having switched, and we don't want a false trigger off that). Deliberately
// conservative: requires an actual marker hit, not just "not English", so ordinary short replies
// ("yes", "okay", "sure") never trigger a switch.
const DETECT_MARKERS = {
  en: [/\b(the|is|are|and|you|please|thanks|yes|okay)\b/i],
  es: [/\b(el|la|los|las|est[aá]|por favor|gracias|s[ií]|hola|qu[eé]|c[oó]mo|quiero|necesito)\b/i, /[ñáéíóúü¿¡]/],
  fr: [/\b(le|la|les|est|s'il vous pla[iî]t|merci|oui|bonjour|c'est|je voudrais)\b/i, /[çàâêëîïôùûœ]/],
  'pt-BR': [/\b(o|a|os|as|[eé]|por favor|obrigad[oa]|sim|ol[aá]|n[aã]o)\b/i, /[ãõç]/],
  it: [/\b(il|lo|la|gli|[eè]|per favore|grazie|s[ií]|ciao|buongiorno)\b/i],
  nl: [/\b(de|het|een|is|alstublieft|dank je|ja|hallo|goedendag)\b/i],
  de: [/\b(der|die|das|ist|bitte|danke|ja|hallo|guten tag)\b/i, /[äöüß]/],
  hi: [/[ऀ-ॿ]/],
  pl: [/\b(tak|dzie[nń] dobry|prosz[eę]|dzi[eę]kuj[eę])\b/i, /[ąćęłńóśźż]/i],
  id: [/\b(saya|anda|tidak|terima kasih|selamat)\b/i],
  ar: [/[؀-ۿ]/],
  ja: [/[぀-ヿ]/],
  ru: [/[Ѐ-ӿ]/],
  zh: [/[一-鿿]/],
  ko: [/[가-힣]/],
  tr: [/\b(evet|hayır|teşekkür|lütfen|merhaba)\b/i, /[ığşüöç]/i],
  el: [/[Ͱ-Ͽ]/],
  bg: [/[А-я]/],
  hr: [/\b(da|ne|hvala|molim|bok)\b/i, /[čćžšđ]/i],
  cs: [/\b(ano|ne|děkuji|prosím|ahoj)\b/i, /[ěščřžýáíé]/i],
  da: [/\b(ja|nej|tak|hej|farvel)\b/i, /[æøå]/i],
  tl: [/\b(oo|hindi|salamat|paalam|po)\b/i],
  fi: [/\b(kyllä|ei|kiitos|hei|moikka)\b/i, /[äö]/i],
  ms: [/\b(ya|tidak|terima kasih|selamat)\b/i],
  ro: [/\b(da|nu|mulțumesc|vă rog|salut)\b/i, /[ășțîâ]/i],
  sk: [/\b(áno|nie|ďakujem|prosím|ahoj)\b/i, /[ľĺŕäô]/i],
  sv: [/\b(ja|nej|tack|hej|hejdå)\b/i, /[åäö]/i],
  ta: [/[஀-௿]/],
  uk: [/[Ѐ-ӿ]/],
};

// Guesses which of `allowedCodes` the caller's transcript `text` is in. Returns a code from
// `allowedCodes` (which may include 'en') or null when no marker fires — null means "stay on the
// current language", never "switch to something". Not ASR-grade; only used when a flow explicitly
// opts in via globalSettings.allowLanguageSwitching.
export function detectSpokenLanguage(text, allowedCodes) {
  if (typeof text !== 'string' || !Array.isArray(allowedCodes) || allowedCodes.length === 0) return null;
  const t = text.trim();
  if (t.length < 4) return null; // too short to guess reliably
  let best = null;
  let bestHits = 0;
  for (const code of allowedCodes) {
    const markers = DETECT_MARKERS[code];
    if (!markers) continue;
    const hits = markers.reduce((n, re) => n + (re.test(t) ? 1 : 0), 0);
    if (hits > bestHits) { best = code; bestHits = hits; }
  }
  return bestHits > 0 ? best : null;
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
