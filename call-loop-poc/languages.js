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
  // The 13 entries below (hu..he) use Cartesia's sonic-3.6 model instead of the single global
  // ElevenLabs voice above — sonic-3.6 covers 44 languages, past eleven_multilingual_v2's 29, and
  // these 13 are exactly the ones Cartesia covers that ElevenLabs doesn't (real Deepgram Nova-3
  // codes confirmed for each; no entry added here without one — e.g. Malayalam/Odia were skipped,
  // Cartesia supports them but Deepgram's documented language list doesn't). voiceId per entry was
  // picked from Cartesia's real, live public voice library (GET /voices/?language=<code>), each a
  // clear/professional customer-support-toned voice, same bar as the existing CARTESIA_VOICE_ID.
  // say phrases are kept simple and correct rather than idiomatic — not independently reviewed by a
  // native speaker, unlike the hand-tuned es/fr/pt-BR entries above; flag for review if this matters.
  hu: {
    name: 'Hungarian', dg: { kind: 'nova3', code: 'hu' },
    tts: { backend: 'cartesia', voiceId: 'e97c3b37-1aa5-46af-afb7-9545086aaa92' }, // Eszter - Customer Companion
    say: {
      backchannel: ['Aha.', 'Értem.', 'Egy pillanat.', 'Persze.'],
      warmup: 'Egy pillanat, mindjárt jövök.',
      calendar: 'Egy pillanat, megnézem a naptárat.',
      goodbye: 'Köszönöm szépen a hívást! Szép napot kívánok!',
      transfer: 'Most továbbkapcsolom, egy pillanat, kérem.',
    },
    phoneAskRe: /\b(telefonszám(ot|a)?|mobilszám)\b/i,
    closingRe: /\b(viszontlátásra|szép napot|köszönöm|minden jót)\b/i,
  },
  no: {
    name: 'Norwegian', dg: { kind: 'nova3', code: 'no' },
    tts: { backend: 'cartesia', voiceId: '4f7b1820-6263-4615-87a7-b105768d8f64' }, // Kari - Crisp Coordinator
    say: {
      backchannel: ['Mm.', 'Jeg forstår.', 'Et øyeblikk.', 'Selvfølgelig.'],
      warmup: 'Et øyeblikk, jeg gjør meg klar.',
      calendar: 'Et øyeblikk, jeg sjekker kalenderen.',
      goodbye: 'Tusen takk for at du ringte! Ha en fin dag!',
      transfer: 'Jeg kobler deg videre nå, et øyeblikk.',
    },
    phoneAskRe: /\btelefonnummer(et)?\b|\bmobilnummer\b/i,
    closingRe: /\b(ha det|ha en fin dag|takk for samtalen|vi snakkes)\b/i,
  },
  vi: {
    name: 'Vietnamese', dg: { kind: 'nova3', code: 'vi' },
    tts: { backend: 'cartesia', voiceId: '8e8f222d-c817-4cc5-822b-8bf76ca7e98d' }, // Lien - Gentle Coordinator
    say: {
      backchannel: ['Dạ.', 'Em hiểu rồi.', 'Chờ chút ạ.', 'Vâng ạ.'],
      warmup: 'Chờ một chút, em chuẩn bị nhé.',
      calendar: 'Chờ một chút, em kiểm tra lịch nhé.',
      goodbye: 'Cảm ơn anh/chị đã gọi! Chúc một ngày tốt lành!',
      transfer: 'Em chuyển máy cho anh/chị ngay bây giờ, chờ một chút ạ.',
    },
    phoneAskRe: /\bsố điện thoại\b/i,
    closingRe: /\b(tạm biệt|chào tạm biệt|cảm ơn|một ngày tốt lành)\b/i,
  },
  bn: {
    name: 'Bengali', dg: { kind: 'nova3', code: 'bn' },
    tts: { backend: 'cartesia', voiceId: '48b9e1de-e2fa-4914-8b32-31c437813548' }, // Ananya - Paced Helper
    say: {
      backchannel: ['হ্যাঁ।', 'বুঝেছি।', 'একটু অপেক্ষা করুন।', 'অবশ্যই।'],
      warmup: 'একটু অপেক্ষা করুন, আমি প্রস্তুত হচ্ছি।',
      calendar: 'একটু অপেক্ষা করুন, আমি ক্যালেন্ডার দেখছি।',
      goodbye: 'কল করার জন্য অনেক ধন্যবাদ! আপনার দিনটি ভালো কাটুক!',
      transfer: 'আমি এখন আপনাকে সংযুক্ত করছি, একটু অপেক্ষা করুন।',
    },
    phoneAskRe: /ফোন\s*নম্বর|মোবাইল\s*নম্বর/i,
    closingRe: /(ধন্যবাদ|শুভ দিন|বিদায়)/i,
  },
  th: {
    name: 'Thai', dg: { kind: 'nova3', code: 'th' },
    tts: { backend: 'cartesia', voiceId: '4ff0f045-c140-4aa3-9210-529083f86fca' }, // Supannee - Support Concierge
    say: {
      backchannel: ['ค่ะ.', 'เข้าใจแล้วค่ะ.', 'รอสักครู่นะคะ.', 'ได้ค่ะ.'],
      warmup: 'รอสักครู่นะคะ กำลังเตรียมข้อมูล.',
      calendar: 'รอสักครู่นะคะ กำลังตรวจสอบปฏิทิน.',
      goodbye: 'ขอบคุณมากที่โทรมานะคะ ขอให้มีความสุขตลอดวันค่ะ!',
      transfer: 'ดิฉันจะโอนสายให้ตอนนี้นะคะ รอสักครู่ค่ะ.',
    },
    phoneAskRe: /เบอร์โทร(ศัพท์)?/i,
    closingRe: /(ขอบคุณ|สวัสดีค่ะ|ลาก่อน)/i,
  },
  ka: {
    name: 'Georgian', dg: { kind: 'nova3', code: 'ka' },
    tts: { backend: 'cartesia', voiceId: '0bfbea6c-2f8f-4f86-b411-aa2316561e36' }, // Tamara - Support Specialist
    say: {
      backchannel: ['დიახ.', 'გასაგებია.', 'ერთი წუთი.', 'რა თქმა უნდა.'],
      warmup: 'ერთი წუთი, ვემზადები.',
      calendar: 'ერთი წუთი, კალენდარს ვამოწმებ.',
      goodbye: 'დიდი მადლობა დარეკვისთვის! კარგი დღე გისურვებთ!',
      transfer: 'ახლა შეგაერთებთ, ერთი წუთით მოითმინეთ.',
    },
    phoneAskRe: /ტელეფონის ნომ(ერი|რის)/i,
    closingRe: /(მადლობა|ნახვამდის|კარგი დღე)/i,
  },
  te: {
    name: 'Telugu', dg: { kind: 'nova3', code: 'te' },
    tts: { backend: 'cartesia', voiceId: '82c2afc8-ebbc-4802-8ccf-036dc0fa1e3b' }, // Charan - Clear Concierge
    say: {
      backchannel: ['అలాగే.', 'అర్థమైంది.', 'ఒక్క నిమిషం.', 'తప్పకుండా.'],
      warmup: 'ఒక్క నిమిషం, సిద్ధమవుతున్నాను.',
      calendar: 'ఒక్క నిమిషం, క్యాలెండర్ చూస్తున్నాను.',
      goodbye: 'కాల్ చేసినందుకు చాలా ధన్యవాదాలు! మంచి రోజు గడపండి!',
      transfer: 'ఇప్పుడు మిమ్మల్ని కనెక్ట్ చేస్తున్నాను, ఒక్క నిమిషం.',
    },
    phoneAskRe: /ఫోన్\s*నంబర్/i,
    closingRe: /(ధన్యవాదాలు|వీడ్కోలు|మంచి రోజు)/i,
  },
  gu: {
    name: 'Gujarati', dg: { kind: 'nova3', code: 'gu' },
    tts: { backend: 'cartesia', voiceId: '4590a461-bc68-4a50-8d14-ac04f5923d22' }, // Isha - Learner
    say: {
      backchannel: ['હા.', 'સમજાઈ ગયું.', 'એક મિનિટ.', 'ચોક્કસ.'],
      warmup: 'એક મિનિટ, હું તૈયાર થાઉં છું.',
      calendar: 'એક મિનિટ, હું કેલેન્ડર ચેક કરું છું.',
      goodbye: 'કૉલ કરવા બદલ ખૂબ ખૂબ આભાર! તમારો દિવસ સારો રહે!',
      transfer: 'હું હમણાં તમને જોડું છું, એક મિનિટ રાહ જુઓ.',
    },
    phoneAskRe: /ફોન\s*નંબર/i,
    closingRe: /(આભાર|આવજો|સારો દિવસ)/i,
  },
  kn: {
    name: 'Kannada', dg: { kind: 'nova3', code: 'kn' },
    tts: { backend: 'cartesia', voiceId: '6baae46d-1226-45b5-a976-c7f9b797aae2' }, // Prakash - Instructor
    say: {
      backchannel: ['ಹೌದು.', 'ಅರ್ಥವಾಯಿತು.', 'ಒಂದು ನಿಮಿಷ.', 'ಖಂಡಿತ.'],
      warmup: 'ಒಂದು ನಿಮಿಷ, ನಾನು ಸಿದ್ಧವಾಗುತ್ತಿದ್ದೇನೆ.',
      calendar: 'ಒಂದು ನಿಮಿಷ, ನಾನು ಕ್ಯಾಲೆಂಡರ್ ಪರಿಶೀಲಿಸುತ್ತಿದ್ದೇನೆ.',
      goodbye: 'ಕರೆ ಮಾಡಿದ್ದಕ್ಕಾಗಿ ತುಂಬಾ ಧನ್ಯವಾದಗಳು! ಶುಭ ದಿನ!',
      transfer: 'ನಾನು ಈಗ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಿದ್ದೇನೆ, ಒಂದು ನಿಮಿಷ.',
    },
    phoneAskRe: /ಫೋನ್\s*ನಂಬರ್/i,
    closingRe: /(ಧನ್ಯವಾದ|ಶುಭ ದಿನ|ವಿದಾಯ)/i,
  },
  mr: {
    name: 'Marathi', dg: { kind: 'nova3', code: 'mr' },
    tts: { backend: 'cartesia', voiceId: 'f227bc18-3704-47fe-b759-8c78a450fdfa' }, // Suresh - Instruction Voice
    say: {
      backchannel: ['हो.', 'समजलं.', 'एक क्षण.', 'नक्कीच.'],
      warmup: 'एक क्षण, मी तयार होत आहे.',
      calendar: 'एक क्षण, मी कॅलेंडर तपासत आहे.',
      goodbye: 'कॉल केल्याबद्दल खूप धन्यवाद! तुमचा दिवस चांगला जावो!',
      transfer: 'मी आता तुम्हाला जोडत आहे, एक क्षण थांबा.',
    },
    phoneAskRe: /फोन\s*नंबर/i,
    closingRe: /(धन्यवाद|निरोप|चांगला दिवस)/i,
  },
  pa: {
    name: 'Punjabi', dg: { kind: 'nova3', code: 'pa' },
    tts: { backend: 'cartesia', voiceId: '9bf2ddcd-bb35-4e90-81b3-21c8183b24f4' }, // Navjot - Data Relayer
    say: {
      backchannel: ['ਹਾਂ ਜੀ.', 'ਸਮਝ ਗਿਆ.', 'ਇੱਕ ਮਿੰਟ.', 'ਬਿਲਕੁਲ.'],
      warmup: 'ਇੱਕ ਮਿੰਟ, ਮੈਂ ਤਿਆਰ ਹੋ ਰਿਹਾ ਹਾਂ.',
      calendar: 'ਇੱਕ ਮਿੰਟ, ਮੈਂ ਕੈਲੰਡਰ ਚੈੱਕ ਕਰ ਰਿਹਾ ਹਾਂ.',
      goodbye: 'ਕਾਲ ਕਰਨ ਲਈ ਬਹੁਤ ਧੰਨਵਾਦ! ਤੁਹਾਡਾ ਦਿਨ ਵਧੀਆ ਰਹੇ!',
      transfer: 'ਮੈਂ ਹੁਣ ਤੁਹਾਨੂੰ ਜੋੜ ਰਿਹਾ ਹਾਂ, ਇੱਕ ਮਿੰਟ ਰੁਕੋ.',
    },
    phoneAskRe: /ਫੋਨ\s*ਨੰਬਰ/i,
    closingRe: /(ਧੰਨਵਾਦ|ਅਲਵਿਦਾ|ਵਧੀਆ ਦਿਨ)/i,
  },
  ur: {
    name: 'Urdu', dg: { kind: 'nova3', code: 'ur' },
    tts: { backend: 'cartesia', voiceId: '01fc5e31-71e9-40dc-a220-06dbd4b4ed7e' }, // Zara - Customer Guide
    say: {
      backchannel: ['جی ہاں۔', 'سمجھ گیا۔', 'ایک لمحہ۔', 'بالکل۔'],
      warmup: 'ایک لمحہ، میں تیار ہو رہا ہوں۔',
      calendar: 'ایک لمحہ، میں کیلنڈر چیک کر رہا ہوں۔',
      goodbye: 'کال کرنے کا بہت شکریہ! آپ کا دن اچھا گزرے!',
      transfer: 'میں ابھی آپ کو منسلک کر رہا ہوں، ایک لمحہ انتظار کریں۔',
    },
    phoneAskRe: /فون\s*نمبر/i,
    closingRe: /(شکریہ|خدا حافظ|اچھا دن)/i,
  },
  he: {
    name: 'Hebrew', dg: { kind: 'nova3', code: 'he' },
    tts: { backend: 'cartesia', voiceId: 'ff857c8e-e7f9-4afd-af42-dce9f3c5ab02' }, // Yarden - Trusted Advisor
    say: {
      backchannel: ['כן.', 'הבנתי.', 'רגע אחד.', 'בטח.'],
      warmup: 'רגע אחד, אני מתארגן.',
      calendar: 'רגע אחד, אני בודק את היומן.',
      goodbye: 'תודה רבה שהתקשרת! שיהיה לך יום נהדר!',
      transfer: 'אני מעביר אותך עכשיו, רגע אחד בבקשה.',
    },
    phoneAskRe: /מספר\s*טלפון/i,
    closingRe: /(תודה|להתראות|יום נהדר)/i,
  },
  // The 3 entries below (ca, lt, hy) use Fish Audio — languages neither ElevenLabs nor Cartesia
  // cover with a real voice. Fish's public catalog is a largely-unmoderated community marketplace
  // (many entries are unlicensed character clones or multi-tagged with languages the sample audio
  // isn't actually in), so unlike the Cartesia batch above, each referenceId here was individually
  // vetted by reading its actual sample text against the target language/script before picking it —
  // most candidate languages checked (Estonian, Latvian, Slovenian, Serbian, Persian, Swahili) had
  // no real match in their top results and were left out rather than guessed. These 3 voices are
  // real matches but low-usage/unproven (0-1 likes) — flag for review before high-volume use.
  ca: {
    name: 'Catalan', dg: { kind: 'nova3', code: 'ca' },
    tts: { backend: 'fish', referenceId: '25073f4318ef4137bf4cdf6daf7adeb9' }, // "Adif" — real Catalan transit-announcer sample
    say: {
      backchannel: ['Sí.', 'Entesos.', 'Un moment.', 'És clar.'],
      warmup: 'Un moment, m\'estic preparant.',
      calendar: 'Un moment, estic comprovant el calendari.',
      goodbye: 'Moltes gràcies per trucar! Que tingui un bon dia!',
      transfer: 'Ara el connecto, un moment si us plau.',
    },
    phoneAskRe: /número\s*de\s*telèfon/i,
    closingRe: /(gràcies|adéu|bon dia)/i,
  },
  lt: {
    name: 'Lithuanian', dg: { kind: 'nova3', code: 'lt' },
    tts: { backend: 'fish', referenceId: '9b26eaea17e04f48af794405588281da' }, // "Merge Fellas" — single-tagged, real Lithuanian sample
    say: {
      backchannel: ['Taip.', 'Supratau.', 'Vienas momentas.', 'Žinoma.'],
      warmup: 'Vienas momentas, ruošiuosi.',
      calendar: 'Vienas momentas, tikrinu kalendorių.',
      goodbye: 'Ačiū, kad paskambinote! Geros dienos!',
      transfer: 'Dabar jus sujungsiu, vienas momentas.',
    },
    phoneAskRe: /telefono\s*numer(is|į)/i,
    closingRe: /(ačiū|viso gero|geros dienos)/i,
  },
  hy: {
    name: 'Armenian', dg: { kind: 'nova3', code: 'hy' },
    tts: { backend: 'fish', referenceId: 'a5851dea4f2647d7948b3c0026eb64e8' }, // "Փորձառու հայ տղամարդ" — real Armenian-script sample
    say: {
      backchannel: ['Այո.', 'Հասկացա.', 'Մի պահ.', 'Իհարկե.'],
      warmup: 'Մի պահ, պատրաստվում եմ.',
      calendar: 'Մի պահ, ստուգում եմ օրացույցը.',
      goodbye: 'Շնորհակալություն զանգի համար! Հաջող օր!',
      transfer: 'Հիմա կապում եմ ձեզ, մի պահ սպասեք.',
    },
    phoneAskRe: /հեռախոսահամար/i,
    closingRe: /(շնորհակալություն|ցտեսություն|հաջող օր)/i,
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
  // Default every language to the single global ElevenLabs voice, same as always — a LANGS entry
  // may set its own `tts` (e.g. { backend: 'cartesia', voiceId: '...' }) to override this, for
  // languages ElevenLabs' eleven_multilingual_v2 doesn't cover but Cartesia's sonic-3.6 does.
  const tts = LANGS[key].tts || { backend: 'elevenlabs', elevenVoiceId: EL_VOICE };
  return { code: key, ...LANGS[key], tts };
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
