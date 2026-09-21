from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, ListFlowable, ListItem, HRFlowable)
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("Bodyc", parent=styles["BodyText"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.5, textColor=colors.grey, leading=11))
styles.add(ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=10))

def P(t, style="Bodyc"): return Paragraph(t, styles[style])

def tbl(header, rows, colWidths=None):
    data = [header] + rows
    t = Table(data, colWidths=colWidths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

story = []
story.append(P("Self-Hosted TTS: Voice Model Training Research", "Title"))
story.append(P("Internal research record - realtime-tts repository", "Small"))
story.append(P("Compiled 2026-09-20", "Small"))
story.append(Spacer(1, 14))
story.append(HRFlowable(width="100%", color=colors.HexColor("#999999")))
story.append(Spacer(1, 10))

story.append(P("Summary", "H1c"))
story.append(P(
"This document records the approaches taken to build a self-hosted text-to-speech (TTS) "
"service competitive with ElevenLabs on latency and price without an always-on GPU, and the "
"subsequent work to expand it to more voices, languages, and a customer-facing voice-cloning "
"pipeline. It exists as a research log: what was tried, what was measured, what worked, and "
"what was rejected and why."))

story.append(P("1. Model selection", "H1c"))
story.append(P("1.1 Candidates evaluated", "H2c"))
story.append(ListFlowable([
    ListItem(P("<b>Kokoro-82M</b> - a StyleTTS2-derived model. Could not be fine-tuned in-house; "
               "used only as a hosted GPU engine (Modal, T4) for its existing voice set.")),
    ListItem(P("<b>Matcha-TTS</b> - flow-matching model. Fine-tuned on the same corpus as Piper for "
               "comparison; too slow on CPU to serve without a GPU, so dropped for the CPU-serving goal.")),
    ListItem(P("<b>Piper (VITS/GAN, ONNX, CPU-native)</b> - selected. Small enough to run acceptably "
               "on ordinary CPU cores, which removes the always-on GPU cost entirely.")),
], bulletType="bullet"))

story.append(P("1.2 Full-corpus Piper fine-tune", "H2c"))
story.append(P(
"Piper's medium-quality architecture was fine-tuned from the Lessac base checkpoint on a "
"22,011-clip Polly-Joanna-generated corpus (piper_full_finetune.py). Lightning's max_steps counts "
"optimizer steps; the GAN objective takes two steps per batch. Two full training passes were run: "
"an initial 20,000-step run and a continuation to roughly 60,000 steps (about 12 epochs), after "
"which validation loss (val_mel) plateaued around 0.325 and further training was judged to add "
"little. Judged by ear against the pilot and against Matcha-TTS/Kokoro fine-tune attempts, the "
"full-corpus Piper voice was rated the best available."))

story.append(P("2. Licence-clean base investigation", "H1c"))
story.append(P(
"The production voice's base checkpoint (Lessac) carries a Blizzard 2013 licence restricted to "
"research use, and the training corpus was AWS Polly-generated speech. Both create commercial-use "
"risk. Piper also publishes a checkpoint trained from scratch on the public-domain LJ Speech "
"dataset (rhasspy/piper-checkpoints, en_US/ljspeech/medium). It was exported to ONNX and "
"benchmarked with no fine-tuning required (it is already a complete model). By ear, the "
"LJ Speech voice was rated acceptable but slower-paced and lower quality than the Lessac/Polly "
"voice - a usable fallback if the licensing question resolves against the current voice, not a "
"drop-in replacement."))

story.append(P("3. Small-data voice cloning experiments", "H1c"))
story.append(P(
"To evaluate a customer-facing “train a voice from your own recordings” product, several "
"fine-tunes were run from a licence-clean base against real speakers drawn from LibriTTS-R "
"(CC BY 4.0), holding the recipe (warm-start from a clean base, per-length step count) constant."))
story.append(tbl(
    ["Test", "Speaker / base", "Length", "Steps", "By-ear result"],
    [
        ["1", "Male speaker, LJ Speech (female) base", "10 min", "3,000", "Robotic / shaky"],
        ["2", "Same speaker, LJ Speech base", "~25 min", "4,000", "Still shaky - base mismatch suspected"],
        ["3", "Female speaker, LJ Speech (female) base", "~25 min", "4,000", "On par with the production voice"],
        ["4", "Male speaker (same as #1), male base (\"john\")", "10-25 min", "3,000-15,000", "Shaky at all step counts tried"],
        ["5", "Two other male speakers, male base (\"john\")", "~25 min", "4,000", "Good, only barely noticeable shake"],
    ],
    colWidths=[0.4*inch, 2.3*inch, 0.75*inch, 0.85*inch, 1.9*inch],
))
story.append(P(
"<b>Findings.</b> Training length was not the limiting factor (test 4 tried up to 15,000 steps "
"with no improvement). Gender-matching the base checkpoint to the speaker mattered: the same "
"male speaker stayed shaky on a female base regardless of steps or data length, but two other, "
"more neutral-delivery male speakers were good on a male base at the same 25-minute/4,000-step "
"recipe. The original male speaker's very wide intonation range is the suspected remaining "
"cause of residual shakiness even on a matched base - an advisory “delivery variation” "
"warning (median pitch deviation across the training clips) was added to the pipeline to flag "
"speakers of that kind. Raw training clips were separately confirmed to be clean-sounding "
"audio, ruling out bad source recordings as the cause. The minimum viable recording length "
"(10 vs 20+ minutes) remains only partially verified by ear and is conservatively set to 20 "
"minutes pending further listening.", "Caption"))

story.append(P("4. Production voice pipeline", "H1c"))
story.append(P(
"A server-to-server pipeline (voice-pipeline/) turns a customer-supplied dataset into a served "
"voice: (1) a consent record is required before any training; (2) audio is checked for length, "
"noise floor, telephone-bandwidth limiting, multiple-speaker contamination, and transcript/"
"audio-length mismatch, each calibrated against real LibriTTS-R data; (3) the fine-tuning base "
"is chosen automatically from the speaker's estimated pitch (median F0, autocorrelation-based); "
"(4) training runs on a Modal T4 GPU (roughly $0.40-1 per voice); (5) the result is exported to "
"ONNX with preview samples for approval before deployment; (6) approved voices are pushed to the "
"serving fleet's registry, gated to the owning customer's API key(s) or user id, and can be "
"deleted on request, removing both the model and the training data. One full end-to-end run "
"(a 22-minute male speaker) trained and deployed cleanly, with one audible artifact - an unnatural "
"pause on a transition between two sentence halves - noted for follow-up."))

story.append(PageBreak())
story.append(P("5. House voice expansion and licensing", "H1c"))
story.append(P(
"To broaden language and accent coverage beyond the single production voice, every candidate "
"Piper community checkpoint was audited for licence and training lineage, since many are "
"themselves fine-tuned from the same restricted Lessac base."))
story.append(tbl(
    ["Tier", "Definition", "Examples", "Publishing decision"],
    [
        ["A", "Clean training data and clean (from-scratch or public-domain) lineage",
         "en-us ljspeech/kristin/john, en-gb cori, de/fr/nl “mls” voices, it-it serena",
         "Published"],
        ["B", "Clean data licence, but fine-tuned from the research-only Lessac base",
         "VCTK accent voices (AU/CA/UK/IE/IN/NZ/ZA), pl, pt-BR, ru, most es/fr variants",
         "Published after owner accepted the lineage risk"],
        ["C", "Non-commercial or unclear data licence",
         "*-low family (NC-SA lineage), several unverifiable “see URL” licences",
         "Not published"],
    ],
    colWidths=[0.5*inch, 2.2*inch, 2.4*inch, 1.1*inch],
))
story.append(P(
"53 voices are currently served: 16 tier-A and 37 tier-B, spanning English (US, UK, Australian, "
"Canadian, Irish, Indian, New Zealand, South African), German, French, Dutch, Italian, Polish, "
"Brazilian Portuguese, Spanish, and Russian. Multi-speaker source models (VCTK, MLS) required "
"adding speaker-pinning support to the serving engine so a specific speaker can be selected "
"and served as a fixed voice.", "Caption"))

story.append(P("5.1 Cross-lingual pilot: training a new language from scratch", "H2c"))
story.append(P(
"For languages with no acceptable existing checkpoint (e.g. clean Spanish), a pilot fine-tuned "
"a two-speaker Spanish voice (CML-TTS, CC BY 4.0, ~12h/speaker) warm-started from the "
"English LJ Speech-derived base. This works because Piper's phoneme table is a fixed-size "
"slot table shared across languages, so the bulk of the network's weights transfer even "
"across a language change; only the speaker embedding and discriminator are effectively "
"re-learned. The pilot became intelligible by roughly 4,000 steps and reached a 21% Whisper-"
"measured word error rate on held-out call-centre-style sentences at 15,400 steps, for about "
"$5 of GPU time. Voice-quality judgement by ear is still pending; the same recipe is expected "
"to generalize to other uncovered languages (Portuguese, Polish, Italian, French) at similar cost."))

story.append(P("6. Serving and latency work", "H1c"))
story.append(P("6.1 Piper (CPU) serving", "H2c"))
story.append(P(
"Piper is served from a single always-on Fly.io machine (shared CPU, ~$13/month) rather than "
"an autoscaled GPU worker, eliminating cold starts entirely at that cost. Measured end-to-end "
"time to first audio was 168-171ms warm, versus 173-174ms measured for ElevenLabs Flash v2.5 "
"under the same interleaved methodology - a statistical tie, not a claimed win. ElevenLabs "
"Multilingual v2 measured roughly 1.0-1.1s. Model-only synthesis time (excluding phonemizing "
"and network) was found to scale with the length of the first sentence sent (85ms for a 3-word "
"sentence up to 229ms for 17 words), motivating an (currently opt-in, unshipped) first-sentence "
"splitting feature to cut worst-case first-chunk latency; a listening test found the naive "
"word-boundary split point can introduce an audible discontinuity at the seam, and refinement "
"(trimming, cross-fade, and a punctuation-aware cut point) is in progress. Capacity testing found "
"a single shared-CPU machine sustains only 2-4 concurrent calls before latency degrades sharply, "
"versus 8-16 on a 10x-costlier dedicated-CPU machine - a scale-when-revenue-justifies-it decision."))

story.append(P("6.2 Kokoro (GPU) latency breakdown", "H2c"))
story.append(P(
"A separate investigation into Kokoro's larger end-to-end latency (0.7-1.3s versus ~130ms of "
"actual model compute) found the majority of the gap was outside the model entirely: Modal's "
"ingress/WebSocket-handshake overhead (~300-400ms) and, on a fresh connection, an oversized "
"first PCM audio frame that delays first playback (a 144KB first frame vs. ~10KB in small-frame "
"tests). Splitting the first chunk at a word boundary and optionally slicing PCM into small "
"frames cut fresh-connection first-audio time from ~643ms to ~368ms in testing; the model's own "
"contribution (95-130ms) was confirmed not to be the bottleneck."))

story.append(P("7. Speech-to-text (exploratory)", "H1c"))
story.append(P(
"As a complementary product, batch and realtime speech-to-text were evaluated on Whisper-family "
"and dedicated streaming ASR models. Batch transcription (faster-whisper large-v3-turbo, GPU, "
"batched inference) reached 2.6-2.8% word error rate on clean and simulated telephone audio at "
"roughly $0.011-0.03 per audio-hour of compute, and shipped as a paid API at $0.11/audio-hour "
"(half of a comparable published competitor rate). Realtime/streaming transcription is harder to "
"do cheaply on CPU-only infrastructure: a permissively-licensed streaming transducer model "
"(Zipformer, Apache-2.0) achieved 3.4-4.4% WER at low CPU cost (~0.06-0.08 cores per stream) "
"with an end-of-speech-to-final latency around 0.6s at a 500ms silence threshold - workable for "
"a voice-agent turn-taking use case, but not competitive with realtime services quoting ~150ms, "
"since that gap is dominated by the endpointing wait rather than model compute."))

story.append(P("8. Open questions", "H1c"))
story.append(ListFlowable([
    ListItem(P("Whether the production (Lessac/Polly-derived) voice can be used commercially, "
               "or must be replaced by the licence-clean LJ Speech-derived voice.")),
    ListItem(P("Whether a 10-minute customer recording is sufficient, or whether 20 minutes should "
               "remain the enforced minimum, pending a further by-ear listening pass.")),
    ListItem(P("Whether the first-sentence-splitting latency optimization can be shipped without "
               "an audible seam artifact.")),
    ListItem(P("Full quality validation (by ear) of the cross-lingual Spanish pilot before it is "
               "used as the template for further language expansion.")),
], bulletType="bullet"))

story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
story.append(Spacer(1, 6))
story.append(P("Source: realtime-tts repository (training-data/, voice-pipeline/, worker-piper-fly/, "
               "worker-modal-readaloud/, voices/). This document summarizes work already committed "
               "to that repository; see individual README files and commit history for full detail, "
               "raw measurements, and listening samples.", "Small"))

doc = SimpleDocTemplate(
    "/tmp/voice_model_training_research.pdf", pagesize=letter,
    topMargin=0.75*inch, bottomMargin=0.75*inch, leftMargin=0.75*inch, rightMargin=0.75*inch,
    title="Self-Hosted TTS: Voice Model Training Research", author="ReadAloud AI / realtime-tts",
)
doc.build(story)
print("built")
