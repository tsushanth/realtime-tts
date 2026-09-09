// Blind A/B judge for the mystery-shopper framework (see
// MYSTERY_SHOPPER_DECISIONS.md decision 4 for why it's blind: an unblinded
// judge told "here's ours, here's Retell's" carries a real labeling-bias
// risk, so this always randomizes which transcript is "Call A" vs "Call B"
// and only de-anonymizes in the final printed report.
//
// Runs the judge via the `claude` CLI (OAuth session auth), NOT the
// Anthropic SDK + ANTHROPIC_API_KEY -- this is a one-off local analysis
// task, not a cron/server workload, so it shouldn't burn metered API
// credits (see the repo's own CLAUDE.md rule on this). Requires `claude`
// on PATH and an active `claude login` session.
//
// Usage:
//   node mystery-shopper-judge.mjs --ours transcript-ours.txt --retell transcript-retell.txt

import fs from 'node:fs';
import { spawnSync } from 'node:child_process';

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? null : process.argv[i + 1];
}

const oursPath = arg('ours');
const retellPath = arg('retell');
if (!oursPath || !retellPath) {
  console.error('usage: node mystery-shopper-judge.mjs --ours <file> --retell <file>');
  process.exit(1);
}

const oursTranscript = fs.readFileSync(oursPath, 'utf8');
const retellTranscript = fs.readFileSync(retellPath, 'utf8');

const oursIsA = Math.random() < 0.5;
const [labelA, labelB] = oursIsA ? ['ours', 'retell'] : ['retell', 'ours'];
const [transcriptA, transcriptB] = oursIsA ? [oursTranscript, retellTranscript] : [retellTranscript, oursTranscript];

const PROMPT = `You are an expert voice-AI conversation quality judge. You will be shown two
real phone call transcripts, "Call A" and "Call B", both of the SAME customer
persona/goal calling two different backend implementations of a booking
assistant. You do not know which is which — score them purely on the
transcript content, blind.

Score each call 1-5 (5 = best) on:
1. Turn efficiency — fewer unnecessary back-and-forths to complete the task
2. Naturalness — does it sound like a real conversation, not a form to fill out
3. Slot-filling design — does it ask for related info together sensibly, or
   split things awkwardly across turns
4. Error recovery / confirmation — does it verify captured info (names,
   times) before committing, and handle correction gracefully if needed
5. Closing quality — a clean, confident single closing vs. a disjointed
   multi-beat wrap-up
6. Any awkward or robotic-sounding phrasing (quote it if present)

Then give an overall winner (A, B, or tie) with a one-paragraph justification,
and 2-4 CONCRETE, ACTIONABLE suggestions for how the worse-performing call's
system could be improved to close the gap — specific enough to hand directly
to an engineer (e.g. "combine the name and time ask into one question" not
"be more efficient").

Respond in this exact structure:
## Call A
- Turn efficiency: X/5 — reasoning
- Naturalness: X/5 — reasoning
- Slot-filling design: X/5 — reasoning
- Error recovery: X/5 — reasoning
- Closing quality: X/5 — reasoning
- Awkward phrasing: [quote or "none noted"]

## Call B
(same structure)

## Verdict
Winner: A / B / tie
Justification: ...

## Suggestions to close the gap
1. ...
2. ...

## Call A transcript
${transcriptA}

## Call B transcript
${transcriptB}
`;

const result = spawnSync('claude', ['-p', '--output-format', 'text'], {
  input: PROMPT,
  encoding: 'utf8',
  maxBuffer: 10 * 1024 * 1024,
});

if (result.status !== 0) {
  console.error('claude CLI failed:', result.stderr);
  process.exit(1);
}

console.log(result.stdout);
console.log('\n\n=== DE-ANONYMIZED ===');
console.log(`Call A = ${labelA}`);
console.log(`Call B = ${labelB}`);
