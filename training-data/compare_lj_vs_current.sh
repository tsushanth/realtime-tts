#!/bin/bash
# For each test sentence: plays the licence-clean LJ Speech voice, then our current voice.
cd "$(dirname "$0")"
for i in 0 1 2 3 4; do
  echo "== sentence $i: (1) LJ Speech clean base, (2) current Piper full fine-tune"
  afplay ljspeech_output/sample_$i.wav; sleep 0.6; afplay piper_full_output/sample_$i.wav; sleep 1.2
done
