#!/bin/bash
# Per sentence: (1) LJ base, (2) 10-min fine-tune, (3) ~25-min fine-tune if present, (4) current Piper voice.
cd "$(dirname "$0")"
for i in 0 1 2 3 4; do
  echo "== sentence $i"
  for f in ljspeech_output lj_small_output/small/samples lj_small_output/full/samples piper_full_output; do
    [ -f $f/sample_$i.wav ] && { echo "  $f"; afplay $f/sample_$i.wav; sleep 0.5; }
  done; sleep 1
done
