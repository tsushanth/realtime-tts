#!/bin/bash
# Plays the same sentence in each voice, back to back, for A/B listening.
# Usage: ./compare_by_ear.sh [sentence-number 0-4 | all]   (default: all)
cd "$(dirname "$0")"
SENT=("Thanks for calling... (account details)" "Your order should arrive... (3-5 days)" "I understand your frustration..." "Is there anything else...?" "Your extension is six six three five.")
play() { echo "   > $1"; afplay "$2"; sleep 0.6; }
run() {
  i=$1; echo; echo "=== Sentence $i: ${SENT[$i]} ==="
  play "Piper (ours, full corpus)"      piper_full_output/sample_$i.wav
  play "Matcha-TTS (ours, full corpus)" full_ft_cpu_output/sample_$i.wav
  play "ElevenLabs Flash v2.5"          elevenlabs_output/flash_$i.mp3
  play "ElevenLabs Multilingual v2"     elevenlabs_output/multilingual_$i.mp3
}
if [ "${1:-all}" = "all" ]; then for i in 0 1 2 3 4; do run $i; done; else run "$1"; fi
