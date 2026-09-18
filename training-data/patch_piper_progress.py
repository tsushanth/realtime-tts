"""Build-time patch for piper1-gpl's dataset.py: adds progress logging every
500 utterances to prepare_data(), which otherwise logs nothing between
"Processing utterances..." at the start and "Processed N utterance(s)" at
the end - silent by design, indistinguishable from a hang at full-corpus
scale. See piper_full_finetune.py's docstring for the real incident this
fixes (two 8+ hour "hangs" that were actually just this silent path).
"""
path = "src/piper/train/vits/dataset.py"
old = "num_utterances += 1\n                if report_prepare:"
new = (
    "num_utterances += 1\n"
    "                if num_utterances % 500 == 0:\n"
    '                    _LOGGER.info("Processed %s utterances so far...", num_utterances)\n'
    "                if report_prepare:"
)

with open(path) as f:
    content = f.read()

assert old in content, "patch target not found - dataset.py may have changed upstream"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print("Patched dataset.py with periodic progress logging.")
