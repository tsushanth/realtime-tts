"""
Actually decode the fine-tuned model's output back to audio and save WAVs,
so the pilot can be judged by ear instead of by token-generation speed.

finetune_pilot.py's first validation pass used TextIteratorStreamer, which
decodes generated ids as TEXT -- but most of what this model emits for a TTS
prompt are SNAC audio-token ids, not text tokens. That pass only proved the
model generates *something* quickly; it never proved the audio is valid or
sounds like the cloned voice. This script fixes that: extract the audio-token
span from a raw generate() call, undo the interleave/offset, decode through
SNAC, and write real WAV files to the checkpoint volume.

Run: modal run --detach voice-pipeline/orpheus_clone_pilot/decode_validate.py
"""
import modal

# Inlined from finetune_pilot.py -- Modal only mounts the entrypoint file by
# default, not sibling local imports, so importing that module fails inside
# the container. Keep these in sync with finetune_pilot.py by hand; they are
# the exact constants verified against canopylabs' data-prep notebook.
VOICE_TAG = "joanna"
TOKENISER_LENGTH = 128256
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # = 128266
CODEBOOK_SIZE = 4096
END_OF_TEXT = 128009
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258

VAL_PROMPTS = [
    f"{VOICE_TAG}: Thanks for calling, how can I help you today?",
    f"{VOICE_TAG}: Your order will arrive in three to five business days.",
    f"{VOICE_TAG}: I'm sorry, I didn't quite catch that. Could you repeat it?",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.3", "datasets", "accelerate", "peft==0.13.2",
        "snac", "soundfile", "huggingface_hub", "numpy<2",
    )
)

app = modal.App("orpheus-clone-pilot-decode", image=image)
checkpoint_volume = modal.Volume.from_name("orpheus-clone-pilot-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(gpu="A10G", timeout=1800, volumes={"/checkpoints": checkpoint_volume}, secrets=[hf_secret])
def decode_and_validate():
    import json
    import os
    import time

    import soundfile as sf
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = f"/checkpoints/{VOICE_TAG}_merged"
    print(f"=== Loading merged model from {model_dir} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval().cuda()

    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    def audio_tokens_to_wav(audio_ids: list[int]) -> "torch.Tensor":
        """Invert build_dataset's/finetune_pilot's interleave+offset to
        recover the 3 SNAC code levels, then decode to a waveform."""
        n_frames = len(audio_ids) // 7
        c0, c1, c2 = [], [], []
        for i in range(n_frames):
            f = audio_ids[i * 7:(i + 1) * 7]
            c0.append(f[0] - AUDIO_TOKENS_START)
            c1.append(f[1] - AUDIO_TOKENS_START - 1 * CODEBOOK_SIZE)
            c2.append(f[2] - AUDIO_TOKENS_START - 2 * CODEBOOK_SIZE)
            c2.append(f[3] - AUDIO_TOKENS_START - 3 * CODEBOOK_SIZE)
            c1.append(f[4] - AUDIO_TOKENS_START - 4 * CODEBOOK_SIZE)
            c2.append(f[5] - AUDIO_TOKENS_START - 5 * CODEBOOK_SIZE)
            c2.append(f[6] - AUDIO_TOKENS_START - 6 * CODEBOOK_SIZE)
        codes = [
            torch.tensor(c0, device="cuda").unsqueeze(0),
            torch.tensor(c1, device="cuda").unsqueeze(0),
            torch.tensor(c2, device="cuda").unsqueeze(0),
        ]
        # clamp defensively: a bad sample can emit an out-of-range code, which
        # would otherwise crash the codec decode on an otherwise-fine sequence
        codes = [c.clamp(0, CODEBOOK_SIZE - 1) for c in codes]
        with torch.inference_mode():
            audio = snac.decode(codes)
        return audio.squeeze().float().cpu()

    results = []
    for i, prompt in enumerate(VAL_PROMPTS):
        full_prompt_ids = (
            [START_OF_HUMAN]
            + tokenizer(prompt, add_special_tokens=False).input_ids
            + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        )
        input_ids = torch.tensor([full_prompt_ids]).cuda()

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids, max_new_tokens=1200, do_sample=True,
                temperature=0.6, top_p=0.8, repetition_penalty=1.3,
                eos_token_id=END_OF_SPEECH,
            )
        gen_time = time.time() - t0

        generated = out[0][len(full_prompt_ids):].tolist()
        # keep only audio-range tokens, drop any trailing special tokens
        audio_ids = [t for t in generated if t >= AUDIO_TOKENS_START]
        audio_ids = audio_ids[: (len(audio_ids) // 7) * 7]  # truncate to whole frames

        entry = {
            "prompt": prompt,
            "generated_tokens": len(generated),
            "audio_tokens": len(audio_ids),
            "gen_time_s": round(gen_time, 2),
        }

        if len(audio_ids) < 7:
            entry["status"] = "no_audio_tokens_generated"
            print(f"  [{i}] {prompt!r}: NO AUDIO TOKENS generated ({len(generated)} raw tokens)", flush=True)
        else:
            try:
                wav = audio_tokens_to_wav(audio_ids)
                out_path = f"/checkpoints/pilot_sample_{i}.wav"
                sf.write(out_path, wav.numpy(), 24000)
                duration_s = len(wav) / 24000
                rms = float((wav.pow(2).mean().sqrt()))
                entry.update({
                    "status": "ok",
                    "wav_path": out_path,
                    "duration_s": round(duration_s, 2),
                    "rms": round(rms, 4),
                })
                print(f"  [{i}] {prompt!r}: {duration_s:.2f}s audio, rms={rms:.4f} -> {out_path}", flush=True)
            except Exception as e:
                entry["status"] = f"decode_failed: {e!r}"
                print(f"  [{i}] {prompt!r}: DECODE FAILED: {e!r}", flush=True)

        results.append(entry)

    with open("/checkpoints/pilot_decode_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    print("Saved WAVs and pilot_decode_validation.json to the checkpoint volume.", flush=True)
    return results


@app.local_entrypoint()
def main():
    call = decode_and_validate.spawn()
    print(f"Spawned function call: {call.object_id}")
