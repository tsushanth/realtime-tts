"""
Orpheus voice-cloning pilot: LoRA-finetune canopylabs/orpheus-tts-0.1-pretrained
on a single voice (250 clips from build_dataset.py) and validate the result.

Internal proof-of-concept only -- see build_dataset.py's docstring re: the
Polly-Joanna source data and why it's fine to use here but not in the shipped
product.

Why LoRA, not full fine-tune: canopylabs' own guidance treats 50-300
examples/voice as the expected scale for this recipe, and full fine-tuning a
3B model on <300 examples overfits/forgets fast. LoRA on the LM backbone
matches the recipe canopylabs describes as "any huggingface-compatible
process like LoRA" on top of finetune/train.py.

Why we tokenize with SNAC ourselves instead of using canopylabs' Colab
notebook: same codec (24kHz, 3-level RVQ, 7 interleaved tokens/frame,
~83 tok/s), no need for the HF Hub round-trip for a 250-clip local pilot.

Pipeline (single Modal function, spawned):
  1. Load dataset_manifest.jsonl + clips/ (built by build_dataset.py).
  2. Encode each clip with SNAC, interleave codes, append after text tokens
     -> input_ids per canopylabs' documented layout.
  3. LoRA-finetune the LM (r=16, alpha=32, targeting attention + MLP proj
     layers) for a few epochs -- 250 examples is small enough that this
     finishes in well under an hour even on a single A10G.
  4. Merge the adapter into the base weights and save the merged model to a
     Modal Volume, so it's a normal HF checkpoint OrpheusModel can load with
     no runtime LoRA machinery.
  5. Run a handful of validation prompts through the merged model and save
     the audio + wall-clock time-to-first-chunk, so the PoC's win condition
     (does it sound like the source voice, is TTFB anywhere near budget) is
     checkable without a separate deploy step.

Deploy/run: modal run voice-pipeline/orpheus_clone_pilot/finetune_pilot.py \
    --dataset-dir ./pilot_upload (a local copy of build_dataset.py's --out dir)
"""
import modal

BASE_MODEL = "canopylabs/orpheus-tts-0.1-pretrained"  # fine-tune from pretrained, not finetune-prod
SAMPLE_RATE = 24000
VOICE_TAG = "joanna"

# Exact layout verified against canopylabs' own data-prep notebook (linked from
# github.com/canopyai/Orpheus-TTS README, "Finetune Model" step 2) -- do not
# reconstruct these from memory, the offsets and interleave order are easy to
# get subtly wrong and it fails silently (trains on garbage, no error).
TOKENISER_LENGTH = 128256  # Llama-3 base vocab size
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # = 128266
CODEBOOK_SIZE = 4096  # SNAC per-level codebook size
END_OF_TEXT = 128009
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
PAD_TOKEN = 128263  # per canopylabs finetune/config.yaml

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

app = modal.App("orpheus-clone-pilot", image=image)

dataset_volume = modal.Volume.from_name("orpheus-clone-pilot-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("orpheus-clone-pilot-checkpoints", create_if_missing=True)

VAL_PROMPTS = [
    f"{VOICE_TAG}: Thanks for calling, how can I help you today?",
    f"{VOICE_TAG}: Your order will arrive in three to five business days.",
    f"{VOICE_TAG}: I'm sorry, I didn't quite catch that. Could you repeat it?",
]


@app.function(
    gpu="A10G",
    timeout=4 * 3600,
    volumes={"/data": dataset_volume, "/checkpoints": checkpoint_volume},
)
def run_pilot(epochs: int = 3, lora_r: int = 16, lora_alpha: int = 32):
    import json
    import os
    import time

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from snac import SNAC
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    print("=== Loading dataset manifest ===", flush=True)
    manifest_path = "/data/dataset_manifest.jsonl"
    rows = [json.loads(l) for l in open(manifest_path)]
    print(f"{len(rows)} text/audio pairs", flush=True)

    print("=== Loading SNAC codec (24kHz) ===", flush=True)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode_audio_to_tokens(wav_path: str) -> list[int]:
        """SNAC-encode one clip and interleave the 3 hierarchical code levels
        into 7 tokens/frame, exactly per canopylabs' data-prep notebook
        (level order [0,1,2,2,1,2,2], each occurrence in its own 4096-wide
        offset band, stacked in that order -- NOT simply grouped by level)."""
        audio, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.mean(dim=0, keepdim=True).unsqueeze(0).cuda()  # mono, batch=1
        with torch.inference_mode():
            codes = snac.encode(audio)  # list of 3 tensors, one per RVQ level
        # codes[0]: coarsest (1 tok/frame), codes[1]: 2 tok/frame, codes[2]: 4 tok/frame
        c0, c1, c2 = [c.squeeze(0).squeeze(0).tolist() for c in codes]
        n_frames = len(c0)
        frames = []
        for i in range(n_frames):
            frames.append([
                AUDIO_TOKENS_START + c0[i],
                AUDIO_TOKENS_START + 1 * CODEBOOK_SIZE + c1[2 * i],
                AUDIO_TOKENS_START + 2 * CODEBOOK_SIZE + c2[4 * i],
                AUDIO_TOKENS_START + 3 * CODEBOOK_SIZE + c2[4 * i + 1],
                AUDIO_TOKENS_START + 4 * CODEBOOK_SIZE + c1[2 * i + 1],
                AUDIO_TOKENS_START + 5 * CODEBOOK_SIZE + c2[4 * i + 2],
                AUDIO_TOKENS_START + 6 * CODEBOOK_SIZE + c2[4 * i + 3],
            ])
        # Dedup: drop any frame whose coarsest-level token matches the
        # previous frame's, per canopylabs' remove_duplicate_frames pass --
        # skipping this makes sequence lengths diverge from their recipe
        # even with correct interleaving.
        deduped = [frames[0]] if frames else []
        for f in frames[1:]:
            if f[0] != deduped[-1][0]:
                deduped.append(f)
        interleaved = []
        for f in deduped:
            interleaved.extend(f)
        return interleaved

    print("=== Tokenizing clips (text + SNAC codes) ===", flush=True)
    examples = []
    t0 = time.time()
    for i, row in enumerate(rows):
        text_ids = tokenizer(row["text"], add_special_tokens=False).input_ids
        audio_ids = encode_audio_to_tokens(os.path.join("/data", row["audio"]))
        # Exact sequence layout from canopylabs' notebook: human turn wraps
        # the text, ai turn wraps the speech codes.
        input_ids = (
            [START_OF_HUMAN] + text_ids + [END_OF_TEXT, END_OF_HUMAN]
            + [START_OF_AI, START_OF_SPEECH] + audio_ids + [END_OF_SPEECH, END_OF_AI]
        )
        examples.append({"input_ids": input_ids, "labels": list(input_ids)})
        if (i + 1) % 50 == 0:
            print(f"  tokenized {i + 1}/{len(rows)} ({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"Tokenized {len(examples)} examples in {time.time() - t0:.0f}s", flush=True)

    ds = Dataset.from_list(examples)

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [PAD_TOKEN] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
            attn.append([1] * len(x["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    print("=== Loading base model + attaching LoRA ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="/tmp/orpheus_lora_out",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)

    print("=== Training ===", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"Training finished in {time.time() - t0:.0f}s", flush=True)

    print("=== Merging LoRA into base weights ===", flush=True)
    merged = model.merge_and_unload()
    out_dir = f"/checkpoints/{VOICE_TAG}_merged"
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    checkpoint_volume.commit()
    print(f"Merged model saved to {out_dir}", flush=True)

    print("=== Validation: sample generation + latency ===", flush=True)
    # Reload as a plain causal LM for a quick greedy sanity check (not the
    # vLLM streaming path used in prod -- that's the next wiring step once
    # this proves the voice is cloned and where TTFB roughly lands).
    from transformers import TextIteratorStreamer
    import threading

    merged.eval().cuda()
    results = []
    for prompt in VAL_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        t0 = time.time()
        first_token_time = None
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        def _gen():
            merged.generate(
                input_ids, max_new_tokens=700, do_sample=True,
                temperature=0.6, top_p=0.8, repetition_penalty=1.3,
                streamer=streamer,
            )

        th = threading.Thread(target=_gen)
        th.start()
        token_count = 0
        for _ in streamer:
            if first_token_time is None:
                first_token_time = time.time() - t0
            token_count += 1
        th.join()
        results.append({
            "prompt": prompt,
            "ttft_s": round(first_token_time, 3) if first_token_time else None,
            "total_s": round(time.time() - t0, 3),
            "tokens": token_count,
        })
        print(f"  {prompt!r} -> TTFT={results[-1]['ttft_s']}s total={results[-1]['total_s']}s", flush=True)

    with open("/checkpoints/pilot_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    print("Note: this LM-only TTFT does not include SNAC decode to audio or "
          "vLLM's batching speedup -- treat it as an upper bound, not the "
          "final streaming-path number. Wire the merged checkpoint into "
          "orpheus_tts_dev.py's OrpheusModel/vLLM path for the real figure.",
          flush=True)
    return results


@app.local_entrypoint()
def main(dataset_dir: str = "", epochs: int = 3):
    if dataset_dir:
        print(f"Uploading {dataset_dir} to the orpheus-clone-pilot-data volume...")
        with dataset_volume.batch_upload(force=True) as batch:
            batch.put_directory(dataset_dir, "/")
        print("Upload complete.")

    call = run_pilot.spawn(epochs=epochs)
    print(f"Spawned function call: {call.object_id}")
    print("Retrieve later with modal.FunctionCall.from_id(<id>).get(), "
          "or check `modal app logs orpheus-clone-pilot`.")
