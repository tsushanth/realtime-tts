"""
Real streaming-serving latency for the fine-tuned joanna checkpoint, using the
same vLLM/OrpheusModel path orpheus_tts_dev.py uses in prod -- NOT the plain
HF generate() call decode_validate.py used (that measured LM-only decode
speed with no batching/streaming engine, ~2.6-4.3s/sentence, not a real
number). This is the number that actually answers "does it hit 500ms".

Metric definitions (mirrors this repo's STT latency harness conventions --
first-chunk time, then n runs for median/p90 rather than a single sample):
  - ttfb_s: wall-clock from request start to the FIRST audio chunk yielded by
    generate_speech() -- the number that matters against the 500ms target.
  - total_s: wall-clock to the last chunk (full-utterance synthesis time).
  - audio_s: duration of the synthesized audio, from chunk byte count
    (PCM16 mono @ 24kHz -> bytes / 2 / 24000).
  - rtf: total_s / audio_s (real-time factor; <1 means faster than real-time).

Run: modal run --detach voice-pipeline/orpheus_clone_pilot/serve_benchmark.py
"""
import json
import time

import modal

VOICE_TAG = "joanna"
SAMPLE_RATE = 24000
N_RUNS = 5  # per prompt, first run's cold-start effects reported separately

PROMPTS = [
    f"{VOICE_TAG}: Thanks for calling, how can I help you today?",
    f"{VOICE_TAG}: Your order will arrive in three to five business days.",
    f"{VOICE_TAG}: I'm sorry, I didn't quite catch that. Could you repeat it?",
]

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # vllm==0.7.3 requires transformers>=4.48.2 (pinning to 4.46.3, which
        # worked for finetune_pilot.py's save/load, is a hard pip conflict
        # here). An unpinned "transformers" (matching orpheus_tts_dev.py's
        # prod image) resolved to something newer than 4.48.2 and crashed
        # vLLM with "TokenizersBackend has no attribute
        # all_special_tokens_extended" -- a real tokenizers/transformers
        # version-compat bug, likely present in the prod image too since it
        # was never confirmed to have actually run. Pinning to exactly the
        # floor vLLM declares, closer to what 0.7.3 was tested against.
        "transformers==4.48.2", "huggingface_hub", "numpy<2", "soundfile",
        "snac", "vllm==0.7.3", "orpheus-speech",
    )
)

app = modal.App("orpheus-clone-pilot-serve", image=gpu_image)
checkpoint_volume = modal.Volume.from_name("orpheus-clone-pilot-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=300,  # tight cap while diagnosing the earlier 40+ minute hang
    scaledown_window=300,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[hf_secret],
)
class ClonedVoiceEngine:
    """Mirrors orpheus_tts_dev.py's OrpheusEngine, but loads the merged
    joanna checkpoint from the volume instead of downloading the 8-voice
    finetune-prod model from HF.

    tokenizer is a HF hub id (not the local checkpoint dir), matching
    orpheus_tts_dev.py's own pattern (model_name=local path, tokenizer=hub
    id) -- unverified whether OrpheusModel's internals handle a local
    tokenizer path the same way, so this avoids relying on that."""

    @modal.enter()
    def load(self):
        import functools
        import os
        import torch
        import vllm
        import orpheus_tts.engine_class as orpheus_engine_mod
        from orpheus_tts import OrpheusModel

        # OrpheusModel._setup_engine() hard-codes AsyncEngineArgs(model=...,
        # dtype=...) with no way to pass max_model_len through the public
        # constructor -- it then defaults to the Llama-3 backbone's native
        # max_position_embeddings (131072), which doesn't fit an A10G's KV
        # cache ("max seq len (131072) is larger than ... KV cache (113984)").
        # Our sequences are ~1-2K tokens; patch the class orpheus_tts.engine_class
        # actually holds a reference to, not vllm's own module (already-bound
        # names don't get un-bound by patching the source module).
        orpheus_engine_mod.AsyncEngineArgs = functools.partial(
            vllm.AsyncEngineArgs, max_model_len=2048
        )

        model_dir = f"/checkpoints/{VOICE_TAG}_merged"
        t0 = time.time()
        # Ground truth from inspecting the installed orpheus-speech source
        # directly (two earlier guesses -- tokenizer=, max_model_len= --
        # each crash-looped every @modal.enter() retry until the function
        # timeout, burning GPU time; orpheus_tts_dev.py's prod script passes
        # both and was never confirmed to have actually run). The real
        # signature is __init__(self, model_name, dtype=...) only; it loads
        # its own tokenizer from model_name internally.
        self.model = OrpheusModel(
            model_name=model_dir,
            dtype=torch.bfloat16,
        )
        self.container_load_seconds = round(time.time() - t0, 2)
        print(f"Cloned-voice engine loaded in {self.container_load_seconds}s", flush=True)

    @modal.method()
    def benchmark(self, prompt: str, n_runs: int, max_wall_s: float = 60.0, max_tokens: int = 1200):
        """max_wall_s is a hard per-run cutoff: an under-trained checkpoint
        can fail to emit an end-of-speech token and generate until
        max_model_len every time -- that hung a full run for 40+ minutes
        with vLLM's internal generation silent the whole time. Chunk-level
        logging plus this cutoff make a hang visible and bounded instead."""
        results = []
        for run_idx in range(n_runs):
            t0 = time.time()
            ttfb = None
            total_bytes = 0
            n_chunks = 0
            timed_out = False
            for chunk in self.model.generate_speech(
                prompt=prompt,
                voice=VOICE_TAG,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.3,
                max_tokens=max_tokens,
                # orpheus-speech's own default (stop_token_ids=[49158]) does
                # not match our tokenization's real end-of-speech token
                # (128258, verified against canopylabs' notebook and used
                # throughout finetune_pilot.py/decode_validate.py) -- that
                # mismatch is why identical prompts produced 1-12s of audio
                # across runs with no reliable stopping point.
                stop_token_ids=[128258],
            ):
                now = time.time()
                if ttfb is None:
                    ttfb = now - t0
                    print(f"    run {run_idx}: first chunk at {ttfb:.3f}s", flush=True)
                total_bytes += len(chunk)
                n_chunks += 1
                if now - t0 > max_wall_s:
                    timed_out = True
                    print(f"    run {run_idx}: exceeded {max_wall_s}s wall clock after {n_chunks} chunks, aborting this run", flush=True)
                    break
            total_s = time.time() - t0
            audio_s = total_bytes / 2 / SAMPLE_RATE  # PCM16 mono
            results.append({
                "run": run_idx,
                "ttfb_s": round(ttfb, 3) if ttfb is not None else None,
                "total_s": round(total_s, 3),
                "audio_s": round(audio_s, 3),
                "rtf": round(total_s / audio_s, 3) if audio_s > 0 else None,
                "n_chunks": n_chunks,
                "timed_out": timed_out,
            })
            print(f"    run {run_idx} done: {results[-1]}", flush=True)
        return results


@app.function(image=gpu_image, timeout=300, volumes={"/checkpoints": checkpoint_volume})
def run_benchmark(smoke_test: bool = False):
    import statistics

    engine = ClonedVoiceEngine()
    all_results = {}
    prompts = PROMPTS[:1] if smoke_test else PROMPTS
    n_runs = 1 if smoke_test else N_RUNS
    for prompt in prompts:
        print(f"=== Benchmarking: {prompt!r} ===", flush=True)
        runs = engine.benchmark.remote(prompt, n_runs, max_wall_s=60.0, max_tokens=400 if smoke_test else 1200)
        for r in runs:
            print(f"  run {r['run']}: ttfb={r['ttfb_s']}s total={r['total_s']}s audio={r['audio_s']}s rtf={r['rtf']}", flush=True)
        # First run includes any first-call vLLM warmup inside the already-loaded
        # engine; report it separately rather than silently averaging it in.
        warm_runs = runs[1:] if len(runs) > 1 else runs
        ttfbs = [r["ttfb_s"] for r in warm_runs if r["ttfb_s"] is not None]
        all_results[prompt] = {
            "runs": runs,
            "cold_first_run_ttfb_s": runs[0]["ttfb_s"],
            "warm_ttfb_median_s": round(statistics.median(ttfbs), 3) if ttfbs else None,
            "warm_ttfb_p90_s": round(sorted(ttfbs)[max(0, int(len(ttfbs) * 0.9) - 1)], 3) if ttfbs else None,
        }

    with open("/checkpoints/serve_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
    checkpoint_volume.commit()
    print("=== Summary ===", flush=True)
    for prompt, r in all_results.items():
        print(f"{prompt!r}: cold={r['cold_first_run_ttfb_s']}s warm_median={r['warm_ttfb_median_s']}s warm_p90={r['warm_ttfb_p90_s']}s", flush=True)
    return all_results


@app.local_entrypoint()
def main(smoke_test: bool = False):
    call = run_benchmark.spawn(smoke_test=smoke_test)
    print(f"Spawned function call: {call.object_id}")
