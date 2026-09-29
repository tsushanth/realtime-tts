"""
Production serving for the Orpheus streaming-clone service.

Carries forward three real, verified-by-running fixes from
serve_benchmark.py (do not regress any of them -- see Global Constraints):
  1. OrpheusModel(model_name=..., dtype=...) only -- no tokenizer=/max_model_len=.
  2. AsyncEngineArgs monkey-patched to max_model_len=2048 before constructing
     OrpheusModel, or vLLM tries to size the KV cache for the base model's
     native 131072 context and fails on an A10G.
  3. stop_token_ids=[128258] passed explicitly to every generate_speech()
     call -- the library's own default ([49158]) does not match this
     tokenization's real end-of-speech token.

Runs in the vllm==0.7.3 / transformers==4.48.2-pinned serving image (see
Global Constraints) -- a DIFFERENT image than train_job.py's
transformers==4.46.3 training image.
"""
import re
import time

from orpheus_clone_prod.storage import VoiceRecordStore

VOICE_ID_RE = re.compile(r"^v-[0-9a-f]{10}$")
CUSTOM_FAST_PREFIX = "custom-fast:"


class UnknownVoiceError(Exception):
    pass


class GenerationTimeoutError(Exception):
    pass


def resolve_voice_dir(voice: str, store: VoiceRecordStore) -> str:
    if not voice.startswith(CUSTOM_FAST_PREFIX):
        raise UnknownVoiceError(f"not a custom-fast voice: {voice!r}")
    vid = voice[len(CUSTOM_FAST_PREFIX):]
    if not VOICE_ID_RE.match(vid):
        raise UnknownVoiceError(f"malformed voice id: {vid!r}")
    status = store.read_status(vid)
    if status is None or status.get("status") != "ready":
        raise UnknownVoiceError(f"voice not ready: {vid!r}")
    return store.checkpoint_dir(vid)


def _bounded_generate(chunk_iter, timeout_s: float):
    """Wraps a chunk generator with a wall-clock deadline -- reproduces and
    bounds the pilot's own confirmed runaway-generation failure mode (RTF>1,
    unreliable stop-token behavior) instead of letting a request hang."""
    deadline = time.time() + timeout_s
    for chunk in chunk_iter:
        if time.time() > deadline:
            raise GenerationTimeoutError(f"generation exceeded {timeout_s}s")
        yield chunk


def load_engine_for_checkpoint(checkpoint_dir: str):
    """Constructs a fresh OrpheusModel for one voice's checkpoint, applying
    the AsyncEngineArgs max_model_len patch first. One engine instance per
    warm container; a new voice request on a cold container calls this once
    in @modal.enter()-equivalent setup (wired in Task 6's Modal wrapper)."""
    import functools

    import torch
    import vllm
    import orpheus_tts.engine_class as orpheus_engine_mod
    from orpheus_tts import OrpheusModel

    orpheus_engine_mod.AsyncEngineArgs = functools.partial(
        vllm.AsyncEngineArgs, max_model_len=2048
    )
    return OrpheusModel(model_name=checkpoint_dir, dtype=torch.bfloat16)


def synthesize(model, text: str, voice_tag: str, timeout_s: float = 15.0):
    """voice_tag is the tag baked into the checkpoint's training data
    (e.g. the vid itself, per train_job.py) -- the prompt prefix, not the
    "custom-fast:" API-facing voice string."""
    chunks = model.generate_speech(
        prompt=text,
        voice=voice_tag,
        temperature=0.6,
        top_p=0.8,
        repetition_penalty=1.3,
        stop_token_ids=[128258],
    )
    yield from _bounded_generate(chunks, timeout_s)
