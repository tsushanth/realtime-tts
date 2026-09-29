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
    if store.is_delete_requested(vid):
        # A delete was requested; never serve it even if a training job's
        # commit left it "ready" before the deletion was completed.
        raise UnknownVoiceError(f"voice deleted: {vid!r}")
    return store.checkpoint_dir(vid)


def _bounded_generate(chunk_iter, timeout_s: float):
    """Bounds elapsed time between yields from the wrapped generator.

    Raises GenerationTimeoutError if no chunk is yielded within timeout_s.

    LIMITATION: Does NOT interrupt a call that blocks entirely inside a single
    next() with no yield at all (a true hang, as opposed to a slow-but-yielding
    generation). The deadline check only runs once a chunk actually arrives. The
    real backstop against a fully-blocked generation is the outer Modal
    function-level timeout, configured where this engine is wired into a
    deployable Modal app (Task 6), not this in-process check alone.

    The pilot's confirmed runaway-generation failure mode (RTF>1, unreliable
    stop-token behavior) manifested as slow or stalled yields, which this
    check is designed to catch; true complete hangs would require thread-based
    or async-cancellation mechanisms outside the scope of this wrapper.

    NOTE: raising here abandons the generator but does NOT cancel the
    underlying vLLM request -- it keeps generating on the GPU (competing with
    later requests) until it hits its own stop token or max length."""
    deadline = time.time() + timeout_s
    for chunk in chunk_iter:
        if time.time() > deadline:
            raise GenerationTimeoutError(f"generation exceeded {timeout_s}s")
        yield chunk


def load_engine_for_checkpoint(checkpoint_dir: str):
    """Constructs a fresh OrpheusModel for one voice's checkpoint, applying
    the AsyncEngineArgs max_model_len patch first. Called once from
    main.py's OrpheusCloneEngine @modal.enter(): that class is parameterized
    by vid, so each GPU container serves exactly one voice and never swaps
    engines (vLLM 0.7.3 doesn't reliably free GPU memory on engine delete).

    Regression-guarded by tests/test_serve.py::test_load_engine_* -- do not
    add tokenizer=/max_model_len= kwargs or move the patch target."""
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
