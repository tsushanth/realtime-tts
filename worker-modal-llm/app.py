"""
Self-hosted open-weight LLM, on Modal, serving vLLM's own OpenAI-compatible
API — the open-weight leg of the Haiku/Luna/self-hosted three-way model
comparison for call-loop-poc's flow-node LLM (see server.js's
MODEL_PROVIDER map / runOpenAiCompatibleTurn adapter, which this is designed
to be a drop-in provider for, same shape as the OpenAI/Groq legs).

Model: Qwen2.5-7B-Instruct — "try small model first" per the 2026-09-29
request. Chosen specifically for tool-calling reliability, not just size:
the Groq-hosted gpt-oss-120b leg tested the same day failed to extract more
than 1 of 3 fields per turn and needed a follow-up round-trip just to speak
at all (see server.js's runOpenAiCompatibleTurn comment). Qwen2.5-Instruct's
chat template has solid, well-exercised OpenAI-style function-calling
support at small sizes, which matters more here than raw parameter count.

vLLM's own `vllm serve` already IS an OpenAI-compatible server (chat
completions, streaming, tool calling) — no custom FastAPI wrapper needed,
unlike the Kokoro worker in ../worker-modal/app.py which had to hand-roll
its own WebSocket protocol. `--api-key` makes it check the same
`Authorization: Bearer <token>` header call-loop-poc's OpenAI-compatible
adapter already sends to every provider.

Cost note (explicit, not a recurring commitment): scaledown_window=120 (no
min_containers) — same idle-teardown-after-calls, cold-start-after-idle
tradeoff already accepted for the Kokoro worker, not the ~$425/mo A10G
always-on alternative. A cold start here is a full vLLM engine boot
(CUDA init + weight load), likely SLOWER than Kokoro's, so expect the first
call after any idle gap to be materially slower than the ones that follow —
worth measuring explicitly before drawing latency conclusions from a single
call, the same lesson worth carrying over from the Kokoro cold-start find.
"""
import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "huggingface_hub==0.25.2",
    )
    .run_commands(
        # Real bug, found live (2026-09-29): vllm==0.6.3 unconditionally
        # calls build_guided_decoding_logits_processor_async on every
        # /v1/chat/completions request (not just guided-JSON/tool-choice
        # ones), which imports outlines.types.airports ->
        # pyairports.airports.AIRPORT_LIST. vllm pins outlines<0.1, and every
        # outlines 0.0.x release has this same import; the pyairports
        # actually published on PyPI (0.0.1) is an unrelated package with no
        # such module — every chat completion 500'd until this was found.
        # We never use guided JSON/airport-code validation, so a stub with
        # an empty list is sufficient — this exists purely to satisfy the
        # unconditional import, not to provide real airport data.
        # Can't `import outlines` to locate site-packages first — that
        # import is exactly what fails. Use the fixed debian_slim python3.11
        # site-packages path directly instead.
        "mkdir -p /usr/local/lib/python3.11/site-packages/pyairports && "
        "touch /usr/local/lib/python3.11/site-packages/pyairports/__init__.py && "
        "echo 'AIRPORT_LIST = []' > /usr/local/lib/python3.11/site-packages/pyairports/airports.py && "
        "python3 -c \"import outlines\""  # verify the stub actually fixes the import before baking it in
    )
    .run_commands(
        # Bake weights into the image at build time, same reasoning as the
        # Kokoro worker's baked-in voice: a cold start should never also pay
        # for a weights download over the network.
        f"python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}')\""
    )
)

app = modal.App("selfhosted-llm-worker", image=image)


@app.function(
    gpu="A10G",
    # See module docstring — deliberately NOT min_containers=1. This is a
    # bounded, usage-proportional idle-teardown, same tradeoff as the Kokoro
    # worker's scaledown_window, not a new recurring spend commitment.
    scaledown_window=120,
    max_containers=5,
    secrets=[modal.Secret.from_name("selfhosted-llm-auth-token")],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    import os
    import subprocess

    api_key = os.environ["LLM_API_KEY"]
    # Real bug found live (2026-09-29), fixed via the pyairports stub in the
    # image build above: vllm==0.6.3's build_guided_decoding_logits_processor
    # eagerly imports outlines.types.airports on EVERY request, tool call or
    # not — it 500'd unconditionally, not just when these flags were set.
    # These flags are what actually enable Qwen's tool-calling support
    # (hermes parser matches Qwen2.5's chat template) and can stay now that
    # the underlying import bug is fixed at the image layer.
    subprocess.Popen(
        [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--served-model-name", "qwen2.5-7b-instruct-selfhosted",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--api-key", api_key,
            "--max-model-len", "8192",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
    )
