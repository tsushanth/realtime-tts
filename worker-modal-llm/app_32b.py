"""
Larger step-up from app.py's Qwen2.5-7B-Instruct, same family and same
vLLM tool-call parser ("hermes") deliberately — this isolates PARAMETER
COUNT as the one variable changing versus the working 7B deployment,
rather than also switching model family/generation at the same time
(requested 2026-09-29: "want to see how a larger model performs").

Model: Qwen2.5-32B-Instruct. Not Qwen3 — a background research pass the
same day flagged Qwen3 as plausibly stronger on BFCL, but with an
unverified tool-call-parser story for vLLM (open GitHub discussion, not a
settled "hermes" parser confirmation like Qwen2.5 already has proven live
here). Worth trying Qwen3 as a FOLLOW-UP once this same-family, same-parser
comparison establishes whether size alone fixes the confirmation-readback
bug Qwen2.5-7B showed (dropping the callback number from a required
three-field readback) — see app.py's docstring and
call-loop-poc/server.js's OPENAI_COMPAT_ENDPOINT 'qwen2.5-32b-...' entry.

GPU: A100-80GB, not A10G — 32B in bf16 needs ~65GB weights alone, doesn't
fit A10G's 24GB or even L40S's 48GB with room for KV cache. This is a real,
larger recurring-if-used cost than the 7B's A10G (~$1.10/hr vs A100-80GB's
~$3.95/hr list price on Modal) — still the SAME scaledown_window=120,
no-min-containers tradeoff as every other worker here, so it's usage-
proportional, not an always-on commitment, but flagging the per-hour delta
explicitly since "try a larger model" has a real cost shape difference
worth knowing going in, not discovering later.
"""
import modal

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "huggingface_hub==0.25.2",
    )
    .run_commands(
        # Same pyairports/outlines bug as app.py — see that file's comment
        # for the full explanation. Identical fix, baked in here too since
        # this is a separate image build.
        "mkdir -p /usr/local/lib/python3.11/site-packages/pyairports && "
        "touch /usr/local/lib/python3.11/site-packages/pyairports/__init__.py && "
        "echo 'AIRPORT_LIST = []' > /usr/local/lib/python3.11/site-packages/pyairports/airports.py && "
        "python3 -c \"import outlines\""
    )
    .run_commands(
        f"python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}')\""
    )
)

app = modal.App("selfhosted-llm-worker-32b", image=image)


@app.function(
    gpu="A100-80GB",
    scaledown_window=120,
    max_containers=3,
    secrets=[modal.Secret.from_name("selfhosted-llm-auth-token")],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=420)
def serve():
    import os
    import subprocess

    api_key = os.environ["LLM_API_KEY"]
    subprocess.Popen(
        [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--served-model-name", "qwen2.5-32b-instruct-selfhosted",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--api-key", api_key,
            "--max-model-len", "8192",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
    )
