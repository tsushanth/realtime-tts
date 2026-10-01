"""
Qwen3-32B, self-hosted on Modal — newer generation than app_32b.py's
Qwen2.5-32B-Instruct, requested 2026-09-29 to keep the model comparison
going after Qwen2.5-32B's surprising result (fastest of everything tested,
only one to interleave speech with tool calls, but WORSE at following the
explicit "restate all three fields" confirmation-node rule than the
smaller Qwen2.5-7B with the identical prompt fix — see
../../calldesktech/bench/MODEL-COMPARISON-2026-09-29.md).

Why Qwen3 specifically, not a different family: isolates "generation"
as the next variable to test, holding family constant — Qwen2.5-7B and
Qwen2.5-32B already established a same-family baseline at two sizes.
Background research the same day flagged Qwen3-235B-A22B as the current
BFCL-v4 leader among open models, with Qwen3-32B/30B-A3B as smaller
same-family options — but also flagged that the vLLM "hermes" tool-call
parser's compatibility with Qwen3 was, at the time of that research, an
open GitHub discussion, not a settled confirmation like Qwen2.5's already
proven-live-here support. This deployment is that verification.

UPDATE (still 2026-09-29): the vllm==0.6.3 attempt failed outright —
`transformers` at that pin doesn't recognize the `qwen3` model_type at all
("Transformers does not recognize this architecture"), a clean model-
loading error confirming vLLM 0.6.3 simply predates Qwen3 (real-world vLLM
added Qwen3 support around 0.8.5). Bumped to vllm==0.9.2 — well past that,
not bleeding-edge latest (0.30.0 available), to limit how many other
things might have changed at once (CLI flags, tool-parser naming, outlines
dependency shape) while still definitely supporting Qwen3.

Removed the pyairports/outlines stub build step for this version — it's
specific to vllm==0.6.3's outlines<0.1 pin and would either be a no-op or,
worse, fail the build outright if 0.9.2's dependency shape doesn't match
(e.g. outlines removed as a hard dependency, or a different version with a
different missing-module story). If chat completions 500 the same way on
this version, that's the next thing to check — don't assume the exact same
fix applies without verifying first.
"""
import modal

MODEL_NAME = "Qwen/Qwen3-32B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # UPDATE: vllm==0.9.2 turned out to be a dead end regardless of the
        # transformers pin — its vendored transformers_utils/configs/ovis.py
        # unconditionally calls AutoConfig.register("aimv2", ...) at import
        # time, and EVERY transformers release tried (unpinned 5.17.0, and
        # transformers<5.0 which resolved to 4.57.6) already registers
        # "aimv2" natively, so the collision isn't about which transformers
        # version — it's that 0.9.2's vendored ovis config predates
        # transformers absorbing that model type natively, and there's no
        # transformers version where both can coexist. Jumping to a
        # meaningfully newer vllm on the assumption this vendored-config
        # approach was refactored away once transformers added native
        # support (a common pattern) — not verified, next thing to check if
        # this also fails.
        "vllm==0.11.0",
        # vllm 0.11.0's own PyPI metadata declares only a floor
        # (transformers>=4.55.2) with NO ceiling — real packaging gap, not a
        # pin mistake on this end. Unpinned, pip resolves the newest
        # transformers, which has already dropped the
        # all_special_tokens_extended property vllm's tokenizer loader
        # still calls (AttributeError, found live). Pinning to the exact
        # floor version vllm declares as its minimum, on the reasoning that
        # it's the version most likely still tested against at release time.
        "transformers==4.55.2",
    )
    .run_commands(
        f"python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}')\""
    )
)

app = modal.App("selfhosted-llm-worker-qwen3-32b", image=image)


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
            "--served-model-name", "qwen3-32b-selfhosted",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--api-key", api_key,
            "--max-model-len", "8192",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            # Real bug found live: vllm 0.11.0's default torch.compile +
            # CUDA-graph capture across ~69 batch sizes took 230s+ and still
            # hadn't finished when Modal killed the container at "Runner has
            # been initializing for too long: 300 seconds" — a fixed
            # platform check independent of this function's own
            # startup_timeout=420. That compile investment only pays off
            # under high-throughput batched serving; this deployment serves
            # one voice call's turns at a time. --enforce-eager skips
            # compilation/cudagraphs entirely, trading steady-state
            # throughput (irrelevant here) for a startup fast enough to
            # actually finish.
            "--enforce-eager",
        ]
    )
