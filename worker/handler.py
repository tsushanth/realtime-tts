"""RunPod Serverless entrypoint.

Input:  {"text": "...", "voice": "af_heart", "speed": 1.0}
Output: a stream of {"text", "gen_ms", "audio_s", "pcm16_b64", "sample_rate", "providers"}
        objects, one per synthesized clause, terminated by RunPod's generator completion.
        "providers" is onnxruntime's actual active execution providers for that inference
        call — included so GPU-vs-CPU-fallback is directly observable per request instead
        of inferred from timing (see DECISIONS.md — timing alone was misleading).
"""
import base64
import subprocess

import numpy as np
import runpod

from synth import synthesize_stream, SAMPLE_RATE


def gpu_debug_info():
    """One-shot diagnostic: does this container even see an NVIDIA GPU device? If not,
    no amount of CUDA library wiring will make CUDAExecutionProvider register — that's
    an onnxruntime/onnx GPU-detection precondition, not just a linking problem."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return {"nvidia_smi_rc": r.returncode, "nvidia_smi_out": (r.stdout or r.stderr)[:500]}
    except Exception as e:
        return {"nvidia_smi_error": str(e)}


def to_pcm16_b64(samples: np.ndarray) -> str:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm16).decode("ascii")


def handler(job):
    inp = job.get("input", {})
    text = inp.get("text", "")
    voice = inp.get("voice", "af_heart")
    speed = float(inp.get("speed", 1.0))

    first = True
    for chunk in synthesize_stream(text, voice=voice, speed=speed):
        out = {
            "text": chunk["text"],
            "gen_ms": chunk["gen_ms"],
            "audio_s": chunk["audio_s"],
            "sample_rate": SAMPLE_RATE,
            "pcm16_b64": to_pcm16_b64(chunk["samples"]),
            "providers": chunk["providers"],
        }
        if first:
            out["gpu_debug"] = gpu_debug_info()
            first = False
        yield out


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
