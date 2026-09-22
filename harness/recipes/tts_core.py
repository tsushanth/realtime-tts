"""TTS-core recipe: tries different Piper model-size/quality-tier variants,
reusing training-data/piper_full_finetune.py's exact training approach
(same Modal image, same piper.train CLI invocation) parameterized instead
of hardcoded to one checkpoint. See
docs/superpowers/specs/2026-09-22-voice-research-harness-design.md."""
import json
import os

from harness.recipe import Candidate, TrainedModel

TRIED_PATH = os.path.join(os.path.dirname(__file__), "tts_core_tried.json")

# rhasspy/piper-checkpoints ships three quality tiers per voice as genuinely
# different-sized models (not just different training - "low"/"medium"/"high"
# are architecturally smaller/larger VITS configs). Using the same Lessac
# English base this repo's own full-finetune pipeline already uses
# (training-data/piper_full_finetune.py) so results are comparable against
# that existing, already-verified run.
KNOWN_SIZE_VARIANTS = [
    {
        "size_label": "low",
        "warmstart_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/low/epoch%3D2218-step%3D1358400.ckpt",
        "max_steps": 8000,
    },
    {
        "size_label": "medium",
        "warmstart_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt",
        "max_steps": 20000,
    },
    {
        "size_label": "high",
        "warmstart_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/high/epoch%3D2218-step%3D1358400.ckpt",
        "max_steps": 20000,
    },
]

# Basis for the $/step estimate: piper_full_finetune.py's own measured rate,
# ~1.2 it/s (avg of 1.14-1.39) at batch_size=8 on a Modal T4. Re-verify the
# T4 hourly rate against Modal's current published pricing before trusting
# this for a real budget decision - it is not re-fetched at runtime.
_T4_HOURLY_USD = 0.59
_MEASURED_STEPS_PER_SECOND = 1.2


class TTSCoreRecipe:
    name = "tts-core"

    def discover_candidates(self) -> list[Candidate]:
        tried = self._load_tried()
        candidates = []
        for variant in KNOWN_SIZE_VARIANTS:
            if variant["size_label"] in tried:
                continue
            candidates.append(
                Candidate(
                    id=variant["size_label"],
                    description=f"Piper Lessac {variant['size_label']}-quality-tier fine-tune, {variant['max_steps']} steps",
                    train_config={
                        "warmstart_url": variant["warmstart_url"],
                        "max_steps": variant["max_steps"],
                        "size_label": variant["size_label"],
                    },
                )
            )
        return candidates

    def estimate_cost_usd(self, candidate: Candidate) -> float:
        max_steps = candidate.train_config["max_steps"]
        seconds = max_steps / _MEASURED_STEPS_PER_SECOND
        hours = seconds / 3600
        return round(hours * _T4_HOURLY_USD, 2)

    def _load_tried(self) -> dict:
        if not os.path.exists(TRIED_PATH):
            return {}
        try:
            with open(TRIED_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File exists but is corrupted/truncated/invalid JSON. Treat as
            # "no candidates tried yet" and degrade gracefully instead of
            # crashing. This can happen if a write is interrupted mid-flight.
            return {}

    def train(self, candidate: Candidate, workdir: str) -> TrainedModel:
        import time

        from harness.recipes.tts_core_train_job import run_piper_finetune

        cfg = candidate.train_config
        t0 = time.time()
        call = run_piper_finetune.spawn(
            warmstart_url=cfg["warmstart_url"],
            max_steps=cfg["max_steps"],
            size_label=cfg["size_label"],
        )
        artifact_path = call.get(timeout=8 * 3600)  # blocks the harness process for this candidate's whole run - fine for an on-demand CLI tool, not a long-lived service
        elapsed_hours = (time.time() - t0) / 3600
        actual_cost = round(elapsed_hours * _T4_HOURLY_USD, 2)

        return TrainedModel(candidate=candidate, artifact_path=artifact_path, actual_cost_usd=actual_cost)
