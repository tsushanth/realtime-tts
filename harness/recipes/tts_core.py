"""TTS-core recipe: tries different Piper model-size/quality-tier variants,
reusing training-data/piper_full_finetune.py's exact training approach
(same Modal image, same piper.train CLI invocation) parameterized instead
of hardcoded to one checkpoint. See
docs/superpowers/specs/2026-09-22-voice-research-harness-design.md."""
import datetime
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

# The `engine` label the harness candidate's rows carry in eval/'s results
# files. eval/results/{latency,quality}.json are FLAT lists holding EVERY
# engine's rows in one file (74 rows across piper/kokoro/elevenlabs-flash/
# elevenlabs-multilingual in the committed run), so aggregation MUST filter
# by engine or it silently mixes ElevenLabs' numbers into the candidate's.
CANDIDATE_ENGINE_NAME = "harness-candidate"


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

    def record_tried(self, candidate_id: str, report_path: str | None = None) -> None:
        """Mark a candidate as tried so discover_candidates() stops returning
        it on later cycles. Called (duck-typed, optionally) by the harness core
        after the report is written. A trained-but-eval-failed candidate still
        counts as tried: the GPU money is already spent, and retraining it
        every cycle would burn that money again without fixing the evaluation
        side, which is a separate problem."""
        tried = self._load_tried()
        entry: dict = {"date": datetime.date.today().isoformat()}
        if report_path:
            entry["report_path"] = report_path
        tried[candidate_id] = entry
        with open(TRIED_PATH, "w") as f:
            json.dump(tried, f, indent=2, sort_keys=True)
            f.write("\n")

    def train(self, candidate: Candidate, workdir: str) -> TrainedModel:
        """actual_cost_usd is wall-clock from just before .spawn() until the
        call returns, times the T4 hourly rate. That window includes Modal
        queue and cold-start time, which Modal does not bill for, so this is a
        conservative UPPER BOUND on the real billed cost, not Modal's exact
        figure - it can trip the budget cap early, never overspend. Integrating
        Modal's usage-billing API for an exact number is separate follow-up work."""
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

    def evaluate(self, model: TrainedModel) -> dict[str, float]:
        """Aggregate this candidate's rows from eval/'s two result files into
        summary metrics. Only rows whose `engine` is CANDIDATE_ENGINE_NAME are
        counted - see that constant's comment.

        Missing measurements (score.py writes wer/naturalness_mos as null on a
        scoring failure or for the reference clip itself; synth.mjs writes a
        row with `error` and no first_byte_ms when synthesis failed) are
        skipped rather than coerced to 0.0, which would read as a perfect WER.
        With nothing measurable at all the metric is NaN, not 0.0, so a
        totally failed run can never look like the best candidate."""
        import json
        import math
        import statistics

        latency_path, quality_path = self._run_eval_pipeline(model)

        with open(quality_path) as f:
            quality_rows = [r for r in json.load(f) if r.get("engine") == CANDIDATE_ENGINE_NAME]
        with open(latency_path) as f:
            latency_rows = [r for r in json.load(f) if r.get("engine") == CANDIDATE_ENGINE_NAME]

        wers = [r["wer"] for r in quality_rows if r.get("wer") is not None]
        mos_scores = [r["naturalness_mos"] for r in quality_rows if r.get("naturalness_mos") is not None]
        latencies = [r["first_byte_ms"] for r in latency_rows if r.get("first_byte_ms") is not None]

        nan = math.nan
        return {
            "wer_mean": round(statistics.mean(wers), 4) if wers else nan,
            "naturalness_mos_median": round(statistics.median(mos_scores), 3) if mos_scores else nan,
            "latency_ms_median": round(statistics.median(latencies), 1) if latencies else nan,
            # Provenance for the report: how much of the test set actually
            # produced a number, so a 2-of-20-sentence run is not read as
            # comparable to a full one.
            "n_scored": float(len(wers)),
            "n_failed": float(len(quality_rows) - len(wers)),
        }

    def _run_eval_pipeline(self, model: TrainedModel) -> tuple[str, str]:
        """Returns (latency_path, quality_path). CONTRACT FOR WHOEVER
        IMPLEMENTS THIS: the rows it writes for this candidate must carry
        `engine == CANDIDATE_ENGINE_NAME`, because evaluate() filters on
        exactly that and will silently aggregate zero rows otherwise.

        NOT IMPLEMENTED - deliberately, see .superpowers/sdd/
        2026-09-22-voice-research-harness/task-6-report.md for the full
        write-up. The intended sequence (export the checkpoint to ONNX,
        publish it as a temporary non-public custom: voice, run
        eval/synth.mjs + eval/score.py scoped to it, then DELETE it) hits
        three concrete blockers that need decisions this task cannot make
        unilaterally:

        1. owner.json: the brief prescribes {"public": false, "key_ids": []},
           which worker-piper-fly/server.py's PUT /admin/voices/<id> handler
           rejects with a 400 - it asserts `public is True or non-empty
           key_ids or non-empty user_ids`. Publishing a research candidate
           non-publicly therefore requires the gateway KEY ID that
           TTS_GATEWAY_API_KEY resolves to (the value verify_session_claims
           returns and get_engine matches against owner.json's key_ids). That
           id is minted by gateway/keys.js and is not derivable from the API
           key locally. Publishing with {"public": true} instead would expose
           an unvetted research candidate to every caller of production - the
           exact outcome the brief forbids.

        2. Scoping the run to one voice: eval/synth.mjs hardcodes its engine
           list and its PIPER_VOICE map, and always synthesizes the full
           multi-language testset through Piper, Kokoro AND both ElevenLabs
           models (real per-candidate spend, ~20-30 min wall clock). There is
           no env var or CLI flag to scope it to a single voice. Scoping needs
           either a change to eval/synth.mjs (which this task's interface
           contract says to leave unmodified) or a harness-owned copy of it -
           a structural decision, not an implementation detail.

        3. score.py's MOS is reference-based: it scores every clip against a
           per-language elevenlabs-multilingual reference clip taken FROM THE
           SAME RUN. A single-voice run therefore has no reference and would
           emit naturalness_mos=None for every row, making
           naturalness_mos_median NaN. Any single-voice scoping must still
           synthesize the ElevenLabs reference clips (so it is not free), or
           reuse a pinned reference set - another decision to make explicitly.

        The export step itself (1) is NOT a blocker and is well precedented in
        this repo: training-data/synthesize_piper_full.py already exports a
        raw training checkpoint from the tts-checkpoints Volume via
        `piper.train.export_onnx --checkpoint <ckpt> --output-file model.onnx`
        plus copying the training root's config.json to model.onnx.json, and
        Task 5's job writes both (config at
        /checkpoints/harness_tts_core_<label>/config.json, checkpoints under
        lightning_logs/version_*/checkpoints/). Implementing export alone
        without resolving 1-3 would produce a pipeline that cannot run, so it
        is not stubbed in half here."""
        raise NotImplementedError(
            "_run_eval_pipeline is unimplemented by design - see this method's docstring "
            "and .superpowers/sdd/2026-09-22-voice-research-harness/task-6-report.md for "
            "the three blockers (owner.json visibility, single-voice scoping of "
            "eval/synth.mjs, score.py's reference-based MOS)."
        )


from harness.recipes import register

register("tts-core", TTSCoreRecipe)
