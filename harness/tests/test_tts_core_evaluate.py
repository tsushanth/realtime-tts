"""Tests for TTSCoreRecipe.evaluate() - the metric-aggregation half of Task 6.

The fixtures here deliberately mirror the SHAPE of this repo's REAL
eval/results/{latency,quality}.json (checked against the committed files,
not just the plan's illustrative example): a flat list of rows, one per
(engine, lang, id) job, with ALL engines' rows in the same file, `wer` and
`naturalness_mos` nullable, and failed-synthesis rows carrying an `error`
key and no `wav`/`first_byte_ms`.
"""
import json

from unittest.mock import patch

from harness.recipe import Candidate, TrainedModel
from harness.recipes.tts_core import CANDIDATE_ENGINE_NAME, TTSCoreRecipe


def _model():
    candidate = Candidate(
        id="low",
        description="test",
        train_config={"warmstart_url": "x", "max_steps": 100, "size_label": "low"},
    )
    return TrainedModel(candidate=candidate, artifact_path="/checkpoints/x", actual_cost_usd=1.0)


def _evaluate_over(tmp_path, quality_rows, latency_rows):
    quality_json = tmp_path / "quality.json"
    quality_json.write_text(json.dumps(quality_rows))
    latency_json = tmp_path / "latency.json"
    latency_json.write_text(json.dumps(latency_rows))
    recipe = TTSCoreRecipe()
    with patch.object(recipe, "_run_eval_pipeline", return_value=(str(latency_json), str(quality_json))):
        return recipe.evaluate(_model())


def test_evaluate_parses_quality_json_into_summary_metrics(tmp_path):
    metrics = _evaluate_over(
        tmp_path,
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "wer": 0.05, "naturalness_mos": 4.2},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s2", "wer": 0.09, "naturalness_mos": 4.4},
        ],
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "first_byte_ms": 200},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s2", "first_byte_ms": 220},
        ],
    )
    assert abs(metrics["wer_mean"] - 0.07) < 0.001  # mean of 0.05, 0.09
    assert abs(metrics["naturalness_mos_median"] - 4.3) < 0.001  # median of 4.2, 4.4
    assert abs(metrics["latency_ms_median"] - 210) < 0.001  # median of 200, 220


def test_evaluate_ignores_rows_from_other_engines(tmp_path):
    """The real results files hold EVERY engine's rows in one list (74 rows
    across piper/kokoro/elevenlabs-* in the committed run). Aggregating them
    all would silently report ElevenLabs' numbers as the candidate's."""
    metrics = _evaluate_over(
        tmp_path,
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "wer": 0.10, "naturalness_mos": 4.0},
            {"engine": "elevenlabs-flash", "lang": "en", "id": "s1", "wer": 0.00, "naturalness_mos": 4.9},
            {"engine": "piper", "lang": "en", "id": "s1", "wer": 0.50, "naturalness_mos": 3.0},
        ],
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "first_byte_ms": 300},
            {"engine": "elevenlabs-flash", "lang": "en", "id": "s1", "first_byte_ms": 1400},
        ],
    )
    assert abs(metrics["wer_mean"] - 0.10) < 0.001
    assert abs(metrics["naturalness_mos_median"] - 4.0) < 0.001
    assert abs(metrics["latency_ms_median"] - 300) < 0.001


def test_evaluate_skips_null_and_failed_rows(tmp_path):
    """score.py emits naturalness_mos=None for the reference clip itself and on
    MOS failure, and wer=None on transcription failure; synth.mjs emits rows
    with `error` and no first_byte_ms when synthesis failed."""
    metrics = _evaluate_over(
        tmp_path,
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "wer": 0.2, "naturalness_mos": None},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s2", "wer": 0.4, "naturalness_mos": 3.5},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s3", "wer": None,
             "naturalness_mos": None, "note": "no audio (synthesis failed)"},
        ],
        [
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s1", "first_byte_ms": 100},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s2", "first_byte_ms": 300},
            {"engine": CANDIDATE_ENGINE_NAME, "lang": "en", "id": "s3", "error": "timeout"},
        ],
    )
    assert abs(metrics["wer_mean"] - 0.3) < 0.001
    assert abs(metrics["naturalness_mos_median"] - 3.5) < 0.001
    assert abs(metrics["latency_ms_median"] - 200) < 0.001
    assert metrics["n_scored"] == 2  # s1, s2 - the failed row is not counted
    assert metrics["n_failed"] == 1


def test_evaluate_returns_nan_when_no_candidate_rows(tmp_path):
    """A run where every synthesis failed must not crash the harness; NaN
    reads as 'no measurement', which report.py/run_cycle.py can surface as
    such rather than a falsely good 0.0 WER."""
    import math

    metrics = _evaluate_over(
        tmp_path,
        [{"engine": "piper", "lang": "en", "id": "s1", "wer": 0.0, "naturalness_mos": 4.0}],
        [{"engine": "piper", "lang": "en", "id": "s1", "first_byte_ms": 100}],
    )
    assert math.isnan(metrics["wer_mean"])
    assert math.isnan(metrics["naturalness_mos_median"])
    assert math.isnan(metrics["latency_ms_median"])
    assert metrics["n_scored"] == 0


def test_evaluate_matches_the_real_committed_results_field_names():
    """Guards against the aggregation drifting away from what score.py and
    synth.mjs actually write. Reads this repo's real committed eval output."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    quality = json.load(open(os.path.join(root, "eval", "results", "quality.json")))
    latency = json.load(open(os.path.join(root, "eval", "results", "latency.json")))
    assert isinstance(quality, list) and isinstance(latency, list)
    assert {"engine", "lang", "id", "wer", "naturalness_mos"} <= set(quality[0])
    assert {"engine", "lang", "id", "first_byte_ms"} <= set(latency[0])
