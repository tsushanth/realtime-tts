import os
import tempfile

from harness.recipe import Candidate
from harness.report import CandidateResult, write_report


def test_write_report_includes_all_candidate_statuses():
    candidates_and_results = [
        CandidateResult(
            candidate=Candidate(id="small", description="small model", train_config={}),
            status="trained",
            cost_usd=2.50,
            metrics={"wer": 0.08, "latency_ms": 180.0},
            error=None,
        ),
        CandidateResult(
            candidate=Candidate(id="broken", description="a candidate that failed", train_config={}),
            status="failed",
            cost_usd=0.50,
            metrics=None,
            error="Modal job raised: out of memory",
        ),
        CandidateResult(
            candidate=Candidate(id="skipped", description="never attempted", train_config={}),
            status="skipped_over_budget",
            cost_usd=0.0,
            metrics=None,
            error=None,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = write_report(
            recipe_name="tts-core",
            budget_usd=30.0,
            total_spent_usd=3.00,
            results=candidates_and_results,
            out_dir=tmp,
        )
        assert os.path.exists(path)
        assert "tts-core" in os.path.basename(path)
        content = open(path).read()

    assert "small" in content
    assert "0.08" in content or "8.0" in content  # wer shows up somehow
    assert "broken" in content
    assert "out of memory" in content
    assert "skipped" in content
    assert "$30.00" in content or "30.00" in content
    assert "$3.00" in content or "3.00" in content


def test_write_report_filename_has_todays_date():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_report(recipe_name="tts-core", budget_usd=30.0, total_spent_usd=0.0, results=[], out_dir=tmp)
    import datetime
    today = datetime.date.today().isoformat()
    assert today in os.path.basename(path)


def test_reading_this_names_cheapest_and_metric_min_max():
    results = [
        CandidateResult(
            candidate=Candidate(id="cheapie", description="cheap one", train_config={}),
            status="trained", cost_usd=1.00,
            metrics={"wer": 0.20, "mos": 3.0}, error=None,
        ),
        CandidateResult(
            candidate=Candidate(id="pricey", description="expensive one", train_config={}),
            status="trained", cost_usd=9.00,
            metrics={"wer": 0.05, "mos": 4.5}, error=None,
        ),
        CandidateResult(
            candidate=Candidate(id="nope", description="failed one", train_config={}),
            status="failed", cost_usd=0.10, metrics=None, error="boom",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        content = open(write_report("tts-core", 30.0, 10.10, results, out_dir=tmp)).read()

    assert "Cheapest trained candidate: **cheapie** at $1.00." in content
    # min/max per metric, not a single "winner" - the core is domain-free and
    # cannot know that lower WER but higher MOS is better.
    assert "**wer**: min 0.05 (pricey), max 0.2 (cheapie)" in content
    assert "**mos**: min 3.0 (cheapie), max 4.5 (pricey)" in content
    assert "boom" in content


def test_reading_this_omits_cost_metric_summary_when_nothing_trained():
    results = [
        CandidateResult(
            candidate=Candidate(id="x", description="skipped", train_config={}),
            status="skipped_over_budget", cost_usd=0.0, metrics=None, error=None,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        content = open(write_report("tts-core", 30.0, 0.0, results, out_dir=tmp)).read()

    assert "Cheapest trained candidate" not in content
    assert "Metric range" not in content


def test_reading_this_reports_trained_but_eval_failed_candidates():
    results = [
        CandidateResult(
            candidate=Candidate(id="e1", description="trained, eval broke", train_config={}),
            status="trained", cost_usd=4.00, metrics=None,
            error="evaluation failed: not implemented",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        content = open(write_report("tts-core", 30.0, 4.0, results, out_dir=tmp)).read()

    assert "Trained but not evaluated" in content
    assert "evaluation failed: not implemented" in content
    # no metrics anywhere, so no metric-range section, but cost summary stands
    assert "Cheapest trained candidate: **e1** at $4.00." in content
    assert "Metric range" not in content
