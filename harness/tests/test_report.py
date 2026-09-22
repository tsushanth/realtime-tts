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
