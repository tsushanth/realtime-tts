import tempfile

from harness.recipe import Candidate, TrainedModel
from harness.run_cycle import run_cycle


class BudgetTestRecipe:
    """Three candidates costing $10 each; with a $25 budget the third
    must be skipped (10+10=20 spent, 10+20=30 > 25 remaining check on
    the third)."""

    name = "budget-test"

    def discover_candidates(self):
        return [
            Candidate(id="c1", description="first", train_config={}),
            Candidate(id="c2", description="second", train_config={}),
            Candidate(id="c3", description="third", train_config={}),
        ]

    def estimate_cost_usd(self, candidate):
        return 10.0

    def train(self, candidate, workdir):
        return TrainedModel(candidate=candidate, artifact_path=f"{workdir}/{candidate.id}.onnx", actual_cost_usd=10.0)

    def evaluate(self, model):
        return {"score": 1.0}


def test_budget_enforcement_stops_cycle_before_overspending():
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        recipe = BudgetTestRecipe()
        report_path = run_cycle(recipe, budget_usd=25.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert "c1" in content and "trained" in content
    assert "c2" in content
    assert "c3" in content and "skipped_over_budget" in content


class FailingTrainRecipe:
    """One candidate whose train() raises - the cycle must record it as
    failed, not crash."""

    name = "failing-test"

    def discover_candidates(self):
        return [Candidate(id="broken", description="always fails", train_config={})]

    def estimate_cost_usd(self, candidate):
        return 1.0

    def train(self, candidate, workdir):
        raise RuntimeError("simulated training failure")

    def evaluate(self, model):
        raise AssertionError("evaluate() should never be called for a failed train()")


def test_failed_training_is_recorded_not_raised():
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        recipe = FailingTrainRecipe()
        report_path = run_cycle(recipe, budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert "broken" in content
    assert "failed" in content
    assert "simulated training failure" in content
