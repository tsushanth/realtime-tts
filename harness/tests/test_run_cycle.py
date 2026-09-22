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
    # c1 and c2 train ($20); c3's check is 20+10=30 > 25, which STOPS the
    # cycle (break, per spec) - so exactly one candidate is skipped here.
    assert content.count("skipped_over_budget") == 1
    assert "Spent: $20.00" in content


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


class EvalFailsRecipe:
    """train() succeeds (real money spent) but evaluate() raises - the cycle
    must still write the report. This is the exact shape of TTSCoreRecipe
    today, whose _run_eval_pipeline is NotImplementedError by design."""

    name = "eval-fails-test"

    def discover_candidates(self):
        return [Candidate(id="e1", description="evaluates badly", train_config={})]

    def estimate_cost_usd(self, candidate):
        return 5.0

    def train(self, candidate, workdir):
        return TrainedModel(candidate=candidate, artifact_path="/tmp/e1.onnx", actual_cost_usd=4.0)

    def evaluate(self, model):
        raise NotImplementedError("eval pipeline not wired up")


def test_evaluate_failure_still_writes_report_and_records_trained():
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(EvalFailsRecipe(), budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert "e1" in content
    assert "trained" in content  # training DID succeed
    assert "evaluation failed: eval pipeline not wired up" in content


class SingleExpensiveRecipe:
    name = "too-expensive-test"

    def __init__(self):
        self.trained = []

    def discover_candidates(self):
        return [Candidate(id="huge", description="blows the whole budget", train_config={})]

    def estimate_cost_usd(self, candidate):
        return 100.0

    def train(self, candidate, workdir):
        self.trained.append(candidate.id)
        raise AssertionError("train() must not be called for an over-budget candidate")

    def evaluate(self, model):
        raise AssertionError("unreachable")


def test_first_candidate_over_entire_budget_is_skipped_not_trained():
    recipe = SingleExpensiveRecipe()
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(recipe, budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert recipe.trained == []
    assert "skipped_over_budget" in content
    assert "Spent: $0.00" in content


class ExpensiveThenCheapRecipe:
    """Priority order matters: the expensive candidate comes FIRST. Once it is
    over budget the cycle must STOP (spec), so the cheap one that follows must
    NOT be trained even though it would have fit."""

    name = "break-not-continue-test"

    def __init__(self):
        self.trained = []

    def discover_candidates(self):
        return [
            Candidate(id="expensive", description="priority 1, too expensive", train_config={}),
            Candidate(id="cheap", description="priority 2, would have fit", train_config={}),
        ]

    def estimate_cost_usd(self, candidate):
        return 100.0 if candidate.id == "expensive" else 1.0

    def train(self, candidate, workdir):
        self.trained.append(candidate.id)
        return TrainedModel(candidate=candidate, artifact_path="/tmp/m.onnx", actual_cost_usd=1.0)

    def evaluate(self, model):
        return {"score": 1.0}


def test_over_budget_candidate_breaks_cycle_and_later_cheaper_one_is_also_skipped():
    recipe = ExpensiveThenCheapRecipe()
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(recipe, budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert recipe.trained == [], "the cheaper later candidate must not be trained"
    assert content.count("skipped_over_budget") == 2
    assert "expensive" in content and "cheap" in content


class CostDriftRecipe:
    """estimate says $5, train() actually costs $12. The running total must
    track the ACTUAL spend or the harness silently drifts from real money."""

    name = "cost-drift-test"

    def discover_candidates(self):
        return [Candidate(id="d1", description="costs more than estimated", train_config={})]

    def estimate_cost_usd(self, candidate):
        return 5.0

    def train(self, candidate, workdir):
        return TrainedModel(candidate=candidate, artifact_path="/tmp/d1.onnx", actual_cost_usd=12.0)

    def evaluate(self, model):
        return {"score": 1.0}


def test_running_total_uses_actual_cost_not_estimate():
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(CostDriftRecipe(), budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert "Spent: $12.00" in content
    assert "Spent: $5.00" not in content


def test_failed_training_charges_the_estimate_to_the_running_total():
    """Training didn't complete, so there's no actual cost to use - the
    conservative choice is to charge the estimate, not $0."""
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(FailingTrainRecipe(), budget_usd=30.0, workdir=workdir, out_dir=out_dir)
        content = open(report_path).read()

    assert "Spent: $1.00" in content  # FailingTrainRecipe's estimate is $1.00


class RecordingRecipe:
    """Exercises the optional duck-typed record_tried() write-back."""

    name = "record-tried-test"

    def __init__(self):
        self.recorded = []

    def discover_candidates(self):
        return [
            Candidate(id="ok", description="trains and evaluates", train_config={}),
            Candidate(id="evalbad", description="trains, eval raises", train_config={}),
            Candidate(id="bad", description="train raises", train_config={}),
        ]

    def estimate_cost_usd(self, candidate):
        return 1.0

    def train(self, candidate, workdir):
        if candidate.id == "bad":
            raise RuntimeError("nope")
        return TrainedModel(candidate=candidate, artifact_path="/tmp/m.onnx", actual_cost_usd=1.0)

    def evaluate(self, model):
        if model.candidate.id == "evalbad":
            raise RuntimeError("eval blew up")
        return {"score": 1.0}

    def record_tried(self, candidate_id, report_path):
        self.recorded.append((candidate_id, report_path))


def test_record_tried_is_called_for_trained_candidates_including_eval_failures():
    recipe = RecordingRecipe()
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        report_path = run_cycle(recipe, budget_usd=30.0, workdir=workdir, out_dir=out_dir)

    assert [cid for cid, _ in recipe.recorded] == ["ok", "evalbad"]
    assert all(p == report_path for _, p in recipe.recorded)


def test_recipe_without_record_tried_still_works():
    """The core must not require every recipe to implement the write-back."""
    with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as out_dir:
        recipe = BudgetTestRecipe()
        assert not hasattr(recipe, "record_tried")
        run_cycle(recipe, budget_usd=30.0, workdir=workdir, out_dir=out_dir)
