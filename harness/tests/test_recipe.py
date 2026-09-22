from dataclasses import dataclass

from harness.recipe import Candidate, Recipe, TrainedModel


class FakeRecipe:
    name = "fake"

    def discover_candidates(self) -> list[Candidate]:
        return [Candidate(id="c1", description="test candidate", train_config={})]

    def estimate_cost_usd(self, candidate: Candidate) -> float:
        return 1.0

    def train(self, candidate: Candidate, workdir: str) -> TrainedModel:
        return TrainedModel(candidate=candidate, artifact_path=f"{workdir}/model.onnx", actual_cost_usd=1.0)

    def evaluate(self, model: TrainedModel) -> dict[str, float]:
        return {"wer": 0.05}


def test_fake_recipe_satisfies_protocol():
    recipe: Recipe = FakeRecipe()
    candidates = recipe.discover_candidates()
    assert len(candidates) == 1
    assert candidates[0].id == "c1"
    model = recipe.train(candidates[0], "/tmp/workdir")
    assert model.candidate is candidates[0]
    metrics = recipe.evaluate(model)
    assert metrics["wer"] == 0.05


def test_candidate_is_a_plain_dataclass():
    c = Candidate(id="x", description="desc", train_config={"size": "small"})
    assert c.id == "x"
    assert c.train_config == {"size": "small"}


def test_trained_model_carries_real_cost():
    c = Candidate(id="x", description="desc", train_config={})
    m = TrainedModel(candidate=c, artifact_path="/tmp/m.onnx", actual_cost_usd=2.5)
    assert m.actual_cost_usd == 2.5
