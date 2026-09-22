"""The plugin interface every research vertical implements once. See
docs/superpowers/specs/2026-09-22-voice-research-harness-design.md for the
full design rationale - in short, the harness core (run_cycle.py,
report.py) knows only these types; it never knows what a "candidate"
means for a given vertical or how to evaluate one, since that's proven
to differ per vertical (WER works for TTS, was useless for audio
isolation earlier this project)."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Candidate:
    id: str
    description: str
    train_config: dict = field(default_factory=dict)


@dataclass
class TrainedModel:
    candidate: Candidate
    artifact_path: str
    actual_cost_usd: float


class Recipe(Protocol):
    name: str

    def discover_candidates(self) -> list[Candidate]:
        """Return candidates not yet tried, in priority order."""
        ...

    def estimate_cost_usd(self, candidate: Candidate) -> float:
        """A conservative cost estimate, checked against remaining budget
        before training starts."""
        ...

    def train(self, candidate: Candidate, workdir: str) -> TrainedModel:
        """Actually run training. Raises on failure."""
        ...

    def evaluate(self, model: TrainedModel) -> dict[str, float]:
        """Metric name -> value, entirely recipe-defined."""
        ...
