# Voice Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an on-demand CLI research harness (`harness/run_cycle.py --recipe tts-core --budget 30`) that discovers untried TTS model-size candidates, trains each on Modal within a hard budget cap, evaluates each with this project's existing WER/MOS/latency framework, and writes a markdown report — plus the `Recipe` plugin interface that lets future verticals (music, dubbing, etc.) reuse the same core.

**Architecture:** A generic orchestration core (`harness/run_cycle.py`, `harness/report.py`) that knows nothing about TTS or training — only how to sequence candidates against a budget and write a report — plus one concrete `Recipe` implementation (`harness/recipes/tts_core.py` + `harness/recipes/tts_core_train_job.py`) that does know about Piper training, modeled directly on this repo's existing `training-data/piper_full_finetune.py` (same Modal image/volume/CLI-invocation pattern, parameterized instead of hardcoded) and reusing `eval/`'s existing synth+score scripts wholesale for evaluation.

**Tech Stack:** Python (harness core + recipe), Modal (GPU training dispatch, mirroring `training-data/piper_full_finetune.py`'s image/volume setup), the existing `eval/synth.mjs` (Node) + `eval/score.py` (Python) for evaluation, pytest for the harness's own tests.

**Spec:** `docs/superpowers/specs/2026-09-22-voice-research-harness-design.md`

## Global Constraints

- On-demand only: no scheduler, no cron, no autonomous trigger — the harness only runs when invoked via CLI.
- Modal only for v1 (no RunPod/Vast) — per spec's scope decision.
- Budget is a hard cap enforced by the harness core before each training dispatch, not a post-hoc report. Default $30/cycle, overridable via `--budget`.
- Reports are markdown files committed to the repo at `harness/reports/YYYY-MM-DD-<recipe-name>.md`, matching the style of `eval/results/REPORT_DRAFT.md`.
- The harness core (`run_cycle.py`, `report.py`) must not import anything TTS-specific — it only knows the `Recipe` protocol's types (`Candidate`, `TrainedModel`, and `dict[str, float]` metrics).
- "Already tried" candidates for `tts-core` are tracked in `harness/recipes/tts_core_tried.json`, committed to the repo.
- No new database, no new always-on service — matches this project's existing script-based tooling (`eval/`, `voice-pipeline/`).

---

## File Structure

- `harness/recipe.py` — the `Recipe` protocol, `Candidate`/`TrainedModel` dataclasses. No dependencies beyond stdlib.
- `harness/run_cycle.py` — CLI entrypoint + harness core orchestration (discovery, budget enforcement, error handling, calling into `report.py`).
- `harness/report.py` — markdown report writer, takes the cycle's results and writes the file. No dependency on `Recipe` beyond its result types.
- `harness/recipes/__init__.py` — recipe name → class registry.
- `harness/recipes/tts_core.py` — `TTSCoreRecipe`: `discover_candidates()`, `estimate_cost_usd()` (pure, no Modal), `evaluate()` (shells out to `eval/`). `train()` delegates to `tts_core_train_job.py`.
- `harness/recipes/tts_core_train_job.py` — the actual Modal app/function for training a candidate, parameterized version of `training-data/piper_full_finetune.py`.
- `harness/recipes/tts_core_tried.json` — starts as `{}`, updated by `discover_candidates()`'s caller after each cycle (see Task 4).
- `harness/tests/test_run_cycle.py` — core orchestration tests using a fake in-test `Recipe`.
- `harness/tests/test_report.py` — report-writer tests.
- `harness/tests/test_tts_core.py` — `discover_candidates()`/`estimate_cost_usd()` pure-function tests (no Modal, no network).

---

## Task 1: `Recipe` protocol and shared types

**Files:**
- Create: `harness/recipe.py`
- Test: `harness/tests/test_recipe.py`

**Interfaces:**
- Consumes: nothing (foundation module)
- Produces: `Candidate(id: str, description: str, train_config: dict)`, `TrainedModel(candidate: Candidate, artifact_path: str, actual_cost_usd: float)`, `Recipe` Protocol with `name: str`, `discover_candidates(self) -> list[Candidate]`, `estimate_cost_usd(self, candidate: Candidate) -> float`, `train(self, candidate: Candidate, workdir: str) -> TrainedModel`, `evaluate(self, model: TrainedModel) -> dict[str, float]`

- [ ] **Step 1: Write the failing test — a minimal fake Recipe satisfies the Protocol**

```python
# harness/tests/test_recipe.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_recipe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness'` or `'harness.recipe'`

- [ ] **Step 3: Write minimal implementation**

```python
# harness/recipe.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_recipe.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Create the empty package files needed for imports to work**

```bash
mkdir -p harness/tests harness/recipes
touch harness/__init__.py harness/tests/__init__.py harness/recipes/__init__.py
```

- [ ] **Step 6: Re-run to confirm package structure resolves imports cleanly**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_recipe.py -v`
Expected: PASS (3 tests), no import warnings

- [ ] **Step 7: Commit**

```bash
git add harness/__init__.py harness/recipe.py harness/tests/__init__.py harness/tests/test_recipe.py harness/recipes/__init__.py
git commit -m "harness: add Recipe protocol and shared Candidate/TrainedModel types"
```

---

## Task 2: Report writer

**Files:**
- Create: `harness/report.py`
- Test: `harness/tests/test_report.py`

**Interfaces:**
- Consumes: `Candidate`, `TrainedModel` from Task 1 (`harness/recipe.py`)
- Produces: `CandidateResult` dataclass (`candidate: Candidate`, `status: str` one of `"trained"`/`"failed"`/`"skipped_over_budget"`, `cost_usd: float`, `metrics: dict[str, float] | None`, `error: str | None`), `write_report(recipe_name: str, budget_usd: float, total_spent_usd: float, results: list[CandidateResult], out_dir: str = "harness/reports") -> str` (returns the path it wrote)

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_report.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# harness/report.py
"""Writes one markdown report per research cycle, matching the style of
eval/results/REPORT_DRAFT.md. Knows nothing about what a metric means -
it prints whatever dict[str, float] a recipe's evaluate() returned as
table columns, unioned across all trained candidates in the cycle."""
import datetime
import os
from dataclasses import dataclass

from harness.recipe import Candidate


@dataclass
class CandidateResult:
    candidate: Candidate
    status: str  # "trained" | "failed" | "skipped_over_budget"
    cost_usd: float
    metrics: dict[str, float] | None
    error: str | None


def write_report(
    recipe_name: str,
    budget_usd: float,
    total_spent_usd: float,
    results: list[CandidateResult],
    out_dir: str = "harness/reports",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = os.path.join(out_dir, f"{today}-{recipe_name}.md")

    metric_names: list[str] = []
    for r in results:
        if r.metrics:
            for k in r.metrics:
                if k not in metric_names:
                    metric_names.append(k)

    lines = [
        f"# Research cycle: {recipe_name} ({today})",
        "",
        f"Budget: ${budget_usd:.2f} | Spent: ${total_spent_usd:.2f}",
        "",
        "## Results",
        "",
        "| Candidate | Description | Status | Cost | " + " | ".join(metric_names) + " |",
        "|---|---|---|---|" + "---|" * len(metric_names),
    ]
    for r in results:
        metric_cells = ""
        if metric_names:
            values = [str(r.metrics.get(m, "")) if r.metrics else "" for m in metric_names]
            metric_cells = " | ".join(values) + " |"
        lines.append(
            f"| {r.candidate.id} | {r.candidate.description} | {r.status} | ${r.cost_usd:.2f} | {metric_cells}"
        )

    lines += ["", "## Reading this", ""]
    trained = [r for r in results if r.status == "trained"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped_over_budget"]
    lines.append(f"{len(trained)} trained, {len(failed)} failed, {len(skipped)} skipped (over budget).")
    if failed:
        lines.append("")
        lines.append("Failures:")
        for r in failed:
            lines.append(f"- **{r.candidate.id}**: {r.error}")
    if skipped:
        lines.append("")
        lines.append("Not attempted this cycle (budget exhausted before reaching these):")
        for r in skipped:
            lines.append(f"- {r.candidate.id}: {r.candidate.description}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add harness/report.py harness/tests/test_report.py
git commit -m "harness: add markdown report writer"
```

---

## Task 3: Harness core orchestration

**Files:**
- Create: `harness/run_cycle.py`
- Test: `harness/tests/test_run_cycle.py`

**Interfaces:**
- Consumes: `Recipe`, `Candidate`, `TrainedModel` (Task 1); `CandidateResult`, `write_report` (Task 2)
- Produces: `run_cycle(recipe: Recipe, budget_usd: float, workdir: str) -> str` (returns the report path), `main()` CLI entrypoint parsing `--recipe`/`--budget`

- [ ] **Step 1: Write the failing test — budget enforcement stops the cycle before overspending**

```python
# harness/tests/test_run_cycle.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_run_cycle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.run_cycle'`

- [ ] **Step 3: Write minimal implementation**

```python
# harness/run_cycle.py
"""The harness core: sequences a Recipe's candidates against a hard
budget cap, training and evaluating each in turn, and writes one report
per cycle. Contains zero domain logic - see harness/recipe.py's Recipe
protocol for the boundary this file never crosses."""
import argparse
import os

from harness.recipe import Recipe
from harness.report import CandidateResult, write_report
from harness.recipes import get_recipe


def run_cycle(recipe: Recipe, budget_usd: float, workdir: str, out_dir: str = "harness/reports") -> str:
    candidates = recipe.discover_candidates()
    results: list[CandidateResult] = []
    spent = 0.0

    for candidate in candidates:
        estimate = recipe.estimate_cost_usd(candidate)
        if spent + estimate > budget_usd:
            results.append(CandidateResult(candidate=candidate, status="skipped_over_budget", cost_usd=0.0, metrics=None, error=None))
            continue

        try:
            model = recipe.train(candidate, workdir)
        except Exception as e:  # noqa: BLE001 - one bad candidate must not kill the cycle
            results.append(CandidateResult(candidate=candidate, status="failed", cost_usd=estimate, metrics=None, error=str(e)))
            spent += estimate  # attempted spend, even on failure - conservative accounting
            continue

        spent += model.actual_cost_usd
        metrics = recipe.evaluate(model)
        results.append(CandidateResult(candidate=candidate, status="trained", cost_usd=model.actual_cost_usd, metrics=metrics, error=None))

    return write_report(recipe_name=recipe.name, budget_usd=budget_usd, total_spent_usd=spent, results=results, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(description="Run one voice-model research cycle.")
    parser.add_argument("--recipe", required=True, help="Recipe name, e.g. tts-core")
    parser.add_argument("--budget", type=float, default=30.0, help="Hard USD cap for this cycle (default: $30)")
    parser.add_argument("--workdir", default="/tmp/harness-workdir", help="Scratch directory for training artifacts")
    args = parser.parse_args()

    recipe = get_recipe(args.recipe)
    os.makedirs(args.workdir, exist_ok=True)
    report_path = run_cycle(recipe, budget_usd=args.budget, workdir=args.workdir)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the recipe registry stub so imports resolve (real recipe comes in Task 4)**

```python
# harness/recipes/__init__.py
"""Recipe name -> class registry. Adding a new vertical means adding one
entry here, never touching harness/run_cycle.py."""
from harness.recipe import Recipe

_REGISTRY: dict[str, type] = {}


def register(name: str, recipe_class: type):
    _REGISTRY[name] = recipe_class


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown recipe {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_run_cycle.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full harness test suite so far**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/ -v`
Expected: PASS (7 tests: 3 from test_recipe.py, 2 from test_report.py, 2 from test_run_cycle.py)

- [ ] **Step 7: Commit**

```bash
git add harness/run_cycle.py harness/recipes/__init__.py
git commit -m "harness: add budget-enforced core orchestration and recipe registry"
```

---

## Task 4: `TTSCoreRecipe` — candidate discovery and cost estimation

**Files:**
- Create: `harness/recipes/tts_core.py` (discovery/estimate parts only this task; `train`/`evaluate` come in Tasks 5-6)
- Create: `harness/recipes/tts_core_tried.json` (starts as `{}`)
- Test: `harness/tests/test_tts_core.py`

**Interfaces:**
- Consumes: `Candidate` (Task 1)
- Produces: `TTSCoreRecipe.discover_candidates(self) -> list[Candidate]`, `TTSCoreRecipe.estimate_cost_usd(self, candidate: Candidate) -> float`. `train_config` shape for this recipe's candidates: `{"warmstart_url": str, "max_steps": int, "size_label": str}` — Task 5 consumes exactly these three keys.

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_tts_core.py
import json
import os
import tempfile

from harness.recipe import Candidate
from harness.recipes.tts_core import KNOWN_SIZE_VARIANTS, TTSCoreRecipe


def test_discover_candidates_returns_known_size_variants_not_yet_tried(tmp_path, monkeypatch):
    tried_path = tmp_path / "tried.json"
    tried_path.write_text(json.dumps({}))
    monkeypatch.setattr("harness.recipes.tts_core.TRIED_PATH", str(tried_path))

    recipe = TTSCoreRecipe()
    candidates = recipe.discover_candidates()

    assert len(candidates) == len(KNOWN_SIZE_VARIANTS)
    ids = {c.id for c in candidates}
    assert ids == {v["size_label"] for v in KNOWN_SIZE_VARIANTS}
    for c in candidates:
        assert set(c.train_config.keys()) == {"warmstart_url", "max_steps", "size_label"}


def test_discover_candidates_excludes_already_tried(tmp_path, monkeypatch):
    already_tried_id = KNOWN_SIZE_VARIANTS[0]["size_label"]
    tried_path = tmp_path / "tried.json"
    tried_path.write_text(json.dumps({already_tried_id: {"date": "2026-01-01", "report_path": "x.md"}}))
    monkeypatch.setattr("harness.recipes.tts_core.TRIED_PATH", str(tried_path))

    recipe = TTSCoreRecipe()
    candidates = recipe.discover_candidates()

    ids = {c.id for c in candidates}
    assert already_tried_id not in ids
    assert len(candidates) == len(KNOWN_SIZE_VARIANTS) - 1


def test_estimate_cost_scales_with_max_steps():
    recipe = TTSCoreRecipe()
    small = Candidate(id="small", description="", train_config={"warmstart_url": "x", "max_steps": 5000, "size_label": "small"})
    large = Candidate(id="large", description="", train_config={"warmstart_url": "x", "max_steps": 20000, "size_label": "large"})

    small_cost = recipe.estimate_cost_usd(small)
    large_cost = recipe.estimate_cost_usd(large)

    assert large_cost > small_cost
    # Basis: piper_full_finetune.py's own measured rate, ~1.2 it/s at batch_size=8
    # on a T4 ($0.59/hr per Modal's published T4 rate at time of writing -
    # verify against `modal pricing` or the Modal dashboard before trusting this
    # exact number for a real budget decision), giving 20000 steps ~= 4.6h ~= $2.50-3,
    # matching the docstring in training-data/piper_full_finetune.py.
    assert 2.0 <= large_cost <= 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.recipes.tts_core'`

- [ ] **Step 3: Create the "already tried" tracking file**

```bash
echo '{}' > harness/recipes/tts_core_tried.json
```

- [ ] **Step 4: Write minimal implementation**

```python
# harness/recipes/tts_core.py
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
        with open(TRIED_PATH) as f:
            return json.load(f)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add harness/recipes/tts_core.py harness/recipes/tts_core_tried.json harness/tests/test_tts_core.py
git commit -m "harness: add TTSCoreRecipe candidate discovery and cost estimation"
```

---

## Task 5: `TTSCoreRecipe.train()` — parameterized Modal training job

**Files:**
- Create: `harness/recipes/tts_core_train_job.py`
- Modify: `harness/recipes/tts_core.py` (add `train()` method)
- Test: `harness/tests/test_tts_core_train_job.py`

**Interfaces:**
- Consumes: `Candidate.train_config` keys `warmstart_url`, `max_steps`, `size_label` (Task 4)
- Produces: `run_piper_finetune(warmstart_url: str, max_steps: int, size_label: str) -> str` (a Modal function, returns the checkpoint volume path where the trained model landed), `TTSCoreRecipe.train(self, candidate: Candidate, workdir: str) -> TrainedModel`

- [ ] **Step 1: Write the failing test — the training dispatch function is parameterized correctly (unit test on argument construction, not a real Modal run)**

```python
# harness/tests/test_tts_core_train_job.py
from harness.recipes.tts_core_train_job import build_piper_train_args


def test_build_piper_train_args_uses_candidate_config():
    args = build_piper_train_args(
        warmstart_url="https://example.com/checkpoint.ckpt",
        max_steps=8000,
        size_label="low",
        warmstart_local_path="/ckpt/warmstart.ckpt",
    )

    assert "--model.warmstart_ckpt" in args
    idx = args.index("--model.warmstart_ckpt")
    assert args[idx + 1] == "/ckpt/warmstart.ckpt"

    assert "--trainer.max_steps" in args
    idx = args.index("--trainer.max_steps")
    assert args[idx + 1] == "8000"

    assert "--trainer.default_root_dir" in args
    idx = args.index("--trainer.default_root_dir")
    assert "low" in args[idx + 1]  # size_label keeps different candidates' checkpoints from colliding


def test_build_piper_train_args_different_sizes_get_different_output_dirs():
    args_low = build_piper_train_args(warmstart_url="x", max_steps=8000, size_label="low", warmstart_local_path="/ckpt/x.ckpt")
    args_high = build_piper_train_args(warmstart_url="x", max_steps=20000, size_label="high", warmstart_local_path="/ckpt/x.ckpt")

    idx_low = args_low.index("--trainer.default_root_dir")
    idx_high = args_high.index("--trainer.default_root_dir")
    assert args_low[idx_low + 1] != args_high[idx_high + 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core_train_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.recipes.tts_core_train_job'`

- [ ] **Step 3: Write minimal implementation**

```python
# harness/recipes/tts_core_train_job.py
"""Modal training dispatch for TTSCoreRecipe. This is
training-data/piper_full_finetune.py's exact image/volume/CLI-invocation
approach (see that file's docstring for the hard-won gotchas already
solved there: WAV-copy-before-training, progress-patching,
.spawn()-not-.remote() for long runs), parameterized by candidate config
instead of hardcoded to one checkpoint/step-count. Read that file in full
before changing this one - the gotchas documented there apply here
unchanged."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
    .pip_install("torch==2.1.2", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2")
    .run_commands("git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src")
    .apt_install("python3-dev")
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    .pip_install("numpy<2")
    .add_local_file("../../training-data/patch_piper_progress.py", remote_path="/tmp/patch_piper_progress.py", copy=True)
    .run_commands("python3 /tmp/patch_piper_progress.py")
    .add_local_dir("../../training-data/full", remote_path="/filelists_src")
)

app = modal.App("harness-tts-core-train", image=image)

corpus_volume = modal.Volume.from_name("tts-corpus-full")
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")


def build_piper_train_args(warmstart_url: str, max_steps: int, size_label: str, warmstart_local_path: str) -> list[str]:
    """Pure function so the CLI-argument construction is unit-testable without
    a real Modal container. warmstart_url is accepted (and unused here) so the
    signature mirrors what the caller has available - it's downloaded to
    warmstart_local_path by the Modal function itself, not by this function."""
    return [
        "piper.train", "fit",
        "--data.voice_name", f"harness_tts_core_{size_label}",
        "--data.csv_path", "/tmp/train.csv",
        "--data.audio_dir", "/tmp/wavs_local",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/tmp/piper_cache",
        "--data.config_path", f"/checkpoints/harness_tts_core_{size_label}/config.json",
        "--data.batch_size", "8",
        "--model.sample_rate", "22050",
        "--model.warmstart_ckpt", warmstart_local_path,
        "--trainer.max_steps", str(max_steps),
        "--trainer.default_root_dir", f"/checkpoints/harness_tts_core_{size_label}",
    ]


@app.function(gpu="T4", timeout=8 * 3600, volumes={"/data": corpus_volume, "/checkpoints": checkpoint_volume})
def run_piper_finetune(warmstart_url: str, max_steps: int, size_label: str) -> str:
    import os
    import subprocess
    import sys

    import piper.espeakbridge  # noqa: F401  sanity check, see piper_pilot_finetune.py

    subprocess.run(["wget", "-q", "-O", "/tmp/warmstart.ckpt", warmstart_url], check=True)

    def convert(src_filelist, dst_csv):
        import csv
        rows = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)
                rows.append((fname, text))
        with open(dst_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return len(rows)

    n_train = convert("/filelists_src/train.txt", "/tmp/train.csv")
    n_val = convert("/filelists_src/val.txt", "/tmp/val.csv")
    print(f"{n_train} train / {n_val} val rows")

    import shutil
    os.makedirs("/tmp/wavs_local", exist_ok=True)
    needed = set()
    for csv_path in ("/tmp/train.csv", "/tmp/val.csv"):
        with open(csv_path) as f:
            for line in f:
                fname = line.split("|", 1)[0]
                needed.add(fname)
    for fname in sorted(needed):
        shutil.copy(f"/data/wavs_full/{fname}", f"/tmp/wavs_local/{fname}")

    import torch
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.chdir("/opt/piper-src")
    sys.argv = build_piper_train_args(warmstart_url, max_steps, size_label, "/tmp/warmstart.ckpt")
    print("Running:", " ".join(sys.argv))

    import piper.train.__main__ as piper_main
    piper_main._DEFAULT_CALLBACKS = [piper_main._DEFAULT_CALLBACKS[0]]
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exited with code {e.code}")

    checkpoint_volume.commit()
    out_dir = f"/checkpoints/harness_tts_core_{size_label}"
    print(f"=== Contents of {out_dir} ===")
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            print(os.path.join(root, f))
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core_train_job.py -v`
Expected: PASS (2 tests) — this test only exercises `build_piper_train_args`, a pure function; it does not invoke Modal or GPU

- [ ] **Step 5: Add `TTSCoreRecipe.train()`, dispatching the real Modal job**

Add to `harness/recipes/tts_core.py` (after the class's existing `_load_tried` method):

```python
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
```

- [ ] **Step 6: Write a test confirming `train()` calls Modal's spawn/get pattern correctly, without a real GPU run**

```python
# append to harness/tests/test_tts_core_train_job.py
from unittest.mock import MagicMock, patch

from harness.recipe import Candidate
from harness.recipes.tts_core import TTSCoreRecipe


def test_train_spawns_and_waits_on_modal_call():
    candidate = Candidate(
        id="low",
        description="test",
        train_config={"warmstart_url": "https://example.com/x.ckpt", "max_steps": 100, "size_label": "low"},
    )

    fake_call = MagicMock()
    fake_call.get.return_value = "/checkpoints/harness_tts_core_low"

    with patch("harness.recipes.tts_core_train_job.run_piper_finetune") as fake_fn:
        fake_fn.spawn.return_value = fake_call
        recipe = TTSCoreRecipe()
        result = recipe.train(candidate, workdir="/tmp/workdir")

    fake_fn.spawn.assert_called_once_with(warmstart_url="https://example.com/x.ckpt", max_steps=100, size_label="low")
    fake_call.get.assert_called_once()
    assert result.artifact_path == "/checkpoints/harness_tts_core_low"
    assert result.candidate is candidate
    assert result.actual_cost_usd >= 0.0
```

- [ ] **Step 7: Run to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core_train_job.py -v`
Expected: PASS (3 tests), no real Modal call made

- [ ] **Step 8: Commit**

```bash
git add harness/recipes/tts_core_train_job.py harness/recipes/tts_core.py harness/tests/test_tts_core_train_job.py
git commit -m "harness: add parameterized Modal training dispatch for TTSCoreRecipe"
```

---

## Task 6: `TTSCoreRecipe.evaluate()` — wire into the existing `eval/` framework

**Files:**
- Modify: `harness/recipes/tts_core.py` (add `evaluate()` method)
- Test: `harness/tests/test_tts_core_evaluate.py`

**Interfaces:**
- Consumes: `TrainedModel` (Task 1), `eval/synth.mjs` and `eval/score.py` (existing, unmodified)
- Produces: `TTSCoreRecipe.evaluate(self, model: TrainedModel) -> dict[str, float]` returning at minimum `{"wer_mean": float, "naturalness_mos_median": float, "latency_ms_median": float}`

- [ ] **Step 1: Read the existing eval framework's real entrypoints before wiring against them**

Run: `cat ~/Documents/GitHub/realtime-tts/eval/README.md` and skim `eval/synth.mjs`'s and `eval/score.py`'s top-of-file usage comments. Confirm the exact env vars they need (`TTS_GATEWAY_API_KEY`, `ELEVENLABS_API_KEY` for `synth.mjs`) and their exact output file paths (`eval/results/latency.json`, `eval/results/quality.json`) before writing the wiring code below — this step has no code output, it's a required read.

- [ ] **Step 2: Write the failing test — evaluate() parses score.py's real output shape**

```python
# harness/tests/test_tts_core_evaluate.py
import json
import tempfile
from unittest.mock import patch

from harness.recipe import Candidate, TrainedModel
from harness.recipes.tts_core import TTSCoreRecipe


def test_evaluate_parses_quality_json_into_summary_metrics(tmp_path):
    quality_json = tmp_path / "quality.json"
    quality_json.write_text(json.dumps([
        {"engine": "harness-candidate", "lang": "en", "id": "s1", "wer": 0.05, "naturalness_mos": 4.2},
        {"engine": "harness-candidate", "lang": "en", "id": "s2", "wer": 0.09, "naturalness_mos": 4.4},
    ]))
    latency_json = tmp_path / "latency.json"
    latency_json.write_text(json.dumps([
        {"engine": "harness-candidate", "lang": "en", "id": "s1", "first_byte_ms": 200},
        {"engine": "harness-candidate", "lang": "en", "id": "s2", "first_byte_ms": 220},
    ]))

    candidate = Candidate(id="low", description="test", train_config={"warmstart_url": "x", "max_steps": 100, "size_label": "low"})
    model = TrainedModel(candidate=candidate, artifact_path="/checkpoints/x", actual_cost_usd=1.0)

    recipe = TTSCoreRecipe()
    with patch.object(recipe, "_run_eval_pipeline", return_value=(str(latency_json), str(quality_json))):
        metrics = recipe.evaluate(model)

    assert abs(metrics["wer_mean"] - 0.07) < 0.001  # mean of 0.05, 0.09
    assert abs(metrics["naturalness_mos_median"] - 4.3) < 0.001  # median of 4.2, 4.4
    assert abs(metrics["latency_ms_median"] - 210) < 0.001  # median of 200, 220
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core_evaluate.py -v`
Expected: FAIL with `AttributeError: 'TTSCoreRecipe' object has no attribute 'evaluate'` (or `_run_eval_pipeline`)

- [ ] **Step 4: Write minimal implementation**

Add to `harness/recipes/tts_core.py`:

```python
    def evaluate(self, model: TrainedModel) -> dict[str, float]:
        import json
        import statistics

        latency_path, quality_path = self._run_eval_pipeline(model)

        with open(quality_path) as f:
            quality_rows = json.load(f)
        with open(latency_path) as f:
            latency_rows = json.load(f)

        wers = [r["wer"] for r in quality_rows if r.get("wer") is not None]
        mos_scores = [r["naturalness_mos"] for r in quality_rows if r.get("naturalness_mos") is not None]
        latencies = [r["first_byte_ms"] for r in latency_rows if r.get("first_byte_ms") is not None]

        return {
            "wer_mean": round(statistics.mean(wers), 4) if wers else float("nan"),
            "naturalness_mos_median": round(statistics.median(mos_scores), 3) if mos_scores else float("nan"),
            "latency_ms_median": round(statistics.median(latencies), 1) if latencies else float("nan"),
        }

    def _run_eval_pipeline(self, model: TrainedModel) -> tuple[str, str]:
        """Exports the trained candidate as a temporary custom: voice, runs it
        through eval/synth.mjs + eval/score.py exactly like this project's
        existing eval framework already does for any other voice, then
        returns (latency_json_path, quality_json_path). This is real,
        untested-by-this-plan integration work - see the "Not fully specified"
        note below rather than treating this stub as complete.

        NOT FULLY SPECIFIED IN THIS PLAN: exporting model.artifact_path (a
        raw Piper checkpoint directory on the tts-checkpoints Modal Volume)
        into the .onnx + .onnx.json + owner.json shape eval/synth.mjs's
        PIPER_VOICE expects, and registering it as a temporary, non-public
        custom: voice via the same admin endpoint voices/publish_voice.py
        uses, is real work this task's own brief could not fully specify
        without deeper knowledge of Piper's ONNX export step (see
        voices/export_voices.py for the existing export pattern to follow).
        The implementer should read voices/export_voices.py and
        voices/publish_voice.py in full, adapt the export step for a raw
        training-run checkpoint (not a downloaded rhasspy/piper-checkpoints
        .ckpt), publish it under a temporary id (e.g.
        f"harness-{model.candidate.id}-{uuid4()}"), run eval/synth.mjs and
        eval/score.py against just that one voice (check synth.mjs's
        PIPER_VOICE mapping for how to scope a run to one voice), then
        DELETE the temporary voice via the admin endpoint before returning
        - a harness research candidate must never stay published. If this
        turns out to need real design decisions (e.g. the ONNX export step
        fails for a checkpoint that hasn't gone through Piper's normal
        export tooling), STOP and escalate rather than guessing at export
        internals - this is exactly the kind of judgment call this plan's
        author flagged as not fully specifiable in advance."""
        raise NotImplementedError("see docstring - implementer must design the export+publish+eval+cleanup sequence")
```

- [ ] **Step 5: Run test to verify it passes (the test patches `_run_eval_pipeline`, so its `NotImplementedError` body is never reached)**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_tts_core_evaluate.py -v`
Expected: PASS (1 test) — confirms the metric-aggregation logic (`evaluate()`'s own body) is correct in isolation from the export/publish integration work

- [ ] **Step 6: Implement `_run_eval_pipeline` for real, following the export→publish→eval→cleanup sequence documented in its own docstring above**

This step has no pre-written code — per the docstring in Step 4, the exact export mechanics depend on details (Piper's ONNX export CLI, `voices/export_voices.py`'s real structure) that require reading those files first. Read `voices/export_voices.py` and `voices/publish_voice.py` in full, then implement:
1. Export `model.artifact_path`'s checkpoint to `model.onnx`/`model.onnx.json` (following `export_voices.py`'s export step).
2. Write a temporary `owner.json` (`{"public": false, "key_ids": [], "license": "internal-research", "attribution": "harness research candidate, not for production"}`).
3. Publish under a temporary id via the same admin PUT flow `voices/publish_voice.py` uses.
4. Run `eval/synth.mjs` and `eval/score.py` scoped to just that one voice (check `synth.mjs`'s `PIPER_VOICE` mapping and testset for how to scope a run — may need a temporary one-voice testset rather than the full multi-language one, to keep evaluation cost/time bounded per candidate).
5. Delete the temporary voice via the admin `DELETE` endpoint.
6. Return the two result file paths.

Write a real test for this step once implemented (an actual small end-to-end run, mirroring how `eval/`'s own test/verification runs were done earlier in this project) rather than leaving it test-free — if a genuinely low-cost real test isn't feasible here, that itself is a finding to report back, not a step to silently skip.

- [ ] **Step 7: Commit**

```bash
git add harness/recipes/tts_core.py harness/tests/test_tts_core_evaluate.py
git commit -m "harness: wire TTSCoreRecipe.evaluate() into the existing eval/ framework"
```

---

## Task 7: CLI wiring and end-to-end smoke test

**Files:**
- Modify: `harness/run_cycle.py` (register `TTSCoreRecipe` — currently the registry has no entries)
- Modify: `harness/recipes/tts_core.py`'s module (register itself)
- Test: `harness/tests/test_cli_wiring.py`

**Interfaces:**
- Consumes: `register`, `get_recipe` (Task 3); `TTSCoreRecipe` (Tasks 4-6)
- Produces: `harness/recipes/__init__.py` importable with `tts-core` already registered as a side effect of import

- [ ] **Step 1: Write the failing test**

```python
# harness/tests/test_cli_wiring.py
from harness.recipes import get_recipe
import harness.recipes.tts_core  # noqa: F401 - import triggers registration


def test_tts_core_is_registered_by_default():
    recipe = get_recipe("tts-core")
    assert recipe.name == "tts-core"


def test_unknown_recipe_raises_with_helpful_message():
    import pytest
    with pytest.raises(ValueError, match="Unknown recipe"):
        get_recipe("does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_cli_wiring.py -v`
Expected: FAIL — `get_recipe("tts-core")` raises `ValueError: Unknown recipe 'tts-core'` since nothing registers it yet

- [ ] **Step 3: Register `TTSCoreRecipe` at the bottom of its own module**

Add to the end of `harness/recipes/tts_core.py`:

```python
from harness.recipes import register

register("tts-core", TTSCoreRecipe)
```

- [ ] **Step 4: Import the recipe module from `run_cycle.py` so registration always happens before CLI dispatch**

In `harness/run_cycle.py`, add near the top (after the existing `from harness.recipes import get_recipe` line):

```python
import harness.recipes.tts_core  # noqa: F401 - side-effecting import, registers "tts-core"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/tests/test_cli_wiring.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full harness suite one final time**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m pytest harness/ -v`
Expected: PASS (all tests from Tasks 1-7; `_run_eval_pipeline`'s real integration test from Task 6 Step 6 either passes or is explicitly documented as a follow-up if infeasible to test cheaply)

- [ ] **Step 7: Manually confirm the CLI actually parses and dispatches (not a full real run — budget=0 forces every candidate to skip, proving wiring without spending anything)**

Run: `cd ~/Documents/GitHub/realtime-tts && python3 -m harness.run_cycle --recipe tts-core --budget 0`
Expected: exits cleanly, prints "Report written to harness/reports/YYYY-MM-DD-tts-core.md", and that file's content shows all 3 size-variant candidates as `skipped_over_budget` (proving discovery + registry + CLI argument parsing + report writing all work end to end, with zero real spend)

- [ ] **Step 8: Commit**

```bash
git add harness/run_cycle.py harness/recipes/tts_core.py harness/tests/test_cli_wiring.py
git commit -m "harness: wire TTSCoreRecipe into the CLI registry, confirm end-to-end dispatch"
```

---

## Final integration check (after all 7 tasks)

- [ ] Re-read the spec's "Not in scope" section — confirm nothing in this plan built a scheduler, a second GPU provider, or a second recipe.
- [ ] Confirm `harness/run_cycle.py` and `harness/report.py` contain zero references to Piper, Modal, WER, or any TTS-specific concept — grep for `piper\|modal\|wer` (case-insensitive) in both files and expect no hits.
- [ ] Run `python3 -m harness.run_cycle --recipe tts-core --budget 0` once more after all tasks land, confirm it still exits cleanly.
- [ ] Flag Task 6's `_run_eval_pipeline` real-implementation step (Step 6) explicitly in the final report to whoever reviews this plan's execution — it's the one place this plan deliberately didn't pre-write code, per its own "No Placeholders" exception reasoning, and deserves a specific look before the harness is trusted for a real (non-`--budget 0`) run.
