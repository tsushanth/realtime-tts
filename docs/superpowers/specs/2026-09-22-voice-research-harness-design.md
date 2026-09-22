# Voice research harness: core + TTS-size/latency recipe

Status: approved by user 2026-09-22, pending implementation plan.

## Context

This product (realtime-tts / ReadAloudAI) competes on latency and price for
self-hosted voice infra. Today's exploratory work (voice mining, hardening
MVPs for audio isolation/dubbing/voice-design/speech-to-speech, the model
comparison during audio isolation's real-speech investigation) was all done
by hand, one research question at a time, dispatched ad hoc. The user wants
a standing, reusable harness: pick a research question, let the harness
find candidates, train them on real GPU, evaluate them against this
project's own quality bar, and produce a report — repeatably, not as a
one-off investigation each time.

The full scope named by the user spans many verticals with genuinely
different evaluation needs (proven directly by this project's own recent
work: WER worked for TTS, was useless for audio isolation, which needed
SNR and eventually a real-noise WER test instead) — core TTS
latency/size/cost tradeoffs, music, background noise filtering, dubbing,
voice cloning. This spec covers two things only, by explicit decomposition
decision: the harness **core** (the reusable orchestration: discovery →
train → evaluate → report → budget enforcement), and **one concrete
recipe** (core TTS: CPU vs GPU, smaller vs larger models) that proves the
core's plugin interface is real and not accidentally shaped around one
vertical's assumptions. Every other vertical (music, noise filtering,
dubbing, voice cloning) is an explicitly separate, later sub-project, each
getting its own spec once the core exists to build on.

## Scope decisions (from brainstorming)

- **Trigger model**: on-demand CLI tool, not a scheduled/autonomous
  service. The user runs a cycle when they want one (`harness/run_cycle.py
  --recipe tts-core --budget 30`); no unattended GPU spend on a timer.
  Scheduling can be added later once the harness is trusted.
- **Budget**: a hard cap enforced by the harness core, not a suggestion.
  Default $30/cycle (user-adjustable via `--budget`); the harness refuses
  to start a training job whose estimated cost would exceed the remaining
  budget, and stops the cycle early rather than guessing or overspending.
- **GPU provider**: Modal only for v1. Already used throughout this project
  (Kokoro, voice-design, speech-to-speech all run on Modal today) — reuses
  existing credentials and job-dispatch patterns. RunPod/Vast are explicitly
  deferred, not designed for in this spec; adding a second provider later
  means implementing a second `TrainingBackend`, not a rewrite (see
  Component 3).
- **First recipe's candidate scope**: different model sizes/configs of
  architectures already in production (smaller/larger Piper-style VITS
  variants, quantization, quality tiers) — not new architecture families.
  Bounded, comparable apples-to-apples against this project's existing
  WER/MOS/latency eval, lower research risk than chasing novel
  architectures for the harness's first real run.
- **Report destination**: markdown files committed to the repo, matching
  the existing pattern (`eval/results/REPORT_DRAFT.md`,
  `voices/MINING_BATCH_*.md`) — no new infra, reviewed the same way the
  user already reviews this project's other reports.

## Architecture

A single CLI entrypoint, not a service. `harness/run_cycle.py --recipe
<name> --budget <dollars>` runs one research cycle end to end: it asks the
named recipe to discover candidates, trains each (within budget) via Modal,
evaluates each trained result using the recipe's own evaluation logic, and
writes one report file. The core (discovery loop, budget tracking, Modal
job dispatch, report writing) is generic; a `Recipe` is a small interface
each vertical implements once. This mirrors how this project's existing
tooling is already structured (`eval/`, `voice-pipeline/` are each
standalone scripts, not a service) — the harness is a new sibling in that
same family, not a new kind of infrastructure.

## Components

### 1. `Recipe` protocol

```python
# harness/recipe.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Candidate:
    id: str                    # stable identifier, e.g. "piper-vits-small-en"
    description: str           # human-readable, goes straight into the report
    train_config: dict         # recipe-specific, opaque to the harness core

@dataclass
class TrainedModel:
    candidate: Candidate
    artifact_path: str         # where the trained weights ended up (e.g. a Modal Volume path)
    actual_cost_usd: float     # real, measured cost of this training run, not an estimate

class Recipe(Protocol):
    name: str  # e.g. "tts-core" — used as the --recipe CLI value and in report filenames

    def discover_candidates(self) -> list[Candidate]:
        """Return candidates not yet tried, in priority order. The harness core
        trains them in this order until the budget runs out."""
        ...

    def estimate_cost_usd(self, candidate: Candidate) -> float:
        """A cost estimate the harness core checks against remaining budget
        BEFORE starting training — must be conservative (better to
        overestimate and skip a candidate than underestimate and blow the
        budget mid-run)."""
        ...

    def train(self, candidate: Candidate, workdir: str) -> TrainedModel:
        """Actually run training (via Modal, dispatched by this method — the
        harness core does not know how to train anything, only how to
        budget and sequence). Raises on failure; the harness core catches
        and records the failure in the report rather than crashing the cycle."""
        ...

    def evaluate(self, model: TrainedModel) -> dict[str, float]:
        """Metric name -> value. Metrics are entirely recipe-defined — the
        harness core only knows how to put them in a table, never how to
        compute or compare them. This is deliberate: today's own audio
        isolation work proved a shared 'evaluate' formula across verticals
        doesn't hold."""
        ...
```

### 2. Harness core (`harness/run_cycle.py`)

Orchestration only, no domain logic:

1. Load the named recipe (a simple registry, `harness/recipes/__init__.py`
   mapping `"tts-core"` → the `TTSCoreRecipe` class — adding a recipe later
   means adding one entry here, not touching the core).
2. Call `recipe.discover_candidates()`.
3. For each candidate, in order: call `recipe.estimate_cost_usd(candidate)`;
   if `running_total + estimate > budget`, stop the cycle here (do not skip
   ahead to a cheaper later candidate — order from `discover_candidates`
   is the recipe's priority, and the harness respects it) and note in the
   report which candidates were never attempted due to budget.
4. Otherwise call `recipe.train(candidate, workdir)`, catching exceptions
   and recording them as a failed candidate in the report rather than
   aborting the whole cycle.
5. Call `recipe.evaluate(model)` on every successfully trained model.
6. Write the report (Component 4).

### 3. Modal training dispatch

Recipes call Modal directly from their own `train()` method (the harness
core never touches Modal) — following this project's existing pattern
(`voice-pipeline/train_job.py`, `voice-pipeline/convert_job.py`) of each
domain owning its own Modal app rather than a shared dispatch layer. This
keeps the `Recipe` protocol's `train()` free to do whatever a given
vertical's training actually requires (different base images, different
GPU classes, different data-loading) without the harness core needing to
know any of it. If a second GPU provider (RunPod/Vast) is added later, it's
a second recipe implementation detail, not a harness-core change — the
`TrainedModel.actual_cost_usd` field is provider-agnostic by design.

### 4. Report (`harness/reports/YYYY-MM-DD-<recipe-name>.md`)

Structure, matching `eval/results/REPORT_DRAFT.md`'s existing style:

- Header: cycle date, recipe name, budget cap, actual total spend.
- Results table: one row per attempted candidate — id, description,
  status (trained/failed/skipped-over-budget), cost, and every metric
  `evaluate()` returned (columns are whatever the recipe's metrics were,
  not a fixed schema).
- "Reading this" section: the harness core writes a mechanical summary
  (cheapest candidate, best-per-metric candidate, candidates that failed
  and why) — no editorializing/recommendation, since the harness core
  doesn't understand what the metrics mean for a given vertical. Recipes
  should not add their own interpretation either for the first version —
  YAGNI: the user reviews the table and decides, as stated in the original
  request ("publish reports which i can look at and productionize").
- Explicit "not attempted this cycle" list for anything `discover_candidates`
  returned that never got trained (budget-exhausted or earlier failure).

### 5. `TTSCoreRecipe` (`harness/recipes/tts_core.py`)

- `discover_candidates()`: enumerates model-size/config variants of the
  existing Piper VITS architecture not yet tried, e.g. a smaller-parameter
  VITS config, a more aggressively quantized ONNX export, a
  larger/higher-quality config. "Already tried" is tracked in a plain JSON
  file, `harness/recipes/tts_core_tried.json` (`{candidate_id: {date,
  report_path}}`), committed to the repo alongside the reports themselves —
  matching this project's existing file-based state pattern (no database
  anywhere in this repo's tooling) and letting `git log` on that one file
  double as the recipe's own history. Reuses the "diff against what we
  already have" pattern already used for voice mining
  (`voices/export_v2.py`'s `SPEC` table approach) rather than inventing a
  new discovery mechanism.
- `estimate_cost_usd()`: based on the same GPU-second math already
  established for this project's pricing (Piper/Kokoro cost-basis
  reasoning, and the per-second Modal T4/A10G rates already measured for
  voice-design/speech-to-speech earlier this project).
- `train()`: a Modal job following `voice-pipeline/train_job.py`'s existing
  fine-tune pattern, parameterized by the candidate's `train_config`
  (model size/quantization settings).
- `evaluate()`: reuses `eval/`'s existing framework wholesale — WER via
  faster-whisper, naturalness via SQUIM, latency via the warm-connection
  benchmark methodology already established in `eval/synth.mjs`. No new
  evaluation code for this recipe; this is exactly why TTS-core was chosen
  as the first recipe — the hard eval-methodology problem is already
  solved for this vertical specifically.

## Error handling

- A failed `train()` call: logged in the report as `status: failed`, cycle
  continues to the next candidate (one bad candidate doesn't waste the
  whole budget/cycle).
- Budget exhaustion mid-cycle: cycle stops cleanly, already-trained
  candidates still get evaluated and reported, remaining candidates listed
  as not-attempted.
- Modal/GPU unavailability: surfaces as a `train()` exception, same
  failed-candidate handling as above — not a special case the harness core
  needs to know about.

## Testing

- `Recipe` protocol: a fake/test recipe (fixed candidate list, fake
  train/evaluate that don't touch Modal) exercises the harness core's
  budget-enforcement and report-writing logic without spending real money
  or real GPU time — this is the primary test surface for the core.
- `TTSCoreRecipe`: `discover_candidates()` and `estimate_cost_usd()` are
  pure functions, directly testable. `train()`/`evaluate()` are
  integration-tested with real (small, cheap) Modal runs, following the
  same "small real test, not a mock" pattern already used throughout this
  project's Modal-based work today (voice-design, speech-to-speech).

## Not in scope for this spec

- Any recipe other than `tts-core` (music, noise filtering, dubbing, voice
  cloning) — each is its own future spec, reusing this core.
- RunPod/Vast support — Modal only, per the scope decision above.
- Scheduling/autonomous runs — on-demand only, per the scope decision above.
- Any automatic "productionize this candidate" action — the harness stops
  at the report; promoting a candidate to production voice/model is a
  separate, human-decided action outside this system, matching the user's
  own framing ("publish reports which i can look at and productionize").

## Open items before implementation planning

- The exact default `--budget` value ($30 suggested in this spec) should
  be confirmed, not just assumed, before the plan treats it as final —
  this is a real spending decision for the user to make explicitly, not
  something to infer.
