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

    if trained:
        cheapest = min(trained, key=lambda r: r.cost_usd)
        lines.append("")
        lines.append(f"Cheapest trained candidate: **{cheapest.candidate.id}** at ${cheapest.cost_usd:.2f}.")

    # Min and max per metric, NOT a "winner": the harness core is domain-free
    # and cannot know whether higher or lower is better for an arbitrary metric
    # name (lower WER is better, higher MOS is better). The reader decides.
    scored = [r for r in trained if r.metrics]
    if scored and metric_names:
        lines.append("")
        lines.append("Metric range across trained candidates (min/max - the harness does not know which direction is better):")
        for m in metric_names:
            having = [r for r in scored if m in r.metrics]
            if not having:
                continue
            lo = min(having, key=lambda r: r.metrics[m])
            hi = max(having, key=lambda r: r.metrics[m])
            lines.append(f"- **{m}**: min {lo.metrics[m]} ({lo.candidate.id}), max {hi.metrics[m]} ({hi.candidate.id})")

    eval_failed = [r for r in trained if r.error]
    if eval_failed:
        lines.append("")
        lines.append("Trained but not evaluated (training spend was still incurred):")
        for r in eval_failed:
            lines.append(f"- **{r.candidate.id}**: {r.error}")

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
