"""The harness core: sequences a Recipe's candidates against a hard
budget cap, training and evaluating each in turn, and writes one report
per cycle. Contains zero domain logic - see harness/recipe.py's Recipe
protocol for the boundary this file never crosses."""
import argparse
import os

from harness.recipe import Recipe
from harness.report import CandidateResult, write_report
from harness.recipes import get_recipe
import harness.recipes.tts_core  # noqa: F401 - side-effecting import, registers "tts-core"


def run_cycle(recipe: Recipe, budget_usd: float, workdir: str, out_dir: str = "harness/reports") -> str:
    candidates = recipe.discover_candidates()
    results: list[CandidateResult] = []
    spent = 0.0

    for i, candidate in enumerate(candidates):
        estimate = recipe.estimate_cost_usd(candidate)
        if spent + estimate > budget_usd:
            # Spec: stop the cycle here. Do NOT skip ahead to a cheaper later
            # candidate - discover_candidates() order is the recipe's priority
            # and the harness respects it. The cycle has definitively ended;
            # "not re-costing" means never checking a later estimate against
            # the budget (which could change the stop/go decision), not
            # withholding the estimate itself - the reader needs the real
            # number to judge whether a small budget bump would have covered
            # it. estimate_cost_usd() is a pure, side-effect-free function
            # (per the Recipe protocol), so calling it here for display is free.
            results.append(CandidateResult(candidate=candidate, status="skipped_over_budget", cost_usd=estimate, metrics=None, error=None))
            for remaining in candidates[i + 1:]:
                remaining_estimate = recipe.estimate_cost_usd(remaining)
                results.append(CandidateResult(candidate=remaining, status="skipped_over_budget", cost_usd=remaining_estimate, metrics=None, error=None))
            break

        try:
            model = recipe.train(candidate, workdir)
        except Exception as e:  # noqa: BLE001 - one bad candidate must not kill the cycle
            results.append(CandidateResult(candidate=candidate, status="failed", cost_usd=estimate, metrics=None, error=str(e)))
            spent += estimate  # attempted spend, even on failure - conservative accounting
            continue

        spent += model.actual_cost_usd

        # evaluate() gets its own guard: training already spent real money, so
        # an evaluation failure must never cost us the report. The candidate is
        # still "trained" (that part succeeded) with no metrics and the error.
        try:
            metrics = recipe.evaluate(model)
            error = None
        except Exception as e:  # noqa: BLE001
            metrics = None
            error = f"evaluation failed: {e}"
        results.append(CandidateResult(candidate=candidate, status="trained", cost_usd=model.actual_cost_usd, metrics=metrics, error=error))

    report_path = write_report(recipe_name=recipe.name, budget_usd=budget_usd, total_spent_usd=spent, results=results, out_dir=out_dir)

    # Optional, duck-typed write-back so a recipe can remember what it already
    # tried. Deliberately not required by the Recipe protocol - the core stays
    # domain-free and recipes without persistent state need not implement it.
    if hasattr(recipe, "record_tried"):
        for r in results:
            if r.status == "trained":
                recipe.record_tried(r.candidate.id, report_path)

    return report_path


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
