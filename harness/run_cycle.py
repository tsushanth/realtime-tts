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
