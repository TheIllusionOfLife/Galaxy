#!/usr/bin/env python3
"""Analyze full-scale conservation prompts validation results."""

import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_run_data(run_dir: Path) -> dict[str, Any]:
    """Load evolution history and extract key metrics."""
    history_path = run_dir / "evolution_history.json"
    with open(history_path) as f:
        data = json.load(f)

    # Extract all models' energy drift (skip None/inf values)
    energy_drifts = []
    momentum_drifts = []
    valid_models = 0
    invalid_models = 0

    for gen in data["history"]:
        for model in gen["population"]:
            # Skip models with invalid fitness (validation failed)
            if (
                model["fitness"] is None
                or model["fitness"] == float("-inf")
                or math.isinf(model["fitness"])
            ):
                invalid_models += 1
                continue

            valid_models += 1

            if "energy_drift" in model and model["energy_drift"] is not None:
                energy_drifts.append(model["energy_drift"])

            if "angular_momentum_drift" in model and model["angular_momentum_drift"] is not None:
                momentum_drifts.append(model["angular_momentum_drift"])

    # Calculate statistics
    if not energy_drifts:
        return {
            "run_dir": str(run_dir),
            "test_problem": data["metadata"]["test_problem"],
            "error": "No valid energy drift measurements",
            "invalid_models": invalid_models,
        }

    return {
        "run_dir": str(run_dir),
        "test_problem": data["metadata"]["test_problem"],
        "best_fitness": data["summary"]["best_overall_fitness"],
        "best_drift": min(energy_drifts),
        "mean_drift": sum(energy_drifts) / len(energy_drifts),
        "median_drift": statistics.median(energy_drifts),
        "models_below_1pct": sum(1 for d in energy_drifts if d < 0.01),
        "models_below_10pct": sum(1 for d in energy_drifts if d < 0.10),
        "models_below_100pct": sum(1 for d in energy_drifts if d < 1.00),
        "total_valid_models": valid_models,
        "total_invalid_models": invalid_models,
        "energy_drifts": energy_drifts,
        "momentum_drifts": momentum_drifts,
        "api_calls": data["summary"]["total_models_evaluated"],
        "cost": data["summary"].get("total_cost_usd", 0.0),
    }


def compare_to_baseline(new_data: dict, baseline_data: dict) -> dict:
    """Compare new run to PR #45 baseline."""
    if "error" in new_data or "error" in baseline_data:
        return {"error": "Cannot compare - invalid data"}

    best_improvement = (
        ((baseline_data["best_drift"] - new_data["best_drift"]) / baseline_data["best_drift"] * 100)
        if baseline_data.get("best_drift")
        else float("inf")
    )
    mean_improvement = (
        ((baseline_data["mean_drift"] - new_data["mean_drift"]) / baseline_data["mean_drift"] * 100)
        if baseline_data.get("mean_drift")
        else float("inf")
    )

    conservation_rate_new = new_data["models_below_1pct"] / new_data["total_valid_models"] * 100
    conservation_rate_baseline = (
        baseline_data["models_below_1pct"] / baseline_data["total_valid_models"] * 100
    )

    return {
        "best_drift_improvement_pct": best_improvement,
        "mean_drift_improvement_pct": mean_improvement,
        "conservation_rate_new": conservation_rate_new,
        "conservation_rate_baseline": conservation_rate_baseline,
        "conservation_rate_delta": conservation_rate_new - conservation_rate_baseline,
    }


def main():
    # Load baseline (PR #45)
    baseline_dir = Path("results/run_20251102_215639")

    # Manually set baseline data from PR #45 analysis
    baseline = {
        "run_dir": str(baseline_dir),
        "test_problem": "plummer",
        "best_fitness": 1518.11,
        "best_drift": 0.0421,  # 4.21%
        "mean_drift": 3.9167,  # 391.67%
        "models_below_1pct": 0,
        "models_below_10pct": 0,
        "total_valid_models": 50,
    }

    # Load new runs
    two_body_dir = Path("results/run_20251103_111004")
    plummer_dir = Path("results/run_20251103_111514")

    print("Loading run data...")
    two_body = load_run_data(two_body_dir)
    plummer = load_run_data(plummer_dir)

    # Compare plummer to baseline
    if "error" not in plummer:
        plummer_comparison = compare_to_baseline(plummer, baseline)
    else:
        plummer_comparison = {"error": plummer["error"]}

    # Print results
    print()
    print("=" * 70)
    print("FULL-SCALE CONSERVATION PROMPTS VALIDATION RESULTS")
    print("=" * 70)
    print()

    # two_body results
    if "error" in two_body:
        print(f"❌ two_body (N=2): {two_body['error']}")
    else:
        print("two_body (N=2):")
        print(f"  Best drift: {two_body['best_drift'] * 100:.2f}%")
        print(f"  Mean drift: {two_body['mean_drift'] * 100:.2f}%")
        print(f"  Median drift: {two_body['median_drift'] * 100:.2f}%")
        print(
            f"  Models <1%: {two_body['models_below_1pct']}/{two_body['total_valid_models']} ({two_body['models_below_1pct'] / two_body['total_valid_models'] * 100:.0f}%)"
        )
        print(
            f"  Models <10%: {two_body['models_below_10pct']}/{two_body['total_valid_models']} ({two_body['models_below_10pct'] / two_body['total_valid_models'] * 100:.0f}%)"
        )
        print(f"  Invalid models: {two_body['total_invalid_models']}")
        print(f"  Best fitness: {two_body['best_fitness']:,.0f}")
        print(f"  Cost: ${two_body['cost']:.4f}")

    print()

    # plummer results
    if "error" in plummer:
        print(f"❌ plummer (N=50): {plummer['error']}")
    else:
        print("plummer (N=50):")
        print(f"  Best drift: {plummer['best_drift'] * 100:.2f}%")
        print(f"  Mean drift: {plummer['mean_drift'] * 100:.2f}%")
        print(f"  Median drift: {plummer['median_drift'] * 100:.2f}%")
        print(
            f"  Models <1%: {plummer['models_below_1pct']}/{plummer['total_valid_models']} ({plummer['models_below_1pct'] / plummer['total_valid_models'] * 100:.0f}%)"
        )
        print(
            f"  Models <10%: {plummer['models_below_10pct']}/{plummer['total_valid_models']} ({plummer['models_below_10pct'] / plummer['total_valid_models'] * 100:.0f}%)"
        )
        print(f"  Invalid models: {plummer['total_invalid_models']}")
        print(f"  Best fitness: {plummer['best_fitness']:,.0f}")
        print(f"  Cost: ${plummer['cost']:.4f}")

    print()

    # Comparison
    if "error" not in plummer_comparison:
        print("Comparison to PR #45 baseline (plummer, WITHOUT conservation prompts):")
        print(f"  Best drift improvement: {plummer_comparison['best_drift_improvement_pct']:+.1f}%")
        print(f"  Mean drift improvement: {plummer_comparison['mean_drift_improvement_pct']:+.1f}%")
        print(
            f"  Conservation rate: {plummer_comparison['conservation_rate_new']:.1f}% (vs {plummer_comparison['conservation_rate_baseline']:.1f}% baseline)"
        )
        print(
            f"  Conservation rate delta: {plummer_comparison['conservation_rate_delta']:+.1f} percentage points"
        )

    print()
    print("=" * 70)

    # Save detailed results
    results = {
        "two_body": two_body,
        "plummer": plummer,
        "baseline": baseline,
        "comparison": plummer_comparison,
    }

    output_path = Path("validation_results_detailed.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
