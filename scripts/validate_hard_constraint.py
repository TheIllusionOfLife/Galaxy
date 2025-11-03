#!/usr/bin/env python3
"""Validate PR #50 hard constraint effectiveness.

Compares evolution results with hard constraint (PR #50) against
baseline results without hard constraint (PR #49).
"""

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

    # Extract all models' energy drift and momentum drift
    energy_drifts = []
    momentum_drifts = []
    valid_models = 0
    eliminated_models = 0  # Models with fitness=-inf (hard constraint triggered)
    catastrophic_violators = 0  # Models with >10% energy OR >50% momentum

    for gen in data["history"]:
        for model in gen["population"]:
            # Check if model was eliminated by hard constraint
            fitness = model.get("fitness")
            if fitness is None or (
                isinstance(fitness, (int, float)) and math.isinf(fitness) and fitness < 0
            ):
                eliminated_models += 1
                # Still count drift for eliminated models if available
                energy_drift = model.get("energy_drift")
                momentum_drift = model.get("angular_momentum_drift")

                if energy_drift is not None:
                    energy_drifts.append(energy_drift)
                    if energy_drift > 0.10 or (
                        momentum_drift is not None and momentum_drift > 0.50
                    ):
                        catastrophic_violators += 1
                continue

            valid_models += 1

            energy_drift = model.get("energy_drift")
            momentum_drift = model.get("angular_momentum_drift")

            if energy_drift is not None:
                energy_drifts.append(energy_drift)
                if energy_drift > 0.10 or (momentum_drift is not None and momentum_drift > 0.50):
                    catastrophic_violators += 1

            if momentum_drift is not None:
                momentum_drifts.append(momentum_drift)

    # Calculate statistics
    if not energy_drifts:
        return {
            "run_dir": str(run_dir),
            "test_problem": data["metadata"]["test_problem"],
            "error": "No energy drift measurements",
            "eliminated_models": eliminated_models,
            "total_models": eliminated_models + valid_models,
        }

    return {
        "run_dir": str(run_dir),
        "test_problem": data["metadata"]["test_problem"],
        "best_fitness": data["summary"]["best_overall_fitness"],
        "best_drift": min(energy_drifts) if energy_drifts else None,
        "mean_drift": sum(energy_drifts) / len(energy_drifts) if energy_drifts else None,
        "median_drift": statistics.median(energy_drifts) if energy_drifts else None,
        "models_below_1pct": sum(1 for d in energy_drifts if d < 0.01),
        "models_below_10pct": sum(1 for d in energy_drifts if d < 0.10),
        "models_below_100pct": sum(1 for d in energy_drifts if d < 1.00),
        "catastrophic_violators": catastrophic_violators,
        "total_valid_models": valid_models,
        "total_eliminated_models": eliminated_models,
        "total_models": eliminated_models + valid_models,
        "energy_drifts": energy_drifts,
        "momentum_drifts": momentum_drifts,
        "api_calls": data["summary"]["total_models_evaluated"],
    }


def print_results(label: str, data: dict, baseline: dict | None = None):
    """Print formatted results with optional baseline comparison."""
    print(f"\n{label}:")
    print("=" * 70)

    if "error" in data:
        print(f"  ❌ Error: {data['error']}")
        print(f"  Total models: {data['total_models']}")
        print(f"  Eliminated: {data['eliminated_models']}")
        return

    # Key metrics
    print(f"  Best drift: {data['best_drift'] * 100:.2f}%")
    print(f"  Mean drift: {data['mean_drift'] * 100:.2f}%")
    print(f"  Median drift: {data['median_drift'] * 100:.2f}%")
    print()

    # Conservation rates
    print(
        f"  Models <1% drift:  {data['models_below_1pct']}/{data['total_models']} ({data['models_below_1pct'] / data['total_models'] * 100:.0f}%)"
    )
    print(
        f"  Models <10% drift: {data['models_below_10pct']}/{data['total_models']} ({data['models_below_10pct'] / data['total_models'] * 100:.0f}%)"
    )
    print()

    # Hard constraint metrics
    print(
        f"  Catastrophic violators: {data['catastrophic_violators']}/{data['total_models']} ({data['catastrophic_violators'] / data['total_models'] * 100:.0f}%)"
    )
    print(
        f"  Models eliminated: {data['total_eliminated_models']}/{data['total_models']} ({data['total_eliminated_models'] / data['total_models'] * 100:.0f}%)"
    )
    print(
        f"  Models survived: {data['total_valid_models']}/{data['total_models']} ({data['total_valid_models'] / data['total_models'] * 100:.0f}%)"
    )
    print()

    # Fitness
    print(f"  Best fitness: {data['best_fitness']:,.2f}")

    # Baseline comparison
    if baseline and "error" not in baseline:
        print()
        print("  Comparison to PR #49 baseline (WITHOUT hard constraint):")

        if baseline["best_drift"] and data["best_drift"]:
            best_improvement = (
                (baseline["best_drift"] - data["best_drift"]) / baseline["best_drift"] * 100
            )
            print(
                f"    Best drift: {baseline['best_drift'] * 100:.2f}% → {data['best_drift'] * 100:.2f}% ({best_improvement:+.1f}%)"
            )

        if baseline["mean_drift"] and data["mean_drift"]:
            mean_improvement = (
                (baseline["mean_drift"] - data["mean_drift"]) / baseline["mean_drift"] * 100
            )
            print(
                f"    Mean drift: {baseline['mean_drift'] * 100:.2f}% → {data['mean_drift'] * 100:.2f}% ({mean_improvement:+.1f}%)"
            )

        baseline_catast_rate = baseline["catastrophic_violators"] / baseline["total_models"] * 100
        current_catast_rate = data["catastrophic_violators"] / data["total_models"] * 100
        print(
            f"    Catastrophic rate: {baseline_catast_rate:.0f}% → {current_catast_rate:.0f}% ({current_catast_rate - baseline_catast_rate:+.0f}pp)"
        )


def main():
    print("=" * 70)
    print("PR #50 HARD CONSTRAINT VALIDATION")
    print("=" * 70)
    print()
    print("Comparing evolution with hard constraint (PR #50) vs without (PR #49)")
    print()

    # PR #49 baseline data (manually extracted from FULL_SCALE_CONSERVATION_VALIDATION.md)
    pr49_two_body_baseline = {
        "test_problem": "two_body",
        "best_drift": 0.0016,  # 0.16%
        "mean_drift": 1610.61,  # 161,061%
        "models_below_1pct": 1,
        "models_below_10pct": 24,
        "catastrophic_violators": 25,  # Estimated from 48% <10% → 52% ≥10%
        "total_models": 50,
        "best_fitness": 320157,
    }

    pr49_plummer_baseline = {
        "test_problem": "plummer",
        "best_drift": 0.1625,  # 16.25%
        "mean_drift": 16.58,  # 1,658%
        "models_below_1pct": 0,
        "models_below_10pct": 0,
        "catastrophic_violators": 49,  # All models exceeded thresholds
        "total_models": 49,  # 1 invalid
        "best_fitness": 1345,
    }

    # Load PR #50 results
    two_body_dir = Path("results/run_20251103_180806")
    plummer_dir = Path("results/run_20251103_181334")

    print("Loading PR #50 evolution results...")
    pr50_two_body = load_run_data(two_body_dir)
    pr50_plummer = load_run_data(plummer_dir)

    # Print results
    print_results("PR #50: two_body (N=2)", pr50_two_body, pr49_two_body_baseline)
    print_results("\nPR #50: plummer (N=50)", pr50_plummer, pr49_plummer_baseline)

    # Summary verdict
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)
    print()

    # Check success criteria
    two_body_success = (
        "error" not in pr50_two_body
        and pr50_two_body["catastrophic_violators"] / pr50_two_body["total_models"]
        < 0.50  # <50% catastrophic
        and pr50_two_body["mean_drift"] < 1000  # <1000% mean drift
    )

    plummer_success = (
        "error" not in pr50_plummer
        and pr50_plummer["catastrophic_violators"] / pr50_plummer["total_models"]
        < pr49_plummer_baseline["catastrophic_violators"] / pr49_plummer_baseline["total_models"]
    )

    if two_body_success:
        print("✅ two_body: PASS - Hard constraint effectively eliminated catastrophic violators")
    else:
        print("❌ two_body: FAIL - Still significant catastrophic violators or mean drift")

    if plummer_success:
        print("✅ plummer: PASS - Hard constraint improved catastrophic violator rate")
    else:
        print("⚠️  plummer: CONCERN - Hard constraint may be too strict for complex N-body")
        if "error" in pr50_plummer or pr50_plummer["total_valid_models"] == 0:
            print("    ALL models eliminated - consider relaxing threshold")

    print()

    # Save results
    results = {
        "pr49_baseline": {
            "two_body": pr49_two_body_baseline,
            "plummer": pr49_plummer_baseline,
        },
        "pr50_results": {
            "two_body": pr50_two_body,
            "plummer": pr50_plummer,
        },
        "verdict": {
            "two_body_success": two_body_success,
            "plummer_success": plummer_success,
        },
    }

    output_path = Path("results/hard_constraint_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
