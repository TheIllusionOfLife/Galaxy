#!/usr/bin/env python3
"""Analyze crossover effectiveness from experimental results."""

import json
import statistics
from pathlib import Path


def analyze_run(run_dir: Path) -> dict:
    """Extract metrics from a single run."""
    history_file = run_dir / "evolution_history.json"
    with open(history_file) as f:
        data = json.load(f)

    history = data["history"]
    summary = data["summary"]

    # Extract metrics
    gen0_best = history[0]["best_fitness"]
    gen4_best = history[-1]["best_fitness"]
    best_overall = summary["best_overall_fitness"]
    improvement_rate = (best_overall - gen0_best) / gen0_best * 100

    # Fitness variance (diversity metric) per generation
    fitness_variances = []
    for gen in history:
        pop_fitnesses = [civ["fitness"] for civ in gen["population"]]
        if len(pop_fitnesses) > 1:
            fitness_variances.append(statistics.variance(pop_fitnesses))

    avg_variance = statistics.mean(fitness_variances) if fitness_variances else 0

    # Track best fitness progression across generations
    best_progression = [gen["best_fitness"] for gen in history]

    return {
        "gen0_best": gen0_best,
        "gen4_best": gen4_best,
        "best_overall": best_overall,
        "improvement_rate_pct": improvement_rate,
        "avg_fitness_variance": avg_variance,
        "total_models": summary["total_models_evaluated"],
        "best_progression": best_progression,
    }


def main():
    # Analyze all three runs
    base_dir = Path(__file__).parent

    runs = {
        "Control (Disabled)": "control_disabled",
        "Crossover 30%": "crossover_30pct",
        "Crossover 50%": "crossover_50pct",
    }

    print("=" * 70)
    print("CROSSOVER EFFECTIVENESS ANALYSIS - October 31, 2025")
    print("=" * 70)

    results = {}
    for name, dirname in runs.items():
        run_dir = base_dir / dirname
        if run_dir.exists():
            results[name] = analyze_run(run_dir)
            metrics = results[name]
            print(f"\n{name}:")
            print(f"  Gen 0 Best Fitness:    {metrics['gen0_best']:10.2f}")
            print(f"  Gen 4 Best Fitness:    {metrics['gen4_best']:10.2f}")
            print(f"  Best Overall Fitness:  {metrics['best_overall']:10.2f}")
            print(f"  Improvement Rate:      {metrics['improvement_rate_pct']:9.1f}%")
            print(f"  Avg Fitness Variance:  {metrics['avg_fitness_variance']:10.2f}")
        else:
            print(f"\n{name}: NOT FOUND")

    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    if len(results) == 3:
        control = results["Control (Disabled)"]
        cross30 = results["Crossover 30%"]
        cross50 = results["Crossover 50%"]

        print("\nBest Overall Fitness:")
        print(f"  Control:      {control['best_overall']:10.2f}")
        print(
            f"  Crossover 30%: {cross30['best_overall']:10.2f} ({(cross30['best_overall'] - control['best_overall']) / control['best_overall'] * 100:+.1f}%)"
        )
        print(
            f"  Crossover 50%: {cross50['best_overall']:10.2f} ({(cross50['best_overall'] - control['best_overall']) / control['best_overall'] * 100:+.1f}%)"
        )

        print("\nImprovement Rate (Gen 0 → Best):")
        print(f"  Control:      {control['improvement_rate_pct']:6.1f}%")
        print(
            f"  Crossover 30%: {cross30['improvement_rate_pct']:6.1f}% ({cross30['improvement_rate_pct'] - control['improvement_rate_pct']:+.1f}% points)"
        )
        print(
            f"  Crossover 50%: {cross50['improvement_rate_pct']:6.1f}% ({cross50['improvement_rate_pct'] - control['improvement_rate_pct']:+.1f}% points)"
        )

        print("\nPopulation Diversity (Avg Variance):")
        print(f"  Control:      {control['avg_fitness_variance']:10.2f}")
        print(f"  Crossover 30%: {cross30['avg_fitness_variance']:10.2f}")
        print(f"  Crossover 50%: {cross50['avg_fitness_variance']:10.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
