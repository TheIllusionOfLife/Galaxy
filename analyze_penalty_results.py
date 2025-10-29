#!/usr/bin/env python3
"""Analyze code length penalty comparative testing results."""

import json
from pathlib import Path


def analyze_run(results_dir: Path, weight: float) -> dict:
    """Extract key metrics from a results directory."""
    json_path = results_dir / "evolution_history.json"

    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    history = data["history"]
    summary = data["summary"]

    # Collect all token counts
    all_tokens = []
    gen_stats = []

    for gen in history:
        gen_tokens = [
            m.get("token_count", 0)
            for m in gen["population"]
            if m.get("token_count") and m.get("token_count") > 0
        ]
        if gen_tokens:
            gen_stats.append(
                {
                    "generation": gen["generation"],
                    "avg": sum(gen_tokens) / len(gen_tokens),
                    "min": min(gen_tokens),
                    "max": max(gen_tokens),
                    "count": len(gen_tokens),
                }
            )
            all_tokens.extend(gen_tokens)

    # Calculate overall statistics
    if all_tokens:
        avg_tokens = sum(all_tokens) / len(all_tokens)
        min_tokens = min(all_tokens)
        max_tokens = max(all_tokens)
    else:
        avg_tokens = min_tokens = max_tokens = 0

    return {
        "weight": weight,
        "best_fitness": summary["best_overall_fitness"],
        "avg_fitness": summary["avg_overall_fitness"],
        "total_models": summary["total_models_evaluated"],
        "avg_tokens": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "gen_stats": gen_stats,
    }


def main():
    """Analyze all penalty test results."""
    results_base = Path("results")

    # Find test results directories
    test_configs = [
        ("run_20251030_002207", 0.1, "Test 2 (Default)"),
        ("run_20251030_002858", 0.2, "Test 3 (Aggressive)"),
    ]

    print("=" * 70)
    print("CODE LENGTH PENALTY COMPARATIVE TESTING RESULTS")
    print("=" * 70)
    print()

    all_results = []

    for dir_name, weight, label in test_configs:
        results_dir = results_base / dir_name
        if not results_dir.exists():
            print(f"⚠️  {label}: Directory not found")
            continue

        result = analyze_run(results_dir, weight)
        if result:
            all_results.append(result)
            print(f"✓ {label} (weight={weight})")
            print(f"  Avg tokens: {result['avg_tokens']:.1f}")
            print(f"  Token range: {result['min_tokens']}-{result['max_tokens']}")
            print(f"  Best fitness: {result['best_fitness']:.2f}")
            print()

    # Comparison table
    if len(all_results) > 1:
        print()
        print("COMPARISON TABLE")
        print("-" * 70)
        print(f"{'Weight':<10} {'Avg Tokens':<15} {'Token Reduction':<20} {'Best Fitness':<15}")
        print("-" * 70)

        baseline = all_results[0]  # First result is baseline
        for r in all_results:
            reduction = (
                ((baseline["avg_tokens"] - r["avg_tokens"]) / baseline["avg_tokens"] * 100)
                if baseline["avg_tokens"] > 0
                else 0
            )
            print(
                f"{r['weight']:<10.2f} {r['avg_tokens']:<15.1f} {reduction:<20.1f}% {r['best_fitness']:<15.2f}"
            )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
