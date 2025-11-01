"""Compare evolution results across multiple test problems.

This script compares best evolved models from multiple evolution runs,
each using a different test problem (two_body, figure_eight, plummer).

Usage:
    python scripts/compare_problems.py \\
        results/two_body_run_XXX \\
        results/figure_eight_run_YYY \\
        results/plummer_run_ZZZ

Output:
    - multi_problem_comparison.md: Markdown table comparing all problems
    - multi_problem_comparison.json: JSON data for programmatic access
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_evolution_history(run_dir: Path) -> dict:
    """Load evolution history from a run directory.

    Args:
        run_dir: Path to results/run_XXX directory

    Returns:
        Evolution history dictionary with metadata

    Raises:
        FileNotFoundError: If evolution_history.json doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    history_path = run_dir / "evolution_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Evolution history not found: {history_path}")

    with open(history_path) as f:
        return json.load(f)


def extract_best_model(history_data: dict) -> dict:
    """Extract best model from evolution history.

    Args:
        history_data: Evolution history dictionary

    Returns:
        Best model metrics dictionary
    """
    best_fitness = float("-inf")
    best_model = None

    for generation in history_data.get("history", []):
        for model in generation.get("population", []):
            fitness = model.get("fitness", float("-inf"))
            # Skip non-numeric or infinite fitness (validation failures, NaN converted to null)
            if (
                isinstance(fitness, (int, float))
                and math.isfinite(fitness)
                and fitness > best_fitness
            ):
                best_fitness = fitness
                best_model = model

    if best_model is None:
        return {
            "fitness": 0.0,
            "accuracy": 0.0,
            "speed": 0.0,
        }

    return {
        "fitness": best_model.get("fitness", 0.0),
        "accuracy": best_model.get("accuracy", 0.0),
        "speed": best_model.get("speed", 0.0),
    }


def create_comparison_table(results: list[dict]) -> str:
    """Create markdown table comparing all test problems.

    Args:
        results: List of result dictionaries with test_problem, num_particles, metrics

    Returns:
        Markdown formatted table string
    """
    lines = []
    lines.append("# Multi-Problem Validation Results")
    lines.append("")
    lines.append("## Comparison Table")
    lines.append("")
    lines.append("| Test Problem | N | Best Fitness | Accuracy | Speed (s) |")
    lines.append("|--------------|---|--------------|----------|-----------|")

    for result in results:
        test_problem = result["test_problem"]
        num_particles = result["num_particles"]
        fitness = result["best_fitness"]
        accuracy = result["best_accuracy"]
        speed = result["best_speed"]

        lines.append(
            f"| {test_problem:12} | {num_particles:3} | {int(fitness):12,} | {accuracy * 100:8.2f}% | {speed:9.6f} |"
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    # Find best problem by fitness
    if results:
        best_result = max(results, key=lambda r: r["best_fitness"])
        lines.append(
            f"- **Best fitness**: {best_result['test_problem']} "
            f"(fitness={int(best_result['best_fitness']):,})"
        )

        # Find most accurate
        best_accuracy = max(results, key=lambda r: r["best_accuracy"])
        lines.append(
            f"- **Highest accuracy**: {best_accuracy['test_problem']} "
            f"(accuracy={best_accuracy['best_accuracy'] * 100:.2f}%)"
        )

        # Find fastest
        fastest = min(
            results, key=lambda r: r["best_speed"] if r["best_speed"] > 0 else float("inf")
        )
        lines.append(
            f"- **Fastest speed**: {fastest['test_problem']} (speed={fastest['best_speed']:.6f}s)"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    """Main entry point for compare_problems script."""
    parser = argparse.ArgumentParser(
        description="Compare evolution results across multiple test problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Paths to evolution run directories (results/run_XXX)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "multi_problem_comparison",
        help="Output base path (default: results/multi_problem_comparison)",
    )

    args = parser.parse_args()

    # Collect results from all runs
    results = []
    for run_dir in args.run_dirs:
        try:
            print(f"Loading {run_dir}...")
            history_data = load_evolution_history(run_dir)

            # Extract metadata
            metadata = history_data.get("metadata", {})
            test_problem = metadata.get("test_problem", "unknown")
            num_particles = metadata.get("num_particles", 0)

            # Extract best model
            best_model = extract_best_model(history_data)

            results.append(
                {
                    "test_problem": test_problem,
                    "num_particles": num_particles,
                    "best_fitness": best_model["fitness"],
                    "best_accuracy": best_model["accuracy"],
                    "best_speed": best_model["speed"],
                    "run_dir": str(run_dir),
                }
            )

            print(f"  ✓ {test_problem} (N={num_particles}): fitness={best_model['fitness']:.2f}")

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            continue

    if not results:
        print("Error: No valid results found", file=sys.stderr)
        sys.exit(1)

    # Sort results by test_problem name for consistency
    results.sort(key=lambda r: r["test_problem"])

    # Generate markdown table
    markdown_output = create_comparison_table(results)

    # Write outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)

    md_path = args.output.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(markdown_output)
    print(f"\n✓ Comparison table saved: {md_path}")

    json_path = args.output.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Comparison data saved: {json_path}")

    print("\n" + "=" * 70)
    print("MULTI-PROBLEM COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
