"""Compare evolved model against benchmark baselines.

This script compares a best evolved model's performance against the
KDTree and direct N-body baselines from the benchmark suite.

Usage:
    python scripts/compare_to_baselines.py \\
        --evolved results/run_XXX/best_model_info.json \\
        --benchmarks results/benchmarks/run_YYY/benchmark_results.json \\
        --particles 50

Output:
    - comparison.md: Markdown table comparing all models
    - comparison.json: JSON data for programmatic access
"""

import argparse
import json
import sys
from pathlib import Path


def load_benchmark_results(benchmark_path: Path) -> list:
    """Load benchmark results from JSON file.

    Args:
        benchmark_path: Path to benchmark_results.json

    Returns:
        List of benchmark result dictionaries

    Raises:
        FileNotFoundError: If benchmark file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_path) as f:
        return json.load(f)


def filter_benchmark_by_particles(benchmarks: list, num_particles: int) -> list:
    """Filter benchmarks by particle count.

    Args:
        benchmarks: List of benchmark dictionaries
        num_particles: Target particle count

    Returns:
        Filtered list containing only matching particle count
    """
    return [b for b in benchmarks if b["num_particles"] == num_particles]


def calculate_fitness(accuracy: float, speed: float) -> float:
    """Calculate fitness score (accuracy / speed).

    Args:
        accuracy: Accuracy metric (0.0 - 1.0)
        speed: Speed in seconds

    Returns:
        Fitness score (higher is better)
    """
    if speed == 0.0:
        return float("inf")
    return accuracy / speed


def find_baseline_by_name(benchmarks: list, baseline_name: str, num_particles: int) -> dict | None:
    """Find specific baseline from benchmark results.

    Args:
        benchmarks: List of benchmark dictionaries
        baseline_name: Name of baseline (e.g., 'kdtree', 'direct_nbody')
        num_particles: Particle count to match

    Returns:
        Benchmark dictionary or None if not found
    """
    for b in benchmarks:
        if b["baseline_name"] == baseline_name and b["num_particles"] == num_particles:
            return b
    return None


def parse_model_metrics(model: dict, name: str) -> dict:
    """Parse metrics from model dictionary.

    Args:
        model: Model dictionary from evolution or benchmark
        name: Display name for the model

    Returns:
        Dictionary with standardized metric fields
    """
    return {
        "name": name,
        "accuracy": model.get("accuracy", 0.0),
        "speed": model.get("speed", 0.0),
        "fitness": model.get("fitness", 0.0),
        "energy_drift": model.get("energy_drift", None),
        "trajectory_rmse": model.get("trajectory_rmse", None),
    }


def create_comparison_row(data: dict) -> str:
    """Create a markdown table row from data.

    Args:
        data: Dictionary with metrics

    Returns:
        Formatted markdown table row
    """
    name = data["name"]
    accuracy = data["accuracy"]
    speed = data["speed"]
    fitness = data["fitness"]
    energy_drift = data.get("energy_drift", "N/A")
    trajectory_rmse = data.get("trajectory_rmse", "N/A")

    # Format numeric values
    acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
    speed_str = f"{speed:.6f}" if isinstance(speed, (int, float)) else str(speed)
    fitness_str = f"{fitness:.2f}" if isinstance(fitness, (int, float)) else str(fitness)

    energy_str = (
        f"{energy_drift:.2f}" if isinstance(energy_drift, (int, float)) else str(energy_drift)
    )
    rmse_str = (
        f"{trajectory_rmse:.2f}"
        if isinstance(trajectory_rmse, (int, float))
        else str(trajectory_rmse)
    )

    return f"| {name} | {acc_str} | {speed_str} | {fitness_str} | {energy_str} | {rmse_str} |"


def create_comparison_table(rows: list[dict]) -> str:
    """Create full comparison table in markdown.

    Args:
        rows: List of data dictionaries

    Returns:
        Formatted markdown table
    """
    table = [
        "| Model | Accuracy | Speed (s) | Fitness | Energy Drift (%) | Trajectory RMSE |",
        "|-------|----------|-----------|---------|------------------|-----------------|",
    ]

    for row in rows:
        table.append(create_comparison_row(row))

    return "\n".join(table)


def format_comparison_summary(evolved: dict, kdtree: dict, direct: dict) -> str:
    """Format comparison summary with key findings.

    Args:
        evolved: Evolved model metrics
        kdtree: KDTree baseline metrics
        direct: Direct N-body baseline metrics

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "Phase 3: Evolution vs Baseline Comparison",
        "=" * 70,
        "",
        "## Models Compared",
        "",
        f"1. **{evolved['name']}**",
        f"   - Fitness: {evolved['fitness']:.2f}",
        f"   - Accuracy: {evolved['accuracy']:.4f}",
        f"   - Speed: {evolved['speed']:.6f}s",
        "",
        f"2. **{kdtree['name']}**",
        f"   - Fitness: {kdtree['fitness']:.2f}",
        f"   - Accuracy: {kdtree['accuracy']:.4f}",
        f"   - Speed: {kdtree['speed']:.6f}s",
        "",
        f"3. **{direct['name']}**",
        f"   - Fitness: {direct['fitness']:.2f}",
        f"   - Accuracy: {direct['accuracy']:.4f}",
        f"   - Speed: {direct['speed']:.6f}s",
        "",
        "## Key Comparisons",
        "",
    ]

    # Compare evolved vs KDTree
    if evolved["fitness"] > kdtree["fitness"]:
        ratio = evolved["fitness"] / kdtree["fitness"] if kdtree["fitness"] > 0 else float("inf")
        lines.append(f"✅ Evolved model beats KDTree by {ratio:.2f}x fitness")
    else:
        ratio = kdtree["fitness"] / evolved["fitness"] if evolved["fitness"] > 0 else float("inf")
        lines.append(f"❌ KDTree beats evolved model by {ratio:.2f}x fitness")

    if evolved["accuracy"] > kdtree["accuracy"]:
        diff = (evolved["accuracy"] - kdtree["accuracy"]) * 100
        lines.append(f"✅ Evolved model has {diff:.1f}% higher accuracy than KDTree")
    else:
        diff = (kdtree["accuracy"] - evolved["accuracy"]) * 100
        lines.append(f"❌ KDTree has {diff:.1f}% higher accuracy than evolved model")

    # Compare evolved vs Direct
    lines.append("")
    if evolved["speed"] < direct["speed"]:
        ratio = direct["speed"] / evolved["speed"] if evolved["speed"] > 0 else float("inf")
        lines.append(f"✅ Evolved model is {ratio:.2f}x faster than direct N-body")
    else:
        ratio = evolved["speed"] / direct["speed"] if direct["speed"] > 0 else float("inf")
        lines.append(f"❌ Direct N-body is {ratio:.2f}x faster than evolved model")

    if evolved["accuracy"] > 0.9 * direct["accuracy"]:
        lines.append("✅ Evolved model maintains >90% of direct N-body accuracy")
    else:
        if direct["accuracy"] > 0:
            acc_pct = (evolved["accuracy"] / direct["accuracy"]) * 100
            lines.append(f"⚠️  Evolved model has {acc_pct:.1f}% of direct N-body accuracy")
        else:
            lines.append("⚠️  Cannot calculate accuracy percentage (direct N-body accuracy is zero)")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_comparison_markdown(comparison_text: str, output_path: Path) -> None:
    """Save comparison to markdown file.

    Args:
        comparison_text: Formatted comparison text
        output_path: Path to save file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(comparison_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare evolved model against benchmark baselines"
    )
    parser.add_argument("--evolved", type=Path, required=True, help="Path to best_model_info.json")
    parser.add_argument(
        "--benchmarks", type=Path, required=True, help="Path to benchmark_results.json"
    )
    parser.add_argument(
        "--particles", type=int, default=50, help="Number of particles (default: 50)"
    )
    parser.add_argument(
        "--output", type=Path, help="Output markdown path (default: <evolved_dir>/comparison.md)"
    )

    args = parser.parse_args()

    # Load evolved model
    print(f"Loading evolved model from: {args.evolved}")
    if not args.evolved.exists():
        print(f"❌ Error: Evolved model file not found: {args.evolved}")
        return 1

    with open(args.evolved) as f:
        evolved_model = json.load(f)

    # Load benchmarks
    print(f"Loading benchmarks from: {args.benchmarks}")
    try:
        benchmarks = load_benchmark_results(args.benchmarks)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing benchmark JSON: {e}")
        return 1

    # Filter by particle count
    filtered = filter_benchmark_by_particles(benchmarks, args.particles)
    if not filtered:
        print(f"❌ Error: No benchmarks found for {args.particles} particles")
        return 1

    print(f"Found {len(filtered)} benchmark(s) for N={args.particles}")

    # Find specific baselines
    kdtree = find_baseline_by_name(filtered, "kdtree", args.particles)
    direct = find_baseline_by_name(filtered, "direct_nbody", args.particles)

    if kdtree is None:
        print("❌ Error: KDTree baseline not found")
        return 1

    if direct is None:
        print("❌ Error: Direct N-body baseline not found")
        return 1

    # Parse metrics
    evolved_metrics = parse_model_metrics(evolved_model, "Best Evolved")
    kdtree_metrics = parse_model_metrics(kdtree, "KDTree Baseline")
    direct_metrics = parse_model_metrics(direct, "Direct N-body (Ground Truth)")

    # Calculate fitness if not already present
    if evolved_metrics["fitness"] == 0.0:
        evolved_metrics["fitness"] = calculate_fitness(
            evolved_metrics["accuracy"], evolved_metrics["speed"]
        )
    if kdtree_metrics["fitness"] == 0.0:
        kdtree_metrics["fitness"] = calculate_fitness(
            kdtree_metrics["accuracy"], kdtree_metrics["speed"]
        )
    if direct_metrics["fitness"] == 0.0:
        direct_metrics["fitness"] = calculate_fitness(
            direct_metrics["accuracy"], direct_metrics["speed"]
        )

    # Create comparison
    print()
    summary = format_comparison_summary(evolved_metrics, kdtree_metrics, direct_metrics)
    print(summary)
    print()

    # Create table
    table = create_comparison_table([direct_metrics, kdtree_metrics, evolved_metrics])
    print("## Detailed Metrics")
    print()
    print(table)
    print()

    # Save to file
    output_path = args.output or (args.evolved.parent / "comparison.md")
    full_text = f"{summary}\n\n## Detailed Metrics\n\n{table}\n"
    save_comparison_markdown(full_text, output_path)
    print(f"✅ Comparison saved to: {output_path}")

    # Save JSON
    json_path = output_path.parent / "comparison.json"
    comparison_data = {
        "evolved": evolved_metrics,
        "kdtree": kdtree_metrics,
        "direct_nbody": direct_metrics,
        "num_particles": args.particles,
    }
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✅ Comparison data saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
