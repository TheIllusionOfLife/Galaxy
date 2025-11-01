#!/usr/bin/env python3
"""Run comprehensive benchmark suite for Galaxy Prometheus N-body simulation.

This script runs systematic benchmarks of baseline surrogate models across
multiple test problems and particle counts, generating:
  - Scaling analysis plots (log-log complexity)
  - Accuracy heatmaps
  - Pareto front (accuracy vs speed trade-off)
  - Performance tables (markdown + JSON)

Usage:
    python scripts/run_benchmarks.py

Output:
    results/benchmarks/run_YYYYMMDD_HHMMSS/
        ├── benchmark_results.json
        ├── performance_table.md
        ├── scaling_comparison.png
        ├── accuracy_heatmap.png
        └── pareto_front.png
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path (must be before imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# flake8: noqa: E402 (module level import not at top of file - intentional for path setup)
from benchmarks.benchmark_runner import BenchmarkRunner  # noqa: E402
from benchmarks.scaling_analysis import export_markdown_table, measure_scaling  # noqa: E402
from benchmarks.visualization import (  # noqa: E402
    export_results_json,
    plot_accuracy_bars,
    plot_pareto_front,
    plot_scaling_comparison,
)
from config import Settings  # noqa: E402


def main() -> None:
    """Run benchmark suite and generate all outputs."""
    print("=" * 70)
    print("Galaxy Prometheus - Comprehensive Benchmark Suite")
    print("=" * 70)
    print()

    # Load configuration
    print("Loading configuration...")
    config = Settings.load_from_yaml()

    if not config.benchmark_enabled:
        print("ERROR: Benchmarks are disabled in config.yaml")
        print("Set benchmark.enabled = true to run benchmarks")
        sys.exit(1)

    print(f"  Baselines: {', '.join(config.benchmark_baselines)}")
    print(f"  Test problems: {', '.join(config.benchmark_test_problems)}")
    print(f"  Particle counts: {config.benchmark_particle_counts}")
    print(f"  Timesteps per run: {config.benchmark_timesteps}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "benchmarks" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 70)

    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()

    print(f"\n✓ Completed {len(results)} benchmark runs")
    print()

    # Export results
    print("Exporting results...")

    # 1. JSON export
    json_path = output_dir / "benchmark_results.json"
    export_results_json(results, str(json_path))
    print(f"  ✓ JSON results: {json_path}")

    # 2. Markdown table
    markdown_table = export_markdown_table(results)
    table_path = output_dir / "performance_table.md"
    with open(table_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(markdown_table)
    print(f"  ✓ Markdown table: {table_path}")

    # 3. Scaling analysis summary
    scaling_path = output_dir / "scaling_analysis.txt"
    with open(scaling_path, "w") as f:
        f.write("Scaling Analysis Summary\n")
        f.write("=" * 70 + "\n\n")

        for baseline in config.benchmark_baselines:
            for problem in config.benchmark_test_problems:
                try:
                    scaling = measure_scaling(results, baseline, problem)
                    f.write(f"{baseline} on {problem}:\n")
                    f.write(f"  Empirical: O(N^{scaling['exponent']:.2f})\n")
                    f.write(f"  Theoretical: {scaling['theoretical']}\n")
                    f.write(f"  Coefficient: {scaling['coefficient']:.6f}\n")
                    f.write("\n")
                except ValueError:
                    # Not enough data points for this combination
                    pass

    print(f"  ✓ Scaling analysis: {scaling_path}")

    # Generate plots
    print("\nGenerating plots...")

    # 1. Scaling comparison
    scaling_plot_path = output_dir / "scaling_comparison.png"
    plot_scaling_comparison(results, str(scaling_plot_path))
    print(f"  ✓ Scaling comparison: {scaling_plot_path}")

    # 2. Accuracy bars
    accuracy_plot_path = output_dir / "accuracy_bars.png"
    plot_accuracy_bars(results, str(accuracy_plot_path))
    print(f"  ✓ Accuracy bars: {accuracy_plot_path}")

    # 3. Pareto front
    pareto_plot_path = output_dir / "pareto_front.png"
    plot_pareto_front(results, str(pareto_plot_path))
    print(f"  ✓ Pareto front: {pareto_plot_path}")

    print()
    print("=" * 70)
    print("Benchmark suite completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
