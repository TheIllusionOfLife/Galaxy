"""Benchmark visualization utilities.

This module provides visualization functions for benchmark results,
following the same patterns as the main visualization.py module.

All plots use 300 DPI, consistent styling, and publication-quality output.

Example usage:
    from benchmarks import BenchmarkRunner
    from benchmarks.visualization import plot_scaling_comparison

    runner = BenchmarkRunner(settings)
    results = runner.run_all_benchmarks()

    plot_scaling_comparison(results, "output/scaling.png")
"""

from pathlib import Path

import matplotlib.pyplot as plt

from benchmarks.benchmark_runner import BenchmarkResult
from benchmarks.scaling_analysis import compare_baselines


def plot_scaling_comparison(results: list[BenchmarkResult], output_path: str) -> None:
    """Plot log-log scaling comparison for all baselines.

    Creates a log-log plot showing execution time vs particle count for each
    baseline, with fitted power law curves to visualize empirical complexity.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path where plot image will be saved

    Example:
        >>> plot_scaling_comparison(results, "results/scaling_comparison.png")
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique baselines
    baselines = sorted(set(r.baseline_name for r in results))
    test_problems = sorted(set(r.test_problem for r in results))

    # Use different colors and markers for combinations
    colors = {"kdtree": "blue", "direct_nbody": "green"}
    markers = {"two_body": "o", "figure_eight": "s", "plummer": "^"}

    for baseline in baselines:
        for problem in test_problems:
            # Filter results for this combination
            filtered = [
                r for r in results if r.baseline_name == baseline and r.test_problem == problem
            ]

            if len(filtered) < 2:
                continue  # Need at least 2 points to plot

            # Sort by particle count
            filtered.sort(key=lambda r: r.num_particles)
            n_values = [r.num_particles for r in filtered]
            times = [r.speed for r in filtered]

            # Plot data points
            color = colors.get(baseline, "black")
            marker = markers.get(problem, "o")
            label = f"{baseline} - {problem}"

            ax.loglog(
                n_values,
                times,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                label=label,
                alpha=0.7,
            )

    # Styling
    ax.set_xlabel("Number of Particles (N)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title("Baseline Scaling Comparison (Log-Log)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_heatmap(results: list[BenchmarkResult], output_path: str) -> None:
    """Plot accuracy heatmap across baselines and test problems.

    Creates a bar plot showing accuracy for each baseline/problem combination.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path where plot image will be saved

    Example:
        >>> plot_accuracy_heatmap(results, "results/accuracy_heatmap.png")
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by baseline and test problem, use max particle count for each
    comparison = compare_baselines(results)

    baselines = comparison["baselines"]
    test_problems = comparison["test_problems"]

    # Prepare data for bar plot
    x_labels = []
    accuracies = []
    colors_list = []

    color_map = {"kdtree": "blue", "direct_nbody": "green"}

    for baseline in baselines:
        for problem in test_problems:
            problem_data = comparison["data"].get(baseline, {}).get(problem, {})
            if problem_data:
                # Use largest particle count
                max_n = max(problem_data.keys())
                metrics = problem_data[max_n]

                x_labels.append(f"{baseline}\n{problem}")
                accuracies.append(metrics["accuracy"])
                colors_list.append(color_map.get(baseline, "gray"))

    # Create bar plot
    x_positions = range(len(x_labels))
    ax.bar(x_positions, accuracies, color=colors_list, alpha=0.7, edgecolor="black")

    # Styling
    ax.set_xlabel("Baseline / Test Problem", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Baseline and Test Problem", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_front(results: list[BenchmarkResult], output_path: str) -> None:
    """Plot accuracy vs speed trade-off (Pareto front).

    Creates a scatter plot showing the trade-off between accuracy and speed,
    with different colors/markers for each baseline.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path where plot image will be saved

    Example:
        >>> plot_pareto_front(results, "results/pareto_front.png")
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by baseline
    baselines = sorted(set(r.baseline_name for r in results))

    colors = {"kdtree": "blue", "direct_nbody": "green"}
    markers = {"kdtree": "o", "direct_nbody": "s"}

    for baseline in baselines:
        filtered = [r for r in results if r.baseline_name == baseline]

        accuracies = [r.accuracy for r in filtered]
        speeds = [r.speed for r in filtered]

        color = colors.get(baseline, "black")
        marker = markers.get(baseline, "o")

        ax.scatter(
            speeds,
            accuracies,
            c=color,
            marker=marker,
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidths=1,
            label=baseline,
        )

    # Styling
    ax.set_xlabel("Speed (seconds, lower is better)", fontsize=12)
    ax.set_ylabel("Accuracy (higher is better)", fontsize=12)
    ax.set_title("Accuracy vs Speed Trade-off", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_results_json(results: list[BenchmarkResult], output_path: str) -> None:
    """Export benchmark results to JSON file.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path where JSON file will be saved

    Example:
        >>> export_results_json(results, "results/benchmark_results.json")
    """
    import json

    # Convert results to dictionaries
    results_data = [
        {
            "baseline_name": r.baseline_name,
            "test_problem": r.test_problem,
            "num_particles": r.num_particles,
            "accuracy": r.accuracy,
            "speed": r.speed,
            "energy_drift": r.energy_drift,
            "trajectory_rmse": r.trajectory_rmse,
        }
        for r in results
    ]

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
