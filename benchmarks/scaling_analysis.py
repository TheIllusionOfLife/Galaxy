"""Scaling analysis utilities for benchmark results.

This module provides tools for analyzing computational complexity scaling
and generating performance comparison tables from benchmark results.

Example usage:
    from benchmarks import BenchmarkRunner
    from benchmarks.scaling_analysis import measure_scaling, compare_baselines

    runner = BenchmarkRunner(settings)
    results = runner.run_all_benchmarks()

    # Measure scaling
    kdtree_scaling = measure_scaling(results, "kdtree", "plummer")
    print(f"KDTree exponent: {kdtree_scaling['exponent']:.2f}")

    # Compare baselines
    comparison_df = compare_baselines(results)
    print(comparison_df.to_markdown())
"""

import math
from typing import Any

from benchmarks.benchmark_runner import BenchmarkResult


def measure_scaling(
    results: list[BenchmarkResult], baseline_name: str, test_problem: str
) -> dict[str, Any]:
    """Measure computational complexity scaling for a specific baseline and problem.

    Fits execution time to power law: time = a * N^b
    to determine empirical complexity exponent b.

    Args:
        results: List of BenchmarkResult objects
        baseline_name: Name of baseline to analyze ("kdtree" or "direct_nbody")
        test_problem: Name of test problem to analyze

    Returns:
        Dictionary containing:
            - n_values: List of particle counts
            - times: List of execution times
            - exponent: Empirical complexity exponent (b in a*N^b)
            - coefficient: Scaling coefficient (a in a*N^b)
            - theoretical: Theoretical complexity (e.g., "O(N²)" or "O(N² log N)")

    Example:
        >>> scaling = measure_scaling(results, "kdtree", "plummer")
        >>> print(f"Empirical: O(N^{scaling['exponent']:.2f})")
        >>> print(f"Theoretical: {scaling['theoretical']}")
    """
    # Filter results for this baseline and test problem
    filtered = [
        r for r in results if r.baseline_name == baseline_name and r.test_problem == test_problem
    ]

    if not filtered:
        raise ValueError(f"No results found for baseline={baseline_name}, problem={test_problem}")

    # Sort by particle count
    filtered.sort(key=lambda r: r.num_particles)

    n_values = [r.num_particles for r in filtered]
    times = [r.speed for r in filtered]

    # Fit to a*N^b using log-log linear regression
    # log(time) = log(a) + b*log(N)
    # This is a linear regression: y = c + b*x where y=log(time), x=log(N), c=log(a)

    if len(n_values) < 2:
        raise ValueError("Need at least 2 data points to measure scaling")

    log_n = [math.log(n) for n in n_values]
    log_times = [math.log(t) for t in times]

    # Simple linear regression
    n_points = len(log_n)
    mean_log_n = sum(log_n) / n_points
    mean_log_time = sum(log_times) / n_points

    # Calculate slope (exponent b)
    numerator = sum(
        (log_n[i] - mean_log_n) * (log_times[i] - mean_log_time) for i in range(n_points)
    )
    denominator = sum((log_n[i] - mean_log_n) ** 2 for i in range(n_points))

    if denominator == 0:
        exponent = 0.0
        coefficient = times[0] if times else 1.0
    else:
        exponent = numerator / denominator
        log_a = mean_log_time - exponent * mean_log_n
        coefficient = math.exp(log_a)

    # Determine theoretical complexity
    if baseline_name == "direct_nbody":
        theoretical = "O(N²)"
    elif baseline_name == "kdtree":
        theoretical = "O(N² log N)"  # KDTree query is O(log N) per particle
    else:
        theoretical = "Unknown"

    return {
        "n_values": n_values,
        "times": times,
        "exponent": exponent,
        "coefficient": coefficient,
        "theoretical": theoretical,
    }


def compare_baselines(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate performance comparison table across baselines and test problems.

    Creates a structured comparison showing accuracy, speed, and physics metrics
    for all baseline/problem/particle count combinations.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Dictionary with comparison data organized by:
            - baselines: List of baseline names
            - test_problems: List of test problem names
            - particle_counts: List of particle counts
            - data: Nested dict [baseline][problem][n_particles] -> metrics

    Example:
        >>> comparison = compare_baselines(results)
        >>> for baseline in comparison['baselines']:
        ...     for problem in comparison['test_problems']:
        ...         metrics = comparison['data'][baseline][problem]
        ...         print(f"{baseline} on {problem}: {metrics}")
    """
    # Extract unique values
    baselines = sorted(set(r.baseline_name for r in results))
    test_problems = sorted(set(r.test_problem for r in results))
    particle_counts = sorted(set(r.num_particles for r in results))

    # Organize data
    data = {}
    for baseline in baselines:
        data[baseline] = {}
        for problem in test_problems:
            data[baseline][problem] = {}
            for n in particle_counts:
                # Find result for this combination
                matching = [
                    r
                    for r in results
                    if r.baseline_name == baseline
                    and r.test_problem == problem
                    and r.num_particles == n
                ]
                if matching:
                    result = matching[0]
                    data[baseline][problem][n] = {
                        "accuracy": result.accuracy,
                        "speed": result.speed,
                        "energy_drift": result.energy_drift,
                        "trajectory_rmse": result.trajectory_rmse,
                    }

    return {
        "baselines": baselines,
        "test_problems": test_problems,
        "particle_counts": particle_counts,
        "data": data,
    }


def export_markdown_table(results: list[BenchmarkResult]) -> str:
    """Export results as a markdown-formatted table.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Markdown table string ready for README or documentation

    Example:
        >>> table = export_markdown_table(results)
        >>> print(table)
        | Baseline | Test Problem | N | Accuracy | Speed (s) | Energy Drift | RMSE |
        |----------|-------------|---|----------|-----------|--------------|------|
        | KDTree   | two_body    | 2 | 0.987    | 0.0021    | 0.024        | 0.15 |
        ...
    """
    # Sort results for consistent ordering
    sorted_results = sorted(
        results, key=lambda r: (r.baseline_name, r.test_problem, r.num_particles)
    )

    # Build table
    lines = [
        "| Baseline | Test Problem | N | Accuracy | Speed (s) | Energy Drift | RMSE |",
        "|----------|-------------|---|----------|-----------|--------------|------|",
    ]

    for r in sorted_results:
        line = (
            f"| {r.baseline_name:12} | {r.test_problem:11} | {r.num_particles:3} | "
            f"{r.accuracy:8.3f} | {r.speed:9.6f} | {r.energy_drift:12.6f} | "
            f"{r.trajectory_rmse:8.6f} |"
        )
        lines.append(line)

    return "\n".join(lines)
