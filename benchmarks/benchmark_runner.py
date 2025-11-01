"""Core benchmark runner for Galaxy Prometheus N-body simulation.

This module provides systematic benchmarking of baseline surrogate models
across multiple test problems and particle counts. It measures accuracy,
speed, and physics metrics (energy conservation, trajectory error).

Example usage:
    from config import Settings
    from benchmarks import BenchmarkRunner

    settings = Settings.load_from_yaml()
    runner = BenchmarkRunner(settings)
    results = runner.run_all_benchmarks()
"""

import time
from dataclasses import dataclass

from baselines import create_direct_nbody_baseline, create_kdtree_baseline
from config import Settings
from initial_conditions import plummer_sphere, three_body_figure_eight, two_body_circular_orbit
from validation_metrics import (
    compute_energy_drift,
    compute_trajectory_rmse,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        baseline_name: Name of baseline model ("kdtree" or "direct_nbody")
        test_problem: Name of test problem ("two_body", "figure_eight", "plummer")
        num_particles: Number of particles in simulation
        accuracy: Accuracy metric (0.0-1.0, higher is better)
        speed: Execution time in seconds (lower is better)
        energy_drift: Relative energy conservation error (lower is better)
        trajectory_rmse: Position error vs ground truth (lower is better)
    """

    baseline_name: str
    test_problem: str
    num_particles: int
    accuracy: float
    speed: float
    energy_drift: float
    trajectory_rmse: float


class BenchmarkRunner:
    """Orchestrates benchmark execution across test problems and baselines.

    This class manages the systematic evaluation of baseline surrogate models
    on standard N-body test problems, measuring both computational performance
    and physical accuracy.

    Attributes:
        config: Settings instance with benchmark configuration

    Example:
        >>> from config import Settings
        >>> settings = Settings.load_from_yaml()
        >>> runner = BenchmarkRunner(settings)
        >>> results = runner.run_all_benchmarks()
        >>> print(f"Ran {len(results)} benchmarks")
    """

    def __init__(self, config: Settings):
        """Initialize benchmark runner with configuration.

        Args:
            config: Settings instance containing benchmark parameters
        """
        self.config = config

    def _get_test_problem_particles(self, problem_name: str, n_particles: int) -> list[list[float]]:
        """Get initial particles for specified test problem.

        Args:
            problem_name: Name of test problem ("two_body", "figure_eight", "plummer")
            n_particles: Number of particles to generate

        Returns:
            List of particles, each as [x, y, z, vx, vy, vz, mass]

        Raises:
            ValueError: If problem_name is not recognized
        """
        if problem_name == "two_body":
            # Two-body circular orbit (always 2 particles)
            return two_body_circular_orbit()
        elif problem_name == "figure_eight":
            # Three-body figure-8 (always 3 particles)
            return three_body_figure_eight()
        elif problem_name == "plummer":
            # Plummer sphere (use requested n_particles)
            return plummer_sphere(n_particles=n_particles, random_seed=42)
        else:
            raise ValueError(
                f"Unknown test problem: {problem_name}. "
                f"Valid options: two_body, figure_eight, plummer"
            )

    def _get_baseline_model(self, baseline_name: str):
        """Get baseline surrogate model by name.

        Args:
            baseline_name: Name of baseline ("kdtree" or "direct_nbody")

        Returns:
            SurrogateGenome instance with compiled predict function

        Raises:
            ValueError: If baseline_name is not recognized
        """
        if baseline_name == "kdtree":
            return create_kdtree_baseline(k_neighbors=self.config.benchmark_kdtree_k)
        elif baseline_name == "direct_nbody":
            return create_direct_nbody_baseline()
        else:
            raise ValueError(
                f"Unknown baseline: {baseline_name}. Valid options: kdtree, direct_nbody"
            )

    def run_single_benchmark(
        self, baseline_name: str, test_problem: str, n_particles: int
    ) -> BenchmarkResult:
        """Run a single benchmark for given baseline, problem, and particle count.

        Args:
            baseline_name: Name of baseline model ("kdtree" or "direct_nbody")
            test_problem: Name of test problem ("two_body", "figure_eight", "plummer")
            n_particles: Number of particles to simulate

        Returns:
            BenchmarkResult containing all metrics

        Example:
            >>> result = runner.run_single_benchmark("kdtree", "two_body", 10)
            >>> print(f"Accuracy: {result.accuracy:.3f}, Speed: {result.speed:.4f}s")
        """
        # Get initial particles
        initial_particles = self._get_test_problem_particles(test_problem, n_particles)

        # Get baseline model
        genome = self._get_baseline_model(baseline_name)
        model_func = genome.compiled_predict

        # Get ground truth for comparison (direct N-body)
        if baseline_name != "direct_nbody":
            ground_truth_genome = create_direct_nbody_baseline()
            ground_truth_func = ground_truth_genome.compiled_predict
        else:
            # Direct N-body is its own ground truth
            ground_truth_func = model_func

        # Simulate for N timesteps
        num_timesteps = self.config.benchmark_timesteps

        # Run baseline model
        start_time = time.time()
        current_particles = [p[:] for p in initial_particles]  # Deep copy
        for _ in range(num_timesteps):
            # Update each particle
            new_particles = []
            for particle in current_particles:
                new_particle = model_func(particle, current_particles)
                new_particles.append(new_particle)
            current_particles = new_particles
        speed = time.time() - start_time

        # Run ground truth (if different from baseline)
        if baseline_name != "direct_nbody":
            ground_truth_particles = [p[:] for p in initial_particles]
            for _ in range(num_timesteps):
                new_particles = []
                for particle in ground_truth_particles:
                    new_particle = ground_truth_func(particle, ground_truth_particles)
                    new_particles.append(new_particle)
                ground_truth_particles = new_particles
        else:
            ground_truth_particles = current_particles

        # Compute metrics
        # 1. Trajectory RMSE (vs ground truth)
        trajectory_rmse = compute_trajectory_rmse(current_particles, ground_truth_particles)

        # 2. Accuracy metric (0-1 scale, higher is better)
        # Use formula from prototype.py: accuracy = 1.0 / (1.0 + sqrt(error))
        import math

        accuracy = 1.0 / (1.0 + math.sqrt(max(trajectory_rmse, 0.0)))

        # 3. Energy drift (conservation)
        energy_drift = compute_energy_drift(initial_particles, current_particles)

        return BenchmarkResult(
            baseline_name=baseline_name,
            test_problem=test_problem,
            num_particles=len(initial_particles),
            accuracy=accuracy,
            speed=speed,
            energy_drift=energy_drift,
            trajectory_rmse=trajectory_rmse,
        )

    def run_all_benchmarks(self) -> list[BenchmarkResult]:
        """Run all configured benchmarks.

        Executes benchmarks for all combinations of:
        - Baselines (from config.benchmark_baselines)
        - Test problems (from config.benchmark_test_problems)
        - Particle counts (from config.benchmark_particle_counts)

        Returns:
            List of BenchmarkResult objects, one per benchmark

        Example:
            >>> results = runner.run_all_benchmarks()
            >>> print(f"Completed {len(results)} benchmarks")
            >>> print(f"Average accuracy: {sum(r.accuracy for r in results) / len(results):.3f}")
        """
        results = []

        for baseline_name in self.config.benchmark_baselines:
            for test_problem in self.config.benchmark_test_problems:
                for n_particles in self.config.benchmark_particle_counts:
                    # Skip if test problem has fixed particle count
                    if test_problem == "two_body" and n_particles != 2:
                        continue
                    if test_problem == "figure_eight" and n_particles != 3:
                        continue

                    result = self.run_single_benchmark(baseline_name, test_problem, n_particles)
                    results.append(result)

        return results
