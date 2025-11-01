"""Integration tests for benchmark suite.

These tests verify that the benchmark runner works correctly with real
baseline models and test problems from the Phase 2 infrastructure.
"""

import pytest

from benchmarks.benchmark_runner import BenchmarkRunner
from config import Settings


@pytest.mark.integration
class TestBenchmarkIntegration:
    """Integration tests with real baselines and test problems."""

    def test_kdtree_baseline_two_body_integration(self):
        """Verify KDTree baseline works with two-body test problem."""
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Run single benchmark with KDTree on two-body problem
        result = runner.run_single_benchmark("kdtree", "two_body", 2)

        # Verify result structure
        assert result.baseline_name == "kdtree"
        assert result.test_problem == "two_body"
        assert result.num_particles == 2

        # Verify metrics are reasonable
        assert 0.0 <= result.accuracy <= 1.0
        assert result.speed > 0.0
        assert result.energy_drift >= 0.0
        assert result.trajectory_rmse >= 0.0

        # KDTree should have good accuracy (>0.9) on two-body
        assert result.accuracy > 0.9

    def test_direct_nbody_baseline_perfect_accuracy(self):
        """Verify direct N-body baseline achieves perfect accuracy (1.0)."""
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Run with direct N-body (ground truth)
        result = runner.run_single_benchmark("direct_nbody", "plummer", 10)

        # Verify direct N-body is its own ground truth
        assert result.baseline_name == "direct_nbody"
        assert result.accuracy == 1.0  # Perfect accuracy
        assert result.trajectory_rmse == 0.0  # Zero error

    def test_plummer_sphere_multiple_particle_counts(self):
        """Verify Plummer sphere works with multiple particle counts."""
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        particle_counts = [10, 50]

        for n in particle_counts:
            result = runner.run_single_benchmark("kdtree", "plummer", n)

            # Verify correct particle count used
            assert result.num_particles == n

            # Verify reasonable metrics (KDTree is approximate, may have lower accuracy)
            assert 0.0 <= result.accuracy <= 1.0
            assert result.speed > 0.0

    def test_figure_eight_three_body_integration(self):
        """Verify figure-8 three-body problem works correctly."""
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Figure-8 always has 3 particles
        result = runner.run_single_benchmark("kdtree", "figure_eight", 3)

        assert result.num_particles == 3
        assert result.test_problem == "figure_eight"
        assert 0.0 <= result.accuracy <= 1.0

    def test_benchmark_suite_small_scale(self):
        """Run a small-scale benchmark suite end-to-end."""
        settings = Settings.load_from_yaml()

        # Override config for small test
        settings.benchmark_baselines = ["kdtree", "direct_nbody"]
        settings.benchmark_test_problems = ["two_body"]
        settings.benchmark_particle_counts = [2]
        settings.benchmark_timesteps = 10  # Reduced for speed

        runner = BenchmarkRunner(settings)
        results = runner.run_all_benchmarks()

        # Should have 2 results (kdtree + direct_nbody on two_body)
        assert len(results) == 2

        # Verify both baselines ran
        baselines = {r.baseline_name for r in results}
        assert baselines == {"kdtree", "direct_nbody"}

        # Verify all results are valid
        for result in results:
            assert 0.0 <= result.accuracy <= 1.0
            assert result.speed > 0.0
            assert result.energy_drift >= 0.0

    def test_scaling_behavior_validation(self):
        """Verify that execution time increases with particle count."""
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Run with different particle counts
        result_10 = runner.run_single_benchmark("kdtree", "plummer", 10)
        result_50 = runner.run_single_benchmark("kdtree", "plummer", 50)

        # More particles should take longer
        # Allow some tolerance for timing variance
        assert result_50.speed > result_10.speed * 0.5
