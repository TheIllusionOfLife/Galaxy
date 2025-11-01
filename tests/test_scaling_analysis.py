"""Tests for scaling analysis utilities."""

import pytest

from benchmarks.benchmark_runner import BenchmarkResult
from benchmarks.scaling_analysis import compare_baselines, export_markdown_table, measure_scaling


class TestMeasureScaling:
    """Test scaling analysis functionality."""

    def test_measure_scaling_basic(self):
        """Verify scaling measurement with simple data."""
        # Create synthetic results with known scaling
        # time = 0.001 * N^2 (perfect quadratic scaling)
        results = [
            BenchmarkResult("direct_nbody", "plummer", 10, 0.99, 0.001 * 10**2, 0.01, 0.1),
            BenchmarkResult("direct_nbody", "plummer", 20, 0.99, 0.001 * 20**2, 0.01, 0.1),
            BenchmarkResult("direct_nbody", "plummer", 50, 0.99, 0.001 * 50**2, 0.01, 0.1),
        ]

        scaling = measure_scaling(results, "direct_nbody", "plummer")

        # Check structure
        assert "n_values" in scaling
        assert "times" in scaling
        assert "exponent" in scaling
        assert "coefficient" in scaling
        assert "theoretical" in scaling

        # Check values
        assert scaling["n_values"] == [10, 20, 50]
        assert len(scaling["times"]) == 3
        assert scaling["theoretical"] == "O(N²)"

        # Exponent should be close to 2.0 for quadratic scaling
        assert 1.95 <= scaling["exponent"] <= 2.05

    def test_measure_scaling_kdtree(self):
        """Verify KDTree theoretical complexity is labeled correctly."""
        results = [
            BenchmarkResult("kdtree", "plummer", 10, 0.95, 0.05, 0.02, 0.2),
            BenchmarkResult("kdtree", "plummer", 50, 0.95, 0.15, 0.02, 0.2),
        ]

        scaling = measure_scaling(results, "kdtree", "plummer")

        assert scaling["theoretical"] == "O(N² log N)"

    def test_measure_scaling_insufficient_data(self):
        """Verify error raised with insufficient data points."""
        results = [
            BenchmarkResult("kdtree", "plummer", 10, 0.95, 0.05, 0.02, 0.2),
        ]

        with pytest.raises(ValueError, match="at least 2 data points"):
            measure_scaling(results, "kdtree", "plummer")

    def test_measure_scaling_no_matching_results(self):
        """Verify error raised when no results match criteria."""
        results = [
            BenchmarkResult("kdtree", "plummer", 10, 0.95, 0.05, 0.02, 0.2),
        ]

        with pytest.raises(ValueError, match="No results found"):
            measure_scaling(results, "direct_nbody", "plummer")


class TestCompareBaselines:
    """Test baseline comparison functionality."""

    def test_compare_baselines_structure(self):
        """Verify comparison data structure."""
        results = [
            BenchmarkResult("kdtree", "two_body", 2, 0.95, 0.01, 0.01, 0.1),
            BenchmarkResult("direct_nbody", "two_body", 2, 1.0, 0.02, 0.001, 0.0),
            BenchmarkResult("kdtree", "plummer", 50, 0.93, 0.15, 0.02, 0.3),
        ]

        comparison = compare_baselines(results)

        # Check structure
        assert "baselines" in comparison
        assert "test_problems" in comparison
        assert "particle_counts" in comparison
        assert "data" in comparison

        # Check lists
        assert set(comparison["baselines"]) == {"direct_nbody", "kdtree"}
        assert set(comparison["test_problems"]) == {"plummer", "two_body"}
        assert set(comparison["particle_counts"]) == {2, 50}

    def test_compare_baselines_data_access(self):
        """Verify comparison data can be accessed correctly."""
        results = [
            BenchmarkResult("kdtree", "two_body", 2, 0.95, 0.01, 0.01, 0.1),
        ]

        comparison = compare_baselines(results)

        # Access data for kdtree on two_body with 2 particles
        metrics = comparison["data"]["kdtree"]["two_body"][2]

        assert metrics["accuracy"] == 0.95
        assert metrics["speed"] == 0.01
        assert metrics["energy_drift"] == 0.01
        assert metrics["trajectory_rmse"] == 0.1


class TestExportMarkdownTable:
    """Test markdown export functionality."""

    def test_export_markdown_table_format(self):
        """Verify markdown table is correctly formatted."""
        results = [
            BenchmarkResult("kdtree", "two_body", 2, 0.95, 0.01, 0.01, 0.1),
            BenchmarkResult("direct_nbody", "two_body", 2, 1.0, 0.02, 0.001, 0.0),
        ]

        table = export_markdown_table(results)

        # Check header row
        assert (
            "| Baseline | Test Problem | N | Accuracy | Speed (s) | Energy Drift | RMSE |" in table
        )

        # Check separator row
        assert (
            "|----------|-------------|---|----------|-----------|--------------|------|" in table
        )

        # Check data rows contain baseline names
        assert "direct_nbody" in table
        assert "kdtree" in table

        # Check data rows contain test problem names
        assert "two_body" in table

    def test_export_markdown_table_sorted(self):
        """Verify markdown table is sorted correctly."""
        results = [
            BenchmarkResult("kdtree", "plummer", 50, 0.93, 0.15, 0.02, 0.3),
            BenchmarkResult("direct_nbody", "two_body", 2, 1.0, 0.02, 0.001, 0.0),
            BenchmarkResult("kdtree", "two_body", 2, 0.95, 0.01, 0.01, 0.1),
        ]

        table = export_markdown_table(results)

        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.startswith("| ") and "Baseline" not in line and "-" not in line
        ]

        # Should be sorted: direct_nbody/two_body, kdtree/plummer, kdtree/two_body
        assert "direct_nbody" in data_lines[0]
        assert "kdtree" in data_lines[1]
        assert "plummer" in data_lines[1]
        assert "kdtree" in data_lines[2]
        assert "two_body" in data_lines[2]
