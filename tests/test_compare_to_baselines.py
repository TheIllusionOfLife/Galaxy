"""Tests for scripts/compare_to_baselines.py

Tests the comparison of evolved models against benchmark baselines.
"""

import json
import tempfile
from pathlib import Path


def test_load_benchmark_results():
    """Test loading benchmark results from JSON."""
    from scripts.compare_to_baselines import load_benchmark_results

    benchmark_data = [
        {
            "baseline_name": "kdtree",
            "test_problem": "plummer",
            "num_particles": 50,
            "accuracy": 0.083,
            "speed": 0.166,
            "energy_drift": 4.5,
            "trajectory_rmse": 120.6,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        bench_path = Path(tmpdir) / "benchmark_results.json"
        bench_path.write_text(json.dumps(benchmark_data))

        loaded = load_benchmark_results(bench_path)
        assert loaded == benchmark_data


def test_filter_benchmark_by_particles():
    """Test filtering benchmarks by particle count."""
    from scripts.compare_to_baselines import filter_benchmark_by_particles

    benchmarks = [
        {"num_particles": 10, "baseline_name": "kdtree"},
        {"num_particles": 50, "baseline_name": "kdtree"},
        {"num_particles": 50, "baseline_name": "direct_nbody"},
        {"num_particles": 100, "baseline_name": "kdtree"},
    ]

    filtered = filter_benchmark_by_particles(benchmarks, 50)

    assert len(filtered) == 2
    assert all(b["num_particles"] == 50 for b in filtered)


def test_calculate_fitness():
    """Test fitness calculation (accuracy / speed)."""
    from scripts.compare_to_baselines import calculate_fitness

    fitness = calculate_fitness(accuracy=0.8, speed=0.002)
    assert fitness == 400.0

    fitness = calculate_fitness(accuracy=1.0, speed=0.05)
    assert fitness == 20.0


def test_calculate_fitness_zero_speed():
    """Test fitness with zero speed (should return inf)."""
    from scripts.compare_to_baselines import calculate_fitness

    fitness = calculate_fitness(accuracy=0.8, speed=0.0)
    assert fitness == float("inf")


def test_create_comparison_row():
    """Test creating a comparison table row."""
    from scripts.compare_to_baselines import create_comparison_row

    data = {
        "name": "KDTree",
        "accuracy": 0.0835,
        "speed": 0.166,
        "fitness": 502.4,
        "energy_drift": 4.5,
        "trajectory_rmse": 120.6,
    }

    row = create_comparison_row(data)

    assert "KDTree" in row
    assert "0.0835" in row
    assert "0.166" in row
    assert "502.4" in row or "502.40" in row
    assert "4.5" in row or "4.50" in row
    assert "120.6" in row or "120.60" in row


def test_create_comparison_table():
    """Test creating full comparison table."""
    from scripts.compare_to_baselines import create_comparison_table

    rows = [
        {
            "name": "Direct N-body",
            "accuracy": 1.0,
            "speed": 0.055,
            "fitness": 18.18,
            "energy_drift": 2.09,
            "trajectory_rmse": 0.0,
        },
        {
            "name": "KDTree",
            "accuracy": 0.083,
            "speed": 0.166,
            "fitness": 0.50,
            "energy_drift": 4.5,
            "trajectory_rmse": 120.6,
        },
    ]

    table = create_comparison_table(rows)

    # Check table structure
    assert "| Model" in table
    assert "| Accuracy" in table
    assert "| Speed" in table
    assert "| Fitness" in table
    assert "| Energy Drift" in table
    assert "| Trajectory RMSE" in table

    # Check data rows
    assert "Direct N-body" in table
    assert "KDTree" in table
    assert "1.0000" in table
    assert "0.0830" in table


def test_format_comparison_summary():
    """Test formatting comparison summary."""
    from scripts.compare_to_baselines import format_comparison_summary

    evolved = {"name": "Best Evolved", "fitness": 450.0, "accuracy": 0.75, "speed": 0.00167}
    kdtree = {"name": "KDTree", "fitness": 500.0, "accuracy": 0.083, "speed": 0.166}
    direct = {"name": "Direct N-body", "fitness": 18.18, "accuracy": 1.0, "speed": 0.055}

    summary = format_comparison_summary(evolved, kdtree, direct)

    assert "Best Evolved" in summary
    assert "KDTree" in summary
    assert "Direct N-body" in summary
    assert "450.0" in summary or "450.00" in summary


def test_save_comparison_markdown():
    """Test saving comparison to markdown file."""
    from scripts.compare_to_baselines import save_comparison_markdown

    comparison_text = "# Phase 3 Comparison\n\n## Results\n\nTest data"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "comparison.md"
        save_comparison_markdown(comparison_text, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "Phase 3 Comparison" in content
        assert "Test data" in content


def test_parse_model_metrics():
    """Test parsing metrics from model dict."""
    from scripts.compare_to_baselines import parse_model_metrics

    model = {
        "civ_id": "civ_3_7",
        "fitness": 12345.67,
        "accuracy": 0.85,
        "speed": 0.0025,
    }

    metrics = parse_model_metrics(model, name="Best Evolved")

    assert metrics["name"] == "Best Evolved"
    assert metrics["accuracy"] == 0.85
    assert metrics["speed"] == 0.0025
    assert metrics["fitness"] == 12345.67


def test_parse_model_metrics_missing_fields():
    """Test parsing with missing optional fields."""
    from scripts.compare_to_baselines import parse_model_metrics

    model = {"civ_id": "civ_0_0", "fitness": 100.0}

    metrics = parse_model_metrics(model, name="Test Model")

    assert metrics["name"] == "Test Model"
    assert metrics["fitness"] == 100.0
    assert "accuracy" in metrics  # Should have default or None
    assert "speed" in metrics


def test_find_baseline_by_name():
    """Test finding specific baseline from results."""
    from scripts.compare_to_baselines import find_baseline_by_name

    benchmarks = [
        {"baseline_name": "kdtree", "num_particles": 50, "accuracy": 0.083},
        {"baseline_name": "direct_nbody", "num_particles": 50, "accuracy": 1.0},
    ]

    kdtree = find_baseline_by_name(benchmarks, "kdtree", num_particles=50)
    assert kdtree is not None
    assert kdtree["baseline_name"] == "kdtree"
    assert kdtree["accuracy"] == 0.083

    direct = find_baseline_by_name(benchmarks, "direct_nbody", num_particles=50)
    assert direct is not None
    assert direct["accuracy"] == 1.0


def test_find_baseline_by_name_not_found():
    """Test handling missing baseline."""
    from scripts.compare_to_baselines import find_baseline_by_name

    benchmarks = [{"baseline_name": "kdtree", "num_particles": 50}]

    result = find_baseline_by_name(benchmarks, "missing_baseline", num_particles=50)
    assert result is None
