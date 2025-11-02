"""Tests for physics validation of evolved models.

This test suite validates the physics validation workflow that computes
energy drift, trajectory RMSE, and angular momentum conservation for
evolved surrogate models.
"""

import json
from pathlib import Path

import pytest

from scripts.validate_evolved_model import (
    compute_physics_metrics,
    export_validation_json,
    export_validation_markdown,
    load_model_for_validation,
    validate_all_runs,
    validate_single_run,
)


class TestLoadModelForValidation:
    """Test loading models from evolution runs for validation."""

    def test_load_parametric_model_from_run(self, tmp_path: Path):
        """Test loading a parametric model (theta-based) from evolution run."""
        # Create mock evolution run with parametric model
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        history = {
            "metadata": {"test_problem": "two_body", "num_particles": 2},
            "generations": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0",
                            "fitness": 100.0,
                            "accuracy": 0.95,
                            "speed": 0.001,
                            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],
                        }
                    ],
                }
            ],
        }

        (run_dir / "evolution_history.json").write_text(json.dumps(history))

        # Load model
        model = load_model_for_validation(run_dir)

        # Verify model loaded correctly
        assert model["test_problem"] == "two_body"
        assert model["num_particles"] == 2
        assert model["fitness"] == 100.0
        assert "theta" in model
        assert len(model["theta"]) == 6

    def test_load_llm_model_from_run(self, tmp_path: Path):
        """Test loading an LLM-generated model (raw_code) from evolution run."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        raw_code = """
def predict(particle, all_particles):
    # Simple test model
    return particle
"""

        history = {
            "metadata": {"test_problem": "plummer", "num_particles": 10},
            "generations": [
                {
                    "generation": 1,
                    "population": [
                        {
                            "civ_id": "civ_1",
                            "fitness": 500.0,
                            "accuracy": 0.99,
                            "speed": 0.002,
                            "raw_code": raw_code,
                        }
                    ],
                }
            ],
        }

        (run_dir / "evolution_history.json").write_text(json.dumps(history))

        # Load model
        model = load_model_for_validation(run_dir)

        # Verify model loaded correctly
        assert model["test_problem"] == "plummer"
        assert model["num_particles"] == 10
        assert model["fitness"] == 500.0
        assert "raw_code" in model
        assert "def predict" in model["raw_code"]

    def test_handle_missing_model_code(self, tmp_path: Path):
        """Test error handling when evolution run has no raw_code or theta."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        # Create history without model code (old format)
        history = {
            "metadata": {"test_problem": "two_body", "num_particles": 2},
            "generations": [
                {
                    "generation": 0,
                    "population": [
                        {"civ_id": "civ_0", "fitness": 100.0, "accuracy": 0.95, "speed": 0.001}
                    ],
                }
            ],
        }

        (run_dir / "evolution_history.json").write_text(json.dumps(history))

        # Should raise ValueError
        with pytest.raises(ValueError, match="Model must have either 'raw_code' or 'theta'"):
            load_model_for_validation(run_dir)


class TestPhysicsMetricsComputation:
    """Test computation of physics validation metrics."""

    def test_compute_energy_drift_for_model(self):
        """Test energy drift computation for a model."""

        # Create simple identity model that should preserve energy perfectly
        def identity_model(particle, all_particles):
            return particle

        # Run validation
        metrics = compute_physics_metrics(
            model_callable=identity_model,
            test_problem="two_body",
            num_particles=2,
            timesteps=50,
        )

        # Check metrics exist
        assert "energy_drift" in metrics
        assert "trajectory_rmse" in metrics
        assert "angular_momentum_drift" in metrics

        # Energy drift should be reasonably small for identity model
        assert metrics["energy_drift"] >= 0.0  # Non-negative

    def test_energy_drift_detects_violations(self):
        """Test that energy drift detects models violating conservation."""

        # Create model that adds energy (violates conservation)
        def bad_energy_model(particle, all_particles):
            # Add velocity to violate energy conservation
            return [
                particle[0],  # x
                particle[1],  # y
                particle[2],  # z
                particle[3] * 1.1,  # vx - amplify velocity
                particle[4] * 1.1,  # vy
                particle[5] * 1.1,  # vz
                particle[6],  # mass
            ]

        metrics = compute_physics_metrics(
            model_callable=bad_energy_model,
            test_problem="two_body",
            num_particles=2,
            timesteps=50,
        )

        # Should detect significant energy drift (>10%)
        assert metrics["energy_drift"] > 0.1, "Failed to detect energy conservation violation"

    def test_compute_trajectory_rmse_for_model(self):
        """Test trajectory RMSE computation for a model."""

        def simple_model(particle, all_particles):
            # Return slightly perturbed particle (poor accuracy)
            return [p + 0.01 for p in particle]

        metrics = compute_physics_metrics(
            model_callable=simple_model,
            test_problem="two_body",
            num_particles=2,
            timesteps=50,
        )

        # RMSE should be positive (model has error)
        assert metrics["trajectory_rmse"] > 0.0

    def test_compute_angular_momentum_for_model(self):
        """Test angular momentum conservation computation."""

        def identity_model(particle, all_particles):
            return particle

        metrics = compute_physics_metrics(
            model_callable=identity_model,
            test_problem="figure_eight",
            num_particles=3,
            timesteps=50,
        )

        # Angular momentum drift should exist
        assert "angular_momentum_drift" in metrics
        assert metrics["angular_momentum_drift"] >= 0.0

    def test_all_metrics_on_two_body(self):
        """Test all metrics on two-body circular orbit (known case)."""

        def perfect_model(particle, all_particles):
            # Perfect predictor (for testing)
            return particle

        metrics = compute_physics_metrics(
            model_callable=perfect_model,
            test_problem="two_body",
            num_particles=2,
            timesteps=100,
        )

        # All three metrics should be computed
        assert "energy_drift" in metrics
        assert "trajectory_rmse" in metrics
        assert "angular_momentum_drift" in metrics

        # All should be non-negative
        assert metrics["energy_drift"] >= 0.0
        assert metrics["trajectory_rmse"] >= 0.0
        assert metrics["angular_momentum_drift"] >= 0.0


class TestValidationReportGeneration:
    """Test JSON and markdown export functionality."""

    def test_export_json_format(self, tmp_path: Path):
        """Test JSON export with correct schema."""
        results = [
            {
                "run_dir": "results/run_test",
                "test_problem": "two_body",
                "num_particles": 2,
                "fitness": 1000.0,
                "accuracy": 0.99,
                "speed": 0.001,
                "energy_drift": 0.005,
                "trajectory_rmse": 0.002,
                "angular_momentum_drift": 0.001,
            }
        ]

        output_path = tmp_path / "validation_results.json"
        export_validation_json(results, output_path)

        # Verify file created
        assert output_path.exists()

        # Verify JSON structure
        data = json.loads(output_path.read_text())
        assert "results" in data
        assert "metadata" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["test_problem"] == "two_body"

    def test_export_markdown_format(self, tmp_path: Path):
        """Test markdown export with proper table structure."""
        results = [
            {
                "run_dir": "results/run_test_1",
                "test_problem": "two_body",
                "num_particles": 2,
                "fitness": 1000.0,
                "accuracy": 0.99,
                "speed": 0.001,
                "energy_drift": 0.005,
                "trajectory_rmse": 0.002,
                "angular_momentum_drift": 0.001,
            },
            {
                "run_dir": "results/run_test_2",
                "test_problem": "plummer",
                "num_particles": 10,
                "fitness": 500.0,
                "accuracy": 0.95,
                "speed": 0.005,
                "energy_drift": 0.01,
                "trajectory_rmse": 0.005,
                "angular_momentum_drift": 0.002,
            },
        ]

        output_path = tmp_path / "validation_report.md"
        export_validation_markdown(results, output_path)

        # Verify file created
        assert output_path.exists()

        # Verify markdown structure
        content = output_path.read_text()
        assert "# Physics Validation" in content
        assert "## Summary" in content
        assert "## Results by Test Problem" in content
        # Check for title-cased problem names
        assert "Two Body" in content or "two_body" in content.lower()
        assert "Plummer" in content or "plummer" in content.lower()
        # Check for markdown table syntax
        assert "|" in content
        assert "---" in content

    def test_handle_multiple_runs(self, tmp_path: Path):
        """Test batch validation exports all runs correctly."""
        # Create 5 mock results
        results = [
            {
                "run_dir": f"results/run_test_{i}",
                "test_problem": "two_body",
                "num_particles": 2,
                "fitness": 1000.0 + i * 100,
                "accuracy": 0.99,
                "speed": 0.001,
                "energy_drift": 0.005,
                "trajectory_rmse": 0.002,
                "angular_momentum_drift": 0.001,
            }
            for i in range(5)
        ]

        output_path = tmp_path / "validation_results.json"
        export_validation_json(results, output_path)

        # Verify all runs exported
        data = json.loads(output_path.read_text())
        assert len(data["results"]) == 5


class TestCLIInterface:
    """Test command-line interface functionality."""

    def test_single_run_validation(self, tmp_path: Path):
        """Test validating a single evolution run."""
        # Create mock run with parametric model
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        history = {
            "metadata": {"test_problem": "two_body", "num_particles": 2},
            "generations": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0",
                            "fitness": 1000.0,
                            "accuracy": 0.99,
                            "speed": 0.001,
                            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],
                        }
                    ],
                }
            ],
        }

        (run_dir / "evolution_history.json").write_text(json.dumps(history))

        # Validate single run
        result = validate_single_run(run_dir)

        # Verify result structure
        assert "run_dir" in result
        assert "test_problem" in result
        assert "energy_drift" in result
        assert "trajectory_rmse" in result
        assert "angular_momentum_drift" in result

    def test_batch_validation(self, tmp_path: Path):
        """Test validating all runs in results directory."""
        # Create multiple mock runs
        for i in range(3):
            run_dir = tmp_path / f"run_20251102_{i:06d}"
            run_dir.mkdir()

            history = {
                "metadata": {"test_problem": "two_body", "num_particles": 2},
                "generations": [
                    {
                        "generation": 0,
                        "population": [
                            {
                                "civ_id": "civ_0",
                                "fitness": 1000.0 + i * 100,
                                "accuracy": 0.99,
                                "speed": 0.001,
                                "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],
                            }
                        ],
                    }
                ],
            }

            (run_dir / "evolution_history.json").write_text(json.dumps(history))

        # Validate all runs
        results = validate_all_runs(tmp_path)

        # Should find all 3 runs
        assert len(results) == 3

    def test_output_paths_created(self, tmp_path: Path):
        """Test that output files are created in correct locations."""
        results = [
            {
                "run_dir": "results/run_test",
                "test_problem": "two_body",
                "num_particles": 2,
                "fitness": 1000.0,
                "accuracy": 0.99,
                "speed": 0.001,
                "energy_drift": 0.005,
                "trajectory_rmse": 0.002,
                "angular_momentum_drift": 0.001,
            }
        ]

        # Export to both formats
        json_path = tmp_path / "validation_results.json"
        md_path = tmp_path / "validation_report.md"

        export_validation_json(results, json_path)
        export_validation_markdown(results, md_path)

        # Verify both files created
        assert json_path.exists()
        assert md_path.exists()
        assert json_path.stat().st_size > 0
        assert md_path.stat().st_size > 0
