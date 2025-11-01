"""Tests for cross-problem generalization analysis."""

import json
import math
from pathlib import Path

import pytest

from scripts.cross_validate_problems import (
    CrossValidationResult,
    compute_generalization_penalty,
    create_cross_validation_matrix,
    evaluate_model_on_problem,
    export_results_json,
    export_results_markdown,
    load_best_model,
)


class TestLoadBestModel:
    """Tests for loading best model from evolution run."""

    def test_load_best_model_from_valid_run(self, tmp_path: Path):
        """Test loading best model from valid evolution history."""
        history = {
            "metadata": {"test_problem": "two_body", "num_particles": 2},
            "history": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0_0",
                            "fitness": 100.0,
                            "accuracy": 0.9,
                            "speed": 0.009,
                        },
                        {
                            "civ_id": "civ_0_1",
                            "fitness": 200.0,
                            "accuracy": 0.95,
                            "speed": 0.00475,
                        },
                    ],
                }
            ],
        }

        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        history_file = run_dir / "evolution_history.json"
        history_file.write_text(json.dumps(history))

        result = load_best_model(run_dir)

        assert result["civ_id"] == "civ_0_1"
        assert result["fitness"] == 200.0
        assert result["test_problem"] == "two_body"
        assert result["num_particles"] == 2

    def test_load_best_model_skips_infinite_fitness(self, tmp_path: Path):
        """Test that infinite fitness values are skipped."""
        history = {
            "metadata": {"test_problem": "plummer", "num_particles": 20},
            "history": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0_0",
                            "fitness": float("inf"),
                            "accuracy": 0.0,
                            "speed": 0.0,
                        },
                        {
                            "civ_id": "civ_0_1",
                            "fitness": 150.0,
                            "accuracy": 0.9,
                            "speed": 0.006,
                        },
                    ],
                }
            ],
        }

        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        history_file = run_dir / "evolution_history.json"
        history_file.write_text(json.dumps(history))

        result = load_best_model(run_dir)

        assert result["civ_id"] == "civ_0_1"
        assert result["fitness"] == 150.0

    def test_load_best_model_missing_file(self, tmp_path: Path):
        """Test error handling for missing evolution_history.json."""
        run_dir = tmp_path / "run_nonexistent"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_best_model(run_dir)

    def test_load_best_model_missing_metadata(self, tmp_path: Path):
        """Test handling of missing metadata in evolution history."""
        history = {
            "history": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0_0",
                            "fitness": 100.0,
                            "accuracy": 0.9,
                            "speed": 0.009,
                        }
                    ],
                }
            ]
        }

        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        history_file = run_dir / "evolution_history.json"
        history_file.write_text(json.dumps(history))

        with pytest.raises(KeyError, match="metadata"):
            load_best_model(run_dir)


class TestEvaluateModelOnProblem:
    """Tests for cross-problem model evaluation."""

    def test_evaluate_parametric_model_on_same_problem(self):
        """Test evaluating parametric model on its training problem."""
        model = {
            "civ_id": "civ_0_0",
            "fitness": 500.0,
            "accuracy": 0.95,
            "speed": 0.0019,
            "test_problem": "two_body",
            "num_particles": 2,
            "description": "parametric_model",
            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],  # 6 params for make_parametric_surrogate
        }

        result = evaluate_model_on_problem(model, "two_body", 2)

        assert "fitness" in result
        assert "accuracy" in result
        assert "speed" in result
        assert isinstance(result["fitness"], (int, float))
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["speed"] > 0

    def test_evaluate_model_on_different_problem(self):
        """Test evaluating model on a different test problem."""
        model = {
            "civ_id": "civ_0_0",
            "fitness": 500.0,
            "accuracy": 0.95,
            "speed": 0.0019,
            "test_problem": "two_body",
            "num_particles": 2,
            "description": "parametric_model",
            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],  # 6 params for make_parametric_surrogate
        }

        # Evaluate two_body model on plummer problem
        result = evaluate_model_on_problem(model, "plummer", 20)

        assert "fitness" in result
        assert "accuracy" in result
        assert "speed" in result
        # Fitness may be different (possibly worse) on different problem
        assert isinstance(result["fitness"], (int, float))

    def test_evaluate_model_handles_particle_count_mismatch(self):
        """Test that evaluation works with different particle counts."""
        model = {
            "civ_id": "civ_0_0",
            "fitness": 500.0,
            "accuracy": 0.95,
            "speed": 0.0019,
            "test_problem": "plummer",
            "num_particles": 20,
            "description": "parametric_model",
            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],  # 6 params for make_parametric_surrogate
        }

        # Evaluate 20-particle model on 2-particle problem
        result = evaluate_model_on_problem(model, "two_body", 2)

        assert "fitness" in result
        assert "accuracy" in result
        assert "speed" in result

    def test_evaluate_model_invalid_problem(self):
        """Test error handling for invalid test problem."""
        model = {
            "civ_id": "civ_0_0",
            "fitness": 500.0,
            "test_problem": "two_body",
            "num_particles": 2,
            "theta": [1.0, 0.5, 0.1, 0.1, 1.0, 0.99],  # 6 params for make_parametric_surrogate
        }

        with pytest.raises(ValueError, match="Unknown test_problem"):
            evaluate_model_on_problem(model, "invalid_problem", 10)


class TestGeneralizationPenalty:
    """Tests for generalization penalty calculation."""

    def test_compute_penalty_no_change(self):
        """Test penalty when fitness unchanged (same problem)."""
        penalty = compute_generalization_penalty(1000.0, 1000.0)
        assert penalty == 0.0

    def test_compute_penalty_fitness_drop(self):
        """Test penalty when fitness decreases on new problem."""
        # Original fitness: 1000, new fitness: 500 = 50% drop
        penalty = compute_generalization_penalty(1000.0, 500.0)
        assert penalty == 50.0

    def test_compute_penalty_fitness_increase(self):
        """Test negative penalty when fitness increases (rare but possible)."""
        # Original fitness: 100, new fitness: 150 = -50% (improvement)
        penalty = compute_generalization_penalty(100.0, 150.0)
        assert penalty == -50.0

    def test_compute_penalty_zero_original_fitness(self):
        """Test handling of zero original fitness (division by zero)."""
        penalty = compute_generalization_penalty(0.0, 100.0)
        assert math.isinf(penalty)

    def test_compute_penalty_infinite_new_fitness(self):
        """Test handling of infinite new fitness (validation failure)."""
        penalty = compute_generalization_penalty(1000.0, float("inf"))
        # Should return inf penalty (complete failure)
        assert math.isinf(penalty) and penalty < 0  # Negative infinity


class TestCrossValidationMatrix:
    """Tests for creating full cross-validation matrix."""

    def test_create_matrix_three_problems(self, tmp_path: Path):
        """Test creating 3x3 cross-validation matrix."""
        # Create mock run directories
        problems = ["two_body", "figure_eight", "plummer"]
        particle_counts = [2, 3, 20]

        run_dirs = {}
        for problem, n_particles in zip(problems, particle_counts, strict=True):
            run_dir = tmp_path / f"run_{problem}"
            run_dir.mkdir()

            history = {
                "metadata": {"test_problem": problem, "num_particles": n_particles},
                "history": [
                    {
                        "generation": 0,
                        "population": [
                            {
                                "civ_id": f"civ_{problem}_0",
                                "fitness": 500.0,
                                "accuracy": 0.95,
                                "speed": 0.0019,
                                "theta": [
                                    1.0,
                                    0.5,
                                    0.1,
                                    0.1,
                                    1.0,
                                    0.99,
                                ],  # 6 params for make_parametric_surrogate
                            }
                        ],
                    }
                ],
            }

            history_file = run_dir / "evolution_history.json"
            history_file.write_text(json.dumps(history))
            run_dirs[problem] = run_dir

        results = create_cross_validation_matrix(run_dirs)

        # Should have 3x3 = 9 results
        assert len(results) == 9

        # Check diagonal (same problem) has zero or low penalty
        diagonal_results = [r for r in results if r.trained_on == r.tested_on]
        assert len(diagonal_results) == 3

        # Check all combinations present
        trained_problems = {r.trained_on for r in results}
        tested_problems = {r.tested_on for r in results}
        assert trained_problems == set(problems)
        assert tested_problems == set(problems)

    def test_create_matrix_includes_all_metrics(self, tmp_path: Path):
        """Test that matrix results include all required metrics."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        history = {
            "metadata": {"test_problem": "two_body", "num_particles": 2},
            "history": [
                {
                    "generation": 0,
                    "population": [
                        {
                            "civ_id": "civ_0_0",
                            "fitness": 500.0,
                            "accuracy": 0.95,
                            "speed": 0.0019,
                            "theta": [
                                1.0,
                                0.5,
                                0.1,
                                0.1,
                                1.0,
                                0.99,
                            ],  # 6 params for make_parametric_surrogate
                        }
                    ],
                }
            ],
        }

        history_file = run_dir / "evolution_history.json"
        history_file.write_text(json.dumps(history))

        results = create_cross_validation_matrix({"two_body": run_dir})

        assert len(results) == 1
        result = results[0]

        assert hasattr(result, "trained_on")
        assert hasattr(result, "tested_on")
        assert hasattr(result, "trained_particles")
        assert hasattr(result, "tested_particles")
        assert hasattr(result, "fitness")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "speed")
        assert hasattr(result, "generalization_penalty")


class TestExportResults:
    """Tests for exporting cross-validation results."""

    def test_export_markdown_table(self, tmp_path: Path):
        """Test exporting results as markdown table."""
        results = [
            CrossValidationResult(
                trained_on="two_body",
                tested_on="two_body",
                trained_particles=2,
                tested_particles=2,
                fitness=500.0,
                accuracy=0.95,
                speed=0.0019,
                generalization_penalty=0.0,
            ),
            CrossValidationResult(
                trained_on="two_body",
                tested_on="plummer",
                trained_particles=2,
                tested_particles=20,
                fitness=50.0,
                accuracy=0.7,
                speed=0.014,
                generalization_penalty=90.0,
            ),
        ]

        output_path = tmp_path / "cross_validation_matrix.md"
        export_results_markdown(results, output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Check for table header
        assert "Cross-Validation Matrix" in content
        assert "Trained On" in content or "trained_on" in content

        # Check for data values
        assert "two_body" in content
        assert "plummer" in content
        assert "500" in content or "500.0" in content

    def test_export_json_results(self, tmp_path: Path):
        """Test exporting results as JSON."""
        results = [
            CrossValidationResult(
                trained_on="two_body",
                tested_on="two_body",
                trained_particles=2,
                tested_particles=2,
                fitness=500.0,
                accuracy=0.95,
                speed=0.0019,
                generalization_penalty=0.0,
            )
        ]

        output_path = tmp_path / "cross_validation_results.json"
        export_results_json(results, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())

        assert "matrix" in data
        assert len(data["matrix"]) == 1
        assert data["matrix"][0]["trained_on"] == "two_body"
        assert data["matrix"][0]["fitness"] == 500.0

    def test_export_creates_directory_if_missing(self, tmp_path: Path):
        """Test that export creates output directory if it doesn't exist."""
        results = [
            CrossValidationResult(
                trained_on="two_body",
                tested_on="two_body",
                trained_particles=2,
                tested_particles=2,
                fitness=500.0,
                accuracy=0.95,
                speed=0.0019,
                generalization_penalty=0.0,
            )
        ]

        output_dir = tmp_path / "nonexistent" / "subdir"
        output_path = output_dir / "results.json"

        export_results_json(results, output_path)

        assert output_path.exists()
        assert output_dir.exists()
