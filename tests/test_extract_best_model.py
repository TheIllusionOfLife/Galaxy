"""Tests for scripts/extract_best_model.py

Tests the extraction of best model from evolution history.
"""

import json
import tempfile
from pathlib import Path

import pytest


def test_find_best_model_from_history():
    """Test finding the best model across all generations."""
    from scripts.extract_best_model import find_best_model

    history = {
        "generations": [
            {
                "generation": 0,
                "population": [
                    {
                        "civ_id": "civ_0_0",
                        "fitness": 100.0,
                        "accuracy": 0.5,
                        "speed": 0.005,
                    },
                    {
                        "civ_id": "civ_0_1",
                        "fitness": 200.0,
                        "accuracy": 0.6,
                        "speed": 0.003,
                    },
                ],
            },
            {
                "generation": 1,
                "population": [
                    {
                        "civ_id": "civ_1_0",
                        "fitness": 300.0,
                        "accuracy": 0.8,
                        "speed": 0.0027,
                    },
                    {
                        "civ_id": "civ_1_1",
                        "fitness": 150.0,
                        "accuracy": 0.45,
                        "speed": 0.003,
                    },
                ],
            },
        ]
    }

    best = find_best_model(history)

    assert best["civ_id"] == "civ_1_0"
    assert best["fitness"] == 300.0
    assert best["accuracy"] == 0.8
    assert best["speed"] == 0.0027
    assert best["generation"] == 1


def test_find_best_model_single_generation():
    """Test with single generation."""
    from scripts.extract_best_model import find_best_model

    history = {
        "generations": [
            {
                "generation": 0,
                "population": [
                    {"civ_id": "civ_0_0", "fitness": 100.0},
                    {"civ_id": "civ_0_1", "fitness": 50.0},
                ],
            }
        ]
    }

    best = find_best_model(history)
    assert best["civ_id"] == "civ_0_0"
    assert best["fitness"] == 100.0
    assert best["generation"] == 0


def test_find_best_model_with_inf():
    """Test handling infinite fitness (validation failure)."""
    from scripts.extract_best_model import find_best_model

    history = {
        "generations": [
            {
                "generation": 0,
                "population": [
                    {"civ_id": "civ_0_0", "fitness": float("inf")},
                    {"civ_id": "civ_0_1", "fitness": 100.0},
                ],
            }
        ]
    }

    best = find_best_model(history)
    assert best["civ_id"] == "civ_0_1"
    assert best["fitness"] == 100.0


def test_find_best_model_all_inf():
    """Test when all models have infinite fitness."""
    from scripts.extract_best_model import find_best_model

    history = {
        "generations": [
            {
                "generation": 0,
                "population": [
                    {"civ_id": "civ_0_0", "fitness": float("inf")},
                    {"civ_id": "civ_0_1", "fitness": float("inf")},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="No valid models found"):
        find_best_model(history)


def test_extract_model_code_with_raw_code():
    """Test extracting raw_code field."""
    from scripts.extract_best_model import extract_model_code

    model = {
        "civ_id": "civ_0_0",
        "raw_code": "def predict(particle, all_particles):\n    return particle",
    }

    code = extract_model_code(model)
    assert "def predict" in code
    assert "return particle" in code


def test_extract_model_code_with_theta():
    """Test extracting parametric model (theta parameters)."""
    from scripts.extract_best_model import extract_model_code

    model = {"civ_id": "civ_0_0", "theta": [0.5, 1.0, 2.0], "description": "parametric"}

    code = extract_model_code(model)
    assert code is None  # Parametric models don't have raw code


def test_extract_model_code_missing():
    """Test handling model with no code."""
    from scripts.extract_best_model import extract_model_code

    model = {"civ_id": "civ_0_0", "description": "test"}

    code = extract_model_code(model)
    assert code is None


def test_save_model_to_file():
    """Test saving model code to file."""
    from scripts.extract_best_model import save_model_to_file

    code = "def predict(particle, all_particles):\n    return particle\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "best_model.py"
        save_model_to_file(code, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "def predict" in content
        assert "return particle" in content


def test_load_evolution_history():
    """Test loading evolution history from JSON."""
    from scripts.extract_best_model import load_evolution_history

    history_data = {
        "generations": [{"generation": 0, "population": [{"civ_id": "civ_0_0", "fitness": 100.0}]}]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "evolution_history.json"
        history_path.write_text(json.dumps(history_data))

        loaded = load_evolution_history(history_path)
        assert loaded == history_data


def test_load_evolution_history_invalid_json():
    """Test handling invalid JSON."""
    from scripts.extract_best_model import load_evolution_history

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "bad.json"
        history_path.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_evolution_history(history_path)


def test_format_model_summary():
    """Test formatting model summary for display."""
    from scripts.extract_best_model import format_model_summary

    model = {
        "civ_id": "civ_3_7",
        "generation": 3,
        "fitness": 12345.67,
        "accuracy": 0.85,
        "speed": 0.0025,
        "token_count": 456,
        "description": "gemini_gen3_crossover",
    }

    summary = format_model_summary(model)

    assert "civ_3_7" in summary
    assert "Generation: 3" in summary
    assert "12345.67" in summary
    assert "0.8500" in summary
    assert "0.0025" in summary
    assert "456" in summary
    assert "gemini_gen3_crossover" in summary


def test_format_model_summary_missing_fields():
    """Test summary with minimal fields."""
    from scripts.extract_best_model import format_model_summary

    model = {"civ_id": "civ_0_0", "generation": 0, "fitness": 100.0}

    summary = format_model_summary(model)

    assert "civ_0_0" in summary
    assert "Generation: 0" in summary
    assert "100.00" in summary
    # Should handle missing optional fields gracefully
