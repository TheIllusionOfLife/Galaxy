"""
Integration test for visualization with real API.

Runs a minimal evolution (2 generations, 3 population) to verify:
- History tracking works
- Plots are generated
- JSON export is complete

These tests use the real Gemini API and are marked with @pytest.mark.integration.
Skip them during normal development with: pytest -m "not integration"
"""

import json
import os
from datetime import datetime

import pytest

from config import settings
from gemini_client import CostTracker, GeminiClient
from prototype import CosmologyCrucible, EvolutionaryEngine
from visualization import export_history_json, generate_all_plots

# Skip all tests in this module if no API key available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here",
        reason="No valid GOOGLE_API_KEY in environment",
    ),
]


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary results directory for test outputs."""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    return results_dir


class TestVisualizationIntegration:
    """Test visualization system with real evolution run."""

    def test_full_visualization_pipeline(self, temp_results_dir):
        """Test complete visualization pipeline with mini evolution."""
        # Configuration
        num_generations = 2
        population_size = 3

        # Set up components
        gemini_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        cost_tracker = CostTracker(max_cost_usd=1.0)

        crucible = CosmologyCrucible()
        engine = EvolutionaryEngine(
            crucible,
            population_size=population_size,
            gemini_client=gemini_client,
            cost_tracker=cost_tracker,
        )

        # Run evolution
        engine.initialize_population()
        for _gen in range(num_generations):
            engine.run_evolutionary_cycle()

        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = temp_results_dir / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export history as JSON
        history_path = output_dir / "evolution_history.json"
        export_history_json(engine.history, str(history_path))

        # Generate all plots
        generate_all_plots(engine.history, cost_tracker, str(output_dir))

        # Verify all required files exist
        required_files = [
            "evolution_history.json",
            "fitness_progression.png",
            "accuracy_vs_speed.png",
            "cost_progression.png",
        ]

        for filename in required_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"

        # Verify JSON structure
        with open(output_dir / "evolution_history.json") as f:
            data = json.load(f)

        assert "history" in data, "JSON missing 'history' key"
        assert "summary" in data, "JSON missing 'summary' key"
        assert len(data["history"]) == num_generations, (
            f"Expected {num_generations} generations, got {len(data['history'])}"
        )

        # Verify summary statistics
        summary = data["summary"]
        assert summary["total_generations"] == num_generations
        assert summary["total_models_evaluated"] == num_generations * population_size, (
            "Incorrect model count"
        )
        assert summary["best_overall_fitness"] > 0, "Invalid best fitness"
        assert summary["avg_overall_fitness"] > 0, "Invalid avg fitness"

        # Verify history entries have required fields
        for gen_data in data["history"]:
            assert "generation" in gen_data
            assert "population" in gen_data
            assert len(gen_data["population"]) == population_size

            for model in gen_data["population"]:
                assert "civilization_id" in model
                assert "fitness" in model
                assert "accuracy" in model
                assert "speed" in model
                # Check for finite values
                assert model["fitness"] > 0 and model["fitness"] < float("inf"), (
                    "Invalid fitness value"
                )
                assert 0 <= model["accuracy"] <= 1, "Invalid accuracy value"
                assert model["speed"] > 0, "Invalid speed value"
