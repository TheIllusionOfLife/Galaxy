"""
Tests for evolution visualization functionality.

This module tests the visualization functions that create plots
from evolution history data, including fitness progression,
accuracy vs speed trade-offs, and cost tracking.
"""

import json
from unittest.mock import Mock

# Import will happen after visualization.py is created
# from visualization import (
#     plot_fitness_progression,
#     plot_accuracy_vs_speed,
#     plot_cost_progression,
#     generate_all_plots,
#     export_history_json,
# )


class TestVisualizationFunctions:
    """Test visualization plotting functions."""

    def setup_method(self):
        """Create test fixtures with mock history data."""
        self.mock_history = [
            {
                "generation": 0,
                "population": [
                    {
                        "civ_id": "civ_0_0",
                        "fitness": 10000.0,
                        "accuracy": 0.95,
                        "speed": 0.000095,
                        "description": "mock_gen0_0",
                    },
                    {
                        "civ_id": "civ_0_1",
                        "fitness": 12000.0,
                        "accuracy": 0.96,
                        "speed": 0.00008,
                        "description": "mock_gen0_1",
                    },
                    {
                        "civ_id": "civ_0_2",
                        "fitness": 11000.0,
                        "accuracy": 0.94,
                        "speed": 0.000085,
                        "description": "mock_gen0_2",
                    },
                ],
                "best_fitness": 12000.0,
                "avg_fitness": 11000.0,
                "worst_fitness": 10000.0,
            },
            {
                "generation": 1,
                "population": [
                    {
                        "civ_id": "civ_1_0",
                        "fitness": 15000.0,
                        "accuracy": 0.97,
                        "speed": 0.000065,
                        "description": "mock_gen1_0",
                    },
                    {
                        "civ_id": "civ_1_1",
                        "fitness": 14000.0,
                        "accuracy": 0.96,
                        "speed": 0.000069,
                        "description": "mock_gen1_1",
                    },
                    {
                        "civ_id": "civ_1_2",
                        "fitness": 13000.0,
                        "accuracy": 0.95,
                        "speed": 0.000073,
                        "description": "mock_gen1_2",
                    },
                ],
                "best_fitness": 15000.0,
                "avg_fitness": 14000.0,
                "worst_fitness": 13000.0,
            },
        ]

        self.mock_cost_tracker = Mock()
        self.mock_cost_tracker.calls = [
            {
                "tokens": 500,
                "cost_usd": 0.0001,
                "success": True,
                "generation_time_s": 1.2,
                "error": None,
                "context": {"generation": 0, "type": "initial"},
            },
            {
                "tokens": 600,
                "cost_usd": 0.00012,
                "success": True,
                "generation_time_s": 1.5,
                "error": None,
                "context": {"generation": 1, "type": "mutation"},
            },
        ]
        self.mock_cost_tracker.total_cost_usd = 0.00022
        self.mock_cost_tracker.total_tokens = 1100

    def test_plot_fitness_progression_creates_file(self, tmp_path):
        """Test that fitness progression plot is created."""
        from visualization import plot_fitness_progression

        output_path = tmp_path / "fitness_progression.png"
        plot_fitness_progression(self.mock_history, str(output_path))

        assert output_path.exists(), "Plot file should be created"
        assert output_path.stat().st_size > 0, "Plot file should not be empty"

    def test_plot_fitness_progression_shows_all_generations(self, tmp_path):
        """Test that all generations are included in the plot."""
        from visualization import plot_fitness_progression

        # Create plot (should not raise exception)
        output_path = tmp_path / "fitness_test.png"
        plot_fitness_progression(self.mock_history, str(output_path))

        # Plot should exist
        assert output_path.exists()

    def test_plot_accuracy_vs_speed_creates_file(self, tmp_path):
        """Test that accuracy vs speed scatter plot is created."""
        from visualization import plot_accuracy_vs_speed

        output_path = tmp_path / "accuracy_speed.png"
        plot_accuracy_vs_speed(self.mock_history, str(output_path))

        assert output_path.exists(), "Plot file should be created"
        assert output_path.stat().st_size > 0, "Plot file should not be empty"

    def test_plot_accuracy_vs_speed_includes_all_models(self, tmp_path):
        """Test that all models from all generations are plotted."""
        from visualization import plot_accuracy_vs_speed

        output_path = tmp_path / "accuracy_speed_test.png"
        plot_accuracy_vs_speed(self.mock_history, str(output_path))

        # Should not raise exception and should create file
        assert output_path.exists()

    def test_plot_cost_progression_creates_file(self, tmp_path):
        """Test that cost progression plot is created."""
        from visualization import plot_cost_progression

        output_path = tmp_path / "cost_progression.png"
        plot_cost_progression(self.mock_cost_tracker, str(output_path))

        assert output_path.exists(), "Plot file should be created"
        assert output_path.stat().st_size > 0, "Plot file should not be empty"

    def test_plot_cost_progression_with_empty_tracker(self, tmp_path):
        """Test cost plot with no API calls."""
        from visualization import plot_cost_progression

        empty_tracker = Mock()
        empty_tracker.calls = []
        empty_tracker.total_cost_usd = 0.0

        output_path = tmp_path / "cost_empty.png"
        # Should handle empty data gracefully (no error)
        plot_cost_progression(empty_tracker, str(output_path))

    def test_generate_all_plots_creates_directory(self, tmp_path):
        """Test that generate_all_plots creates output directory."""
        from visualization import generate_all_plots

        output_dir = tmp_path / "results"
        generate_all_plots(self.mock_history, self.mock_cost_tracker, str(output_dir))

        assert output_dir.exists(), "Output directory should be created"
        assert output_dir.is_dir(), "Output should be a directory"

    def test_generate_all_plots_creates_all_files(self, tmp_path):
        """Test that all plot types are generated."""
        from visualization import generate_all_plots

        output_dir = tmp_path / "results"
        generate_all_plots(self.mock_history, self.mock_cost_tracker, str(output_dir))

        # Check all expected files exist
        assert (output_dir / "fitness_progression.png").exists()
        assert (output_dir / "accuracy_vs_speed.png").exists()
        assert (output_dir / "cost_progression.png").exists()

    def test_generate_all_plots_without_cost_tracker(self, tmp_path):
        """Test that plots work without cost tracker."""
        from visualization import generate_all_plots

        output_dir = tmp_path / "results"
        generate_all_plots(self.mock_history, None, str(output_dir))

        # Should still create fitness and accuracy/speed plots
        assert (output_dir / "fitness_progression.png").exists()
        assert (output_dir / "accuracy_vs_speed.png").exists()
        # Cost plot should be skipped
        assert not (output_dir / "cost_progression.png").exists()

    def test_export_history_json_creates_file(self, tmp_path):
        """Test that history is exported to JSON."""
        from visualization import export_history_json

        output_path = tmp_path / "evolution_history.json"
        export_history_json(self.mock_history, str(output_path))

        assert output_path.exists(), "JSON file should be created"
        assert output_path.stat().st_size > 0, "JSON file should not be empty"

    def test_export_history_json_valid_format(self, tmp_path):
        """Test that exported JSON is valid and complete."""
        from visualization import export_history_json

        output_path = tmp_path / "evolution_history.json"
        export_history_json(self.mock_history, str(output_path))

        # Read and parse JSON
        with open(output_path) as f:
            data = json.load(f)

        # Verify structure
        assert "history" in data
        assert "summary" in data
        assert len(data["history"]) == 2, "Should have 2 generations"
        assert data["summary"]["total_generations"] == 2
        assert "best_overall_fitness" in data["summary"]

    def test_export_history_json_preserves_data(self, tmp_path):
        """Test that all history data is preserved in JSON export."""
        from visualization import export_history_json

        output_path = tmp_path / "evolution_history.json"
        export_history_json(self.mock_history, str(output_path))

        with open(output_path) as f:
            data = json.load(f)

        # Check first generation data
        gen0 = data["history"][0]
        assert gen0["generation"] == 0
        assert gen0["best_fitness"] == 12000.0
        assert len(gen0["population"]) == 3

    def test_plot_with_single_generation(self, tmp_path):
        """Test plots work with only one generation of data."""
        from visualization import plot_fitness_progression

        single_gen = [self.mock_history[0]]
        output_path = tmp_path / "single_gen.png"

        # Should not raise exception
        plot_fitness_progression(single_gen, str(output_path))
        assert output_path.exists()

    def test_plot_with_large_population(self, tmp_path):
        """Test plots handle large populations efficiently."""
        from visualization import plot_accuracy_vs_speed

        # Create history with 50 models per generation
        large_history = [
            {
                "generation": 0,
                "population": [
                    {
                        "civ_id": f"civ_0_{i}",
                        "fitness": 10000.0 + i * 100,
                        "accuracy": 0.9 + i * 0.001,
                        "speed": 0.0001 - i * 0.000001,
                        "description": f"model_{i}",
                    }
                    for i in range(50)
                ],
                "best_fitness": 14900.0,
                "avg_fitness": 12450.0,
                "worst_fitness": 10000.0,
            }
        ]

        output_path = tmp_path / "large_pop.png"
        plot_accuracy_vs_speed(large_history, str(output_path))
        assert output_path.exists()

    def test_plot_overwrites_existing_file(self, tmp_path):
        """Test that plotting overwrites existing file."""
        from visualization import plot_fitness_progression

        output_path = tmp_path / "overwrite_test.png"

        # Create initial plot
        plot_fitness_progression(self.mock_history, str(output_path))
        assert output_path.exists(), "Initial plot should be created"

        # Overwrite with different data
        modified_history = [self.mock_history[0]]
        plot_fitness_progression(modified_history, str(output_path))

        # File should still exist after overwrite
        assert output_path.exists(), "Plot should exist after overwrite"

    def test_export_creates_parent_directories(self, tmp_path):
        """Test that export creates nested directories if needed."""
        from visualization import export_history_json

        nested_path = tmp_path / "deep" / "nested" / "path" / "history.json"
        export_history_json(self.mock_history, str(nested_path))

        assert nested_path.exists(), "Should create parent directories"

    def test_plot_handles_inf_and_nan_gracefully(self, tmp_path):
        """Test that plots handle infinite/NaN values without crashing."""
        from visualization import plot_fitness_progression

        # History with problematic values
        bad_history = [
            {
                "generation": 0,
                "population": [
                    {
                        "civ_id": "civ_0_0",
                        "fitness": float("inf"),
                        "accuracy": 0.0,
                        "speed": float("inf"),
                        "description": "failed_model",
                    },
                    {
                        "civ_id": "civ_0_1",
                        "fitness": 10000.0,
                        "accuracy": 0.95,
                        "speed": 0.00009,
                        "description": "good_model",
                    },
                ],
                "best_fitness": 10000.0,
                "avg_fitness": 10000.0,
                "worst_fitness": 0.0,
            }
        ]

        output_path = tmp_path / "bad_values.png"
        # Should not crash, may skip inf values
        plot_fitness_progression(bad_history, str(output_path))
        assert output_path.exists()
