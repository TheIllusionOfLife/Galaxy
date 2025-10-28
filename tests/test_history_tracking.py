"""
Tests for evolution history tracking functionality.

This module tests that the EvolutionaryEngine properly tracks
history data during evolution cycles, including fitness, accuracy,
speed, and population statistics per generation.
"""

import math

from prototype import CosmologyCrucible, EvolutionaryEngine, SurrogateGenome


class TestHistoryTracking:
    """Test evolution history tracking in EvolutionaryEngine."""

    def setup_method(self):
        """Create test fixtures."""
        self.crucible = CosmologyCrucible()
        self.engine = EvolutionaryEngine(
            crucible=self.crucible, population_size=3, gemini_client=None, cost_tracker=None
        )

    def test_history_initialized_empty(self):
        """Test that history list is initialized empty."""
        assert hasattr(self.engine, "history"), "Engine should have history attribute"
        assert isinstance(self.engine.history, list), "History should be a list"
        assert len(self.engine.history) == 0, "History should start empty"

    def test_history_populated_after_cycle(self):
        """Test that history is populated after running one evolutionary cycle."""
        # Initialize population
        self.engine.initialize_population()
        assert len(self.engine.civilizations) == 3, "Should have 3 civilizations"

        # Run one cycle
        self.engine.run_evolutionary_cycle()

        # Check history was populated
        assert len(self.engine.history) == 1, "History should have 1 entry after 1 cycle"

    def test_history_entry_structure(self):
        """Test that history entries have the correct structure."""
        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        entry = self.engine.history[0]

        # Check required keys
        assert "generation" in entry, "Entry should have generation"
        assert "population" in entry, "Entry should have population"
        assert "best_fitness" in entry, "Entry should have best_fitness"
        assert "avg_fitness" in entry, "Entry should have avg_fitness"
        assert "worst_fitness" in entry, "Entry should have worst_fitness"

    def test_history_generation_number(self):
        """Test that history records correct generation number."""
        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        assert self.engine.history[0]["generation"] == 0, "First entry should be generation 0"

        # Run another cycle (generation advances automatically in cycle)
        self.engine.run_evolutionary_cycle()

        assert self.engine.history[1]["generation"] == 1, "Second entry should be generation 1"

    def test_history_population_data(self):
        """Test that history records detailed population data."""
        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        population = self.engine.history[0]["population"]

        assert isinstance(population, list), "Population should be a list"
        assert len(population) == 3, "Population should have 3 entries"

        # Check structure of population entry
        model = population[0]
        assert "civ_id" in model, "Model should have civ_id"
        assert "fitness" in model, "Model should have fitness"
        assert "accuracy" in model, "Model should have accuracy"
        assert "speed" in model, "Model should have speed"
        assert "description" in model, "Model should have description"

    def test_history_fitness_statistics(self):
        """Test that fitness statistics are correctly calculated."""
        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        entry = self.engine.history[0]

        # Get all fitness values from history (not current civilizations, which are next gen)
        fitness_values = [model["fitness"] for model in entry["population"]]

        # Verify statistics match
        assert entry["best_fitness"] == max(fitness_values), "Best fitness should match max"
        assert entry["worst_fitness"] == min(fitness_values), "Worst fitness should match min"
        expected_avg = sum(fitness_values) / len(fitness_values)
        assert math.isclose(entry["avg_fitness"], expected_avg, rel_tol=1e-9), (
            "Avg fitness should match mean"
        )

    def test_history_accumulates_over_generations(self):
        """Test that history accumulates data over multiple generations."""
        self.engine.initialize_population()

        # Run 3 generations
        for _ in range(3):
            self.engine.run_evolutionary_cycle()

        # Check history has 3 entries
        assert len(self.engine.history) == 3, "History should have 3 entries after 3 cycles"

        # Verify generation numbers
        assert self.engine.history[0]["generation"] == 0
        assert self.engine.history[1]["generation"] == 1
        assert self.engine.history[2]["generation"] == 2

    def test_history_preserves_model_details(self):
        """Test that history preserves model-specific details."""
        self.engine.initialize_population()

        # Store original civ_id before cycle (which replaces civilizations with next gen)
        original_civ_id = list(self.engine.civilizations.keys())[0]

        self.engine.run_evolutionary_cycle()

        # Get corresponding entry from history
        history_entry = next(
            m for m in self.engine.history[0]["population"] if m["civ_id"] == original_civ_id
        )

        # Verify data was captured correctly in history (not comparing to next gen)
        assert "fitness" in history_entry
        assert "accuracy" in history_entry
        assert "speed" in history_entry
        assert "description" in history_entry
        assert history_entry["civ_id"] == original_civ_id

    def test_history_handles_evaluation_failures(self):
        """Test that history correctly records models that failed evaluation."""
        self.engine.initialize_population()

        # Inject a genome that will fail evaluation
        bad_genome = SurrogateGenome(theta=[0.0] * 6, description="bad_model")
        bad_genome.raw_code = "def predict(): raise Exception('Intentional failure')"
        self.engine.civilizations["civ_0_0"]["genome"] = bad_genome

        self.engine.run_evolutionary_cycle()

        # History should still be populated
        assert len(self.engine.history) == 1

        # Failed model should have zero fitness
        failed_model = next(
            m for m in self.engine.history[0]["population"] if m["civ_id"] == "civ_0_0"
        )
        assert failed_model["fitness"] == 0.0, "Failed model should have zero fitness"

    def test_history_fitness_ordering(self):
        """Test that history correctly identifies best/worst models."""
        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        entry = self.engine.history[0]
        population = entry["population"]

        # Find models with best and worst fitness
        best_model = max(population, key=lambda m: m["fitness"])
        worst_model = min(population, key=lambda m: m["fitness"])

        assert entry["best_fitness"] == best_model["fitness"]
        assert entry["worst_fitness"] == worst_model["fitness"]

    def test_history_with_single_civilization(self):
        """Test history tracking with population size of 1."""
        engine = EvolutionaryEngine(crucible=self.crucible, population_size=1)
        engine.initialize_population()
        engine.run_evolutionary_cycle()

        assert len(engine.history) == 1
        entry = engine.history[0]

        # With single civ, best = worst = avg
        assert entry["best_fitness"] == entry["worst_fitness"]
        assert entry["best_fitness"] == entry["avg_fitness"]

    def test_history_json_serializable(self):
        """Test that history entries can be serialized to JSON."""
        import json

        self.engine.initialize_population()
        self.engine.run_evolutionary_cycle()

        # Should not raise exception
        json_str = json.dumps(self.engine.history)
        assert isinstance(json_str, str)

        # Should be able to deserialize
        restored = json.loads(json_str)
        assert len(restored) == 1
        assert restored[0]["generation"] == 0
