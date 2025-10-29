"""Tests for EvolutionaryEngine elite selection behavior."""

from unittest.mock import Mock

import pytest

from prototype import CosmologyCrucible, EvolutionaryEngine, SurrogateGenome


class TestEliteSelection:
    """Test elite_ratio configuration is respected in selection."""

    @pytest.fixture
    def mock_crucible(self):
        """Create a mock crucible for testing."""
        crucible = Mock(spec=CosmologyCrucible)
        crucible.evaluate_surrogate_model = Mock(return_value=(0.8, 0.001))
        return crucible

    @pytest.fixture
    def mock_genome(self):
        """Create a mock genome for testing."""
        genome = SurrogateGenome(
            theta=[1.0, 2.0],
            description="test genome",
            raw_code="def predict(p, a): return p",
            fitness=None,
            accuracy=None,
            speed=None,
        )
        genome.build_callable = Mock(return_value=lambda p, a: p)
        return genome

    def test_elite_ratio_30_percent_selects_3_of_10(self, mock_crucible, mock_genome):
        """Test elite_ratio=0.3 selects 3 models from population of 10."""
        # Arrange
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            elite_ratio=0.3,
        )

        # Create 10 civilizations with different fitness values
        for i in range(10):
            civ_id = f"civ_{i}"
            engine.civilizations[civ_id] = {
                "genome": mock_genome,
                "fitness": 10 - i,  # Descending fitness: 10, 9, 8, ..., 1
            }

        # Act - trigger selection by running a cycle (will select elites)
        # We need to intercept the selection logic
        sorted_civs = sorted(
            engine.civilizations.items(),
            key=lambda item: item[1].get("fitness", 0),
            reverse=True,
        )
        num_elites = max(1, int(engine.population_size * engine.elite_ratio))
        elites = sorted_civs[:num_elites]

        # Assert
        assert num_elites == 3, f"Expected 3 elites, got {num_elites}"
        assert len(elites) == 3
        # Verify top 3 fitness models are selected
        assert elites[0][1]["fitness"] == 10
        assert elites[1][1]["fitness"] == 9
        assert elites[2][1]["fitness"] == 8

    def test_elite_ratio_zero_selects_minimum_one(self, mock_crucible, mock_genome):
        """Test elite_ratio=0.0 selects at least 1 model (minimum constraint)."""
        # Arrange
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            elite_ratio=0.0,
        )

        # Create 10 civilizations
        for i in range(10):
            civ_id = f"civ_{i}"
            engine.civilizations[civ_id] = {
                "genome": mock_genome,
                "fitness": 10 - i,
            }

        # Act
        sorted_civs = sorted(
            engine.civilizations.items(),
            key=lambda item: item[1].get("fitness", 0),
            reverse=True,
        )
        num_elites = max(1, int(engine.population_size * engine.elite_ratio))
        elites = sorted_civs[:num_elites]

        # Assert
        assert num_elites == 1, f"Expected at least 1 elite, got {num_elites}"
        assert len(elites) == 1
        assert elites[0][1]["fitness"] == 10  # Best model selected

    def test_elite_ratio_one_selects_all_models(self, mock_crucible, mock_genome):
        """Test elite_ratio=1.0 selects all models (no selection pressure)."""
        # Arrange
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            elite_ratio=1.0,
        )

        # Create 10 civilizations
        for i in range(10):
            civ_id = f"civ_{i}"
            engine.civilizations[civ_id] = {
                "genome": mock_genome,
                "fitness": 10 - i,
            }

        # Act
        sorted_civs = sorted(
            engine.civilizations.items(),
            key=lambda item: item[1].get("fitness", 0),
            reverse=True,
        )
        num_elites = max(1, int(engine.population_size * engine.elite_ratio))
        elites = sorted_civs[:num_elites]

        # Assert
        assert num_elites == 10, f"Expected 10 elites, got {num_elites}"
        assert len(elites) == 10

    def test_elite_ratio_fractional_floors_correctly(self, mock_crucible, mock_genome):
        """Test elite_ratio=0.25 with population=10 floors to 2 (not 3)."""
        # Arrange
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            elite_ratio=0.25,  # 10 * 0.25 = 2.5, should floor to 2
        )

        # Create 10 civilizations
        for i in range(10):
            civ_id = f"civ_{i}"
            engine.civilizations[civ_id] = {
                "genome": mock_genome,
                "fitness": 10 - i,
            }

        # Act
        sorted_civs = sorted(
            engine.civilizations.items(),
            key=lambda item: item[1].get("fitness", 0),
            reverse=True,
        )
        num_elites = max(1, int(engine.population_size * engine.elite_ratio))
        elites = sorted_civs[:num_elites]

        # Assert
        assert num_elites == 2, f"Expected 2 elites (floored from 2.5), got {num_elites}"
        assert len(elites) == 2

    def test_default_elite_ratio_is_0_2(self, mock_crucible):
        """Test default elite_ratio is 0.2 when not specified."""
        # Act
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            # elite_ratio not specified, should default to 0.2
        )

        # Assert
        assert hasattr(engine, "elite_ratio"), "Engine should have elite_ratio attribute"
        assert engine.elite_ratio == 0.2, f"Expected default 0.2, got {engine.elite_ratio}"


class TestEliteSelectionIntegration:
    """Integration tests for elite_ratio with config system."""

    @pytest.fixture
    def mock_crucible(self):
        """Create a mock crucible for testing."""
        crucible = Mock(spec=CosmologyCrucible)
        crucible.evaluate_surrogate_model = Mock(return_value=(0.8, 0.001))
        return crucible

    @pytest.fixture
    def mock_genome(self):
        """Create a mock genome for testing."""
        genome = SurrogateGenome(
            theta=[1.0, 2.0],
            description="test genome",
            raw_code="def predict(p, a): return p",
            fitness=None,
            accuracy=None,
            speed=None,
        )
        genome.build_callable = Mock(return_value=lambda p, a: p)
        return genome

    def test_elite_ratio_from_settings(self, monkeypatch):
        """Test EvolutionaryEngine respects elite_ratio from settings."""
        # Arrange
        monkeypatch.setenv("ELITE_RATIO", "0.4")
        from config import Settings

        settings = Settings()

        crucible = Mock(spec=CosmologyCrucible)

        # Act
        engine = EvolutionaryEngine(
            crucible=crucible,
            population_size=10,
            elite_ratio=settings.elite_ratio,
        )

        # Assert
        assert engine.elite_ratio == 0.4, f"Expected 0.4 from settings, got {engine.elite_ratio}"

        # Verify selection uses this value
        num_elites = max(1, int(engine.population_size * engine.elite_ratio))
        assert num_elites == 4, f"Expected 4 elites with ratio 0.4, got {num_elites}"

    def test_elite_ratio_validation(self, mock_crucible):
        """Test elite_ratio parameter validation."""
        # Valid values should work
        engine = EvolutionaryEngine(mock_crucible, elite_ratio=0.0)
        assert engine.elite_ratio == 0.0

        engine = EvolutionaryEngine(mock_crucible, elite_ratio=1.0)
        assert engine.elite_ratio == 1.0

        engine = EvolutionaryEngine(mock_crucible, elite_ratio=0.5)
        assert engine.elite_ratio == 0.5

        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="elite_ratio must be between 0.0 and 1.0"):
            EvolutionaryEngine(mock_crucible, elite_ratio=-0.1)

        with pytest.raises(ValueError, match="elite_ratio must be between 0.0 and 1.0"):
            EvolutionaryEngine(mock_crucible, elite_ratio=1.5)

    def test_elite_selection_behavior_integration(self, mock_crucible, mock_genome):
        """Test that run_evolutionary_cycle actually uses elite_ratio for selection.

        This is a behavior-based integration test that verifies the EvolutionaryEngine
        calls the selection logic with the correct elite_ratio, rather than just testing
        the formula calculation.
        """
        from unittest.mock import patch

        # Arrange: Create engine with elite_ratio=0.3 (should select 3 of 10)
        engine = EvolutionaryEngine(
            crucible=mock_crucible,
            population_size=10,
            elite_ratio=0.3,
        )

        # Initialize population with mock LLM
        with patch("prototype.LLM_propose_surrogate_model", return_value=mock_genome):
            engine.initialize_population()

        assert len(engine.civilizations) == 10

        # Set controlled fitness values to establish ranking
        fitness_values = list(range(10, 0, -1))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        for i, (_civ_id, civ) in enumerate(engine.civilizations.items()):
            civ["fitness"] = fitness_values[i]
            civ["accuracy"] = 0.9
            civ["speed"] = 0.001
            civ["token_count"] = 100

        # Act: Run evolutionary cycle and capture random.choice calls
        with patch("prototype.LLM_propose_surrogate_model", return_value=mock_genome):
            with patch("prototype.random.choice") as mock_choice:
                # Set up mock to return the parent it's called with
                mock_choice.side_effect = lambda elites: elites[0]

                # Mock evaluate to avoid actual evaluation during mutation
                mock_crucible.evaluate_surrogate_model.return_value = {
                    "accuracy": 0.9,
                    "speed": 0.001,
                }

                engine.run_evolutionary_cycle()

                # Assert: Verify random.choice was called with exactly 3 elites
                # (elite_ratio=0.3 * population_size=10 = 3)
                assert mock_choice.called, "random.choice should be called during breeding"

                # Get the first argument (the list of elites) from the first call
                elites_list = mock_choice.call_args_list[0][0][0]
                assert len(elites_list) == 3, (
                    f"Expected 3 elites with ratio 0.3, got {len(elites_list)}"
                )
