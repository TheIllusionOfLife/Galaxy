"""Test suite for genetic crossover operator.

This module tests the crossover functionality that enables LLM-based
code recombination between two elite parents.
"""

import random
from unittest.mock import Mock, patch

import pytest

from prototype import SurrogateGenome, select_crossover_parents


class TestCrossoverOperator:
    """Test crossover-specific logic."""

    def test_crossover_selects_two_different_parents(self):
        """Verify crossover picks two distinct elite parents."""
        # Setup: 5 elites with different civ_ids
        elites = [(f"civ_0_{i}", {"genome": Mock(), "fitness": 10 - i}) for i in range(5)]

        # Call: select two parents
        parent1, parent2 = select_crossover_parents(elites)

        # Assert: Different civ_ids
        assert parent1[0] != parent2[0], "Crossover should select two different parents"

    def test_crossover_requires_at_least_two_elites(self):
        """Verify crossover raises error with insufficient elites."""
        # Setup: Only 1 elite with raw_code
        genome = Mock()
        genome.raw_code = "def predict(particle, attractor): return particle"
        elites = [("civ_0_0", {"genome": genome, "fitness": 10})]

        # Assert: Raises ValueError
        with pytest.raises(ValueError, match="Need at least 2 LLM elites"):
            select_crossover_parents(elites)

    def test_crossover_with_exactly_two_elites(self):
        """Verify crossover works with minimum elite count."""
        # Setup: Exactly 2 elites
        elites = [
            ("civ_0_0", {"genome": Mock(), "fitness": 10}),
            ("civ_0_1", {"genome": Mock(), "fitness": 9}),
        ]

        # Call: Should succeed
        parent1, parent2 = select_crossover_parents(elites)

        # Assert: Got both elites
        civ_ids = {parent1[0], parent2[0]}
        assert civ_ids == {"civ_0_0", "civ_0_1"}

    def test_crossover_prompt_includes_both_parents(self):
        """Verify crossover prompt contains both parent codes."""
        from prompts import get_crossover_prompt

        # Setup: Two parent genomes with known code
        parent1 = SurrogateGenome(
            theta=[],
            raw_code="def predict(p, a): return [p[0]+1, p[1], p[2], p[3]]",
            fitness=100.0,
            accuracy=0.95,
            speed=0.001,
        )
        parent2 = SurrogateGenome(
            theta=[],
            raw_code="def predict(p, a): return [p[0]*2, p[1], p[2], p[3]]",
            fitness=90.0,
            accuracy=0.90,
            speed=0.002,
        )

        # Call: Generate crossover prompt
        prompt = get_crossover_prompt(parent1, parent2, generation=2)

        # Assert: Prompt contains both codes
        assert "p[0]+1" in prompt, "Should include parent1 code"
        assert "p[0]*2" in prompt, "Should include parent2 code"
        assert "100.0" in prompt or "100.00" in prompt, "Should include parent1 fitness"
        assert "90.0" in prompt or "90.00" in prompt, "Should include parent2 fitness"

    def test_crossover_tracks_parentage(self):
        """Verify offspring stores both parent IDs."""
        from prototype import LLM_propose_surrogate_model

        # Setup: Mock LLM response
        mock_code = "def predict(p, a): return [p[0]+p[0]*2, p[1], p[2], p[3]]"
        mock_response = Mock()
        mock_response.code = mock_code
        mock_response.success = True
        mock_response.tokens_used = 100
        mock_response.cost_usd = 0.0001

        mock_client = Mock()
        mock_client.generate_surrogate_code = Mock(return_value=mock_response)

        mock_tracker = Mock()
        mock_tracker.check_budget_exceeded = Mock(return_value=False)

        # Create valid parent genomes
        parent1 = SurrogateGenome(theta=[], raw_code="def predict(p, a): return p", fitness=100.0)
        parent2 = SurrogateGenome(theta=[], raw_code="def predict(p, a): return p", fitness=90.0)

        # Mock validation to succeed
        with patch("prototype.validate_and_compile") as mock_validate:
            mock_func = Mock(return_value=[1.0, 2.0, 3.0, 4.0])
            mock_result = Mock(valid=True, errors=[], warnings=[])
            mock_validate.return_value = (mock_func, mock_result)

            # Call: Crossover reproduction
            offspring = LLM_propose_surrogate_model(
                parent1,
                generation=2,
                gemini_client=mock_client,
                cost_tracker=mock_tracker,
                second_parent=parent2,
                parent_ids=["civ_1_0", "civ_1_3"],
            )

        # Assert: Offspring has parent IDs
        assert hasattr(offspring, "parent_ids"), "Offspring should track parentage"
        assert offspring.parent_ids == ["civ_1_0", "civ_1_3"]


class TestBreedingWithCrossover:
    """Test integration with breeding loop."""

    def test_breeding_respects_crossover_rate(self):
        """Verify breeding uses crossover at configured rate."""
        from prototype import CosmologyCrucible, EvolutionaryEngine

        # Setup: Mock everything, control randomness
        crucible = Mock(spec=CosmologyCrucible)
        engine = EvolutionaryEngine(crucible, population_size=10)

        # Create mock elites with raw_code for LLM crossover
        elites = []
        for i in range(3):  # 3 elites (30% of 10)
            genome = SurrogateGenome(
                theta=[float(i)],
                fitness=100.0 - i,
                raw_code=f"def predict(particle, attractor): return particle  # elite {i}",
            )
            genome.build_callable = Mock(return_value=lambda p, a: p)
            elites.append((f"civ_0_{i}", {"genome": genome, "fitness": 100.0 - i}))

        engine.civilizations = dict(elites)
        engine.generation = 0

        # Mock LLM to track crossover vs mutation calls
        crossover_calls = []
        mutation_calls = []

        def mock_llm_propose(
            base_genome,
            generation,
            gemini_client,
            cost_tracker,
            second_parent=None,
            parent_ids=None,
        ):
            if second_parent is not None:
                crossover_calls.append(1)
            else:
                mutation_calls.append(1)

            genome = SurrogateGenome(theta=[1.0], fitness=50.0)
            genome.build_callable = Mock(return_value=lambda p, a: p)
            return genome

        with patch("prototype.settings") as mock_settings:
            mock_settings.enable_crossover = True
            mock_settings.crossover_rate = 0.3
            mock_settings.elite_ratio = 0.3

            with patch("prototype.LLM_propose_surrogate_model", side_effect=mock_llm_propose):
                with patch(
                    "prototype.select_crossover_parents", side_effect=lambda e: random.sample(e, 2)
                ):
                    # Set random seed for reproducibility
                    random.seed(42)

                    # Call: Breed next generation
                    engine.breed_next_generation()

        # Assert: Roughly 30% crossover, 70% mutation (±20% tolerance)
        total = len(crossover_calls) + len(mutation_calls)
        assert total == 10, "Should generate 10 offspring"

        crossover_pct = len(crossover_calls) / total
        # Allow wide tolerance since only 10 samples
        assert 0.1 <= crossover_pct <= 0.5, (
            f"Crossover rate should be ~30%, got {crossover_pct * 100:.1f}%"
        )

    def test_crossover_disabled_falls_back_to_mutation(self):
        """Verify enable_crossover=false uses only mutation."""
        from prototype import CosmologyCrucible, EvolutionaryEngine

        # Setup
        crucible = Mock(spec=CosmologyCrucible)
        engine = EvolutionaryEngine(crucible, population_size=5)

        # Create mock elites
        elites = []
        for i in range(2):
            genome = SurrogateGenome(theta=[float(i)], fitness=100.0)
            genome.build_callable = Mock(return_value=lambda p, a: p)
            elites.append((f"civ_0_{i}", {"genome": genome, "fitness": 100.0}))

        engine.civilizations = dict(elites)
        engine.generation = 0

        # Track calls
        crossover_calls = []
        mutation_calls = []

        def mock_llm_propose(
            base_genome,
            generation,
            gemini_client,
            cost_tracker,
            second_parent=None,
            parent_ids=None,
        ):
            if second_parent is not None:
                crossover_calls.append(1)
            else:
                mutation_calls.append(1)

            genome = SurrogateGenome(theta=[1.0], fitness=50.0)
            genome.build_callable = Mock(return_value=lambda p, a: p)
            return genome

        with patch("prototype.settings") as mock_settings:
            mock_settings.enable_crossover = False  # DISABLED
            mock_settings.elite_ratio = 0.4

            with patch("prototype.LLM_propose_surrogate_model", side_effect=mock_llm_propose):
                # Call
                engine.breed_next_generation()

        # Assert: Only mutation, no crossover
        assert len(crossover_calls) == 0, "Crossover should be disabled"
        assert len(mutation_calls) == 5, "All offspring should use mutation"

    def test_crossover_with_insufficient_elites(self):
        """Verify crossover handles edge case of only 1 elite."""
        from prototype import CosmologyCrucible, EvolutionaryEngine

        # Setup
        crucible = Mock(spec=CosmologyCrucible)
        engine = EvolutionaryEngine(
            crucible, population_size=5, elite_ratio=0.1
        )  # 0.1*5=0.5→1 elite

        # Create 1 elite
        genome = SurrogateGenome(theta=[1.0], fitness=100.0)
        genome.build_callable = Mock(return_value=lambda p, a: p)
        engine.civilizations = {"civ_0_0": {"genome": genome, "fitness": 100.0}}
        engine.generation = 0

        # Track calls
        crossover_calls = []
        mutation_calls = []

        def mock_llm_propose(
            base_genome,
            generation,
            gemini_client,
            cost_tracker,
            second_parent=None,
            parent_ids=None,
        ):
            if second_parent is not None:
                crossover_calls.append(1)
            else:
                mutation_calls.append(1)

            genome = SurrogateGenome(theta=[1.0], fitness=50.0)
            genome.build_callable = Mock(return_value=lambda p, a: p)
            return genome

        with patch("prototype.settings") as mock_settings:
            mock_settings.enable_crossover = True
            mock_settings.crossover_rate = 0.5  # 50% crossover rate
            mock_settings.elite_ratio = 0.1

            with patch("prototype.LLM_propose_surrogate_model", side_effect=mock_llm_propose):
                # Call: Should fall back to mutation (can't do crossover with 1 elite)
                engine.breed_next_generation()

        # Assert: Only mutation (crossover impossible with 1 elite)
        assert len(crossover_calls) == 0, "Crossover impossible with 1 elite"
        assert len(mutation_calls) == 5, "Should fall back to mutation"


class TestCrossoverValidation:
    """Test crossover output validation."""

    def test_crossover_code_passes_ast_validation(self):
        """Verify crossover output goes through validation."""
        from prototype import LLM_propose_surrogate_model

        # Setup: Mock LLM to return invalid code (forbidden import)
        mock_response = Mock()
        mock_response.code = "import os\ndef predict(p, a): return p"
        mock_response.success = True
        mock_response.tokens_used = 50
        mock_response.cost_usd = 0.00005

        mock_client = Mock()
        mock_client.generate_surrogate_code = Mock(return_value=mock_response)

        mock_tracker = Mock()
        mock_tracker.check_budget_exceeded = Mock(return_value=False)

        parent1 = SurrogateGenome(theta=[], raw_code="def predict(p, a): return p", fitness=100.0)
        parent2 = SurrogateGenome(theta=[], raw_code="def predict(p, a): return p", fitness=90.0)

        # Mock validation to fail (forbidden import)
        with patch("prototype.validate_and_compile") as mock_validate:
            mock_result = Mock(valid=False, errors=["Forbidden: import statement"], warnings=[])
            mock_validate.return_value = (None, mock_result)

            # Mock fallback to parametric
            with patch("prototype._mock_surrogate_generation") as mock_fallback:
                mock_fallback_genome = SurrogateGenome(
                    theta=[1.0, 2.0], description="mock_fallback"
                )
                mock_fallback.return_value = mock_fallback_genome

                # Call: Should trigger fallback
                LLM_propose_surrogate_model(
                    parent1,
                    generation=2,
                    gemini_client=mock_client,
                    cost_tracker=mock_tracker,
                    second_parent=parent2,
                    parent_ids=["civ_1_0", "civ_1_1"],
                )

                # Assert: Validation was called
                mock_validate.assert_called_once()
                # Assert: Fallback was triggered
                mock_fallback.assert_called_once()

    def test_crossover_preserves_function_signature(self):
        """Verify crossover offspring has predict(particle, attractor)."""
        from code_validator import validate_and_compile

        # Valid crossover code (hybrid of Euler + damping)
        crossover_code = """
def predict(particle, attractor):
    x, y, vx, vy = particle
    ax, ay = attractor

    # Hybrid: Euler integration + damping
    dx = x - ax
    dy = y - ay
    r = max(0.1, (dx**2 + dy**2)**0.5)

    fx = -dx / (r**3)
    fy = -dy / (r**3)

    damping = 0.99
    new_vx = (vx + fx * 0.1) * damping
    new_vy = (vy + fy * 0.1) * damping

    new_x = x + new_vx * 0.1
    new_y = y + new_vy * 0.1

    return [new_x, new_y, new_vx, new_vy]
"""

        # Validate
        compiled_func, result = validate_and_compile(crossover_code, attractor=[0.0, 0.0])

        # Assert: Valid
        assert result.valid, f"Crossover code should be valid: {result.errors}"
        assert compiled_func is not None, "Should compile successfully"

        # Test function signature
        test_output = compiled_func([10.0, 0.0, 0.0, 0.0])
        assert isinstance(test_output, list), "Output should be a list"
        assert len(test_output) == 4, "Output should have 4 elements"
