"""Tests for LLM prompt templates with conservation emphasis.

This test suite ensures that all prompts include proper guidance for
physics conservation (energy and angular momentum) to address the root
cause identified in PR #45: LLMs don't generate conservation-preserving
code without explicit prompting.
"""

import pytest

from prompts import (
    SYSTEM_INSTRUCTION,
    get_crossover_prompt,
    get_initial_prompt,
    get_mutation_prompt,
)
from prototype import SurrogateGenome


class TestSystemInstruction:
    """Test that SYSTEM_INSTRUCTION includes conservation requirements."""

    def test_mentions_energy_conservation(self):
        """System instruction must mention energy conservation."""
        assert "energy conservation" in SYSTEM_INSTRUCTION.lower()

    def test_mentions_angular_momentum(self):
        """System instruction must mention angular momentum conservation."""
        assert "angular momentum" in SYSTEM_INSTRUCTION.lower()

    def test_mentions_symplectic(self):
        """System instruction should mention symplectic integrators."""
        assert "symplectic" in SYSTEM_INSTRUCTION.lower()

    def test_mentions_conservation_drift_threshold(self):
        """System instruction should mention conservation drift thresholds."""
        # Check for 1% threshold mentioned
        assert "1%" in SYSTEM_INSTRUCTION or "0.01" in SYSTEM_INSTRUCTION

    def test_mentions_physics_penalty(self):
        """System instruction should warn about physics penalty."""
        assert "penalty" in SYSTEM_INSTRUCTION.lower() or "penalized" in SYSTEM_INSTRUCTION.lower()


class TestInitialPrompt:
    """Test that initial prompts guide toward conservation-preserving approaches."""

    def test_includes_symplectic_approaches(self):
        """Initial prompts should include symplectic integration approaches."""
        # Test multiple seeds to ensure symplectic approaches are available
        all_prompts = [get_initial_prompt(seed) for seed in range(10)]
        combined = " ".join(all_prompts).lower()

        # At least one symplectic method should be mentioned
        symplectic_methods = ["symplectic", "leapfrog", "semi-implicit euler", "verlet"]
        assert any(method in combined for method in symplectic_methods), (
            f"No symplectic methods found. Expected one of: {symplectic_methods}"
        )

    def test_mentions_conservation_in_all_seeds(self):
        """All seed prompts should mention conservation requirements."""
        # Test all 6 approach variations
        for seed in range(6):
            prompt = get_initial_prompt(seed)
            # Should inherit from SYSTEM_INSTRUCTION or add explicit reminders
            assert "conservation" in prompt.lower() or "energy" in prompt.lower(), (
                f"Seed {seed} prompt lacks conservation guidance"
            )

    def test_includes_conservation_reminder(self):
        """Initial prompts should include conservation reminder before generation."""
        prompt = get_initial_prompt(0)
        # Look for explicit conservation requirements section
        assert "drift" in prompt.lower() and (
            "<1%" in prompt or "< 1%" in prompt or "<10%" in prompt
        )


class TestMutationPrompt:
    """Test that mutation prompts include conservation analysis and guidance."""

    def test_accepts_conservation_parameters(self):
        """Mutation prompt should accept energy_drift and momentum_drift parameters."""
        # Should not raise TypeError when passing conservation metrics
        try:
            prompt = get_mutation_prompt(
                parent_code="def predict(p, ps): return p",
                fitness=100.0,
                accuracy=0.9,
                speed=0.001,
                generation=1,
                mutation_type="explore",
                energy_drift=0.05,
                momentum_drift=0.02,
            )
            assert prompt  # Ensure non-empty
        except TypeError as e:
            pytest.fail(f"Mutation prompt doesn't accept conservation parameters: {e}")

    def test_shows_conservation_analysis_when_poor(self):
        """Mutation prompt should show conservation issues when drift is high."""
        prompt = get_mutation_prompt(
            parent_code="def predict(p, ps): return p",
            fitness=100.0,
            accuracy=0.9,
            speed=0.001,
            generation=1,
            energy_drift=0.15,  # 15% drift (poor)
            momentum_drift=0.08,  # 8% drift (poor)
        )

        # Should include analysis of poor conservation
        assert "15" in prompt or "0.15" in prompt  # Energy drift value
        assert "drift" in prompt.lower()
        assert "✗" in prompt or "poor" in prompt.lower() or "violate" in prompt.lower()

    def test_shows_conservation_analysis_when_good(self):
        """Mutation prompt should acknowledge good conservation."""
        prompt = get_mutation_prompt(
            parent_code="def predict(p, ps): return p",
            fitness=100.0,
            accuracy=0.9,
            speed=0.001,
            generation=1,
            energy_drift=0.005,  # 0.5% drift (good)
            momentum_drift=0.003,  # 0.3% drift (good)
        )

        # Should acknowledge good conservation
        assert "✓" in prompt or "good" in prompt.lower()
        assert "conservation" in prompt.lower()

    def test_explore_strategy_mentions_symplectic(self):
        """Explore mutation should suggest symplectic methods."""
        prompt = get_mutation_prompt(
            parent_code="def predict(p, ps): return p",
            fitness=50.0,
            accuracy=0.7,
            speed=0.01,
            generation=1,
            mutation_type="explore",
            energy_drift=0.2,
        )

        # Should suggest symplectic approaches for exploration
        assert (
            "symplectic" in prompt.lower()
            or "leapfrog" in prompt.lower()
            or "verlet" in prompt.lower()
        )

    def test_exploit_strategy_mentions_conservation_improvement(self):
        """Exploit mutation should focus on conservation refinement."""
        prompt = get_mutation_prompt(
            parent_code="def predict(p, ps): return p",
            fitness=200.0,
            accuracy=0.95,
            speed=0.001,
            generation=3,
            mutation_type="exploit",
            energy_drift=0.05,
        )

        # Should focus on improving conservation
        assert "conservation" in prompt.lower() and (
            "improve" in prompt.lower() or "reduce" in prompt.lower()
        )


class TestCrossoverPrompt:
    """Test that crossover prompts include conservation context."""

    def test_shows_parent_conservation_metrics(self):
        """Crossover prompt should show conservation metrics for both parents."""
        parent1 = SurrogateGenome(
            theta=[1.0],
            raw_code="def predict(p, ps): return p",
            fitness=150.0,
            accuracy=0.92,
            speed=0.002,
            energy_drift=0.03,
            momentum_drift=0.01,
        )
        parent2 = SurrogateGenome(
            theta=[2.0],
            raw_code="def predict(p, ps): return [p[0], p[1], p[2], 0, 0, 0, p[6]]",
            fitness=120.0,
            accuracy=0.88,
            speed=0.001,
            energy_drift=0.08,
            momentum_drift=0.05,
        )

        prompt = get_crossover_prompt(parent1, parent2, generation=2)

        # Should show energy drift for both parents (formatted as X.X%)
        assert "3.0%" in prompt or "3%" in prompt  # Parent 1 energy drift
        assert "8.0%" in prompt or "8%" in prompt  # Parent 2 energy drift
        assert "drift" in prompt.lower()

    def test_emphasizes_conservation_in_task(self):
        """Crossover task should emphasize preserving physics."""
        parent1 = SurrogateGenome(
            theta=[1.0],
            raw_code="def predict(p, ps): return p",
            fitness=100.0,
            energy_drift=0.02,
        )
        parent2 = SurrogateGenome(
            theta=[2.0],
            raw_code="def predict(p, ps): return p",
            fitness=100.0,
            energy_drift=0.04,
        )

        prompt = get_crossover_prompt(parent1, parent2, generation=1)

        # TASK section should mention conservation priority
        # Find the last occurrence of "Generate" for the final generation instruction
        task_start = prompt.find("TASK:")
        assert task_start >= 0, "TASK section not found"

        # Get everything from TASK onwards and check for keywords
        task_and_beyond = prompt[task_start:]
        assert "conservation" in task_and_beyond.lower() or "physics" in task_and_beyond.lower()
        assert "drift" in task_and_beyond.lower()


class TestConservationKeywordCoverage:
    """Test that conservation keywords appear across all prompt types."""

    def test_all_prompts_mention_conservation_concepts(self):
        """All prompt types should include conservation-related terms."""
        conservation_terms = [
            "energy",
            "conservation",
            "drift",
            "angular momentum",
            "symplectic",
        ]

        # System instruction
        system_text = SYSTEM_INSTRUCTION.lower()
        assert any(term in system_text for term in conservation_terms)

        # Initial prompt
        initial_text = get_initial_prompt(0).lower()
        assert any(term in initial_text for term in conservation_terms)

        # Mutation prompt
        mutation_text = get_mutation_prompt(
            parent_code="def predict(p, ps): return p",
            fitness=100.0,
            accuracy=0.9,
            speed=0.001,
            generation=1,
            energy_drift=0.05,
        ).lower()
        assert any(term in mutation_text for term in conservation_terms)

        # Crossover prompt
        parent = SurrogateGenome(
            theta=[1.0],
            raw_code="def predict(p, ps): return p",
            fitness=100.0,
            energy_drift=0.02,
        )
        crossover_text = get_crossover_prompt(parent, parent, generation=1).lower()
        assert any(term in crossover_text for term in conservation_terms)
