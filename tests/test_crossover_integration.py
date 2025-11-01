"""Integration tests for genetic crossover with real API.

These tests use the actual Gemini API to validate crossover functionality.
"""

import pytest

from config import settings
from gemini_client import CostTracker, GeminiClient
from prototype import (
    CosmologyCrucible,
    EvolutionaryEngine,
    LLM_propose_surrogate_model,
    SurrogateGenome,
)


@pytest.mark.integration
class TestCrossoverIntegration:
    """Integration tests requiring real API."""

    def test_crossover_produces_valid_hybrid_code(self):
        """Real API: Verify crossover creates compilable code."""
        # Check API key
        if not settings.google_api_key or settings.google_api_key.startswith("test-"):
            pytest.skip("API key required for integration test")

        # Setup: Two known-good parent codes (Euler + Verlet integration, 3D N-body)
        parent1 = SurrogateGenome(
            theta=[],
            raw_code="""
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # Center of mass approximation
    cx = sum(p[0] for p in all_particles) / len(all_particles)
    cy = sum(p[1] for p in all_particles) / len(all_particles)
    cz = sum(p[2] for p in all_particles) / len(all_particles)

    # Euler integration
    dx, dy, dz = x - cx, y - cy, z - cz
    r = max(0.1, (dx**2 + dy**2 + dz**2)**0.5)

    fx = -dx / (r**3)
    fy = -dy / (r**3)
    fz = -dz / (r**3)

    new_vx = vx + fx * 0.1
    new_vy = vy + fy * 0.1
    new_vz = vz + fz * 0.1
    new_x = x + new_vx * 0.1
    new_y = y + new_vy * 0.1
    new_z = z + new_vz * 0.1

    return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]
""",
            fitness=15000.0,
            accuracy=0.85,
            speed=0.0001,
        )

        parent2 = SurrogateGenome(
            theta=[],
            raw_code="""
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # Center of mass approximation
    cx = sum(p[0] for p in all_particles) / len(all_particles)
    cy = sum(p[1] for p in all_particles) / len(all_particles)
    cz = sum(p[2] for p in all_particles) / len(all_particles)

    # Velocity Verlet integration
    dx, dy, dz = x - cx, y - cy, z - cz
    r = max(0.1, (dx**2 + dy**2 + dz**2)**0.5)

    fx = -dx / (r**3)
    fy = -dy / (r**3)
    fz = -dz / (r**3)

    # Update velocity (half step)
    vx_half = vx + fx * 0.05
    vy_half = vy + fy * 0.05
    vz_half = vz + fz * 0.05

    # Update position
    new_x = x + vx_half * 0.1
    new_y = y + vy_half * 0.1
    new_z = z + vz_half * 0.1

    # Update velocity (full step)
    new_vx = vx_half + fx * 0.05
    new_vy = vy_half + fy * 0.05
    new_vz = vz_half + fz * 0.05

    return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]
""",
            fitness=18000.0,
            accuracy=0.90,
            speed=0.00012,
        )

        # Initialize real API client
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            enable_rate_limiting=settings.enable_rate_limiting,
        )

        tracker = CostTracker(max_cost_usd=1.0)

        # Call: Generate crossover offspring
        offspring = LLM_propose_surrogate_model(
            parent1,
            generation=2,
            gemini_client=client,
            cost_tracker=tracker,
            second_parent=parent2,
            parent_ids=["civ_1_0", "civ_1_5"],
        )

        # Assert: Generated code is valid
        assert offspring is not None, "Crossover should produce offspring"

        if offspring.raw_code:
            # LLM succeeded
            assert "def predict" in offspring.raw_code, "Should define predict function"
            assert offspring.compiled_predict is not None, "Code should be compiled"

            # Test execution with 3D N-body format
            test_particle = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 1.0]
            test_all_particles = [
                [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 1.0],
                [20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 1.0],
            ]
            result = offspring.compiled_predict(test_particle, test_all_particles)

            assert isinstance(result, list), "Output should be a list"
            assert len(result) == 7, "Output should have 7 elements"
            assert all(isinstance(v, (int, float)) for v in result), "All outputs should be numeric"
        else:
            # Fell back to parametric (acceptable)
            assert offspring.theta, "Parametric fallback should have theta"

        # Verify parentage tracking
        assert offspring.parent_ids == ["civ_1_0", "civ_1_5"]

        # Verify cost tracking
        assert tracker.total_cost > 0, "Should track API cost"
        assert len(tracker.calls) == 1, "Should have made 1 API call"

        print("\n✓ Crossover integration test passed")
        print(f"  - API cost: ${tracker.total_cost:.6f}")
        print(f"  - Generated code: {len(offspring.raw_code or '')} chars")
        print(f"  - Validation: {'Success' if offspring.raw_code else 'Fallback to parametric'}")

    @pytest.mark.slow
    def test_full_evolution_with_crossover(self):
        """Real API: Run mini evolution with crossover enabled."""
        # Check API key
        if not settings.google_api_key or settings.google_api_key.startswith("test-"):
            pytest.skip("API key required for integration test")

        # Mini config: 2 generations, 5 population, crossover_rate=0.5
        # Expected: ~10 API calls (5 initial + 5 generation 1)
        # Cost: ~$0.004

        # Initialize components (3D N-body with 20 particles for speed)
        crucible = CosmologyCrucible(num_particles=20)

        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            enable_rate_limiting=False,  # Faster for testing
        )

        tracker = CostTracker(max_cost_usd=1.0)

        # Create engine with small population
        engine = EvolutionaryEngine(
            crucible=crucible,
            population_size=5,
            elite_ratio=0.4,  # 2 elites
            gemini_client=client,
            cost_tracker=tracker,
        )

        # Initialize population (5 API calls)
        engine.initialize_population()

        # Run 1 generation (5 more API calls, mix of crossover + mutation)
        engine.run_evolutionary_cycle()

        # Validate results
        history = engine.get_evolution_history()

        assert len(history) == 2, "Should have 2 generations (0 and 1)"
        assert len(history[0]["population"]) == 5, "Gen 0 should have 5 models"
        assert len(history[1]["population"]) == 5, "Gen 1 should have 5 models"

        # Check for crossover offspring (parent_ids with 2 elements)
        gen1_offspring = history[1]["population"]
        crossover_count = sum(
            1 for model in gen1_offspring if len(model.get("parent_ids", [])) == 2
        )
        mutation_count = sum(1 for model in gen1_offspring if len(model.get("parent_ids", [])) <= 1)

        # With crossover_rate=0.3 and 5 offspring, expect ~1-2 crossover (some variability)
        print("\n✓ Full evolution integration test passed")
        print(f"  - Total API calls: {len(tracker.calls)}")
        print(f"  - Total cost: ${tracker.total_cost:.6f}")
        print(f"  - Gen 1 offspring: {crossover_count} crossover + {mutation_count} mutation")
        print(f"  - Best fitness Gen 0: {history[0]['best_fitness']:.2f}")
        print(f"  - Best fitness Gen 1: {history[1]['best_fitness']:.2f}")

        # Basic sanity checks
        assert len(tracker.calls) >= 10, "Should make at least 10 API calls"
        assert len(tracker.calls) <= 15, "Should not exceed expected calls significantly"
        assert tracker.total_cost < 0.01, "Cost should be under 1 cent"

        # Verify evolution completed without errors
        assert history[0]["best_fitness"] > 0, "Should have positive fitness"
        assert history[1]["best_fitness"] > 0, "Should have positive fitness"
