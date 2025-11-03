#!/usr/bin/env python3
"""Smoke test for 3D N-body implementation with real API.

Tests:
- 1 generation
- 2 population size
- 10 particles (for speed)

Expected cost: ~$0.001
"""

import sys

from config import settings
from gemini_client import CostTracker, GeminiClient
from prototype import CosmologyCrucible, EvolutionaryEngine


def main():
    """Run minimal evolution to verify 3D N-body works."""
    print("=" * 70)
    print("3D N-Body Smoke Test")
    print("=" * 70)
    print("Configuration:")
    print("  - Generations: 1")
    print("  - Population: 5")
    print("  - Particles: 10 (3D N-body)")
    print(f"  - Model: {settings.llm_model}")
    print()

    # Initialize components
    crucible = CosmologyCrucible(num_particles=10)
    print(f"✓ Created crucible with {len(crucible.particles)} particles")
    print(f"  Sample particle: {crucible.particles[0]}")
    print()

    # Verify particle format
    for i, particle in enumerate(crucible.particles):
        if len(particle) != 7:
            print(f"✗ ERROR: Particle {i} has {len(particle)} elements, expected 7")
            return 1

    print("✓ All particles have 7 elements [x,y,z,vx,vy,vz,mass]")
    print()

    # Initialize LLM client
    client = GeminiClient(
        api_key=settings.google_api_key,
        model=settings.llm_model,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        enable_rate_limiting=False,  # Faster for smoke test
    )
    tracker = CostTracker()

    engine = EvolutionaryEngine(
        crucible=crucible,
        population_size=5,
        gemini_client=client,
        cost_tracker=tracker,
    )

    print("✓ Initialized engine")
    print()

    # Initialize population
    print("Initializing population...")
    engine.initialize_population()
    print(f"✓ Created {len(engine.civilizations)} initial genomes")
    print()

    # Run evolution
    print("Running evolution...")
    try:
        for gen in range(1):
            print(f"Generation {gen}...")
            engine.run_evolutionary_cycle()
    except Exception as e:
        print(f"✗ ERROR during evolution: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print("✓ Evolution completed successfully")
    print(f"  - API calls: {len(tracker.calls)}")
    print(f"  - Total cost: ${tracker.total_cost:.6f}")
    print("  - Generations: 1")
    print()

    # Get best genome from civilizations
    if engine.civilizations:
        sorted_civs = sorted(
            engine.civilizations.items(),
            key=lambda item: item[1].get("fitness", 0),
            reverse=True,
        )
        best_civ_id, best_civ_data = sorted_civs[0]
        best_genome = best_civ_data["genome"]
        print(f"Best genome ({best_civ_id}):")
        print(f"  - Fitness: {best_civ_data['fitness']:.2f}")
        if best_genome.accuracy:
            print(f"  - Accuracy: {best_genome.accuracy:.4f}")
        if best_genome.speed:
            print(f"  - Speed: {best_genome.speed:.6f}s")
        if best_genome.raw_code:
            print("  - Type: LLM-generated")
            print(f"  - Code length: {len(best_genome.raw_code)} chars")
        else:
            print("  - Type: Parametric fallback")
        print()

    print("=" * 70)
    print("✓ 3D N-BODY SMOKE TEST PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
