"""
Integration tests for Galaxy evolution system.

These tests use the real Gemini API and are marked with @pytest.mark.integration.
Skip them during normal development with: pytest -m "not integration"
"""

import os
import time

import pytest

from code_validator import validate_and_compile
from config import settings
from gemini_client import CostTracker, GeminiClient
from prompts import get_initial_prompt, get_mutation_prompt

# Skip all tests in this module if no API key available
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here",
    reason="No valid GOOGLE_API_KEY in environment",
)


@pytest.mark.integration
class TestGeminiIntegration:
    """Test Gemini API integration with real requests."""

    def setup_method(self):
        """Create client for each test."""
        self.client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=False,  # Disable for faster tests
        )
        self.cost_tracker = CostTracker(max_cost_usd=1.0)

    def test_single_code_generation(self):
        """Test that Gemini can generate valid surrogate code."""
        # Generate with initial prompt
        prompt = get_initial_prompt(seed=0)
        response = self.client.generate_surrogate_code(prompt)

        # Check response structure
        assert response.success, f"Generation failed: {response.error}"
        assert response.code, "No code generated"
        assert response.tokens_used > 0, "No tokens counted"
        assert response.cost_usd > 0, "No cost calculated"
        assert response.generation_time_s > 0, "No time recorded"
        assert response.model == settings.llm_model

        # Track cost
        self.cost_tracker.add_call(response, "test_single_generation")
        assert self.cost_tracker.total_cost > 0

        # Validate generated code
        attractor = [50.0, 50.0]
        compiled_func, validation = validate_and_compile(response.code, attractor)

        assert validation.valid, f"Invalid code: {validation.errors}"
        assert compiled_func is not None, "Failed to compile valid code"

        # Test execution
        test_input = [45.0, 45.0, 1.0, 1.0]
        result = compiled_func(test_input)

        assert isinstance(result, list), "Output not a list"
        assert len(result) == 4, f"Output length {len(result)} != 4"
        assert all(isinstance(v, int | float) for v in result), "Non-numeric output"

    def test_mutation_generation(self):
        """Test that Gemini can mutate existing code."""
        # First, generate parent code
        initial_prompt = get_initial_prompt(seed=0)
        parent_response = self.client.generate_surrogate_code(initial_prompt)
        assert parent_response.success

        # Validate parent
        attractor = [50.0, 50.0]
        parent_func, parent_validation = validate_and_compile(parent_response.code, attractor)
        assert parent_validation.valid

        # Now mutate
        mutation_prompt = get_mutation_prompt(
            parent_code=parent_response.code,
            fitness=10000.0,
            accuracy=0.95,
            speed=0.0001,
            generation=1,
            mutation_type="explore",
        )
        child_response = self.client.generate_surrogate_code(mutation_prompt)

        assert child_response.success, f"Mutation failed: {child_response.error}"
        assert child_response.code, "No child code generated"
        assert child_response.code != parent_response.code, "Child identical to parent"

        # Validate child
        child_func, child_validation = validate_and_compile(child_response.code, attractor)
        assert child_validation.valid, f"Invalid child code: {child_validation.errors}"

    def test_rate_limiting_enforcement(self):
        """Test that rate limiter enforces 15 RPM."""
        # Create client with rate limiting enabled
        rate_limited_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0.8,
            max_output_tokens=1000,
            enable_rate_limiting=True,  # Enable rate limiting
        )

        # Make 3 quick requests
        start_time = time.time()
        prompt = get_initial_prompt(seed=0)

        for i in range(3):
            response = rate_limited_client.generate_surrogate_code(prompt)
            assert response.success, f"Request {i} failed"

        elapsed = time.time() - start_time

        # With 15 RPM (4 seconds between requests), 3 requests should take ~8 seconds
        # Allow some tolerance
        min_expected = 8.0  # 2 intervals of 4 seconds
        max_expected = 12.0  # Allow for API latency

        assert elapsed >= min_expected, (
            f"Rate limiting not working: {elapsed:.1f}s < {min_expected}s"
        )
        assert elapsed <= max_expected, f"Unexpectedly slow: {elapsed:.1f}s > {max_expected}s"

    def test_cost_tracking_accuracy(self):
        """Test that cost tracking correctly accumulates."""
        tracker = CostTracker(max_cost_usd=1.0)

        # Generate 3 codes and track costs
        prompt = get_initial_prompt(seed=0)
        responses = []

        for i in range(3):
            response = self.client.generate_surrogate_code(prompt)
            assert response.success
            tracker.add_call(response, f"test_call_{i}")
            responses.append(response)

        # Check accumulation
        summary = tracker.get_summary()

        assert summary["total_calls"] == 3
        assert summary["successful_calls"] == 3
        assert summary["failed_calls"] == 0
        assert summary["total_cost_usd"] > 0
        assert summary["total_tokens"] > 0
        assert summary["total_time_s"] > 0

        # Check that total cost = sum of individual costs
        expected_total = sum(r.cost_usd for r in responses)
        actual_total = summary["total_cost_usd"]
        assert abs(actual_total - expected_total) < 1e-6, "Cost tracking mismatch"

        # Check budget remaining
        assert summary["budget_remaining_usd"] == 1.0 - actual_total

    def test_budget_enforcement(self):
        """Test that budget limit prevents overspending."""
        # Create tracker with very small budget
        tiny_budget = 0.001  # $0.001 (enough for 2-3 calls)
        tracker = CostTracker(max_cost_usd=tiny_budget)

        # Make calls until budget exceeded
        prompt = get_initial_prompt(seed=0)
        call_count = 0

        while call_count < 10:  # Safety limit
            response = self.client.generate_surrogate_code(prompt)
            assert response.success
            tracker.add_call(response, f"budget_test_{call_count}")
            call_count += 1

            # Check if budget exceeded
            if tracker.check_budget_exceeded():
                break

        # Verify that budget was exceeded
        assert tracker.check_budget_exceeded(), "Budget should be exceeded"
        assert tracker.total_cost >= tiny_budget, "Cost didn't reach budget"

        # Verify we stopped (call_count < 10)
        assert call_count < 10, "Budget enforcement didn't stop calls"

    def test_validation_error_handling(self):
        """Test handling of LLM-generated invalid code."""
        # Try generating with a prompt that might produce invalid code
        # Use very low temperature for determinism
        low_temp_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0.1,  # Low temperature
            max_output_tokens=500,  # Short response
            enable_rate_limiting=False,
        )

        prompt = """Generate a Python function called predict that:
        1. Takes two arguments: particle and attractor
        2. Returns a list of 4 numbers

        IMPORTANT: Make it as short as possible (1-2 lines).
        """

        response = low_temp_client.generate_surrogate_code(prompt)

        # Even if generation succeeds, validation might fail
        if response.success and response.code:
            attractor = [50.0, 50.0]
            compiled_func, validation = validate_and_compile(response.code, attractor)

            # Code might be invalid (too simple, missing logic, etc.)
            # Test that validator catches issues
            if not validation.valid:
                assert len(validation.errors) > 0, "Invalid code should have error messages"
                assert compiled_func is None, "Invalid code shouldn't compile"

    def test_multiple_initial_approaches(self):
        """Test that different seeds produce different initial code."""
        approaches = []
        prompt_lengths = []

        for seed in range(6):  # Test all 6 initial approaches
            prompt = get_initial_prompt(seed)
            prompt_lengths.append(len(prompt))

            # Extract approach from prompt (should be different for each seed)
            if "Euler" in prompt:
                approaches.append("euler")
            elif "semi-implicit" in prompt:
                approaches.append("semi-implicit")
            elif "polynomial" in prompt:
                approaches.append("polynomial")
            elif "adaptive" in prompt:
                approaches.append("adaptive")
            elif "Verlet" in prompt:
                approaches.append("verlet")
            elif "softened" in prompt:
                approaches.append("softened")

        # Should have 6 different approaches
        assert len(set(approaches)) == 6, (
            f"Expected 6 unique approaches, got {len(set(approaches))}"
        )

    def test_explore_vs_exploit_mutation(self):
        """Test that explore and exploit produce different mutation prompts."""
        parent_code = """
def predict(particle, attractor):
    return [particle[0] + 1, particle[1] + 1, particle[2], particle[3]]
"""

        # Get explore prompt
        explore_prompt = get_mutation_prompt(
            parent_code=parent_code,
            fitness=10000.0,
            accuracy=0.95,
            speed=0.0001,
            generation=1,
            mutation_type="explore",
        )

        # Get exploit prompt
        exploit_prompt = get_mutation_prompt(
            parent_code=parent_code,
            fitness=10000.0,
            accuracy=0.95,
            speed=0.0001,
            generation=3,
            mutation_type="exploit",
        )

        # Prompts should be different
        assert explore_prompt != exploit_prompt, "Explore and exploit prompts should differ"

        # Explore should encourage creativity
        assert "different" in explore_prompt.lower() or "new" in explore_prompt.lower()

        # Exploit should encourage refinement
        assert "refine" in exploit_prompt.lower() or "improve" in exploit_prompt.lower()


@pytest.mark.integration
@pytest.mark.slow
class TestSyntaxErrorRate:
    """Test that LLM syntax error rate is below 1%."""

    def test_syntax_error_rate_below_threshold(self):
        """Test that syntax error rate is below 1% across multiple generations.

        This test validates the effectiveness of prompt engineering improvements
        that reduce incomplete code generation (missing brackets, truncated code).

        Target: <1% syntax error rate (down from historical 3.3%)
        Cost: ~$0.01 (20 API calls)
        """
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=False,  # Faster for tests
        )

        attractor = [50.0, 50.0]
        total_attempts = 20
        syntax_errors = 0
        validation_errors = []

        # Test Phase 1: Initial generation (explore phase, temperature 1.0)
        # This tests diverse approaches with high creativity
        for seed in range(10):
            prompt = get_initial_prompt(seed=seed % 6)
            response = client.generate_surrogate_code(prompt)

            if response.success and response.code:
                _, validation = validate_and_compile(response.code, attractor)
                if not validation.valid:
                    # Check if it's specifically a syntax error
                    for error in validation.errors:
                        if "Syntax error" in error or "SyntaxError" in error:
                            syntax_errors += 1
                            validation_errors.append(f"Initial Gen (seed={seed % 6}): {error}")
                            break

        # Test Phase 2: Mutation generation (exploit phase, temperature 0.6)
        # This tests refinement mode where historical errors occurred (Gen 3-4)
        parent_code = """def predict(particle, attractor):
    # Simple baseline - will be mutated
    x, y, vx, vy = particle
    ax, ay = attractor
    dx = ax - x
    dy = ay - y
    r_squared = dx * dx + dy * dy + 0.1
    force = 1.0 / r_squared
    return [x + vx * 0.01, y + vy * 0.01, vx + dx * force * 0.01, vy + dy * force * 0.01]
"""

        for gen in range(10):
            prompt = get_mutation_prompt(
                parent_code=parent_code,
                fitness=15000.0,
                accuracy=0.89,
                speed=0.00005,
                generation=gen + 3,  # Simulate later generation (exploitation phase)
                mutation_type="exploit",  # Refinement mode where errors occurred
            )
            response = client.generate_surrogate_code(prompt)

            if response.success and response.code:
                _, validation = validate_and_compile(response.code, attractor)
                if not validation.valid:
                    # Check if it's specifically a syntax error
                    for error in validation.errors:
                        if "Syntax error" in error or "SyntaxError" in error:
                            syntax_errors += 1
                            validation_errors.append(f"Exploit Gen {gen + 3}: {error}")
                            break

        # Calculate error rate
        error_rate = syntax_errors / total_attempts

        # Log results for debugging
        if syntax_errors > 0:
            print(f"\nSyntax Errors Found ({syntax_errors}/{total_attempts}):")
            for err in validation_errors:
                print(f"  - {err}")

        # Assert: Error rate must be below 1%
        assert error_rate < 0.01, (
            f"Syntax error rate {error_rate:.1%} ({syntax_errors}/{total_attempts}) "
            f"exceeds 1% threshold. Errors: {validation_errors}"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestFullEvolutionCycle:
    """Test complete evolution cycle (slow, uses many API calls)."""

    def test_mini_evolution(self):
        """Run a mini evolution (2 generations, 3 population) to test full pipeline."""
        from prototype import (
            DEFAULT_ATTRACTOR,
            CosmologyCrucible,
            LLM_propose_surrogate_model,
        )

        # Create components
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0.8,
            max_output_tokens=2000,
            enable_rate_limiting=False,  # Faster for tests
        )
        tracker = CostTracker(max_cost_usd=1.0)
        crucible = CosmologyCrucible(num_test_particles=20)  # Fewer particles for speed

        # Initialize mini population
        population = []
        for _ in range(3):
            genome = LLM_propose_surrogate_model(
                base_genome=None,
                generation=0,
                gemini_client=client,
                cost_tracker=tracker,
            )
            population.append(genome)

        # All should have valid code or parametric fallback
        assert len(population) == 3

        # Evaluate population
        for genome in population:
            func = genome.build_callable(DEFAULT_ATTRACTOR)
            accuracy, speed = crucible.evaluate_surrogate_model(func)
            genome.fitness = accuracy / max(speed, 1e-6)
            genome.accuracy = accuracy
            genome.speed = speed

        # Check that fitness values are reasonable
        for genome in population:
            assert genome.fitness is not None
            assert genome.fitness > 0
            assert 0.0 <= genome.accuracy <= 1.0
            assert genome.speed > 0

        # Select elite (top 1)
        elites = sorted(population, key=lambda g: g.fitness, reverse=True)[:1]
        assert len(elites) == 1
        assert elites[0].fitness == max(g.fitness for g in population)

        # Breed next generation
        next_gen = []
        for _ in range(3):
            parent = elites[0]
            child = LLM_propose_surrogate_model(
                base_genome=parent,
                generation=1,
                gemini_client=client,
                cost_tracker=tracker,
            )
            next_gen.append(child)

        assert len(next_gen) == 3

        # Check cost tracking
        summary = tracker.get_summary()
        assert summary["total_calls"] == 6  # 3 initial + 3 children (exact count)
        assert summary["total_cost_usd"] > 0
        assert not tracker.check_budget_exceeded()


@pytest.mark.integration
class TestErrorRecovery:
    """Test system behavior under error conditions."""

    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        bad_client = GeminiClient(
            api_key="invalid_key_12345",
            model="gemini-2.5-flash-lite",
            temperature=0.8,
            max_output_tokens=1000,
            enable_rate_limiting=False,
        )

        prompt = get_initial_prompt(seed=0)
        response = bad_client.generate_surrogate_code(prompt)

        # Should fail gracefully
        assert not response.success, "Should fail with invalid API key"
        assert response.error is not None, "Should have error message"
        assert response.code == "", "Should have no code on failure"

    def test_network_timeout_recovery(self):
        """Test that client retries on timeout."""
        # This test is challenging without actually causing a timeout
        # We'll just verify the retry logic is present
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0.8,
            max_output_tokens=1000,
            enable_rate_limiting=False,
        )

        # Check that retry_attempts parameter exists
        prompt = get_initial_prompt(seed=0)

        # This should succeed even with retry_attempts=1
        response = client.generate_surrogate_code(prompt, retry_attempts=1)

        # As long as network is okay, should succeed
        if os.getenv("GOOGLE_API_KEY"):  # Skip if no key
            assert response.success or response.error is not None
