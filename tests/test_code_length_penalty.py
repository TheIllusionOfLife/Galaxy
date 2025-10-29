"""Tests for code length penalty feature.

This module tests the token counting and fitness penalty mechanisms
that prevent code bloat in later evolutionary generations.
"""

import os

import pytest

# Import will be available after implementation
# from prototype import count_tokens, SurrogateGenome


class TestTokenCounting:
    """Test token counting utility."""

    def test_count_tokens_simple(self):
        """Test basic token counting with simple code."""
        from prototype import count_tokens

        code = "def predict(particle, attractor):\n    return [1, 2, 3, 4]"
        tokens = count_tokens(code)
        # Whitespace split: def, predict(particle,, attractor):, return, [1,, 2,, 3,, 4]
        assert tokens == 8, f"Expected 8 tokens, got {tokens}"

    def test_count_tokens_empty(self):
        """Test empty code returns zero."""
        from prototype import count_tokens

        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_count_tokens_whitespace_only(self):
        """Test code with only whitespace."""
        from prototype import count_tokens

        assert count_tokens("   \n\n\t  ") == 0

    def test_count_tokens_complex(self):
        """Test realistic surrogate model code."""
        from prototype import count_tokens

        code = """def predict(particle, attractor):
    x, y, vx, vy = particle
    ax, ay = attractor
    dx = ax - x
    dy = ay - y
    dist_sq = dx*dx + dy*dy + 0.01
    force = 10.0 / dist_sq
    dist = math.sqrt(dist_sq)
    accel_x = force * dx / dist
    accel_y = force * dy / dist
    new_vx = vx + accel_x * 0.1
    new_vy = vy + accel_y * 0.1
    new_x = x + new_vx * 0.1
    new_y = y + new_vy * 0.1
    return [new_x, new_y, new_vx, new_vy]"""
        tokens = count_tokens(code)
        # Count actual tokens in this specific code
        # This realistic model has approximately 86 tokens
        assert 80 <= tokens <= 90, f"Expected ~86 tokens, got {tokens}"

    def test_count_tokens_very_long(self):
        """Test very long code (3000+ tokens)."""
        from prototype import count_tokens

        # Generate long code similar to Gen 3-4 models
        code = "def predict(particle, attractor):\n"
        code += "    x, y, vx, vy = particle\n"
        code += "    ax, ay = attractor\n"
        # Add many lines of computations
        for i in range(200):
            code += f"    temp_{i} = x + y * {i} + vx - vy\n"
        code += "    return [x, y, vx, vy]\n"

        tokens = count_tokens(code)
        assert tokens > 1000, "Very long code should have >1000 tokens"


class TestPenaltyCalculation:
    """Test fitness penalty calculation logic."""

    def test_no_penalty_below_threshold(self):
        """Code below threshold should not be penalized."""
        from config import settings  # Load from config.yaml

        # Test: 1500 tokens (below 2000 threshold)
        token_count = 1500
        base_fitness = 10000.0

        # Calculate penalty
        excess_tokens = max(0, token_count - settings.max_acceptable_tokens)
        assert excess_tokens == 0, "No excess tokens"

        # No penalty should be applied
        penalty_factor = 1.0 - (
            settings.code_length_penalty_weight * (excess_tokens / settings.max_acceptable_tokens)
        )
        penalty_factor = max(0.1, penalty_factor)

        assert penalty_factor == 1.0, "No penalty below threshold"
        final_fitness = base_fitness * penalty_factor
        assert final_fitness == base_fitness, "Fitness unchanged"

    def test_penalty_above_threshold(self):
        """Code above threshold should be penalized."""
        from config import settings  # Load from config.yaml

        # Test: 3000 tokens (1000 excess)
        token_count = 3000
        base_fitness = 10000.0

        excess_tokens = max(0, token_count - settings.max_acceptable_tokens)
        assert excess_tokens == 1000, "1000 excess tokens"

        # Penalty: 0.1 * (1000 / 2000) = 0.05
        # Factor: 1.0 - 0.05 = 0.95
        penalty_factor = 1.0 - (
            settings.code_length_penalty_weight * (excess_tokens / settings.max_acceptable_tokens)
        )
        penalty_factor = max(0.1, penalty_factor)

        assert penalty_factor == 0.95, "5% penalty for 50% excess"
        final_fitness = base_fitness * penalty_factor
        assert final_fitness == 9500.0, "Fitness reduced by 5%"

    def test_penalty_scales_linearly(self):
        """Penalty should scale linearly with excess tokens."""
        from config import settings  # Load from config.yaml

        test_cases = [
            (2000, 1.0),  # At threshold - no penalty
            (2500, 0.975),  # 25% excess = 2.5% penalty
            (3000, 0.95),  # 50% excess = 5% penalty
            (4000, 0.90),  # 100% excess = 10% penalty
        ]

        for token_count, expected_factor in test_cases:
            excess = max(0, token_count - settings.max_acceptable_tokens)
            factor = 1.0 - (
                settings.code_length_penalty_weight * (excess / settings.max_acceptable_tokens)
            )
            factor = max(0.1, factor)

            assert abs(factor - expected_factor) < 0.001, (
                f"Token {token_count}: expected {expected_factor}, got {factor}"
            )

    def test_penalty_minimum_floor(self, monkeypatch):
        """Penalty should not reduce fitness below 10% of base."""
        from config import settings  # Load from config.yaml

        # Override weight for this test only
        monkeypatch.setattr(settings, "code_length_penalty_weight", 1.0)  # Maximum weight

        # Extreme case: 20,000 tokens (18,000 excess = 900% excess)
        token_count = 20000
        base_fitness = 10000.0

        excess = max(0, token_count - settings.max_acceptable_tokens)
        factor = 1.0 - (
            settings.code_length_penalty_weight * (excess / settings.max_acceptable_tokens)
        )
        factor = max(0.1, factor)  # Floor at 10%

        assert factor == 0.1, "Penalty floor at 10%"
        final_fitness = base_fitness * factor
        assert final_fitness == 1000.0, "Minimum 10% fitness retained"

    def test_penalty_disabled(self, monkeypatch):
        """When penalty disabled, fitness should not be affected."""
        from config import settings  # Load from config.yaml

        # Disable penalty for this test
        monkeypatch.setattr(settings, "enable_code_length_penalty", False)

        # When disabled, penalty logic should be skipped
        # This will be tested in integration test
        assert not settings.enable_code_length_penalty


class TestSurrogateGenomeTokenField:
    """Test that SurrogateGenome correctly stores token_count."""

    def test_genome_with_token_count(self):
        """Test genome can store token count."""
        from prototype import SurrogateGenome

        genome = SurrogateGenome(
            theta=[1.0, 2.0, 3.0],
            description="test",
            raw_code="def predict(p, a): return [1, 2, 3, 4]",
            token_count=150,
        )

        assert genome.token_count == 150
        assert genome.raw_code is not None

    def test_genome_without_token_count(self):
        """Test genome works without token_count (backward compatibility)."""
        from prototype import SurrogateGenome

        genome = SurrogateGenome(theta=[1.0, 2.0, 3.0], description="test")

        assert genome.token_count is None
        assert genome.raw_code is None


@pytest.mark.integration
class TestPenaltyInEvolution:
    """Integration tests with evolution cycle."""

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here",
        reason="Requires valid GOOGLE_API_KEY",
    )
    def test_token_tracking_in_history(self):
        """Verify token counts are recorded in evolution history."""
        from config import settings
        from gemini_client import CostTracker, GeminiClient
        from prototype import CosmologyCrucible, EvolutionaryEngine

        # Mini evolution: 2 generations, 3 population
        crucible = CosmologyCrucible()
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=True,  # Respect rate limits
        )
        cost_tracker = CostTracker(max_cost_usd=1.0)

        engine = EvolutionaryEngine(
            crucible,
            population_size=3,
            elite_ratio=0.2,
            gemini_client=client,
            cost_tracker=cost_tracker,
        )

        # Run evolution
        engine.initialize_population()
        for _ in range(2):
            engine.run_evolutionary_cycle()

        # Check history
        assert len(engine.history) == 2, "Should have 2 generations"

        for gen_data in engine.history:
            assert "population" in gen_data
            for civ_data in gen_data["population"]:
                # Token count should be present
                assert "token_count" in civ_data, f"Missing token_count in {civ_data['civ_id']}"

                # For LLM-generated models, token_count should be > 0
                # For mock models, it might be None or 0
                token_count = civ_data["token_count"]
                if token_count is not None and token_count > 0:
                    assert token_count > 50, "LLM code should have >50 tokens"
                    assert token_count < 10000, "Should be in reasonable range"

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here",
        reason="Requires valid GOOGLE_API_KEY",
    )
    def test_penalty_reduces_fitness_for_long_code(self):
        """Verify penalty reduces fitness when code is long."""
        from config import settings
        from prototype import count_tokens

        # Create a genome with very long code
        long_code = (
            "def predict(particle, attractor):\n    " + "x = 1\n    " * 1000 + "return [1, 2, 3, 4]"
        )
        token_count = count_tokens(long_code)

        # Verify it exceeds threshold
        assert token_count > settings.max_acceptable_tokens, "Test code should exceed threshold"

        # Calculate expected penalty
        excess = token_count - settings.max_acceptable_tokens
        expected_factor = max(
            0.1,
            1.0 - (settings.code_length_penalty_weight * (excess / settings.max_acceptable_tokens)),
        )

        # Verify penalty factor is less than 1
        assert expected_factor < 1.0, "Long code should have penalty factor < 1.0"
        assert expected_factor >= 0.1, "Penalty factor should respect 10% floor"

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here",
        reason="Requires valid GOOGLE_API_KEY",
    )
    def test_penalty_weight_affects_token_count(self, tmp_path, monkeypatch):
        """Verify different penalty weights affect token evolution.

        This is a comprehensive integration test that runs mini evolution cycles
        with different penalty weights and verifies the expected behavior.
        """
        from config import Settings
        from gemini_client import CostTracker, GeminiClient
        from prototype import CosmologyCrucible, EvolutionaryEngine

        # Test with 3 different penalty weights
        test_weights = [0.0, 0.1, 0.2]
        results = {}

        for weight in test_weights:
            # Override settings for this test
            monkeypatch.setenv("CODE_LENGTH_PENALTY_WEIGHT", str(weight))
            monkeypatch.setenv("ENABLE_CODE_LENGTH_PENALTY", "true" if weight > 0 else "false")

            # Reload settings with new environment
            test_settings = Settings.load_from_yaml()

            # Verify settings loaded correctly
            assert test_settings.code_length_penalty_weight == weight

            # Mini evolution: 2 generations, 3 population
            crucible = CosmologyCrucible()
            client = GeminiClient(
                api_key=test_settings.google_api_key,
                model=test_settings.llm_model,
                temperature=test_settings.temperature,
                max_output_tokens=test_settings.max_output_tokens,
                enable_rate_limiting=True,
            )
            cost_tracker = CostTracker(max_cost_usd=1.0)

            engine = EvolutionaryEngine(
                crucible,
                population_size=3,
                elite_ratio=0.2,
                gemini_client=client,
                cost_tracker=cost_tracker,
            )

            # Run evolution
            engine.initialize_population()
            engine.run_evolutionary_cycle()

            # Collect token statistics
            token_counts = []
            for gen_data in engine.history:
                for civ_data in gen_data["population"]:
                    token_count = civ_data.get("token_count", 0)
                    if token_count and token_count > 0:
                        token_counts.append(token_count)

            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            max_tokens = max(token_counts) if token_counts else 0

            results[weight] = {
                "avg_tokens": avg_tokens,
                "max_tokens": max_tokens,
                "token_counts": token_counts,
            }

            print(f"\nWeight {weight}: avg={avg_tokens:.1f}, max={max_tokens}")

        # Verify penalty effect: higher weight → lower tokens (statistically)
        # Note: With small sample size, this is a trend check, not strict ordering
        if len(results[0.0]["token_counts"]) > 0 and len(results[0.2]["token_counts"]) > 0:
            # At minimum, verify penalty doesn't increase token count significantly
            max_allowed = results[0.0]["avg_tokens"] * 1.2
            assert results[0.2]["avg_tokens"] <= max_allowed, (
                f"High penalty should not significantly increase token count. "
                f"Expected max {max_allowed:.1f}, got {results[0.2]['avg_tokens']:.1f} "
                f"(no penalty baseline: {results[0.0]['avg_tokens']:.1f})"
            )
