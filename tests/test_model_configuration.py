"""Tests for multi-model configuration and validation.

This module tests the model switching functionality, including:
- Model name validation
- Pricing lookup for all supported models
- Rate limiter configuration per model
- Cost calculation accuracy
- Environment variable overrides

Note: All "test-key" strings in this file are test fixtures, not real secrets.
pragma: allowlist secret
"""

import pytest
from pydantic import ValidationError

from config import Settings
from gemini_client import GeminiClient


class TestModelValidation:
    """Test model name validation in Settings."""

    def test_valid_model_names(self, monkeypatch):
        """Test all supported model names are accepted."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Test gemini-2.5-flash-lite
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
        settings = Settings.load_from_yaml()
        assert settings.llm_model == "gemini-2.5-flash-lite"

        # Test gemini-2.5-flash
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
        settings = Settings.load_from_yaml()
        assert settings.llm_model == "gemini-2.5-flash"

        # Test gemini-2.5-pro
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
        settings = Settings.load_from_yaml()
        assert settings.llm_model == "gemini-2.5-pro"

    def test_invalid_model_names(self, monkeypatch):
        """Test invalid model names raise ValidationError."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Test completely invalid model
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        with pytest.raises(ValidationError, match="Unsupported model"):
            Settings.load_from_yaml()

        # Test typo in model name
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro-lite")
        with pytest.raises(ValidationError, match="Unsupported model"):
            Settings.load_from_yaml()

        # Test old version
        monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
        with pytest.raises(ValidationError, match="Unsupported model"):
            Settings.load_from_yaml()

    def test_model_env_override(self, monkeypatch):
        """Test LLM_MODEL environment variable overrides config.yaml."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Default from config.yaml is gemini-2.5-flash-lite
        settings = Settings.load_from_yaml()
        assert settings.llm_model == "gemini-2.5-flash-lite"

        # Override with environment variable
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
        settings = Settings.load_from_yaml()
        assert settings.llm_model == "gemini-2.5-pro"


class TestGeminiClientPricing:
    """Test pricing and rate limit configuration per model."""

    def test_flash_lite_pricing(self):
        """Test gemini-2.5-flash-lite pricing configuration."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash-lite",
            enable_rate_limiting=False,
        )

        assert client.INPUT_COST_PER_MTOK == 0.10
        assert client.OUTPUT_COST_PER_MTOK == 0.40
        assert client.model_name == "gemini-2.5-flash-lite"

        # Test cost calculation (1M input + 1M output tokens)
        cost = client._calculate_cost(1_000_000, 1_000_000)
        assert abs(cost - 0.50) < 0.001  # $0.10 + $0.40 = $0.50

    def test_flash_pricing(self):
        """Test gemini-2.5-flash pricing configuration."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            enable_rate_limiting=False,
        )

        assert client.INPUT_COST_PER_MTOK == 0.30
        assert client.OUTPUT_COST_PER_MTOK == 1.20
        assert client.model_name == "gemini-2.5-flash"

        # Test cost calculation (1M input + 1M output tokens)
        cost = client._calculate_cost(1_000_000, 1_000_000)
        assert abs(cost - 1.50) < 0.001  # $0.30 + $1.20 = $1.50

    def test_pro_pricing(self):
        """Test gemini-2.5-pro pricing configuration."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-pro",
            enable_rate_limiting=False,
        )

        assert client.INPUT_COST_PER_MTOK == 1.25
        assert client.OUTPUT_COST_PER_MTOK == 10.00
        assert client.model_name == "gemini-2.5-pro"

        # Test cost calculation (1M input + 1M output tokens)
        cost = client._calculate_cost(1_000_000, 1_000_000)
        assert abs(cost - 11.25) < 0.001  # $1.25 + $10.00 = $11.25

    def test_cost_relative_differences(self):
        """Test cost multipliers between models match expectations."""
        # Create clients for each model
        flash_lite = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash-lite",
            enable_rate_limiting=False,
        )
        flash = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            enable_rate_limiting=False,
        )
        pro = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-pro",
            enable_rate_limiting=False,
        )

        # Test with typical token counts (500 input, 200 output)
        lite_cost = flash_lite._calculate_cost(500, 200)
        flash_cost = flash._calculate_cost(500, 200)
        pro_cost = pro._calculate_cost(500, 200)

        # Flash should be ~3x more expensive than flash-lite
        flash_multiplier = flash_cost / lite_cost
        assert 2.5 < flash_multiplier < 3.5

        # Pro should be ~20-30x more expensive than flash-lite
        pro_multiplier = pro_cost / lite_cost
        assert 15 < pro_multiplier < 30

    def test_invalid_model_name(self):
        """Test unsupported model names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            GeminiClient(
                api_key="test-key",  # pragma: allowlist secret
                model="gpt-4",
            )

        with pytest.raises(ValueError, match="Unsupported model"):
            GeminiClient(
                api_key="test-key",  # pragma: allowlist secret
                model="claude-3",
            )


class TestRateLimiting:
    """Test rate limiter configuration per model."""

    def test_flash_lite_rate_limit(self):
        """Test gemini-2.5-flash-lite uses 15 RPM."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash-lite",
            enable_rate_limiting=True,
        )

        assert client.rate_limiter is not None
        assert client.rate_limiter.rpm == 15
        assert abs(client.rate_limiter.min_interval - 4.0) < 0.01  # 60/15 = 4.0s

    def test_flash_rate_limit(self):
        """Test gemini-2.5-flash uses 15 RPM."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            enable_rate_limiting=True,
        )

        assert client.rate_limiter is not None
        assert client.rate_limiter.rpm == 15
        assert abs(client.rate_limiter.min_interval - 4.0) < 0.01  # 60/15 = 4.0s

    def test_pro_rate_limit(self):
        """Test gemini-2.5-pro uses 10 RPM."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-pro",
            enable_rate_limiting=True,
        )

        assert client.rate_limiter is not None
        assert client.rate_limiter.rpm == 10
        assert abs(client.rate_limiter.min_interval - 6.0) < 0.01  # 60/10 = 6.0s

    def test_rate_limiting_disabled(self):
        """Test rate limiting can be disabled for all models."""
        for model in ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]:
            client = GeminiClient(
                api_key="test-key",  # pragma: allowlist secret
                model=model,
                enable_rate_limiting=False,
            )
            assert client.rate_limiter is None


class TestCostCalculation:
    """Test accurate cost calculation for different token counts."""

    def test_small_request_cost(self):
        """Test cost for small request (typical mutation)."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash-lite",
            enable_rate_limiting=False,
        )

        # Typical mutation: 500 input tokens, 200 output tokens
        cost = client._calculate_cost(500, 200)
        expected = (500 / 1_000_000) * 0.10 + (200 / 1_000_000) * 0.40
        assert abs(cost - expected) < 0.000001

    def test_large_request_cost(self):
        """Test cost for large request (crossover with context)."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-pro",
            enable_rate_limiting=False,
        )

        # Large crossover: 2000 input tokens, 1000 output tokens
        cost = client._calculate_cost(2000, 1000)
        expected = (2000 / 1_000_000) * 1.25 + (1000 / 1_000_000) * 10.00
        assert abs(cost - expected) < 0.000001

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            enable_rate_limiting=False,
        )

        cost = client._calculate_cost(0, 0)
        assert cost == 0.0

    def test_asymmetric_token_counts(self):
        """Test cost with very different input/output ratios."""
        client = GeminiClient(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            enable_rate_limiting=False,
        )

        # Large input, small output (typical for code analysis)
        cost1 = client._calculate_cost(5000, 50)
        expected1 = (5000 / 1_000_000) * 0.30 + (50 / 1_000_000) * 1.20
        assert abs(cost1 - expected1) < 0.000001

        # Small input, large output (typical for code generation)
        cost2 = client._calculate_cost(100, 2000)
        expected2 = (100 / 1_000_000) * 0.30 + (2000 / 1_000_000) * 1.20
        assert abs(cost2 - expected2) < 0.000001


class TestModelConfigurationIntegration:
    """Test end-to-end model configuration with Settings."""

    def test_settings_to_client_integration(self, monkeypatch):
        """Test Settings correctly configures GeminiClient."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-integration-key")
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("TEMPERATURE", "0.9")
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1500")

        settings = Settings.load_from_yaml()

        # Create client from settings
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=settings.enable_rate_limiting,
        )

        assert client.model_name == "gemini-2.5-pro"
        assert client.generation_config["temperature"] == 0.9
        assert client.generation_config["max_output_tokens"] == 1500
        assert client.INPUT_COST_PER_MTOK == 1.25
        assert client.OUTPUT_COST_PER_MTOK == 10.00

    def test_model_switching_changes_costs(self, monkeypatch):
        """Test switching models changes cost calculations."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Start with flash-lite
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
        settings = Settings.load_from_yaml()
        client1 = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            enable_rate_limiting=False,
        )
        cost1 = client1._calculate_cost(1000, 500)

        # Switch to pro
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
        settings = Settings.load_from_yaml()
        client2 = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            enable_rate_limiting=False,
        )
        cost2 = client2._calculate_cost(1000, 500)

        # Pro should be significantly more expensive
        assert cost2 > cost1 * 10  # At least 10x more expensive
