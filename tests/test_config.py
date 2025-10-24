"""Tests for configuration management."""

import os
import pytest
from pydantic import ValidationError
from config import Settings


class TestSettings:
    """Test Settings class configuration and validation."""

    def test_settings_defaults(self, monkeypatch):
        """Test default values when no environment variables set."""
        # Clear any existing env vars
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        settings = Settings()

        # API keys should be None by default
        assert settings.google_api_key is None
        assert settings.anthropic_api_key is None

        # Model defaults
        assert settings.llm_model == "gemini-2.5-flash-lite"
        assert settings.temperature == 0.8
        assert settings.max_output_tokens == 2000

        # Rate limiting defaults
        assert settings.max_requests_per_run == 50
        assert settings.enable_rate_limiting is True
        assert settings.requests_per_minute == 15

        # Evolution defaults
        assert settings.population_size == 10
        assert settings.num_generations == 5
        assert settings.elite_ratio == 0.2

        # Mutation defaults
        assert settings.early_mutation_temp == 1.0
        assert settings.late_mutation_temp == 0.6

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-123")
        monkeypatch.setenv("TEMPERATURE", "1.5")
        monkeypatch.setenv("POPULATION_SIZE", "20")
        monkeypatch.setenv("NUM_GENERATIONS", "10")

        settings = Settings()

        assert settings.google_api_key == "test-api-key-123"
        assert settings.temperature == 1.5
        assert settings.population_size == 20
        assert settings.num_generations == 10

    def test_temperature_validation(self, monkeypatch):
        """Test temperature must be between 0.0 and 2.0."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Valid temperatures
        monkeypatch.setenv("TEMPERATURE", "0.0")
        settings = Settings()
        assert settings.temperature == 0.0

        monkeypatch.setenv("TEMPERATURE", "2.0")
        settings = Settings()
        assert settings.temperature == 2.0

        monkeypatch.setenv("TEMPERATURE", "1.0")
        settings = Settings()
        assert settings.temperature == 1.0

        # Invalid: too low
        monkeypatch.setenv("TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            Settings()

        # Invalid: too high
        monkeypatch.setenv("TEMPERATURE", "2.1")
        with pytest.raises(ValidationError):
            Settings()

    def test_population_size_validation(self, monkeypatch):
        """Test population_size must be between 1 and 100."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Valid
        monkeypatch.setenv("POPULATION_SIZE", "1")
        settings = Settings()
        assert settings.population_size == 1

        monkeypatch.setenv("POPULATION_SIZE", "100")
        settings = Settings()
        assert settings.population_size == 100

        # Invalid: too low
        monkeypatch.setenv("POPULATION_SIZE", "0")
        with pytest.raises(ValidationError):
            Settings()

        # Invalid: too high
        monkeypatch.setenv("POPULATION_SIZE", "101")
        with pytest.raises(ValidationError):
            Settings()

    def test_elite_ratio_validation(self, monkeypatch):
        """Test elite_ratio must be between 0.0 and 1.0."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Valid
        monkeypatch.setenv("ELITE_RATIO", "0.0")
        settings = Settings()
        assert settings.elite_ratio == 0.0

        monkeypatch.setenv("ELITE_RATIO", "1.0")
        settings = Settings()
        assert settings.elite_ratio == 1.0

        # Invalid: negative
        monkeypatch.setenv("ELITE_RATIO", "-0.1")
        with pytest.raises(ValidationError):
            Settings()

        # Invalid: too high
        monkeypatch.setenv("ELITE_RATIO", "1.1")
        with pytest.raises(ValidationError):
            Settings()

    def test_total_requests_needed(self, monkeypatch):
        """Test total_requests_needed property calculation."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("POPULATION_SIZE", "10")
        monkeypatch.setenv("NUM_GENERATIONS", "5")

        settings = Settings()
        assert settings.total_requests_needed == 50

        monkeypatch.setenv("POPULATION_SIZE", "20")
        monkeypatch.setenv("NUM_GENERATIONS", "10")
        settings = Settings()
        assert settings.total_requests_needed == 200

    def test_estimated_runtime_minutes(self, monkeypatch):
        """Test estimated_runtime_minutes calculation."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # With rate limiting enabled (default)
        monkeypatch.setenv("POPULATION_SIZE", "10")
        monkeypatch.setenv("NUM_GENERATIONS", "5")
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        monkeypatch.setenv("REQUESTS_PER_MINUTE", "15")

        settings = Settings()
        # 50 requests / 15 RPM = 3.33 minutes
        assert abs(settings.estimated_runtime_minutes - 3.33) < 0.01

        # With rate limiting disabled
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")
        settings = Settings()
        assert settings.estimated_runtime_minutes == 0.5

    def test_get_mutation_temperature(self, monkeypatch):
        """Test mutation temperature schedule."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("EARLY_MUTATION_TEMP", "1.2")
        monkeypatch.setenv("LATE_MUTATION_TEMP", "0.5")

        settings = Settings()

        # Early generations (0, 1, 2) use high temp
        assert settings.get_mutation_temperature(0) == 1.2
        assert settings.get_mutation_temperature(1) == 1.2
        assert settings.get_mutation_temperature(2) == 1.2

        # Later generations (3+) use low temp
        assert settings.get_mutation_temperature(3) == 0.5
        assert settings.get_mutation_temperature(4) == 0.5
        assert settings.get_mutation_temperature(10) == 0.5

    def test_max_output_tokens_validation(self, monkeypatch):
        """Test max_output_tokens must be between 100 and 8192."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Valid
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "100")
        settings = Settings()
        assert settings.max_output_tokens == 100

        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "8192")
        settings = Settings()
        assert settings.max_output_tokens == 8192

        # Invalid: too low
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "99")
        with pytest.raises(ValidationError):
            Settings()

        # Invalid: too high
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "8193")
        with pytest.raises(ValidationError):
            Settings()
