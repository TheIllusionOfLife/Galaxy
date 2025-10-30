"""Tests for configuration management."""

import pytest
from pydantic import ValidationError

from config import Settings


class TestSettings:
    """Test Settings class configuration and validation."""

    def test_settings_defaults(self):
        """Test default values loaded from config.yaml."""
        from config import settings

        # Model defaults (from config.yaml)
        assert settings.llm_model == "gemini-2.5-flash-lite"
        assert settings.temperature == 0.8
        assert settings.max_output_tokens == 2000

        # Rate limiting defaults (from config.yaml)
        assert settings.max_requests_per_run == 50
        assert settings.enable_rate_limiting is True
        assert settings.requests_per_minute == 15

        # Evolution defaults (from config.yaml)
        assert settings.population_size == 10
        assert settings.num_generations == 5
        assert settings.elite_ratio == 0.2

        # Mutation defaults (from config.yaml)
        assert settings.early_mutation_temp == 1.0
        assert settings.late_mutation_temp == 0.6

        # Code penalty defaults (from config.yaml)
        assert settings.enable_code_length_penalty is True
        assert settings.code_length_penalty_weight == 0.1
        assert settings.max_acceptable_tokens == 400  # Updated from 2000 based on PR #21 findings

    def test_settings_from_env(self, monkeypatch):
        """Test environment variables can override config.yaml values."""

        # Set environment variables to override YAML defaults
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-override")
        monkeypatch.setenv("TEMPERATURE", "1.5")
        monkeypatch.setenv("POPULATION_SIZE", "20")
        monkeypatch.setenv("NUM_GENERATIONS", "10")

        # Reload settings with environment overrides
        settings = Settings.load_from_yaml()

        assert settings.google_api_key == "test-api-key-override"  # pragma: allowlist secret
        assert settings.temperature == 1.5
        assert settings.population_size == 20
        assert settings.num_generations == 10

    def test_temperature_validation(self, monkeypatch):
        """Test temperature must be between 0.0 and 2.0."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Valid temperatures
        monkeypatch.setenv("TEMPERATURE", "0.0")
        settings = Settings.load_from_yaml()
        assert settings.temperature == 0.0

        monkeypatch.setenv("TEMPERATURE", "2.0")
        settings = Settings.load_from_yaml()
        assert settings.temperature == 2.0

        monkeypatch.setenv("TEMPERATURE", "1.0")
        settings = Settings.load_from_yaml()
        assert settings.temperature == 1.0

        # Invalid: too low
        monkeypatch.setenv("TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

        # Invalid: too high
        monkeypatch.setenv("TEMPERATURE", "2.1")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

    def test_population_size_validation(self, monkeypatch):
        """Test population_size must be between 1 and 100."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Valid
        monkeypatch.setenv("POPULATION_SIZE", "1")
        settings = Settings.load_from_yaml()
        assert settings.population_size == 1

        monkeypatch.setenv("POPULATION_SIZE", "100")
        settings = Settings.load_from_yaml()
        assert settings.population_size == 100

        # Invalid: too low
        monkeypatch.setenv("POPULATION_SIZE", "0")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

        # Invalid: too high
        monkeypatch.setenv("POPULATION_SIZE", "101")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

    def test_elite_ratio_validation(self, monkeypatch):
        """Test elite_ratio must be between 0.0 and 1.0."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Valid
        monkeypatch.setenv("ELITE_RATIO", "0.0")
        settings = Settings.load_from_yaml()
        assert settings.elite_ratio == 0.0

        monkeypatch.setenv("ELITE_RATIO", "1.0")
        settings = Settings.load_from_yaml()
        assert settings.elite_ratio == 1.0

        # Invalid: negative
        monkeypatch.setenv("ELITE_RATIO", "-0.1")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

        # Invalid: too high
        monkeypatch.setenv("ELITE_RATIO", "1.1")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

    def test_total_requests_needed(self, monkeypatch):
        """Test total_requests_needed property calculation."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("POPULATION_SIZE", "10")
        monkeypatch.setenv("NUM_GENERATIONS", "5")

        settings = Settings.load_from_yaml()
        assert settings.total_requests_needed == 50

        monkeypatch.setenv("POPULATION_SIZE", "20")
        monkeypatch.setenv("NUM_GENERATIONS", "10")
        settings = Settings.load_from_yaml()
        assert settings.total_requests_needed == 200

    def test_estimated_runtime_minutes(self, monkeypatch):
        """Test estimated_runtime_minutes calculation."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # With rate limiting enabled (default)
        monkeypatch.setenv("POPULATION_SIZE", "10")
        monkeypatch.setenv("NUM_GENERATIONS", "5")
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
        monkeypatch.setenv("REQUESTS_PER_MINUTE", "15")

        settings = Settings.load_from_yaml()
        # 50 requests / 15 RPM = 3.33 minutes
        assert abs(settings.estimated_runtime_minutes - 3.33) < 0.01

        # With rate limiting disabled
        monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")
        settings = Settings.load_from_yaml()
        assert settings.estimated_runtime_minutes == 0.5

    def test_get_mutation_temperature(self, monkeypatch):
        """Test mutation temperature schedule."""

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("EARLY_MUTATION_TEMP", "1.2")
        monkeypatch.setenv("LATE_MUTATION_TEMP", "0.5")

        settings = Settings.load_from_yaml()

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

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Valid
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "100")
        settings = Settings.load_from_yaml()
        assert settings.max_output_tokens == 100

        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "8192")
        settings = Settings.load_from_yaml()
        assert settings.max_output_tokens == 8192

        # Invalid: too low
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "99")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()

        # Invalid: too high
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", "8193")
        with pytest.raises(ValidationError):
            Settings.load_from_yaml()
