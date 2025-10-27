"""Configuration management for Galaxy Prometheus project.

This module handles all configuration settings loaded from environment variables
using Pydantic for validation and type safety.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via .env file or environment variables.
    See .env.example for all available options.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Configuration
    google_api_key: str | None = Field(
        None,
        description="Google AI API key from https://aistudio.google.com/apikey"
    )
    anthropic_api_key: str | None = Field(
        None,
        description="Optional Anthropic API key for comparison experiments"
    )

    @field_validator("google_api_key", "anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Reject placeholder API key values."""
        if v is None:
            return v
        # Check for common placeholder patterns
        placeholders = [
            "your_api_key_here",
            "your-api-key-here",
            "INSERT_KEY_HERE",
            "paste_your_key_here",
            "replace_with_your_key",
        ]
        if v.lower() in placeholders:
            raise ValueError(
                f"Please replace placeholder API key with a real key from "
                f"https://aistudio.google.com/apikey"
            )
        return v

    # Model Selection
    llm_model: str = Field(
        "gemini-2.5-flash-lite",
        description="Gemini model to use for code generation"
    )
    temperature: float = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0-2.0, higher = more creative)"
    )
    max_output_tokens: int = Field(
        2000,
        ge=100,
        le=8192,
        description="Maximum tokens in LLM response"
    )

    # Rate Limiting (Free tier: 15 RPM, 1000 RPD)
    max_requests_per_run: int = Field(
        50,
        description="Maximum LLM calls per evolution run"
    )
    enable_rate_limiting: bool = Field(
        True,
        description="Enable rate limiting for free tier (15 RPM)"
    )
    requests_per_minute: int = Field(
        15,
        description="Free tier rate limit: 15 requests per minute"
    )

    # Evolution Parameters
    population_size: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of surrogate models per generation"
    )
    num_generations: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of evolutionary generations to run"
    )
    elite_ratio: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of top performers to keep for breeding"
    )

    # Mutation Strategy
    early_mutation_temp: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="High temperature for early generations (exploration)"
    )
    late_mutation_temp: float = Field(
        0.6,
        ge=0.0,
        le=2.0,
        description="Low temperature for late generations (exploitation)"
    )

    @property
    def total_requests_needed(self) -> int:
        """Calculate total LLM calls needed for one complete evolution run.

        Returns:
            Total number of API calls (population_size x num_generations)
        """
        return self.population_size * self.num_generations

    @property
    def estimated_runtime_minutes(self) -> float:
        """Estimate runtime based on rate limiting.

        Returns:
            Estimated minutes to complete evolution run
        """
        if not self.enable_rate_limiting:
            return 0.5  # Assume ~30 seconds without rate limiting
        return self.total_requests_needed / self.requests_per_minute

    def get_mutation_temperature(self, generation: int) -> float:
        """Get appropriate temperature for given generation.

        Uses high temperature early for exploration, low temperature later
        for exploitation/refinement.

        Args:
            generation: Current generation number (0-indexed)

        Returns:
            Temperature value to use for this generation
        """
        if generation < 3:
            return self.early_mutation_temp
        return self.late_mutation_temp


# Global settings instance
# Will be initialized when imported, loading from .env or environment
settings = Settings()
