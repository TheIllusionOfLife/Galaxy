"""Configuration management for Galaxy Prometheus project.

This module handles all configuration settings loaded from config.yaml
(single source of truth) and .env file (secrets only).
Uses Pydantic for validation and type safety.
"""

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from config.yaml and .env.

    - config.yaml: All configuration defaults (model, hyperparameters, feature flags)
    - .env: Secrets only (API keys)
    - Environment variables: Can override any setting for one-off runs

    See config.yaml for all configuration options.
    See .env.example for required API keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Configuration (from .env only)
    google_api_key: str | None = Field(
        None, description="Google AI API key from https://aistudio.google.com/apikey"
    )
    anthropic_api_key: str | None = Field(
        None, description="Optional Anthropic API key for comparison experiments"
    )

    @field_validator("google_api_key", "anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Reject placeholder API key values (except in test environment).

        Skips validation if value starts with 'test-' to allow pytest fixtures
        to set mock API keys before Settings is instantiated.
        """
        if v is None:
            return v

        # Allow test API keys (from conftest.py fixture)
        if v.startswith("test-"):
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
                "Please replace placeholder API key with a real key from "
                "https://aistudio.google.com/apikey"
            )
        return v

    # Model Selection (from config.yaml, no defaults here)
    llm_model: str = Field(description="Gemini model to use for code generation")
    temperature: float = Field(
        ge=0.0, le=2.0, description="LLM temperature (0.0-2.0, higher = more creative)"
    )
    max_output_tokens: int = Field(ge=100, le=8192, description="Maximum tokens in LLM response")

    # Rate Limiting (from config.yaml, no defaults here)
    max_requests_per_run: int = Field(description="Maximum LLM calls per evolution run")
    enable_rate_limiting: bool = Field(description="Enable rate limiting for free tier (15 RPM)")
    requests_per_minute: int = Field(description="Free tier rate limit: 15 requests per minute")

    # Evolution Parameters (from config.yaml, no defaults here)
    population_size: int = Field(
        ge=1, le=100, description="Number of surrogate models per generation"
    )
    num_generations: int = Field(
        ge=1, le=100, description="Number of evolutionary generations to run"
    )
    elite_ratio: float = Field(
        ge=0.0, le=1.0, description="Fraction of top performers to keep for breeding"
    )
    evolution_num_particles: int = Field(
        ge=2, le=1000, description="Number of particles for plummer test problem"
    )
    evolution_test_problem: str = Field(
        description="Test problem to use (two_body, figure_eight, plummer)"
    )

    # Mutation Strategy (from config.yaml, no defaults here)
    early_mutation_temp: float = Field(
        ge=0.0, le=2.0, description="High temperature for early generations (exploration)"
    )
    late_mutation_temp: float = Field(
        ge=0.0, le=2.0, description="Low temperature for late generations (exploitation)"
    )

    # Code Length Penalty (from config.yaml, no defaults here)
    enable_code_length_penalty: bool = Field(
        description="Enable fitness penalty for long code to prevent bloat"
    )
    code_length_penalty_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Penalty weight (0.0 = no penalty, 1.0 = maximum penalty)",
    )
    max_acceptable_tokens: int = Field(
        ge=100,
        le=10000,
        description="Token count threshold before penalty applies",
    )

    # Crossover Configuration (from config.yaml, no defaults here)
    enable_crossover: bool = Field(
        description="Enable crossover operator (LLM-based code recombination)"
    )
    crossover_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction using crossover (rest uses mutation)"
    )
    crossover_temperature: float = Field(
        ge=0.0, le=2.0, description="Crossover creativity level (between explore/exploit)"
    )

    # Benchmark Suite Configuration (from config.yaml, no defaults here)
    benchmark_enabled: bool = Field(description="Enable benchmark suite execution")
    benchmark_particle_counts: list[int] = Field(
        description="Particle counts for scaling analysis (e.g., [10, 50, 100, 200])"
    )
    benchmark_timesteps: int = Field(
        ge=10, le=1000, description="Number of integration timesteps per benchmark run"
    )
    benchmark_test_problems: list[str] = Field(
        description="Test problems to evaluate (two_body, figure_eight, plummer)"
    )
    benchmark_baselines: list[str] = Field(
        description="Baseline models to benchmark (kdtree, direct_nbody)"
    )
    benchmark_kdtree_k: int = Field(
        ge=1, le=100, description="K-nearest neighbors for KDTree baseline"
    )

    @field_validator("benchmark_test_problems")
    @classmethod
    def validate_test_problems(cls, v: list[str]) -> list[str]:
        """Validate test problem names against known problems."""
        valid_problems = {"two_body", "figure_eight", "plummer"}
        invalid = set(v) - valid_problems
        if invalid:
            raise ValueError(f"Invalid test problems: {invalid}. Valid options: {valid_problems}")
        return v

    @field_validator("benchmark_baselines")
    @classmethod
    def validate_baselines(cls, v: list[str]) -> list[str]:
        """Validate baseline names against known baselines."""
        valid_baselines = {"kdtree", "direct_nbody"}
        invalid = set(v) - valid_baselines
        if invalid:
            raise ValueError(f"Invalid baselines: {invalid}. Valid options: {valid_baselines}")
        return v

    @field_validator("evolution_test_problem")
    @classmethod
    def validate_evolution_test_problem(cls, v: str) -> str:
        """Validate evolution test_problem against known problems."""
        valid_problems = {"two_body", "figure_eight", "plummer"}
        if v not in valid_problems:
            raise ValueError(f"Invalid test_problem: {v}. Valid options: {valid_problems}")
        return v

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

    @classmethod
    def load_from_yaml(cls, yaml_path: Path = Path("config.yaml")) -> "Settings":
        """Load configuration from YAML file and .env.

        This is the ONLY way to create Settings instances. It enforces the
        "Once and Only Once" principle by loading ALL defaults from config.yaml
        (single source of truth) and secrets from .env.

        Priority (highest to lowest):
        1. Environment variables (runtime overrides)
        2. .env file (secrets)
        3. config.yaml (defaults)

        Args:
            yaml_path: Path to configuration YAML file (default: config.yaml)

        Returns:
            Settings instance with values from YAML + .env + environment variables

        Raises:
            FileNotFoundError: If config.yaml doesn't exist
            ValueError: If YAML structure is invalid
        """
        import os

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}\n"
                "Please ensure config.yaml exists in the project root.\n"
                "See config.yaml for required structure."
            )

        # Load and parse YAML with error handling
        try:
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Failed to parse {yaml_path}. Please check YAML syntax.\nError: {e}"
            ) from e

        # Validate YAML structure
        required_sections = [
            "model",
            "rate_limiting",
            "evolution",
            "mutation",
            "code_penalty",
            "crossover",
            "benchmark",
        ]
        missing_sections = [s for s in required_sections if s not in yaml_config]
        if missing_sections:
            raise ValueError(
                f"Invalid config.yaml structure. Missing sections: {missing_sections}\n"
                "See config.yaml for required structure."
            )

        # Flatten YAML structure with environment variable overrides
        # Environment variables take precedence over YAML defaults
        try:
            config_data = {
                # Model configuration
                "llm_model": os.getenv("LLM_MODEL", yaml_config["model"]["name"]),
                "temperature": float(os.getenv("TEMPERATURE", yaml_config["model"]["temperature"])),
                "max_output_tokens": int(
                    os.getenv("MAX_OUTPUT_TOKENS", yaml_config["model"]["max_output_tokens"])
                ),
                # Rate limiting
                "enable_rate_limiting": os.getenv(
                    "ENABLE_RATE_LIMITING", str(yaml_config["rate_limiting"]["enabled"])
                ).lower()
                == "true",
                "requests_per_minute": int(
                    os.getenv(
                        "REQUESTS_PER_MINUTE", yaml_config["rate_limiting"]["requests_per_minute"]
                    )
                ),
                "max_requests_per_run": int(
                    os.getenv(
                        "MAX_REQUESTS_PER_RUN", yaml_config["rate_limiting"]["max_requests_per_run"]
                    )
                ),
                # Evolution
                "population_size": int(
                    os.getenv("POPULATION_SIZE", yaml_config["evolution"]["population_size"])
                ),
                "num_generations": int(
                    os.getenv("NUM_GENERATIONS", yaml_config["evolution"]["num_generations"])
                ),
                "elite_ratio": float(
                    os.getenv("ELITE_RATIO", yaml_config["evolution"]["elite_ratio"])
                ),
                "evolution_num_particles": int(
                    os.getenv("NUM_PARTICLES", yaml_config["evolution"]["num_particles"])
                ),
                "evolution_test_problem": os.getenv(
                    "TEST_PROBLEM", yaml_config["evolution"].get("test_problem", "plummer")
                ),
                # Mutation
                "early_mutation_temp": float(
                    os.getenv("EARLY_MUTATION_TEMP", yaml_config["mutation"]["early_temp"])
                ),
                "late_mutation_temp": float(
                    os.getenv("LATE_MUTATION_TEMP", yaml_config["mutation"]["late_temp"])
                ),
                # Code penalty
                "enable_code_length_penalty": os.getenv(
                    "ENABLE_CODE_LENGTH_PENALTY", str(yaml_config["code_penalty"]["enabled"])
                ).lower()
                == "true",
                "code_length_penalty_weight": float(
                    os.getenv("CODE_LENGTH_PENALTY_WEIGHT", yaml_config["code_penalty"]["weight"])
                ),
                "max_acceptable_tokens": int(
                    os.getenv("MAX_ACCEPTABLE_TOKENS", yaml_config["code_penalty"]["max_tokens"])
                ),
                # Crossover
                "enable_crossover": os.getenv(
                    "ENABLE_CROSSOVER", str(yaml_config["crossover"]["enabled"])
                ).lower()
                == "true",
                "crossover_rate": float(
                    os.getenv("CROSSOVER_RATE", yaml_config["crossover"]["crossover_rate"])
                ),
                "crossover_temperature": float(
                    os.getenv("CROSSOVER_TEMPERATURE", yaml_config["crossover"]["temperature"])
                ),
                # Benchmark
                "benchmark_enabled": os.getenv(
                    "BENCHMARK_ENABLED", str(yaml_config["benchmark"]["enabled"])
                ).lower()
                == "true",
                "benchmark_particle_counts": yaml_config["benchmark"]["particle_counts"],
                "benchmark_timesteps": int(
                    os.getenv("BENCHMARK_TIMESTEPS", yaml_config["benchmark"]["num_timesteps"])
                ),
                "benchmark_test_problems": yaml_config["benchmark"]["test_problems"],
                "benchmark_baselines": yaml_config["benchmark"]["baselines"],
                "benchmark_kdtree_k": int(
                    os.getenv("BENCHMARK_KDTREE_K", yaml_config["benchmark"]["kdtree_k_neighbors"])
                ),
            }
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid config.yaml structure. Check that all required keys exist.\n"
                f"Error: {e}\n"
                "See config.yaml for required structure."
            ) from e

        # Create Settings instance
        # Pydantic will automatically load .env file for API keys
        return cls(**config_data)  # type: ignore[arg-type]


# Global settings instance
# Loads from config.yaml (defaults) + .env (secrets) + environment (overrides)
settings = Settings.load_from_yaml()
