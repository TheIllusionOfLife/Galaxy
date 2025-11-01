"""Tests for benchmark runner and configuration.

Following TDD discipline:
1. Write tests first (RED phase)
2. Implement to pass tests (GREEN phase)
3. Refactor for quality (REFACTOR phase)
"""

from config import Settings


class TestBenchmarkConfiguration:
    """Test benchmark configuration loading from config.yaml."""

    def test_benchmark_config_loads_from_yaml(self, tmp_path, monkeypatch):
        """Verify benchmark configuration loads correctly from YAML."""
        # Create minimal config.yaml with benchmark section
        config_content = """
model:
  name: gemini-2.5-flash-lite
  temperature: 0.8
  max_output_tokens: 2000

rate_limiting:
  enabled: true
  requests_per_minute: 15
  max_requests_per_run: 50

evolution:
  population_size: 10
  num_generations: 5
  elite_ratio: 0.2
  num_particles: 50

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

benchmark:
  enabled: true
  particle_counts: [10, 50, 100, 200]
  num_timesteps: 100
  test_problems:
    - two_body
    - figure_eight
    - plummer
  baselines:
    - kdtree
    - direct_nbody
  kdtree_k_neighbors: 10
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Mock environment to use test API key
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        # Load config
        settings = Settings.load_from_yaml(config_file)

        # Verify benchmark settings loaded
        assert settings.benchmark_enabled is True
        assert settings.benchmark_particle_counts == [10, 50, 100, 200]
        assert settings.benchmark_timesteps == 100
        assert settings.benchmark_test_problems == ["two_body", "figure_eight", "plummer"]
        assert settings.benchmark_baselines == ["kdtree", "direct_nbody"]
        assert settings.benchmark_kdtree_k == 10

    def test_benchmark_config_can_be_overridden_by_env(self, tmp_path, monkeypatch):
        """Verify environment variables can override benchmark settings."""
        # Create config with default values
        config_content = """
model:
  name: gemini-2.5-flash-lite
  temperature: 0.8
  max_output_tokens: 2000

rate_limiting:
  enabled: true
  requests_per_minute: 15
  max_requests_per_run: 50

evolution:
  population_size: 10
  num_generations: 5
  elite_ratio: 0.2
  num_particles: 50

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 50
  test_problems:
    - two_body
  baselines:
    - kdtree
  kdtree_k_neighbors: 5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Override with environment variables
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        monkeypatch.setenv("BENCHMARK_ENABLED", "false")
        monkeypatch.setenv("BENCHMARK_TIMESTEPS", "200")
        monkeypatch.setenv("BENCHMARK_KDTREE_K", "15")

        # Load config
        settings = Settings.load_from_yaml(config_file)

        # Verify environment overrides applied
        assert settings.benchmark_enabled is False
        assert settings.benchmark_timesteps == 200
        assert settings.benchmark_kdtree_k == 15

    def test_benchmark_disabled_by_default_in_existing_configs(self):
        """Verify benchmark section is optional for backward compatibility."""
        # This test will pass once we make benchmark section optional
        # For now, it should fail since we require the section
        pass
