"""Unit tests for multi-problem validation configuration.

Tests the test_problem field in config.yaml and Settings class,
following TDD discipline (red → green → refactor).
"""

import pytest

from config import Settings


class TestMultiProblemConfiguration:
    """Test suite for test_problem configuration field."""

    def test_default_test_problem_is_plummer(self, tmp_path):
        """Test that default test_problem is plummer for backward compatibility."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        # Load settings (without test_problem field - should default to plummer)
        settings = Settings.load_from_yaml(config_yaml)

        # Should default to plummer
        assert settings.evolution_test_problem == "plummer"

    def test_test_problem_can_be_set_to_two_body(self, tmp_path):
        """Test that test_problem can be configured as two_body."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  test_problem: two_body
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        settings = Settings.load_from_yaml(config_yaml)
        assert settings.evolution_test_problem == "two_body"

    def test_test_problem_can_be_set_to_figure_eight(self, tmp_path):
        """Test that test_problem can be configured as figure_eight."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  test_problem: figure_eight
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        settings = Settings.load_from_yaml(config_yaml)
        assert settings.evolution_test_problem == "figure_eight"

    def test_test_problem_can_be_set_to_plummer(self, tmp_path):
        """Test that test_problem can be explicitly configured as plummer."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  test_problem: plummer
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        settings = Settings.load_from_yaml(config_yaml)
        assert settings.evolution_test_problem == "plummer"

    def test_invalid_test_problem_raises_validation_error(self, tmp_path):
        """Test that invalid test_problem value raises validation error."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  test_problem: invalid_problem
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        with pytest.raises(ValueError, match="Invalid test_problem"):
            Settings.load_from_yaml(config_yaml)

    def test_test_problem_can_be_overridden_by_env_var(self, tmp_path, monkeypatch):
        """Test that TEST_PROBLEM environment variable overrides config.yaml."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  test_problem: plummer
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        # Override with environment variable
        monkeypatch.setenv("TEST_PROBLEM", "two_body")

        settings = Settings.load_from_yaml(config_yaml)
        assert settings.evolution_test_problem == "two_body"

    def test_num_particles_field_exists(self, tmp_path):
        """Test that num_particles field is accessible in settings."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
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
  num_particles: 100
  test_problem: plummer
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
  particle_counts: [10, 50, 100]
  num_timesteps: 100
  test_problems: [two_body, figure_eight, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        )

        settings = Settings.load_from_yaml(config_yaml)
        assert settings.evolution_num_particles == 100
