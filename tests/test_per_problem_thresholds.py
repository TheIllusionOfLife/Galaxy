"""Tests for per-problem physics threshold configuration."""

from pathlib import Path as PathlibPath

import pytest

from config import Settings


class TestPerProblemThresholds:
    """Test per-problem threshold loading and retrieval."""

    def test_per_problem_threshold_loading(self, tmp_path):
        """Test that per-problem thresholds load correctly from config."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_energy_drift: 0.002
      max_momentum_drift: 0.002
    plummer:
      max_energy_drift: 0.200
      max_momentum_drift: 0.200
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        assert settings.fitness_per_problem_thresholds is not None
        assert "two_body" in settings.fitness_per_problem_thresholds
        assert "plummer" in settings.fitness_per_problem_thresholds
        assert settings.fitness_per_problem_thresholds["two_body"]["max_energy_drift"] == 0.002
        assert settings.fitness_per_problem_thresholds["plummer"]["max_energy_drift"] == 0.200

    def test_threshold_retrieval_with_override(self, tmp_path):
        """Test threshold retrieval when per-problem override exists."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_energy_drift: 0.002
      max_momentum_drift: 0.002
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # two_body should use override
        assert settings.get_physics_threshold("two_body", "energy") == 0.002
        assert settings.get_physics_threshold("two_body", "momentum") == 0.002

        # plummer should fall back to global
        assert settings.get_physics_threshold("plummer", "energy") == 0.10
        assert settings.get_physics_threshold("plummer", "momentum") == 0.50

    def test_backward_compatibility_no_overrides(self, tmp_path):
        """Test that config without per_problem_thresholds still works."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # All problems should use global thresholds
        assert settings.get_physics_threshold("two_body", "energy") == 0.10
        assert settings.get_physics_threshold("plummer", "energy") == 0.10
        assert settings.get_physics_threshold("figure_eight", "momentum") == 0.50

    def test_invalid_problem_name(self, tmp_path):
        """Test that invalid problem names are rejected."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    invalid_problem:
      max_energy_drift: 0.002
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="Invalid problem 'invalid_problem'"):
            Settings.load_from_yaml(PathlibPath(config_path))

    def test_threshold_range_validation(self, tmp_path):
        """Test that threshold values outside valid range are rejected."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_energy_drift: 15.0
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="must be between 0.0 and 10.0"):
            Settings.load_from_yaml(PathlibPath(config_path))

    def test_invalid_threshold_key(self, tmp_path):
        """Test that invalid threshold keys are rejected."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_momentum_drfit: 0.002
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="Invalid threshold key 'max_momentum_drfit'"):
            Settings.load_from_yaml(PathlibPath(config_path))

    def test_invalid_metric_in_get_threshold(self, tmp_path):
        """Test that get_physics_threshold rejects invalid metrics."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # Test invalid metric raises ValueError
        with pytest.raises(ValueError, match="Invalid metric 'invalid'"):
            settings.get_physics_threshold("two_body", "invalid")

    def test_partial_threshold_override(self, tmp_path):
        """Test that partial overrides (only energy or only momentum) work."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_energy_drift: 0.002
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # Energy should use override, momentum should fall back to global
        assert settings.get_physics_threshold("two_body", "energy") == 0.002
        assert settings.get_physics_threshold("two_body", "momentum") == 0.50

    def test_empty_per_problem_thresholds(self, tmp_path):
        """Test that empty per_problem_thresholds dict works (all use global)."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds: {}
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # All problems should fall back to global thresholds
        assert settings.get_physics_threshold("two_body", "energy") == 0.10
        assert settings.get_physics_threshold("plummer", "energy") == 0.10
        assert settings.get_physics_threshold("figure_eight", "momentum") == 0.50

    def test_multiple_problems_in_same_config(self, tmp_path):
        """Test that config with multiple problem overrides loads correctly."""
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
  test_problem: plummer

mutation:
  early_temp: 1.0
  late_temp: 0.6

code_penalty:
  enabled: true
  weight: 0.1
  max_tokens: 400

physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01
  angular_momentum_threshold: 0.01
  validation_timesteps: 10

crossover:
  enabled: true
  crossover_rate: 0.5
  temperature: 0.75

fitness:
  enable_hard_constraint: true
  max_energy_drift: 0.10
  max_momentum_drift: 0.50
  per_problem_thresholds:
    two_body:
      max_energy_drift: 0.002
      max_momentum_drift: 0.002
    figure_eight:
      max_energy_drift: 0.015
      max_momentum_drift: 0.015
    plummer:
      max_energy_drift: 0.200
      max_momentum_drift: 0.200
  use_log_speed: true
  speed_log_base: 10.0

benchmark:
  enabled: true
  particle_counts: [10, 50]
  num_timesteps: 100
  test_problems: [two_body, plummer]
  baselines: [kdtree, direct_nbody]
  kdtree_k_neighbors: 10
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        settings = Settings.load_from_yaml(PathlibPath(config_path))

        # Verify all three problem overrides load correctly
        assert settings.get_physics_threshold("two_body", "energy") == 0.002
        assert settings.get_physics_threshold("two_body", "momentum") == 0.002
        assert settings.get_physics_threshold("figure_eight", "energy") == 0.015
        assert settings.get_physics_threshold("figure_eight", "momentum") == 0.015
        assert settings.get_physics_threshold("plummer", "energy") == 0.200
        assert settings.get_physics_threshold("plummer", "momentum") == 0.200
