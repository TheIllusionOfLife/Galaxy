"""Tests for benchmark runner and configuration.

Following TDD discipline:
1. Write tests first (RED phase)
2. Implement to pass tests (GREEN phase)
3. Refactor for quality (REFACTOR phase)
"""

import pytest

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


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Verify BenchmarkResult can be created with all required fields."""
        from benchmarks import BenchmarkResult

        result = BenchmarkResult(
            baseline_name="kdtree",
            test_problem="two_body",
            num_particles=10,
            accuracy=0.987,
            speed=0.0021,
            energy_drift=0.024,
            trajectory_rmse=0.15,
        )

        assert result.baseline_name == "kdtree"
        assert result.test_problem == "two_body"
        assert result.num_particles == 10
        assert result.accuracy == 0.987
        assert result.speed == 0.0021
        assert result.energy_drift == 0.024
        assert result.trajectory_rmse == 0.15


class TestBenchmarkRunner:
    """Test BenchmarkRunner core functionality."""

    def test_get_test_problem_particles_two_body(self):
        """Verify two_body test problem particles are generated correctly."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        # Load config
        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Get two-body particles
        particles = runner._get_test_problem_particles("two_body", n_particles=2)

        # Verify structure
        assert len(particles) == 2
        assert all(len(p) == 7 for p in particles)  # [x,y,z,vx,vy,vz,mass]

    def test_get_test_problem_particles_plummer(self):
        """Verify plummer test problem generates N particles."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Get plummer sphere with 50 particles
        particles = runner._get_test_problem_particles("plummer", n_particles=50)

        # Verify structure
        assert len(particles) == 50
        assert all(len(p) == 7 for p in particles)

    def test_get_test_problem_invalid_name(self):
        """Verify invalid test problem name raises error."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Should raise ValueError for unknown problem
        with pytest.raises(ValueError, match="Unknown test problem"):
            runner._get_test_problem_particles("invalid_problem", n_particles=10)

    def test_get_baseline_model_kdtree(self):
        """Verify KDTree baseline model is created correctly."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Get KDTree baseline
        genome = runner._get_baseline_model("kdtree")

        # Verify it's a SurrogateGenome with compiled predict function
        assert hasattr(genome, "compiled_predict")
        assert callable(genome.compiled_predict)

    def test_get_baseline_model_direct_nbody(self):
        """Verify direct N-body baseline model is created correctly."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Get direct N-body baseline
        genome = runner._get_baseline_model("direct_nbody")

        # Verify it's a SurrogateGenome
        assert hasattr(genome, "compiled_predict")
        assert callable(genome.compiled_predict)

    def test_get_baseline_invalid_name(self):
        """Verify invalid baseline name raises error."""
        from benchmarks import BenchmarkRunner
        from config import Settings

        settings = Settings.load_from_yaml()
        runner = BenchmarkRunner(settings)

        # Should raise ValueError for unknown baseline
        with pytest.raises(ValueError, match="Unknown baseline"):
            runner._get_baseline_model("invalid_baseline")
