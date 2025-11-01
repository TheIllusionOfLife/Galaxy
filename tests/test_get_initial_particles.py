"""Unit tests for get_initial_particles() helper function.

Tests the mapping from test_problem configuration to initial_conditions functions,
following TDD discipline (red → green → refactor).
"""

import pytest

from prototype import get_initial_particles


class TestGetInitialParticles:
    """Test suite for get_initial_particles() helper function."""

    def test_two_body_returns_two_particles(self):
        """Test that two_body problem returns exactly 2 particles."""
        particles = get_initial_particles("two_body", num_particles=50)

        assert len(particles) == 2, "two_body must always return 2 particles"
        # Verify 7-element format
        assert all(len(p) == 7 for p in particles), "All particles must have 7 elements"

    def test_figure_eight_returns_three_particles(self):
        """Test that figure_eight problem returns exactly 3 particles."""
        particles = get_initial_particles("figure_eight", num_particles=50)

        assert len(particles) == 3, "figure_eight must always return 3 particles"
        # Verify 7-element format
        assert all(len(p) == 7 for p in particles), "All particles must have 7 elements"

    def test_plummer_respects_num_particles_parameter(self):
        """Test that plummer problem uses num_particles parameter."""
        particles_10 = get_initial_particles("plummer", num_particles=10)
        particles_50 = get_initial_particles("plummer", num_particles=50)
        particles_100 = get_initial_particles("plummer", num_particles=100)

        assert len(particles_10) == 10
        assert len(particles_50) == 50
        assert len(particles_100) == 100

        # Verify 7-element format
        assert all(len(p) == 7 for p in particles_10)
        assert all(len(p) == 7 for p in particles_50)
        assert all(len(p) == 7 for p in particles_100)

    def test_plummer_default_particles(self):
        """Test that plummer uses default value when num_particles not provided."""
        particles = get_initial_particles("plummer")

        # Should use default from function signature or parameter
        assert len(particles) >= 2, "plummer must return at least 2 particles"
        assert all(len(p) == 7 for p in particles)

    def test_two_body_ignores_num_particles_parameter(self):
        """Test that two_body always returns 2 particles regardless of num_particles."""
        particles_requested_100 = get_initial_particles("two_body", num_particles=100)

        # Should still be 2, not 100
        assert len(particles_requested_100) == 2, "two_body must ignore num_particles"

    def test_figure_eight_ignores_num_particles_parameter(self):
        """Test that figure_eight always returns 3 particles regardless of num_particles."""
        particles_requested_100 = get_initial_particles("figure_eight", num_particles=100)

        # Should still be 3, not 100
        assert len(particles_requested_100) == 3, "figure_eight must ignore num_particles"

    def test_invalid_test_problem_raises_error(self):
        """Test that invalid test_problem raises descriptive error."""
        with pytest.raises(ValueError, match="Unknown test_problem"):
            get_initial_particles("invalid_problem", num_particles=50)

    def test_particles_have_valid_7_element_format(self):
        """Test that all particles have correct [x,y,z,vx,vy,vz,mass] format."""
        for test_problem in ["two_body", "figure_eight", "plummer"]:
            particles = get_initial_particles(test_problem, num_particles=20)

            for i, particle in enumerate(particles):
                assert len(particle) == 7, f"{test_problem} particle {i} must have 7 elements"
                assert all(isinstance(v, (int, float)) for v in particle), (
                    f"{test_problem} particle {i} must contain numeric values"
                )
                assert particle[6] > 0, f"{test_problem} particle {i} must have positive mass"

    def test_particles_satisfy_conservation_laws(self):
        """Test that initial conditions satisfy basic conservation laws."""
        for test_problem in ["two_body", "figure_eight"]:
            particles = get_initial_particles(test_problem, num_particles=50)

            # Center of mass should be at origin
            com_x = sum(p[0] * p[6] for p in particles) / sum(p[6] for p in particles)
            com_y = sum(p[1] * p[6] for p in particles) / sum(p[6] for p in particles)
            com_z = sum(p[2] * p[6] for p in particles) / sum(p[6] for p in particles)

            assert abs(com_x) < 1e-10, f"{test_problem}: COM x should be at origin"
            assert abs(com_y) < 1e-10, f"{test_problem}: COM y should be at origin"
            assert abs(com_z) < 1e-10, f"{test_problem}: COM z should be at origin"

            # Total momentum should be zero
            momentum_x = sum(p[3] * p[6] for p in particles)
            momentum_y = sum(p[4] * p[6] for p in particles)
            momentum_z = sum(p[5] * p[6] for p in particles)

            assert abs(momentum_x) < 1e-10, f"{test_problem}: momentum x should be zero"
            assert abs(momentum_y) < 1e-10, f"{test_problem}: momentum y should be zero"
            assert abs(momentum_z) < 1e-10, f"{test_problem}: momentum z should be zero"

    def test_plummer_has_positive_random_seed(self):
        """Test that plummer sphere is reproducible with fixed random seed."""
        particles_run1 = get_initial_particles("plummer", num_particles=10)
        particles_run2 = get_initial_particles("plummer", num_particles=10)

        # With default random_seed=42, should be identical
        for i in range(len(particles_run1)):
            for j in range(7):
                assert particles_run1[i][j] == particles_run2[i][j], (
                    f"plummer should be reproducible at particle {i}, element {j}"
                )

    def test_function_signature_flexibility(self):
        """Test that function can be called with different parameter styles."""
        # Positional arguments
        particles1 = get_initial_particles("plummer", 20)

        # Keyword arguments
        particles2 = get_initial_particles(test_problem="plummer", num_particles=20)

        # Mixed
        particles3 = get_initial_particles("plummer", num_particles=20)

        # All should work and return same count
        assert len(particles1) == 20
        assert len(particles2) == 20
        assert len(particles3) == 20
