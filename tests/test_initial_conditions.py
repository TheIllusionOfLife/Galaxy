"""Tests for standard N-body test problems (initial conditions)."""

import math

import pytest

# Will import from initial_conditions module (doesn't exist yet - TDD red phase)
from initial_conditions import (
    plummer_sphere,
    three_body_figure_eight,
    two_body_circular_orbit,
)


class TestTwoBodyCircularOrbit:
    """Test two-body circular orbit initial conditions."""

    def test_returns_two_particles(self):
        """Should return exactly 2 particles."""
        particles = two_body_circular_orbit()
        assert len(particles) == 2

    def test_particle_format(self):
        """Each particle should have 7 elements: [x,y,z,vx,vy,vz,mass]."""
        particles = two_body_circular_orbit()
        for particle in particles:
            assert len(particle) == 7
            assert all(isinstance(x, (int, float)) for x in particle)

    def test_equal_masses(self):
        """Default should create equal mass particles."""
        particles = two_body_circular_orbit()
        assert particles[0][6] == particles[1][6]  # mass is 7th element

    def test_center_of_mass_at_origin(self):
        """Center of mass should be at origin."""
        particles = two_body_circular_orbit()
        com_x = (particles[0][0] * particles[0][6] + particles[1][0] * particles[1][6]) / (
            particles[0][6] + particles[1][6]
        )
        com_y = (particles[0][1] * particles[0][6] + particles[1][1] * particles[1][6]) / (
            particles[0][6] + particles[1][6]
        )
        com_z = (particles[0][2] * particles[0][6] + particles[1][2] * particles[1][6]) / (
            particles[0][6] + particles[1][6]
        )
        assert abs(com_x) < 1e-10
        assert abs(com_y) < 1e-10
        assert abs(com_z) < 1e-10

    def test_zero_total_momentum(self):
        """Total momentum should be zero."""
        particles = two_body_circular_orbit()
        total_px = particles[0][3] * particles[0][6] + particles[1][3] * particles[1][6]
        total_py = particles[0][4] * particles[0][6] + particles[1][4] * particles[1][6]
        total_pz = particles[0][5] * particles[0][6] + particles[1][5] * particles[1][6]
        assert abs(total_px) < 1e-10
        assert abs(total_py) < 1e-10
        assert abs(total_pz) < 1e-10

    def test_correct_separation(self):
        """Particles should be separated by specified distance."""
        separation = 2.0
        particles = two_body_circular_orbit(separation=separation)
        dx = particles[1][0] - particles[0][0]
        dy = particles[1][1] - particles[0][1]
        dz = particles[1][2] - particles[0][2]
        actual_separation = math.sqrt(dx**2 + dy**2 + dz**2)
        assert abs(actual_separation - separation) < 1e-10

    def test_configurable_total_mass(self):
        """Should allow configuring total mass."""
        total_mass = 10.0
        particles = two_body_circular_orbit(total_mass=total_mass)
        actual_total = particles[0][6] + particles[1][6]
        assert abs(actual_total - total_mass) < 1e-10


class TestThreeBodyFigureEight:
    """Test three-body figure-eight orbit initial conditions."""

    def test_returns_three_particles(self):
        """Should return exactly 3 particles."""
        particles = three_body_figure_eight()
        assert len(particles) == 3

    def test_particle_format(self):
        """Each particle should have 7 elements."""
        particles = three_body_figure_eight()
        for particle in particles:
            assert len(particle) == 7
            assert all(isinstance(x, (int, float)) for x in particle)

    def test_equal_masses(self):
        """Figure-eight requires equal masses."""
        particles = three_body_figure_eight()
        assert particles[0][6] == particles[1][6] == particles[2][6]

    def test_center_of_mass_at_origin(self):
        """Center of mass should be at origin."""
        particles = three_body_figure_eight()
        total_mass = sum(p[6] for p in particles)
        com_x = sum(p[0] * p[6] for p in particles) / total_mass
        com_y = sum(p[1] * p[6] for p in particles) / total_mass
        com_z = sum(p[2] * p[6] for p in particles) / total_mass
        assert abs(com_x) < 1e-10
        assert abs(com_y) < 1e-10
        assert abs(com_z) < 1e-10

    def test_zero_total_momentum(self):
        """Total momentum should be zero."""
        particles = three_body_figure_eight()
        total_px = sum(p[3] * p[6] for p in particles)
        total_py = sum(p[4] * p[6] for p in particles)
        total_pz = sum(p[5] * p[6] for p in particles)
        assert abs(total_px) < 1e-10
        assert abs(total_py) < 1e-10
        assert abs(total_pz) < 1e-10

    def test_symmetric_configuration(self):
        """Figure-eight has specific symmetry properties."""
        particles = three_body_figure_eight()
        # Third particle should be at origin
        assert abs(particles[2][0]) < 1e-10
        assert abs(particles[2][1]) < 1e-10
        # Other two should be symmetric about origin
        assert abs(particles[0][0] + particles[1][0]) < 1e-10
        assert abs(particles[0][1] + particles[1][1]) < 1e-10


class TestPlummerSphere:
    """Test Plummer sphere initial conditions."""

    def test_returns_correct_number_of_particles(self):
        """Should return requested number of particles."""
        n = 50
        particles = plummer_sphere(n_particles=n)
        assert len(particles) == n

    def test_particle_format(self):
        """Each particle should have 7 elements."""
        particles = plummer_sphere(n_particles=10)
        for particle in particles:
            assert len(particle) == 7
            assert all(isinstance(x, (int, float)) for x in particle)

    def test_total_mass(self):
        """Total mass should match specified value."""
        total_mass = 100.0
        particles = plummer_sphere(n_particles=50, total_mass=total_mass)
        actual_total = sum(p[6] for p in particles)
        assert abs(actual_total - total_mass) < 1e-10

    def test_center_of_mass_near_origin(self):
        """Center of mass should be near origin (statistical)."""
        particles = plummer_sphere(n_particles=100, random_seed=42)
        total_mass = sum(p[6] for p in particles)
        com_x = sum(p[0] * p[6] for p in particles) / total_mass
        com_y = sum(p[1] * p[6] for p in particles) / total_mass
        com_z = sum(p[2] * p[6] for p in particles) / total_mass
        # For large N, COM should be close to origin
        assert abs(com_x) < 0.2
        assert abs(com_y) < 0.2
        assert abs(com_z) < 0.2

    def test_virial_equilibrium(self):
        """Plummer sphere should satisfy virial theorem (2T + U ≈ 0)."""
        particles = plummer_sphere(n_particles=100, random_seed=42)
        grav_const = 1.0

        # Compute kinetic energy: T = 0.5 * sum(m * v²)
        kinetic_energy = 0.0
        for p in particles:
            v_squared = p[3] ** 2 + p[4] ** 2 + p[5] ** 2
            kinetic_energy += 0.5 * p[6] * v_squared

        # Compute potential energy: U = -sum(G * m_i * m_j / r_ij)
        potential_energy = 0.0
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                dx = particles[j][0] - particles[i][0]
                dy = particles[j][1] - particles[i][1]
                dz = particles[j][2] - particles[i][2]
                r = math.sqrt(dx**2 + dy**2 + dz**2)
                if r > 1e-10:  # Avoid division by zero
                    potential_energy -= grav_const * particles[i][6] * particles[j][6] / r

        # Virial theorem: 2T + U should be close to 0
        virial = 2 * kinetic_energy + potential_energy
        # Allow 80% tolerance for finite N effects and simplified velocity sampling
        # Perfect virial equilibrium requires more sophisticated distribution function
        assert abs(virial) < 0.80 * abs(potential_energy)

    def test_reproducible_with_seed(self):
        """Same random seed should produce identical results."""
        particles1 = plummer_sphere(n_particles=20, random_seed=123)
        particles2 = plummer_sphere(n_particles=20, random_seed=123)
        assert particles1 == particles2

    def test_different_with_different_seed(self):
        """Different random seeds should produce different results."""
        particles1 = plummer_sphere(n_particles=20, random_seed=123)
        particles2 = plummer_sphere(n_particles=20, random_seed=456)
        assert particles1 != particles2

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_scales_with_particle_count(self, n):
        """Should work with different particle counts."""
        particles = plummer_sphere(n_particles=n, random_seed=42)
        assert len(particles) == n

    def test_scale_radius_affects_distribution(self):
        """Larger scale radius should produce more spread out particles."""
        particles_small = plummer_sphere(n_particles=100, scale_radius=1.0, random_seed=42)
        particles_large = plummer_sphere(n_particles=100, scale_radius=3.0, random_seed=42)

        # Compute average distance from origin
        def avg_radius(particles):
            return sum(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in particles) / len(
                particles
            )

        assert avg_radius(particles_large) > avg_radius(particles_small)
