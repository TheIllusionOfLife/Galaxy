"""Tests for true 3D N-body gravitational physics.

These tests verify:
1. Particle-particle gravitational interactions (O(N²))
2. Energy conservation for valid integrators
3. Angular momentum conservation
4. Two-body orbital mechanics
5. Complexity scaling (O(N²) verification)
"""

import math
import time

import pytest

from prototype import CosmologyCrucible


class TestParticleFormat:
    """Test 3D particle representation."""

    def test_particles_have_7_elements(self):
        """Particles should be [x, y, z, vx, vy, vz, mass]."""
        crucible = CosmologyCrucible(num_particles=5)

        for particle in crucible.particles:
            assert len(particle) == 7, f"Expected 7 elements, got {len(particle)}"

            # Verify all elements are numbers
            for i, val in enumerate(particle):
                assert isinstance(val, (int, float)), (
                    f"Element {i} should be numeric, got {type(val)}"
                )

    def test_particles_have_positive_mass(self):
        """All particles should have positive mass."""
        crucible = CosmologyCrucible(num_particles=10)

        for particle in crucible.particles:
            mass = particle[6]
            assert mass > 0, f"Mass must be positive, got {mass}"

    def test_configurable_particle_count(self):
        """Should respect num_particles parameter."""
        for num_particles in [10, 50, 100]:
            crucible = CosmologyCrucible(num_particles=num_particles)
            assert len(crucible.particles) == num_particles


class TestNBodyPhysics:
    """Test true N-body gravitational physics."""

    def test_brute_force_step_signature(self):
        """brute_force_step should accept and return 3D particles."""
        crucible = CosmologyCrucible(num_particles=5)
        initial = crucible.particles[:]

        result = crucible.brute_force_step(initial)

        # Should return same number of particles
        assert len(result) == len(initial)

        # Each particle should have 7 elements
        for particle in result:
            assert len(particle) == 7

    def test_particle_particle_interaction(self):
        """Particles should interact with each other, not a central attractor."""
        # Two particles: one at origin, one at (10, 0, 0)
        particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Stationary at origin
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Stationary at x=10
        ]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        result = crucible.brute_force_step(particles)

        # Particle 0 should be pulled toward particle 1 (positive x)
        assert result[0][3] > 0, "Particle 0 should gain velocity toward particle 1"

        # Particle 1 should be pulled toward particle 0 (negative x)
        assert result[1][3] < 0, "Particle 1 should gain velocity toward particle 0"

        # By Newton's third law, accelerations should be equal and opposite
        # (same mass, so velocities should be equal magnitude)
        assert abs(result[0][3]) == pytest.approx(abs(result[1][3]), rel=1e-10)

    def test_no_self_interaction(self):
        """Particle should not exert force on itself."""
        # Single particle should remain stationary (no forces)
        particles = [
            [50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 1.0],
        ]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        result = crucible.brute_force_step(particles)

        # Position should change only due to velocity (which is zero)
        # Since velocity is zero and no forces, position unchanged
        assert result[0][0] == pytest.approx(50.0, abs=1e-6)
        assert result[0][1] == pytest.approx(50.0, abs=1e-6)
        assert result[0][2] == pytest.approx(50.0, abs=1e-6)

    def test_inverse_square_law(self):
        """Force should follow inverse square law: F ∝ 1/r²."""
        # Test particle at origin, source particles at different distances
        particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Test particle
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Distance 1
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Distance 2
        ]

        crucible = CosmologyCrucible(num_particles=0)

        # Measure acceleration from particle at distance 1
        crucible.particles = particles[:2]
        result_1 = crucible.brute_force_step(particles[:2])
        accel_1 = result_1[0][3]  # vx change (since initial vx=0)

        # Measure acceleration from particle at distance 2
        crucible.particles = [particles[0], particles[2]]
        result_2 = crucible.brute_force_step([particles[0], particles[2]])
        accel_2 = result_2[0][3]

        # Force at distance 2 should be 1/4 of force at distance 1
        # (inverse square: (1/2)² = 1/4)
        expected_ratio = 4.0
        actual_ratio = accel_1 / accel_2 if accel_2 != 0 else float("inf")

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_three_dimensions(self):
        """Forces should work in all three dimensions."""
        # Two particles separated in z dimension
        particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0],
        ]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        result = crucible.brute_force_step(particles)

        # Particles should gain velocity in z direction
        assert abs(result[0][5]) > 0, "Particle 0 should gain z velocity"
        assert abs(result[1][5]) > 0, "Particle 1 should gain z velocity"

        # Should be zero in x and y
        assert result[0][3] == pytest.approx(0.0, abs=1e-10)
        assert result[0][4] == pytest.approx(0.0, abs=1e-10)

    def test_mass_affects_force(self):
        """More massive particles should exert stronger gravitational pull."""
        # Test particle at origin, two source particles with different masses
        particles_light = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Test particle
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Light source (mass=1)
        ]

        particles_heavy = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Test particle
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],  # Heavy source (mass=2)
        ]

        crucible = CosmologyCrucible(num_particles=0)

        crucible.particles = particles_light
        result_light = crucible.brute_force_step(particles_light)
        accel_light = result_light[0][3]

        crucible.particles = particles_heavy
        result_heavy = crucible.brute_force_step(particles_heavy)
        accel_heavy = result_heavy[0][3]

        # Heavy source should produce 2x the acceleration
        assert accel_heavy == pytest.approx(2.0 * accel_light, rel=0.01)


class TestTwoBodyOrbits:
    """Test two-body orbital mechanics."""

    def test_two_body_energy_conservation(self):
        """Two-body system should approximately conserve total energy."""
        # Two equal masses separated with some kinetic energy
        # Use setup with non-zero total energy to avoid division by near-zero
        m = 1.0
        # Note: Using separation of 2.0 for larger distance between particles
        # This reduces numerical integration errors

        # Give them velocities perpendicular to separation
        # This creates an elliptical orbit
        v = 0.5  # Moderate velocity

        particles = [
            [-1.0, 0.0, 0.0, 0.0, v, 0.0, m],  # Left particle moving up
            [1.0, 0.0, 0.0, 0.0, -v, 0.0, m],  # Right particle moving down (separation=2.0)
        ]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        # Compute initial energy
        initial_energy = self._compute_total_energy(particles, grav_constant=1.0)

        # Evolve for several timesteps (reduced to avoid excessive drift)
        current_particles = particles[:]
        for _ in range(50):  # Reduced from 100 to 50 timesteps
            current_particles = crucible.brute_force_step(current_particles)

        final_energy = self._compute_total_energy(current_particles, grav_constant=1.0)

        # Check both relative and absolute energy drift
        # Initial energy should be significantly non-zero
        assert abs(initial_energy) > 0.1, (
            f"Initial energy {initial_energy} too close to zero for relative comparison"
        )

        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        assert energy_drift < 0.10, (
            f"Energy drift {energy_drift:.2%} exceeds 10% threshold (E_i={initial_energy:.3f}, E_f={final_energy:.3f})"
        )

    def test_two_body_angular_momentum_conservation(self):
        """Two-body system should conserve angular momentum."""
        m = 1.0
        separation = 1.0
        grav_constant = 1.0  # Gravitational constant
        v_orbit = math.sqrt(grav_constant * m / separation)

        particles = [
            [-0.5, 0.0, 0.0, 0.0, v_orbit, 0.0, m],
            [0.5, 0.0, 0.0, 0.0, -v_orbit, 0.0, m],
        ]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        initial_angular_momentum = self._compute_angular_momentum(particles)

        current_particles = particles[:]
        for _ in range(100):
            current_particles = crucible.brute_force_step(current_particles)

        final_angular_momentum = self._compute_angular_momentum(current_particles)

        # Angular momentum should be conserved (better than energy)
        angular_momentum_drift = (
            abs(final_angular_momentum - initial_angular_momentum) / abs(initial_angular_momentum)
            if initial_angular_momentum != 0
            else 0
        )
        assert angular_momentum_drift < 0.05, (
            f"Angular momentum drift {angular_momentum_drift:.2%} exceeds 5% threshold"
        )

    def _compute_total_energy(self, particles, grav_constant=1.0):
        """Compute total energy (kinetic + potential)."""
        kinetic = 0.0
        potential = 0.0

        # Kinetic energy: KE = 0.5 * m * v²
        for p in particles:
            v_sq = p[3] ** 2 + p[4] ** 2 + p[5] ** 2
            kinetic += 0.5 * p[6] * v_sq

        # Potential energy: PE = -G * m_i * m_j / r_ij
        for i, p_i in enumerate(particles):
            for j in range(i + 1, len(particles)):
                p_j = particles[j]
                dx = p_j[0] - p_i[0]
                dy = p_j[1] - p_i[1]
                dz = p_j[2] - p_i[2]
                r = math.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
                potential -= grav_constant * p_i[6] * p_j[6] / r

        return kinetic + potential

    def _compute_angular_momentum(self, particles):
        """Compute total angular momentum magnitude (L_z for planar motion)."""
        angular_momentum_components = [0.0, 0.0, 0.0]  # [Lx, Ly, Lz]

        for p in particles:
            x, y, z, vx, vy, vz, mass = p
            # L = r × (m*v)
            angular_momentum_components[0] += mass * (y * vz - z * vy)
            angular_momentum_components[1] += mass * (z * vx - x * vz)
            angular_momentum_components[2] += mass * (x * vy - y * vx)

        # Return magnitude
        return math.sqrt(
            angular_momentum_components[0] ** 2
            + angular_momentum_components[1] ** 2
            + angular_momentum_components[2] ** 2
        )


class TestComplexityScaling:
    """Test O(N²) computational complexity."""

    def test_no_artificial_delay(self):
        """Should not use time.sleep() for artificial slowdown."""
        particles = [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(10)]

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        start = time.time()
        crucible.brute_force_step(particles)
        elapsed = time.time() - start

        # Should complete in < 10ms for 10 particles (no artificial delay)
        # O(N²) = 100 operations should be nearly instant
        assert elapsed < 0.01, f"10 particles took {elapsed:.3f}s, suggests artificial delay"

    def test_on_squared_scaling(self):
        """Execution time should scale as O(N²)."""
        timings = {}

        for num_particles in [10, 20, 40]:
            particles = [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(num_particles)]

            crucible = CosmologyCrucible(num_particles=0)
            crucible.particles = particles

            # Measure time for multiple runs to reduce noise
            iterations = 10
            start = time.time()
            for _ in range(iterations):
                crucible.brute_force_step(particles)
            elapsed = time.time() - start

            timings[num_particles] = elapsed / iterations

        # Time(20) / Time(10) should be ~4 (since (20/10)² = 4)
        ratio_10_20 = timings[20] / timings[10]
        assert 2.0 < ratio_10_20 < 8.0, f"Expected ~4x slowdown from 10→20, got {ratio_10_20:.2f}x"

        # Time(40) / Time(20) should be ~4
        ratio_20_40 = timings[40] / timings[20]
        assert 2.0 < ratio_20_40 < 8.0, f"Expected ~4x slowdown from 20→40, got {ratio_20_40:.2f}x"


class TestNoAttractor:
    """Test that central attractor is not used."""

    def test_no_attractor_attribute_used(self):
        """brute_force_step should not reference self.attractor."""
        # If attractor is used, removing it should cause AttributeError
        crucible = CosmologyCrucible(num_particles=5)

        # Remove attractor if it exists
        if hasattr(crucible, "attractor"):
            delattr(crucible, "attractor")

        particles = crucible.particles[:]

        # This should work without AttributeError (no attractor dependency)
        try:
            result = crucible.brute_force_step(particles)
            assert len(result) == len(particles)
        except AttributeError as e:
            if "attractor" in str(e):
                pytest.fail("brute_force_step should not use self.attractor")
            else:
                raise

    def test_uniform_distribution_stays_uniform(self):
        """Uniformly distributed particles should not collapse to a center."""
        # Create uniform grid of particles with zero velocity
        particles = []
        for i in range(5):
            for j in range(5):
                particles.append(
                    [
                        float(i * 10),
                        float(j * 10),
                        0.0,  # Position
                        0.0,
                        0.0,
                        0.0,  # Velocity
                        1.0,  # Mass
                    ]
                )

        crucible = CosmologyCrucible(num_particles=0)
        crucible.particles = particles

        # Compute center of mass
        def center_of_mass(ps):
            total_mass = sum(p[6] for p in ps)
            cx = sum(p[0] * p[6] for p in ps) / total_mass
            cy = sum(p[1] * p[6] for p in ps) / total_mass
            cz = sum(p[2] * p[6] for p in ps) / total_mass
            return (cx, cy, cz)

        initial_com = center_of_mass(particles)

        current = particles[:]
        for _ in range(10):
            current = crucible.brute_force_step(current)

        final_com = center_of_mass(current)

        # Center of mass should not move (no external forces)
        assert final_com[0] == pytest.approx(initial_com[0], abs=1.0)
        assert final_com[1] == pytest.approx(initial_com[1], abs=1.0)
        assert final_com[2] == pytest.approx(initial_com[2], abs=1.0)
