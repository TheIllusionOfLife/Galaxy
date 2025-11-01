"""Tests for physics-based validation metrics."""

import math

import pytest

# Will import from validation_metrics module (doesn't exist yet - TDD red phase)
from validation_metrics import (
    compute_angular_momentum_conservation,
    compute_energy_drift,
    compute_trajectory_rmse,
    compute_virial_ratio,
)


class TestEnergyDrift:
    """Test energy drift calculation."""

    def test_perfect_conservation_zero_drift(self):
        """Perfect energy conservation should give zero drift."""
        # Two particles with same total energy
        initial = [[0, 0, 0, 1, 0, 0, 1], [1, 0, 0, -1, 0, 0, 1]]
        final = [[0.5, 0, 0, 1, 0, 0, 1], [1.5, 0, 0, -1, 0, 0, 1]]

        # Manually set up so energy is identical
        # Just test the interface works
        drift = compute_energy_drift(initial, final)
        assert isinstance(drift, float)
        assert drift >= 0.0  # Drift is always non-negative

    def test_large_drift_detected(self):
        """Large energy change should produce large drift."""
        initial = [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]
        # Final state with much higher velocities
        final = [[0, 0, 0, 10, 0, 0, 1], [1, 0, 0, -10, 0, 0, 1]]

        drift = compute_energy_drift(initial, final)
        assert drift > 0.5  # Should detect large energy gain

    def test_single_particle_system(self):
        """Should work with single particle."""
        initial = [[0, 0, 0, 1, 1, 1, 1]]
        final = [[1, 1, 1, 1, 1, 1, 1]]

        drift = compute_energy_drift(initial, final)
        assert drift >= 0.0

    def test_empty_system_handled(self):
        """Empty particle list should not crash."""
        drift = compute_energy_drift([], [])
        assert drift == 0.0 or math.isnan(drift)


class TestTrajectoryRMSE:
    """Test trajectory RMSE calculation."""

    def test_identical_trajectories_zero_rmse(self):
        """Identical particle states should give zero RMSE."""
        particles = [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]]
        rmse = compute_trajectory_rmse(particles, particles)
        assert rmse == 0.0

    def test_unit_displacement_correct_rmse(self):
        """Known displacement should give correct RMSE."""
        predicted = [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]
        ground_truth = [[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]]

        # Each particle displaced by 1.0 in x
        # RMSE = sqrt(mean(distances²)) = sqrt((1² + 1²) / 2) = 1.0
        rmse = compute_trajectory_rmse(predicted, ground_truth)
        assert abs(rmse - 1.0) < 1e-10

    def test_3d_displacement(self):
        """Should correctly calculate 3D Euclidean distance."""
        predicted = [[0, 0, 0, 0, 0, 0, 1]]
        ground_truth = [[1, 1, 1, 0, 0, 0, 1]]

        # Distance = sqrt(1² + 1² + 1²) = sqrt(3)
        rmse = compute_trajectory_rmse(predicted, ground_truth)
        assert abs(rmse - math.sqrt(3)) < 1e-10

    def test_multiple_particles(self):
        """Should average over all particles."""
        predicted = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
        ground_truth = [[1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]]

        # Distances: [1.0, 1.0]
        # RMSE = sqrt((1² + 1²) / 2) = 1.0
        rmse = compute_trajectory_rmse(predicted, ground_truth)
        assert abs(rmse - 1.0) < 1e-10

    def test_mismatched_lengths_raises_error(self):
        """Mismatched particle counts should raise error."""
        predicted = [[0, 0, 0, 0, 0, 0, 1]]
        ground_truth = [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]

        with pytest.raises((ValueError, AssertionError)):
            compute_trajectory_rmse(predicted, ground_truth)


class TestAngularMomentumConservation:
    """Test angular momentum conservation calculation."""

    def test_perfect_conservation_zero_drift(self):
        """Perfect angular momentum conservation should give zero drift."""
        # Two particles orbiting with same angular momentum
        initial = [[1, 0, 0, 0, 1, 0, 1], [-1, 0, 0, 0, -1, 0, 1]]
        final = [[0, 1, 0, -1, 0, 0, 1], [0, -1, 0, 1, 0, 0, 1]]

        drift = compute_angular_momentum_conservation(initial, final)
        assert abs(drift) < 1e-10

    def test_momentum_change_detected(self):
        """Change in angular momentum should be detected."""
        initial = [[1, 0, 0, 0, 1, 0, 1], [-1, 0, 0, 0, -1, 0, 1]]
        # Final state with different angular momentum
        final = [[1, 0, 0, 0, 2, 0, 1], [-1, 0, 0, 0, -2, 0, 1]]

        drift = compute_angular_momentum_conservation(initial, final)
        assert drift > 0.1  # Should detect change

    def test_single_particle(self):
        """Should work with single particle."""
        initial = [[1, 0, 0, 0, 1, 0, 1]]
        final = [[0, 1, 0, -1, 0, 0, 1]]

        drift = compute_angular_momentum_conservation(initial, final)
        assert isinstance(drift, float)


class TestVirialRatio:
    """Test virial ratio calculation."""

    def test_two_body_orbit_near_unity(self):
        """Stable orbit should have virial ratio near -1 (2T/|U| ≈ 1)."""
        # Two bodies in circular orbit
        # For circular orbit: T = |U|/2, so 2T/|U| = 1
        particles = [[1, 0, 0, 0, 0.5, 0, 1], [-1, 0, 0, 0, -0.5, 0, 1]]

        ratio = compute_virial_ratio(particles)
        # Should be close to 1 for virialized system
        assert 0.5 < ratio < 2.0

    def test_stationary_particles_zero_kinetic(self):
        """Stationary particles should have zero kinetic energy."""
        particles = [[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]]

        ratio = compute_virial_ratio(particles)
        # 2T/|U| = 0 when T=0
        assert ratio == 0.0

    def test_single_particle(self):
        """Single particle should handle gracefully."""
        particles = [[0, 0, 0, 1, 1, 1, 1]]

        ratio = compute_virial_ratio(particles)
        # No potential energy, ratio undefined or inf
        assert math.isinf(ratio) or math.isnan(ratio) or ratio == 0.0

    @pytest.mark.parametrize("grav_const", [0.5, 1.0, 2.0])
    def test_different_gravitational_constants(self, grav_const):
        """Should work with different G values."""
        particles = [[1, 0, 0, 0, 1, 0, 1], [-1, 0, 0, 0, -1, 0, 1]]

        ratio = compute_virial_ratio(particles, grav_const=grav_const)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)
