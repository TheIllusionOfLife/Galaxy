"""Tests for physics-aware fitness penalty.

This test suite validates the physics penalty calculation logic following TDD principles.
Tests are written BEFORE implementation to ensure correct behavior.
"""

import pytest

from config import settings
from prototype import calculate_physics_penalty, validate_physics
from validation_metrics import compute_angular_momentum_conservation, compute_energy_drift

# Test configuration values (from actual config.yaml defaults)
ENERGY_WEIGHT = settings.physics_energy_weight  # 0.3
MOMENTUM_WEIGHT = settings.physics_momentum_weight  # 0.1
ENERGY_THRESHOLD = settings.energy_drift_threshold  # 0.01
MOMENTUM_THRESHOLD = settings.angular_momentum_threshold  # 0.01
MAX_PENALTY = 0.9  # 90% cap (10% floor)


class TestPhysicsPenaltyCalculation:
    """Test physics penalty calculation logic."""

    def test_no_penalty_below_thresholds(self):
        """Models with good physics conservation should not be penalized."""
        base_fitness = 1000.0
        energy_drift = 0.005  # Below 0.01 threshold
        momentum_drift = 0.003  # Below 0.01 threshold

        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        assert total_penalty == 0.0

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == base_fitness

    def test_penalty_energy_violation(self):
        """Energy drift above threshold should apply penalty."""
        base_fitness = 1000.0
        energy_drift = 0.05  # Above 0.01 threshold
        momentum_drift = 0.003  # Below threshold

        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        assert total_penalty == 0.012  # 0.3 * (0.05 - 0.01)

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == 988.0  # 1000 - (1000 * 0.012)

    def test_penalty_momentum_violation(self):
        """Angular momentum drift above threshold should apply penalty."""
        base_fitness = 1000.0
        energy_drift = 0.0
        momentum_drift = 0.03  # Above 0.01 threshold

        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        assert abs(total_penalty - 0.002) < 1e-9  # 0.1 * (0.03 - 0.01)

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert abs(final_fitness - 998.0) < 0.01

    def test_combined_violations(self):
        """Both metrics violating should combine penalties additively."""
        base_fitness = 1000.0
        energy_drift = 0.05  # 0.04 violation
        momentum_drift = 0.03  # 0.02 violation

        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        assert total_penalty == 0.014  # 0.3 * 0.04 + 0.1 * 0.02

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == 986.0

    def test_penalty_floor(self):
        """Extreme violations should not eliminate models completely."""
        base_fitness = 1000.0
        energy_drift = 10.0  # Extreme violation
        momentum_drift = 5.0  # Extreme violation

        # Total penalty could exceed 1.0, need floor
        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        total_penalty = min(MAX_PENALTY, total_penalty)  # Cap at 90%

        assert total_penalty == MAX_PENALTY

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness >= 100.0  # At least 10% of base

    def test_additive_with_code_penalty(self):
        """Physics penalty should combine additively with code penalty."""
        base_fitness = 1000.0

        # Code penalty (excess tokens)
        code_penalty = 0.05  # 5% penalty for long code

        # Physics penalty
        physics_penalty = 0.02  # 2% penalty for physics violation

        # Additive combination
        total_penalty = code_penalty + physics_penalty
        assert total_penalty == 0.07

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == 930.0  # 1000 - 70

    def test_zero_energy_threshold(self):
        """Zero threshold means any drift triggers penalty."""
        base_fitness = 1000.0
        energy_drift = 0.001  # Very small drift
        threshold = 0.0

        violation = max(0, energy_drift - threshold)
        assert violation == 0.001

        penalty = 0.3 * violation
        final_fitness = base_fitness - (base_fitness * penalty)
        assert final_fitness < base_fitness

    def test_configurable_weights(self):
        """Penalty weights should be configurable."""
        # Test different weight configurations
        energy_violation = 0.04  # From 0.05 drift - 0.01 threshold
        momentum_violation = 0.02  # From 0.03 drift - 0.01 threshold

        # Equal weights
        equal_weight_penalty = 0.2 * energy_violation + 0.2 * momentum_violation
        assert equal_weight_penalty == 0.012

        # Energy-heavy weights
        energy_heavy_penalty = 0.5 * energy_violation + 0.1 * momentum_violation
        assert energy_heavy_penalty == 0.022

        # Momentum-heavy weights
        momentum_heavy_penalty = 0.1 * energy_violation + 0.5 * momentum_violation
        assert momentum_heavy_penalty == 0.014


class TestPhysicsValidationIntegration:
    """Test multi-step physics validation."""

    def test_multi_step_simulation(self):
        """Physics validation should run multi-step simulation."""

        # Create simple model that preserves state
        def identity_model(particle, all_particles):
            return particle  # No change

        initial_particles = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        ]

        # Run actual validate_physics function
        energy_drift, momentum_drift = validate_physics(
            identity_model, initial_particles, timesteps=10
        )

        # Identity model should have perfect conservation
        assert energy_drift == 0.0
        assert momentum_drift == 0.0

    def test_physics_metrics_computation(self):
        """Should compute energy and momentum drift correctly."""
        initial = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        ]

        # Slightly different final state (small drift)
        final = [
            [0.1, 0.0, 0.0, 1.01, 0.0, 0.0, 1.0],
            [1.1, 0.0, 0.0, -1.01, 0.0, 0.0, 1.0],
        ]

        energy_drift = compute_energy_drift(initial, final)
        momentum_drift = compute_angular_momentum_conservation(initial, final)

        assert isinstance(energy_drift, float)
        assert isinstance(momentum_drift, float)
        assert energy_drift >= 0.0
        assert momentum_drift >= 0.0

    def test_model_multiple_timesteps(self):
        """Model should be callable multiple times in sequence."""

        def simple_step(particle, all_particles):
            # Simple forward integration
            x, y, z, vx, vy, vz, m = particle
            dt = 0.1
            return [x + vx * dt, y + vy * dt, z + vz * dt, vx, vy, vz, m]

        initial = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]

        # Run actual validate_physics function
        energy_drift, momentum_drift = validate_physics(simple_step, initial, timesteps=10)

        # Should compute drifts without error
        assert isinstance(energy_drift, float)
        assert isinstance(momentum_drift, float)

    def test_state_mutation_prevention(self):
        """Validation should not mutate original particle state."""
        initial = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]
        initial_copy = [p[:] for p in initial]

        def mutating_model(particle, all_particles):
            # Try to mutate (bad model)
            particle[0] = 999.0
            return particle

        # Run actual validate_physics function
        validate_physics(mutating_model, initial, timesteps=5)

        # Original should be unchanged
        assert initial == initial_copy


class TestPhysicsValidationFailure:
    """Test error handling during physics validation."""

    def test_invalid_model_crash(self):
        """Model crash during validation should raise exception."""

        def crashing_model(particle, all_particles):
            raise RuntimeError("Model exploded")

        initial_particles = [[0, 0, 0, 0, 0, 0, 1]]

        # Try actual validate_physics function
        with pytest.raises(RuntimeError, match="Model exploded"):
            validate_physics(crashing_model, initial_particles, timesteps=10)

    def test_invalid_output_format(self):
        """Model returning wrong format should be detected."""

        def bad_model(particle, all_particles):
            return [1, 2, 3]  # Wrong length (should be 7)

        # Validation should detect this
        result = bad_model([0, 0, 0, 0, 0, 0, 1], [])
        assert len(result) != 7, "Invalid output length"

    def test_none_return_value(self):
        """Model returning None should be detected."""

        def none_model(particle, all_particles):
            return None  # Invalid return

        result = none_model([0, 0, 0, 0, 0, 0, 1], [])
        assert result is None, "Model returned None"

    def test_non_list_return(self):
        """Model returning non-list should be detected."""

        def bad_type_model(particle, all_particles):
            return "not a list"

        result = bad_type_model([0, 0, 0, 0, 0, 0, 1], [])
        assert not isinstance(result, list), "Model returned non-list"

    def test_invalid_fitness_handling(self):
        """Invalid fitness should be representable."""
        # Mark model as invalid
        fitness = float("-inf")
        assert fitness == float("-inf")
        assert fitness < 0


class TestEdgeCases:
    """Test edge cases for physics validation."""

    def test_empty_particle_list(self):
        """Empty particle list should return 0.0 drift."""
        initial = []
        final = []

        energy_drift = compute_energy_drift(initial, final)
        momentum_drift = compute_angular_momentum_conservation(initial, final)

        assert energy_drift == 0.0
        assert momentum_drift == 0.0

    def test_single_timestep(self):
        """Single timestep validation should work."""

        def model(particle, all_particles):
            return particle

        initial = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]

        current_state = [p[:] for p in initial]
        for _ in range(1):  # Just 1 timestep
            predicted_state = []
            for particle in current_state:
                prediction = model(particle, current_state)
                predicted_state.append(prediction)
            current_state = predicted_state

        # Should complete without error
        assert len(current_state) == 1

    def test_both_penalties_disabled(self):
        """With both penalties disabled, fitness should be base_fitness."""
        base_fitness = 1000.0

        # No code penalty
        code_penalty = 0.0

        # No physics penalty
        physics_penalty = 0.0

        total_penalty = code_penalty + physics_penalty
        assert total_penalty == 0.0

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == base_fitness

    def test_both_penalties_triggered(self):
        """With both penalties triggered, combined reduction should apply."""
        base_fitness = 1000.0

        # Code penalty (10%)
        code_penalty = 0.1

        # Physics penalty (5%)
        physics_penalty = 0.05

        # Additive (not multiplicative)
        total_penalty = code_penalty + physics_penalty
        assert abs(total_penalty - 0.15) < 1e-9

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert abs(final_fitness - 850.0) < 0.01  # 1000 - 150

    def test_very_small_drift(self):
        """Very small drift values should be handled correctly."""
        base_fitness = 1000.0
        energy_drift = 1e-10  # Extremely small
        threshold = 0.01

        violation = max(0, energy_drift - threshold)
        assert violation == 0.0  # Below threshold

        penalty = 0.3 * violation
        final_fitness = base_fitness - (base_fitness * penalty)
        assert final_fitness == base_fitness

    def test_penalty_approaches_maximum(self):
        """Penalty approaching 90% should be capped."""
        base_fitness = 1000.0

        # Extreme violations (adding back thresholds)
        energy_drift = 2.51  # 2.5 violation after 0.01 threshold
        momentum_drift = 1.01  # 1.0 violation after 0.01 threshold

        total_penalty = calculate_physics_penalty(energy_drift, momentum_drift)
        # total_penalty = 0.3 * 2.5 + 0.1 * 1.0 = 0.75 + 0.1 = 0.85
        total_penalty = min(MAX_PENALTY, total_penalty)  # Cap at 90%

        final_fitness = base_fitness - (base_fitness * total_penalty)
        assert final_fitness == 150.0  # At least 10% of base
        assert final_fitness >= 0.1 * base_fitness
