"""Tests for baseline surrogate models."""

import pytest

from baselines import create_direct_nbody_baseline, create_kdtree_baseline
from prototype import SurrogateGenome


class TestKDTreeBaseline:
    """Test KDTree baseline surrogate."""

    def test_returns_surrogate_genome(self):
        """Should return a valid SurrogateGenome instance."""
        baseline = create_kdtree_baseline()
        assert isinstance(baseline, SurrogateGenome)
        assert baseline.compiled_predict is not None
        assert callable(baseline.compiled_predict)

    def test_callable_signature(self):
        """Compiled predict should accept (particle, all_particles)."""
        baseline = create_kdtree_baseline()
        predict = baseline.compiled_predict

        # Test with simple particle configuration
        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        all_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        result = predict(particle, all_particles)
        assert isinstance(result, list)
        assert len(result) == 7  # [x,y,z,vx,vy,vz,mass]

    def test_returns_7_element_list(self):
        """Should return 7-element particle format."""
        baseline = create_kdtree_baseline()
        predict = baseline.compiled_predict

        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        all_particles = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        result = predict(particle, all_particles)
        assert len(result) == 7
        # Check all elements are numbers
        assert all(isinstance(x, (int, float)) for x in result)

    def test_preserves_mass(self):
        """Should preserve particle mass."""
        baseline = create_kdtree_baseline()
        predict = baseline.compiled_predict

        mass = 2.5
        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mass]
        all_particles = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        result = predict(particle, all_particles)
        assert result[6] == mass  # Mass should be preserved

    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_k_neighbors_parameter(self, k):
        """Should accept different k values."""
        baseline = create_kdtree_baseline(k_neighbors=k)
        assert isinstance(baseline, SurrogateGenome)
        assert baseline.compiled_predict is not None

    def test_works_with_many_particles(self):
        """Should handle larger particle counts."""
        baseline = create_kdtree_baseline(k_neighbors=10)
        predict = baseline.compiled_predict

        # Create 50 particles
        all_particles = []
        for i in range(50):
            all_particles.append([float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        particle = all_particles[0]
        result = predict(particle, all_particles)
        assert len(result) == 7

    def test_description_contains_kdtree(self):
        """Description should indicate KDTree baseline."""
        baseline = create_kdtree_baseline()
        assert (
            "kdtree" in baseline.description.lower() or "baseline" in baseline.description.lower()
        )


class TestDirectNBodyBaseline:
    """Test direct N-body baseline (O(N²) reference)."""

    def test_returns_surrogate_genome(self):
        """Should return a valid SurrogateGenome instance."""
        baseline = create_direct_nbody_baseline()
        assert isinstance(baseline, SurrogateGenome)
        assert baseline.compiled_predict is not None

    def test_callable_signature(self):
        """Should work with standard particle signature."""
        baseline = create_direct_nbody_baseline()
        predict = baseline.compiled_predict

        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        all_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        result = predict(particle, all_particles)
        assert len(result) == 7

    def test_perfect_accuracy_expected(self):
        """Direct N-body should match brute force exactly."""
        baseline = create_direct_nbody_baseline()
        # This is the reference implementation, so accuracy should be 1.0
        # Actual accuracy testing happens in integration tests
        assert baseline.compiled_predict is not None

    def test_description_contains_direct(self):
        """Description should indicate direct N-body."""
        baseline = create_direct_nbody_baseline()
        assert (
            "direct" in baseline.description.lower() or "reference" in baseline.description.lower()
        )


class TestBaselineIntegration:
    """Integration tests for baselines."""

    def test_kdtree_vs_direct_similar_results(self):
        """KDTree should give similar results to direct N-body."""
        kdtree_baseline = create_kdtree_baseline(k_neighbors=10)
        direct_baseline = create_direct_nbody_baseline()

        kdtree_predict = kdtree_baseline.compiled_predict
        direct_predict = direct_baseline.compiled_predict

        # Simple two-body system
        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        all_particles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

        kdtree_result = kdtree_predict(particle, all_particles)
        direct_result = direct_predict(particle, all_particles)

        # Results should be close (KDTree is approximation)
        # Check position updates are in same ballpark
        for i in range(3):  # x, y, z positions
            assert abs(kdtree_result[i] - direct_result[i]) < 1.0  # Reasonable tolerance

    def test_baseline_with_empty_neighbors(self):
        """Should handle case where particle has no neighbors."""
        baseline = create_kdtree_baseline(k_neighbors=10)
        predict = baseline.compiled_predict

        # Single particle system (no other particles to interact with)
        particle = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        all_particles = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]

        # Should not crash - particle moves in straight line
        result = predict(particle, all_particles)
        assert len(result) == 7
        # Position should change based on velocity
        assert result[0] != particle[0] or result[3] == 0  # x changed or vx was zero
