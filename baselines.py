"""Baseline surrogate models for N-body simulation.

Provides reference implementations for comparison with evolved surrogates:
- KDTree baseline: O(N log N) nearest-neighbor approximation using scipy
- Direct N-body baseline: O(N²) exact calculation matching ground truth

All baselines return SurrogateGenome instances compatible with the evolution system.
"""

import math

from scipy.spatial import KDTree

from prototype import SurrogateGenome


def create_kdtree_baseline(
    k_neighbors: int = 10,
    grav_const: float = 1.0,
    epsilon: float = 0.01,
    dt: float = 0.1,
) -> SurrogateGenome:
    """Create KDTree-based baseline surrogate.

    Uses scipy.spatial.KDTree for O(N log N) neighbor finding,
    then approximates gravitational force using only k nearest neighbors.

    Args:
        k_neighbors: Number of nearest neighbors to consider
        grav_const: Gravitational constant (default 1.0)
        epsilon: Softening parameter to prevent singularities
        dt: Timestep for integration

    Returns:
        SurrogateGenome with compiled KDTree surrogate

    Example:
        >>> baseline = create_kdtree_baseline(k_neighbors=10)
        >>> predict = baseline.compiled_predict
        >>> particle = [0,0,0,0,0,0,1]
        >>> all_particles = [[0,0,0,0,0,0,1], [1,0,0,0,0,0,1]]
        >>> result = predict(particle, all_particles)
        >>> len(result)
        7
    """

    def kdtree_predict(particle: list[float], all_particles: list[list[float]]) -> list[float]:
        """Predict next state using k-nearest neighbors approximation."""
        if not all_particles:
            # No particles to interact with - just update position
            x, y, z, vx, vy, vz, mass = particle
            return [x + vx * dt, y + vy * dt, z + vz * dt, vx, vy, vz, mass]

        if len(all_particles) == 1:
            # Single particle - no forces, just drift
            x, y, z, vx, vy, vz, mass = particle
            return [x + vx * dt, y + vy * dt, z + vz * dt, vx, vy, vz, mass]

        # Extract current position and properties
        x, y, z, vx, vy, vz, mass = particle

        # Build KDTree from all particle positions
        positions = [[p[0], p[1], p[2]] for p in all_particles]
        tree = KDTree(positions)

        # Find k nearest neighbors (including self)
        current_pos = [x, y, z]
        k_actual = min(k_neighbors + 1, len(all_particles))  # +1 to account for self
        distances, indices = tree.query(current_pos, k=k_actual)

        # Compute gravitational force from k nearest neighbors
        ax, ay, az = 0.0, 0.0, 0.0

        for idx in indices:
            other = all_particles[idx]
            dx = other[0] - x
            dy = other[1] - y
            dz = other[2] - z
            r_squared = dx * dx + dy * dy + dz * dz

            # Skip self-interaction
            if r_squared < 1e-10:
                continue

            # Gravitational force with softening
            r = math.sqrt(r_squared + epsilon * epsilon)
            force_mag = grav_const * other[6] / (r * r * r)  # other[6] is mass

            ax += force_mag * dx
            ay += force_mag * dy
            az += force_mag * dz

        # Leapfrog integration (velocity Verlet)
        # Half-step velocity update
        vx_half = vx + ax * dt / 2.0
        vy_half = vy + ay * dt / 2.0
        vz_half = vz + az * dt / 2.0

        # Full-step position update
        new_x = x + vx_half * dt
        new_y = y + vy_half * dt
        new_z = z + vz_half * dt

        # Half-step velocity update (using same acceleration - approximation)
        new_vx = vx_half + ax * dt / 2.0
        new_vy = vy_half + ay * dt / 2.0
        new_vz = vz_half + az * dt / 2.0

        return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]

    # Create SurrogateGenome with compiled predict function
    return SurrogateGenome(
        theta=[float(k_neighbors), grav_const, epsilon, dt],
        description=f"KDTree-baseline-k{k_neighbors}",
        raw_code=None,
        compiled_predict=kdtree_predict,
    )


def create_direct_nbody_baseline(
    grav_const: float = 1.0,
    epsilon: float = 0.01,
    dt: float = 0.1,
) -> SurrogateGenome:
    """Create direct N-body baseline (O(N²) reference implementation).

    Matches CosmologyCrucible.brute_force_step() logic exactly.
    Useful for validation and performance comparison.

    This should achieve perfect accuracy (1.0) as it's the ground truth.

    Args:
        grav_const: Gravitational constant (default 1.0)
        epsilon: Softening parameter
        dt: Timestep

    Returns:
        SurrogateGenome with compiled direct N-body surrogate

    Example:
        >>> baseline = create_direct_nbody_baseline()
        >>> predict = baseline.compiled_predict
        >>> particle = [0,0,0,0,0,0,1]
        >>> all_particles = [[0,0,0,0,0,0,1], [1,0,0,0,0,0,1]]
        >>> result = predict(particle, all_particles)
        >>> len(result)
        7
    """

    def direct_nbody_predict(
        particle: list[float], all_particles: list[list[float]]
    ) -> list[float]:
        """Predict next state using exact O(N²) all-pairs calculation."""
        if not all_particles:
            # No particles - just drift
            x, y, z, vx, vy, vz, mass = particle
            return [x + vx * dt, y + vy * dt, z + vz * dt, vx, vy, vz, mass]

        # Extract current state
        x, y, z, vx, vy, vz, mass = particle

        # Compute gravitational acceleration from ALL particles
        ax, ay, az = 0.0, 0.0, 0.0

        for other in all_particles:
            dx = other[0] - x
            dy = other[1] - y
            dz = other[2] - z
            r_squared = dx * dx + dy * dy + dz * dz

            # Skip self-interaction
            if r_squared < 1e-10:
                continue

            # Gravitational force with softening: F = G * m / (r² + ε²)^(3/2)
            r = math.sqrt(r_squared + epsilon * epsilon)
            force_mag = grav_const * other[6] / (r * r * r)

            ax += force_mag * dx
            ay += force_mag * dy
            az += force_mag * dz

        # Leapfrog integration (velocity Verlet) - matches CosmologyCrucible
        # Half-step velocity update
        vx_half = vx + ax * dt / 2.0
        vy_half = vy + ay * dt / 2.0
        vz_half = vz + az * dt / 2.0

        # Full-step position update
        new_x = x + vx_half * dt
        new_y = y + vy_half * dt
        new_z = z + vz_half * dt

        # Half-step velocity update
        new_vx = vx_half + ax * dt / 2.0
        new_vy = vy_half + ay * dt / 2.0
        new_vz = vz_half + az * dt / 2.0

        return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]

    return SurrogateGenome(
        theta=[grav_const, epsilon, dt],
        description="Direct-N-body-reference",
        raw_code=None,
        compiled_predict=direct_nbody_predict,
    )
