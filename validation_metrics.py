"""Physics-based validation metrics for N-body surrogates.

Provides comprehensive metrics beyond simple accuracy:
- Energy drift: Measures energy conservation violation
- Trajectory RMSE: Position error compared to ground truth
- Angular momentum conservation: Rotational momentum drift
- Virial ratio: Checks virial equilibrium (2T/|U|)
"""

import math


def compute_energy_drift(
    initial_particles: list[list[float]],
    final_particles: list[list[float]],
    grav_const: float = 1.0,
) -> float:
    """Calculate relative energy drift: |E_final - E_initial| / |E_initial|.

    Lower is better. Good integrators achieve < 1% over 100 steps.

    Args:
        initial_particles: Initial state [[x,y,z,vx,vy,vz,mass], ...]
        final_particles: Final state [[x,y,z,vx,vy,vz,mass], ...]
        grav_const: Gravitational constant (default 1.0)

    Returns:
        Relative energy drift (0.0 = perfect conservation)

    Example:
        >>> initial = [[0,0,0,1,0,0,1], [1,0,0,-1,0,0,1]]
        >>> final = [[0.1,0,0,1,0,0,1], [1.1,0,0,-1,0,0,1]]
        >>> drift = compute_energy_drift(initial, final)
        >>> drift < 0.01  # Less than 1% drift
        True
    """
    if not initial_particles or not final_particles:
        return 0.0

    def total_energy(particles):
        """Compute total energy: E = T + U (kinetic + potential)."""
        # Kinetic energy: T = 0.5 * sum(m * v²)
        kinetic = 0.0
        for p in particles:
            v_squared = p[3] ** 2 + p[4] ** 2 + p[5] ** 2
            kinetic += 0.5 * p[6] * v_squared

        # Potential energy: U = -sum(G * m_i * m_j / r_ij) for all pairs
        potential = 0.0
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                dx = particles[j][0] - particles[i][0]
                dy = particles[j][1] - particles[i][1]
                dz = particles[j][2] - particles[i][2]
                r = math.sqrt(dx**2 + dy**2 + dz**2)
                if r > 1e-10:  # Avoid division by zero
                    potential -= grav_const * particles[i][6] * particles[j][6] / r

        return kinetic + potential

    initial_energy = total_energy(initial_particles)
    final_energy = total_energy(final_particles)

    # Relative drift
    if abs(initial_energy) < 1e-10:
        # Avoid division by zero if initial energy is near zero
        return abs(final_energy - initial_energy)

    return abs(final_energy - initial_energy) / abs(initial_energy)


def compute_trajectory_rmse(predicted: list[list[float]], ground_truth: list[list[float]]) -> float:
    """Root-mean-square error in 3D positions.

    Measures positional accuracy of predicted trajectory vs ground truth.

    Args:
        predicted: Predicted particle states [[x,y,z,vx,vy,vz,mass], ...]
        ground_truth: Ground truth states [[x,y,z,vx,vy,vz,mass], ...]

    Returns:
        RMSE in simulation units (0.0 = perfect prediction)

    Raises:
        ValueError: If particle counts don't match

    Example:
        >>> predicted = [[0,0,0,0,0,0,1]]
        >>> truth = [[1,0,0,0,0,0,1]]
        >>> rmse = compute_trajectory_rmse(predicted, truth)
        >>> abs(rmse - 1.0) < 1e-10
        True
    """
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Particle count mismatch: {len(predicted)} predicted vs {len(ground_truth)} ground truth"
        )

    if not predicted:
        return 0.0

    # Compute squared distances
    squared_distances = []
    for pred_p, true_p in zip(predicted, ground_truth, strict=True):
        dx = pred_p[0] - true_p[0]
        dy = pred_p[1] - true_p[1]
        dz = pred_p[2] - true_p[2]
        dist_squared = dx**2 + dy**2 + dz**2
        squared_distances.append(dist_squared)

    # RMSE = sqrt(mean(squared_distances))
    mean_squared_distance = sum(squared_distances) / len(squared_distances)
    return math.sqrt(mean_squared_distance)


def compute_angular_momentum_conservation(
    initial_particles: list[list[float]], final_particles: list[list[float]]
) -> float:
    """Calculate relative angular momentum drift.

    Angular momentum should be better conserved than energy for symplectic integrators.

    Args:
        initial_particles: Initial state [[x,y,z,vx,vy,vz,mass], ...]
        final_particles: Final state [[x,y,z,vx,vy,vz,mass], ...]

    Returns:
        Relative angular momentum drift (0.0 = perfect conservation)

    Example:
        >>> initial = [[1,0,0,0,1,0,1], [-1,0,0,0,-1,0,1]]
        >>> final = [[0,1,0,-1,0,0,1], [0,-1,0,1,0,0,1]]
        >>> drift = compute_angular_momentum_conservation(initial, final)
        >>> drift < 1e-10  # Perfect conservation
        True
    """
    if not initial_particles or not final_particles:
        return 0.0

    def total_angular_momentum(particles):
        """Compute total angular momentum: L = sum(r × p) = sum(r × m*v)."""
        lx, ly, lz = 0.0, 0.0, 0.0

        for p in particles:
            x, y, z = p[0], p[1], p[2]
            vx, vy, vz = p[3], p[4], p[5]
            mass = p[6]

            # L = r × (m*v)
            lx += mass * (y * vz - z * vy)
            ly += mass * (z * vx - x * vz)
            lz += mass * (x * vy - y * vx)

        return (lx, ly, lz)

    initial_lx, initial_ly, initial_lz = total_angular_momentum(initial_particles)
    final_lx, final_ly, final_lz = total_angular_momentum(final_particles)

    # Compute vector difference
    dlx = final_lx - initial_lx
    dly = final_ly - initial_ly
    dlz = final_lz - initial_lz
    delta_l_magnitude = math.sqrt(dlx**2 + dly**2 + dlz**2)

    # Initial angular momentum magnitude
    initial_l_magnitude = math.sqrt(initial_lx**2 + initial_ly**2 + initial_lz**2)

    # Relative drift based on vector difference
    if abs(initial_l_magnitude) < 1e-10:
        # If initial L is zero, return absolute difference
        return delta_l_magnitude

    return delta_l_magnitude / abs(initial_l_magnitude)


def compute_virial_ratio(particles: list[list[float]], grav_const: float = 1.0) -> float:
    """Calculate 2T/|U| for virial equilibrium check.

    For virialized systems, should equal 1.0 (virial theorem: 2T + U = 0).

    Args:
        particles: Particle states [[x,y,z,vx,vy,vz,mass], ...]
        grav_const: Gravitational constant (default 1.0)

    Returns:
        Virial ratio 2T/|U| (1.0 = equilibrium, inf if U=0)

    Example:
        >>> particles = [[1,0,0,0,0.5,0,1], [-1,0,0,0,-0.5,0,1]]
        >>> ratio = compute_virial_ratio(particles)
        >>> 0.5 < ratio < 2.0  # Should be near 1.0
        True
    """
    if not particles:
        return 0.0

    # Kinetic energy: T = 0.5 * sum(m * v²)
    kinetic_energy = 0.0
    for p in particles:
        v_squared = p[3] ** 2 + p[4] ** 2 + p[5] ** 2
        kinetic_energy += 0.5 * p[6] * v_squared

    # Potential energy: U = -sum(G * m_i * m_j / r_ij)
    potential_energy = 0.0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            dx = particles[j][0] - particles[i][0]
            dy = particles[j][1] - particles[i][1]
            dz = particles[j][2] - particles[i][2]
            r = math.sqrt(dx**2 + dy**2 + dz**2)
            if r > 1e-10:  # Avoid division by zero
                potential_energy -= grav_const * particles[i][6] * particles[j][6] / r

    # Virial ratio: 2T / |U|
    if abs(potential_energy) < 1e-10:
        # No potential energy (e.g., single particle or infinite separation)
        if kinetic_energy > 1e-10:
            return float("inf")
        return 0.0

    return (2.0 * kinetic_energy) / abs(potential_energy)
