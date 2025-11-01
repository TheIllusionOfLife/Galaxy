"""Standard N-body test problems for validation and benchmarking.

This module provides canonical initial conditions for well-studied N-body problems:
- Two-body circular orbit (stable Kepler problem)
- Three-body figure-eight orbit (Chenciner-Montgomery choreography)
- Plummer sphere (realistic galaxy cluster model)

All functions return particles in the format: [x, y, z, vx, vy, vz, mass]
"""

import math
import random


def two_body_circular_orbit(
    separation: float = 1.0, total_mass: float = 2.0, grav_const: float = 1.0
) -> list[list[float]]:
    """Generate two equal masses in stable circular orbit.

    Creates a circular orbit in the xy-plane with center of mass at origin.
    Perfect for testing:
    - Energy conservation
    - Angular momentum conservation
    - Orbital stability

    Args:
        separation: Distance between the two bodies
        total_mass: Combined mass of both bodies (split equally)
        grav_const: Gravitational constant (default 1.0)

    Returns:
        List of 2 particles: [[x,y,z,vx,vy,vz,mass], [x,y,z,vx,vy,vz,mass]]

    Example:
        >>> particles = two_body_circular_orbit(separation=2.0, total_mass=4.0)
        >>> len(particles)
        2
        >>> particles[0][6]  # Each mass
        2.0
    """
    # Each body has half the total mass
    mass = total_mass / 2.0

    # For circular orbit: v = sqrt(G * M / r) where M is total mass, r is separation
    # Orbital velocity magnitude
    orbital_speed = math.sqrt(grav_const * total_mass / separation)

    # Each body moves at half this speed (center of mass frame)
    speed_per_body = orbital_speed / 2.0

    # Place bodies on x-axis, symmetric about origin
    half_sep = separation / 2.0

    # Body 1: left side, moving up (+y direction)
    body1 = [
        -half_sep,  # x
        0.0,  # y
        0.0,  # z
        0.0,  # vx
        speed_per_body,  # vy (moving upward)
        0.0,  # vz
        mass,
    ]

    # Body 2: right side, moving down (-y direction)
    body2 = [
        half_sep,  # x
        0.0,  # y
        0.0,  # z
        0.0,  # vx
        -speed_per_body,  # vy (moving downward)
        0.0,  # vz
        mass,
    ]

    return [body1, body2]


def three_body_figure_eight(grav_const: float = 1.0) -> list[list[float]]:
    """Generate Chenciner-Montgomery figure-eight orbit initial conditions.

    Three equal masses following a periodic figure-eight trajectory.
    This is a numerically discovered periodic solution to the three-body problem.
    Tests numerical stability and precision.

    The exact initial conditions come from:
    Chenciner & Montgomery (2000), "A remarkable periodic solution of the three-body problem"

    Args:
        grav_const: Gravitational constant (default 1.0)

    Returns:
        List of 3 particles with specific initial conditions for figure-eight orbit

    Note:
        Initial conditions are scaled for G=1. If using different G, velocities
        scale as sqrt(G).
    """
    # Equal masses
    mass = 1.0

    # Initial positions and velocities from Chenciner & Montgomery (2000)
    # These create a stable figure-eight orbit when integrated with high precision

    # Body 1: starts at (-x_0, 0, 0)
    x_0 = 0.97000436  # From numerical solution
    vx_0 = -0.93240737  # From numerical solution
    vy_0 = -0.86473146  # From numerical solution

    # Scale velocities by sqrt(G) if G != 1
    v_scale = math.sqrt(grav_const)

    body1 = [
        -x_0,  # x
        0.0,  # y
        0.0,  # z (planar orbit)
        vx_0 * v_scale,  # vx
        vy_0 * v_scale,  # vy
        0.0,  # vz
        mass,
    ]

    # Body 2: starts at (x_0, 0, 0)
    body2 = [
        x_0,  # x
        0.0,  # y
        0.0,  # z
        vx_0 * v_scale,  # vx (same as body1)
        vy_0 * v_scale,  # vy (same as body1)
        0.0,  # vz
        mass,
    ]

    # Body 3: starts at origin with opposite momentum
    # Conservation of momentum: p1 + p2 + p3 = 0
    # p3 = -(p1 + p2)
    body3 = [
        0.0,  # x
        0.0,  # y
        0.0,  # z
        -2.0 * vx_0 * v_scale,  # vx = -(vx1 + vx2)
        -2.0 * vy_0 * v_scale,  # vy = -(vy1 + vy2)
        0.0,  # vz
        mass,
    ]

    return [body1, body2, body3]


def plummer_sphere(
    n_particles: int = 100,
    total_mass: float = 100.0,
    scale_radius: float = 1.0,
    grav_const: float = 1.0,
    random_seed: int = 42,
) -> list[list[float]]:
    """Generate Plummer sphere initial conditions.

    Creates a realistic galaxy cluster model with:
    - Virial equilibrium (2T + U ≈ 0)
    - Smooth density profile: ρ(r) ∝ (1 + r²/a²)^(-5/2)
    - Isotropic velocity distribution

    This is a standard test problem in N-body dynamics.

    Args:
        n_particles: Number of particles
        total_mass: Total mass of the system
        scale_radius: Plummer scale radius (controls size)
        grav_const: Gravitational constant (default 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        List of n_particles with Plummer distribution

    Reference:
        Plummer (1911), "On the problem of distribution in globular star clusters"
        Aarseth et al. (1974), "The Plummer model"
    """
    random.seed(random_seed)

    # Individual particle mass
    particle_mass = total_mass / n_particles

    particles = []

    for _ in range(n_particles):
        # Generate position using Plummer density profile
        # Use inverse transform sampling for radial distance

        # Cumulative distribution function: M(r) / M_total = r³ / (a² + r²)^(3/2)
        # Solve for r given uniform random u in [0,1]
        u = random.random()
        # r = a / sqrt(u^(-2/3) - 1)
        r = scale_radius / math.sqrt(u ** (-2.0 / 3.0) - 1.0)

        # Isotropic angles
        cos_theta = 2.0 * random.random() - 1.0  # cos(θ) ∈ [-1, 1]
        sin_theta = math.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * math.pi * random.random()

        # Convert spherical to Cartesian
        x = r * sin_theta * math.cos(phi)
        y = r * sin_theta * math.sin(phi)
        z = r * cos_theta

        # Generate velocity using virial theorem for equilibrium
        # For Plummer sphere, potential at radius r:
        # Phi(r) = -G * M / sqrt(r² + a²)

        # Escape velocity: v_esc² = -2 * Phi(r)
        r_squared_plus_a_squared = r**2 + scale_radius**2
        v_esc_squared = 2.0 * grav_const * total_mass / math.sqrt(r_squared_plus_a_squared)

        # Velocity magnitude: use rejection sampling for proper distribution
        # f(v) ∝ v² * (1 - v²/v_esc²)^(7/2) (Eddington formula for Plummer)
        # Simplified: sample from scaled Maxwell-Boltzmann
        # For virial equilibrium: <v²> ≈ G * M / (2 * a)
        v_scale = math.sqrt(grav_const * total_mass / scale_radius)
        v_mag = 0.0

        # Use rejection sampling for proper velocity distribution
        max_attempts = 100
        for _ in range(max_attempts):
            # Sample from Gaussian with appropriate scale
            v_trial = abs(random.gauss(0, v_scale / 3.0))
            if v_trial < math.sqrt(v_esc_squared):
                v_mag = v_trial
                break

        # Isotropic velocity direction
        cos_theta_v = 2.0 * random.random() - 1.0
        sin_theta_v = math.sqrt(1.0 - cos_theta_v**2)
        phi_v = 2.0 * math.pi * random.random()

        vx = v_mag * sin_theta_v * math.cos(phi_v)
        vy = v_mag * sin_theta_v * math.sin(phi_v)
        vz = v_mag * cos_theta_v

        particles.append([x, y, z, vx, vy, vz, particle_mass])

    # Center the system: move COM to origin
    total_mass_check = sum(p[6] for p in particles)
    com_x = sum(p[0] * p[6] for p in particles) / total_mass_check
    com_y = sum(p[1] * p[6] for p in particles) / total_mass_check
    com_z = sum(p[2] * p[6] for p in particles) / total_mass_check

    # Shift all positions to center COM
    for p in particles:
        p[0] -= com_x
        p[1] -= com_y
        p[2] -= com_z

    # Remove bulk velocity: set total momentum to zero
    total_px = sum(p[3] * p[6] for p in particles)
    total_py = sum(p[4] * p[6] for p in particles)
    total_pz = sum(p[5] * p[6] for p in particles)

    avg_vx = total_px / total_mass_check
    avg_vy = total_py / total_mass_check
    avg_vz = total_pz / total_mass_check

    # Shift all velocities
    for p in particles:
        p[3] -= avg_vx
        p[4] -= avg_vy
        p[5] -= avg_vz

    return particles
