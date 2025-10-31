"""Tests for 3D N-body surrogate model interface.

Tests verify:
1. Surrogate signature: predict(particle, all_particles)
2. 7-element particles: [x, y, z, vx, vy, vz, mass]
3. Code validation accepts new signature
4. Sample execution with 3D particles
"""

from code_validator import CodeValidator


class TestSurrogateSignature:
    """Test surrogate model function signature."""

    def test_predict_accepts_two_parameters(self):
        """predict() should accept (particle, all_particles)."""
        code = """
def predict(particle, all_particles):
    return particle
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Should accept 2 parameters: {result.errors}"

    def test_predict_rejects_one_parameter(self):
        """Should reject old single-parameter signature."""
        code = """
def predict(particle):
    return particle
"""
        result = CodeValidator.validate(code)
        assert not result.valid
        assert any("2 arguments" in e for e in result.errors)

    def test_predict_rejects_three_parameters(self):
        """Should reject too many parameters."""
        code = """
def predict(particle, all_particles, extra):
    return particle
"""
        result = CodeValidator.validate(code)
        assert not result.valid
        assert any("2 arguments" in e for e in result.errors)


class TestParticleFormat:
    """Test 7-element particle format."""

    def test_output_has_7_elements(self):
        """Surrogate should return 7-element particle [x,y,z,vx,vy,vz,mass]."""
        code = """
def predict(particle, all_particles):
    # Simple passthrough surrogate
    x, y, z, vx, vy, vz, mass = particle
    return [x, y, z, vx, vy, vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"

        # Compile and execute
        namespace = {"__builtins__": {}}
        exec(code, namespace)
        predict_func = namespace["predict"]

        # Test with 7-element particle
        particle = [10.0, 20.0, 30.0, 0.1, 0.2, 0.3, 1.5]
        all_particles = [particle]

        output = predict_func(particle, all_particles)

        assert len(output) == 7, f"Expected 7 elements, got {len(output)}"

    def test_3d_positions_and_velocities(self):
        """Surrogate should handle 3D coordinates."""
        code = """
def predict(particle, all_particles):
    import math

    x, y, z, vx, vy, vz, mass = particle

    # Simple example: add small acceleration in z direction
    az = 0.01
    dt = 0.1

    new_vz = vz + az * dt
    new_z = z + new_vz * dt

    return [x, y, new_z, vx, vy, new_vz, mass]
"""
        # This should fail because of import statement
        result = CodeValidator.validate(code)
        assert not result.valid
        assert any("import" in e.lower() for e in result.errors)

    def test_mass_preserved(self):
        """Surrogate should not modify particle mass."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle
    # Do some physics...
    new_x = x + vx * 0.1
    new_y = y + vy * 0.1
    new_z = z + vz * 0.1
    # Mass should remain constant
    return [new_x, new_y, new_z, vx, vy, vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"


class TestAllParticlesParameter:
    """Test that surrogates can access all_particles."""

    def test_can_iterate_all_particles(self):
        """Surrogate should be able to loop over all_particles."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # Count neighbors within distance 10
    count = 0
    for other in all_particles:
        if other is particle:  # Skip self
            continue
        ox, oy, oz = other[0], other[1], other[2]
        dx = ox - x
        dy = oy - y
        dz = oz - z
        dist_sq = dx*dx + dy*dy + dz*dz
        if dist_sq < 100:  # radius = 10
            count += 1

    # Simple passthrough (real surrogate would use count)
    return [x, y, z, vx, vy, vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"

        # Test execution
        namespace = {"__builtins__": {}}
        exec(code, namespace)
        predict_func = namespace["predict"]

        particle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        all_particles = [
            particle,
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Within radius 10
            [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Beyond radius 10
        ]

        output = predict_func(particle, all_particles)
        assert len(output) == 7

    def test_can_access_other_particle_masses(self):
        """Surrogate should access masses of other particles."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # Sum all masses (including self)
    total_mass = sum(p[6] for p in all_particles)

    # Simple passthrough
    return [x, y, z, vx, vy, vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"

        # Test execution
        namespace = {"__builtins__": {"sum": sum}}
        exec(code, namespace)
        predict_func = namespace["predict"]

        all_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        ]

        output = predict_func(all_particles[0], all_particles)
        assert len(output) == 7


class TestNBodyApproximations:
    """Test common N-body approximation strategies."""

    def test_cutoff_radius_approximation(self):
        """Test cutoff radius: ignore distant particles."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # Cutoff radius approximation: only consider nearby particles
    R_cutoff = 20.0
    G = 1.0
    epsilon = 0.01
    dt = 0.1

    ax, ay, az = 0.0, 0.0, 0.0

    for other in all_particles:
        # Skip self (same object reference won't work in sandbox, use position)
        ox, oy, oz, _, _, _, omass = other
        if ox == x and oy == y and oz == z:
            continue

        dx = ox - x
        dy = oy - y
        dz = oz - z
        r_sq = dx*dx + dy*dy + dz*dz + epsilon*epsilon

        # CUTOFF: ignore if too far
        if r_sq > R_cutoff * R_cutoff:
            continue

        r = r_sq ** 0.5  # sqrt without import
        force = G * omass / r_sq
        ax += force * (dx / r)
        ay += force * (dy / r)
        az += force * (dz / r)

    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_vz = vz + az * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_z = z + new_vz * dt

    return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"

        # Test execution with actual particles
        namespace = {"__builtins__": {}}
        exec(code, namespace)
        predict_func = namespace["predict"]

        all_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Test particle
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Close (within cutoff)
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Far (beyond cutoff)
        ]

        output = predict_func(all_particles[0], all_particles)

        assert len(output) == 7
        assert output[6] == 1.0  # Mass preserved
        # Position/velocity should change due to gravity
        assert output[0] != 0.0 or output[3] != 0.0  # x or vx changed

    def test_k_nearest_neighbors_approximation(self):
        """Test K-NN: only consider K closest particles."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    K = 5  # Only consider 5 nearest neighbors

    # Compute distances to all particles
    distances = []
    for i, other in enumerate(all_particles):
        ox, oy, oz = other[0], other[1], other[2]
        if ox == x and oy == y and oz == z:
            continue  # Skip self

        dx = ox - x
        dy = oy - y
        dz = oz - z
        dist_sq = dx*dx + dy*dy + dz*dz
        distances.append((dist_sq, i))

    # Sort by distance (simple bubble sort to avoid imports)
    # For K=5, just find minimum K times
    neighbors = []
    for _ in range(min(K, len(distances))):
        if not distances:
            break
        min_idx = 0
        for i in range(1, len(distances)):
            if distances[i][0] < distances[min_idx][0]:
                min_idx = i
        neighbors.append(distances[min_idx][1])
        distances.pop(min_idx)

    # Compute forces from K nearest neighbors only
    G = 1.0
    epsilon = 0.01
    dt = 0.1
    ax, ay, az = 0.0, 0.0, 0.0

    for idx in neighbors:
        other = all_particles[idx]
        ox, oy, oz, _, _, _, omass = other
        dx = ox - x
        dy = oy - y
        dz = oz - z
        r_sq = dx*dx + dy*dy + dz*dz + epsilon*epsilon
        r = r_sq ** 0.5
        force = G * omass / r_sq
        ax += force * (dx / r)
        ay += force * (dy / r)
        az += force * (dz / r)

    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_vz = vz + az * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_z = z + new_vz * dt

    return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]
"""
        result = CodeValidator.validate(code)
        assert result.valid, f"Valid code rejected: {result.errors}"


class TestValidatorEdgeCases:
    """Test edge cases in code validation."""

    def test_empty_all_particles_list(self):
        """Surrogate should handle empty all_particles gracefully."""
        code = """
def predict(particle, all_particles):
    # Handle edge case: no other particles
    if not all_particles:
        return particle

    # Normal processing...
    return particle
"""
        result = CodeValidator.validate(code)
        assert result.valid

    def test_single_particle_system(self):
        """Surrogate should handle single-particle system (only self)."""
        code = """
def predict(particle, all_particles):
    x, y, z, vx, vy, vz, mass = particle

    # If only one particle (self), no forces
    if len(all_particles) <= 1:
        dt = 0.1
        return [x + vx*dt, y + vy*dt, z + vz*dt, vx, vy, vz, mass]

    # Normal N-body processing...
    return particle
"""
        result = CodeValidator.validate(code)
        assert result.valid

    def test_all_particles_is_list_of_lists(self):
        """all_particles should be list of 7-element lists."""
        code = """
def predict(particle, all_particles):
    # Access all_particles as list of lists
    for other in all_particles:
        # Each 'other' is a list: [x, y, z, vx, vy, vz, mass]
        x_other = other[0]
        y_other = other[1]
        z_other = other[2]
        mass_other = other[6]

    return particle
"""
        result = CodeValidator.validate(code)
        assert result.valid
