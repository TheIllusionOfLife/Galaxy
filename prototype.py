import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

# Import LLM integration modules (will be None if not available)
try:
    from code_validator import validate_and_compile
    from config import settings
    from gemini_client import CostTracker, GeminiClient
    from prompts import get_crossover_prompt, get_initial_prompt, get_mutation_prompt

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    settings = None

# Import initial conditions for multi-problem validation
from initial_conditions import plummer_sphere, three_body_figure_eight, two_body_circular_orbit

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Sample particles for validation (3D N-body format: [x, y, z, vx, vy, vz, mass])
_VALIDATION_PARTICLES = [
    [10.0, 20.0, 30.0, 0.1, 0.2, 0.3, 1.0],
    [40.0, 50.0, 60.0, -0.1, -0.2, -0.3, 1.5],
    [70.0, 80.0, 90.0, 0.0, 0.0, 0.0, 2.0],
]


def get_initial_particles(test_problem: str, num_particles: int = 50) -> list[list[float]]:
    """Get initial particle configuration for the specified test problem.

    Maps test_problem configuration to the appropriate initial_conditions function.
    Note: two_body and figure_eight ignore num_particles (fixed counts).

    Args:
        test_problem: Which test problem to use ("two_body", "figure_eight", "plummer")
        num_particles: Number of particles (only used for plummer, default 50)

    Returns:
        List of particles in format [[x,y,z,vx,vy,vz,mass], ...]

    Raises:
        ValueError: If test_problem is not recognized

    Example:
        >>> particles = get_initial_particles("two_body", num_particles=100)
        >>> len(particles)  # Always 2, ignores num_particles
        2
        >>> particles = get_initial_particles("plummer", num_particles=50)
        >>> len(particles)  # Uses num_particles
        50
    """
    if test_problem == "two_body":
        # Two-body circular orbit (always 2 particles)
        return two_body_circular_orbit()
    elif test_problem == "figure_eight":
        # Three-body figure-eight (always 3 particles)
        return three_body_figure_eight()
    elif test_problem == "plummer":
        # Plummer sphere (uses num_particles)
        return plummer_sphere(n_particles=num_particles, random_seed=42)
    else:
        raise ValueError(
            f"Unknown test_problem: {test_problem}. Valid options: two_body, figure_eight, plummer"
        )


@dataclass
class SurrogateGenome:
    """Genetic representation of a parameterized surrogate model."""

    theta: list[float]
    description: str = "parametric"
    raw_code: str | None = None
    fitness: float | None = None  # Store for LLM prompt context
    accuracy: float | None = None  # Store for LLM prompt context
    speed: float | None = None  # Store for LLM prompt context
    token_count: int | None = None  # Track code length for penalty calculation
    compiled_predict: Callable[[list[float], list[list[float]]], list[float]] | None = (
        None  # Pre-validated callable (3D N-body signature)
    )
    parent_ids: list[str] = field(default_factory=list)  # Track crossover parentage

    def as_readable(self) -> str:
        if self.raw_code:
            head = self.raw_code.strip().splitlines()
            return head[0] if head else "custom-code"
        coeffs = ", ".join(f"{value:.4f}" for value in self.theta)
        return f"theta=[{coeffs}]"

    def build_callable(
        self, all_particles: list[list[float]]
    ) -> Callable[[list[float], list[list[float]]], list[float]]:
        """Build callable surrogate model.

        Args:
            all_particles: Sample particles for validation (3D N-body format)

        Returns:
            Callable with signature predict(particle, all_particles) -> [x,y,z,vx,vy,vz,mass]
        """
        # Prefer the pre-validated callable to avoid TOCTOU security issues
        if self.compiled_predict is not None:
            return self.compiled_predict
        if self.raw_code:
            # Use stricter validator when available; otherwise sandbox fallback
            if LLM_AVAILABLE:
                try:
                    from code_validator import validate_and_compile

                    compiled, validation = validate_and_compile(self.raw_code, all_particles)
                    if not validation.valid or compiled is None:
                        raise ValueError(f"Invalid surrogate code: {validation.errors}")
                    self.compiled_predict = compiled
                    return compiled
                except ImportError:
                    pass
            return compile_external_surrogate(self.raw_code, all_particles)
        return make_parametric_surrogate(self.theta, all_particles)


def make_parametric_surrogate(
    theta: list[float], all_particles: list[list[float]]
) -> Callable[[list[float], list[list[float]]], list[float]]:
    """Generate a surrogate model defined by theta parameters.

    Uses center-of-mass approximation for 3D N-body gravity.

    Args:
        theta: [g_const, epsilon, dt_velocity, dt_position, velocity_correction, damping]
        all_particles: Sample particles (unused, kept for signature compatibility)

    Returns:
        Model with signature predict(particle, all_particles) -> [x,y,z,vx,vy,vz,mass]
    """

    g_const, epsilon, dt_velocity, dt_position, velocity_correction, damping = theta

    def model(particle: list[float], all_particles: list[list[float]]) -> list[float]:
        x, y, z, vx, vy, vz, mass = particle

        # Calculate center of mass from all particles
        if all_particles:
            cx = sum(p[0] for p in all_particles) / len(all_particles)
            cy = sum(p[1] for p in all_particles) / len(all_particles)
            cz = sum(p[2] for p in all_particles) / len(all_particles)
        else:
            cx, cy, cz = x, y, z

        dx = cx - x
        dy = cy - y
        dz = cz - z
        dist_sq = dx * dx + dy * dy + dz * dz + epsilon
        dist = math.sqrt(dist_sq)
        denom = max(dist_sq * dist, 1e-6)
        ax = g_const * dx / denom
        ay = g_const * dy / denom
        az = g_const * dz / denom

        new_vx = (1.0 - damping) * vx + ax * dt_velocity
        new_vy = (1.0 - damping) * vy + ay * dt_velocity
        new_vz = (1.0 - damping) * vz + az * dt_velocity
        new_x = x + (vx + velocity_correction * ax) * dt_position
        new_y = y + (vy + velocity_correction * ay) * dt_position
        new_z = z + (vz + velocity_correction * az) * dt_position

        return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]

    return model


def compile_external_surrogate(
    code: str, validation_particles: list[list[float]]
) -> Callable[[list[float], list[list[float]]], list[float]]:
    """Execute LLM-generated code in a safe namespace and retrieve the predict function.

    Args:
        code: Python code defining predict(particle, all_particles) function
        validation_particles: Sample 3D particles for output validation

    Returns:
        Callable with signature predict(particle, all_particles) -> [x,y,z,vx,vy,vz,mass]

    Raises:
        ValueError: If code doesn't define predict function or output is invalid
    """

    # Use same builtins as CodeValidator.SAFE_BUILTINS to maintain consistency
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "float": float,
        "int": int,
        "list": list,
    }
    sandbox_globals = {"__builtins__": allowed_builtins, "math": math}
    local_namespace: dict[str, Callable] = {}
    exec(code, sandbox_globals, local_namespace)

    if "predict" not in local_namespace:
        raise ValueError(
            "Surrogate model code must define predict(particle, all_particles) function."
        )

    predict_func = local_namespace["predict"]

    # Validate output format with sample particles
    if validation_particles:
        sample = validation_particles[0]
        test_output = predict_func(sample, validation_particles)
        if not isinstance(test_output, (list, tuple)) or len(test_output) != 7:
            raise ValueError(
                "Surrogate model output must be a sequence of length 7 [x, y, z, vx, vy, vz, mass]."
            )

    def wrapped(particle: list[float], all_particles: list[list[float]]) -> list[float]:
        result = predict_func(particle, all_particles)
        return list(result)

    return wrapped


PARAMETER_BOUNDS = [
    (5.0, 15.0),  # g_const
    (1e-6, 5.0),  # epsilon
    (0.05, 0.2),  # dt_velocity
    (0.05, 0.2),  # dt_position
    (0.0, 0.5),  # velocity_correction
    (0.0, 0.3),  # damping
]

BASE_THETA = [10.0, 0.01, 0.1, 0.1, 0.05, 0.02]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mutate_theta(theta: list[float], mutation_scale: float) -> list[float]:
    mutated: list[float] = []
    for value, (lower, upper) in zip(theta, PARAMETER_BOUNDS, strict=False):
        span = upper - lower
        jitter = random.gauss(0, span * mutation_scale)
        mutated.append(clamp(value + jitter, lower, upper))
    return mutated


def count_tokens(code: str | None) -> int:
    """Count approximate tokens in code using whitespace splitting.

    This is a simple approximation sufficient for relative comparison
    between models. More sophisticated tokenization (e.g., tiktoken)
    could be added later if needed.

    Note: This method splits on whitespace, which is adequate for
    comparing relative code complexity. For more accurate tokenization
    matching LLM token counts, consider upgrading to tiktoken library.

    Args:
        code: Python source code string

    Returns:
        Approximate token count (0 if code is None or empty)
    """
    if not code:
        return 0
    # code.split() with no arguments already handles whitespace and empty strings
    return len(code.split())


# ------------------------------------------------------------------------------
# LLM-powered surrogate model generation (with Mock fallback)
# ------------------------------------------------------------------------------


def select_crossover_parents(
    elites: list[tuple[str, dict]],
) -> tuple[tuple[str, dict], tuple[str, dict]]:
    """Select two different elite parents for crossover.

    Args:
        elites: List of (civ_id, civ_data) tuples sorted by fitness

    Returns:
        Two distinct elite tuples (parent1, parent2)

    Raises:
        ValueError: If fewer than 2 LLM elites with raw_code available
    """
    # Filter to only LLM genomes with raw_code for crossover
    # Add defensive validation to ensure genome objects are valid
    llm_elites = []
    for elite in elites:
        try:
            civ_id, civ_data = elite
            # Validate structure and that genome exists with raw_code
            if (
                isinstance(civ_data, dict)
                and "genome" in civ_data
                and hasattr(civ_data["genome"], "raw_code")
                and civ_data["genome"].raw_code is not None
            ):
                llm_elites.append(elite)
        except (TypeError, ValueError, KeyError, AttributeError):
            # Skip malformed elite entries
            continue

    if len(llm_elites) < 2:
        raise ValueError(f"Need at least 2 LLM elites for crossover, got {len(llm_elites)}")

    # Simple strategy: Pick two random distinct elites
    # Future enhancement: Fitness-weighted selection or diversity-based
    parent1, parent2 = random.sample(llm_elites, 2)
    return parent1, parent2


def LLM_propose_surrogate_model(
    base_genome: SurrogateGenome | None,
    generation: int,
    gemini_client: Optional["GeminiClient"] = None,
    cost_tracker: Optional["CostTracker"] = None,
    second_parent: SurrogateGenome | None = None,
    parent_ids: list[str] | None = None,
) -> SurrogateGenome:
    """Generate or evolve a new surrogate model using LLM or Mock fallback.

    Modes:
    - Initial: base_genome=None, second_parent=None → get_initial_prompt()
    - Mutation: base_genome set, second_parent=None → get_mutation_prompt()
    - Crossover: base_genome set, second_parent set → get_crossover_prompt()

    Args:
        base_genome: Parent genome (None for initial generation)
        generation: Current generation number
        gemini_client: Gemini API client (None for Mock mode)
        cost_tracker: Cost tracker
        second_parent: Second parent for crossover (None for mutation)
        parent_ids: List of parent civ_ids for lineage tracking

    Returns:
        New SurrogateGenome
    """
    # Check if we should use LLM
    if gemini_client is None or not LLM_AVAILABLE:
        if generation == 0:
            logger.debug("Using mock mode (no LLM client provided)")
        return _mock_surrogate_generation(base_genome, generation)

    # Check budget
    if cost_tracker and cost_tracker.check_budget_exceeded():
        logger.warning("Budget exceeded, falling back to mock mode")
        return _mock_surrogate_generation(base_genome, generation)

    # Generate prompt
    try:
        temp_override = None  # Initialize temperature override

        if base_genome is None or base_genome.raw_code is None:
            # Initial generation - use diverse approaches
            seed = generation * 100 + random.randint(0, 99)
            prompt = get_initial_prompt(seed)
            logger.info(f"Generating initial model (seed {seed % 6})")
        elif second_parent is not None:
            # Crossover - combine two elite parents
            # Validate both parents have raw_code for LLM crossover
            if base_genome.raw_code is None or second_parent.raw_code is None:
                logger.warning(
                    "Crossover requires LLM parents with raw_code, falling back to parametric generation"
                )
                # Both parents need raw_code; if either is None, use parametric fallback
                return _mock_surrogate_generation(base_genome, generation)
            else:
                prompt = get_crossover_prompt(
                    parent1=base_genome,
                    parent2=second_parent,
                    generation=generation,
                )
                temp_override = settings.crossover_temperature
                logger.info(
                    f"Gen {generation}: Crossover between parents "
                    f"(fitness {(base_genome.fitness or 0.0):.2f} x {(second_parent.fitness or 0.0):.2f})"
                )
        else:
            # Mutation - improve existing model
            mutation_type = "explore" if generation < 3 else "exploit"
            prompt = get_mutation_prompt(
                parent_code=base_genome.raw_code,
                fitness=base_genome.fitness or 0.0,
                accuracy=base_genome.accuracy or 0.5,
                speed=base_genome.speed or 0.01,
                generation=generation,
                mutation_type=mutation_type,
            )
            # Get adaptive temperature for this generation
            temp_override = settings.get_mutation_temperature(generation)
            logger.info(
                f"Mutating model (generation {generation}, type {mutation_type}, temp {temp_override:.2f})"
            )

        # Call Gemini with adaptive temperature
        response = gemini_client.generate_surrogate_code(prompt, temperature=temp_override)

        # Track cost
        if cost_tracker:
            # Determine operation type for cost tracking
            if base_genome is None:
                op_type = "initial"
            elif second_parent is not None:
                op_type = "crossover"
            else:
                op_type = "mutation"

            cost_tracker.add_call(
                response,
                {
                    "generation": generation,
                    "type": op_type,
                },
            )

        if not response.success:
            logger.error(f"LLM call failed: {response.error}")
            return _mock_surrogate_generation(base_genome, generation)

        # Validate code with sample 3D N-body particles
        compiled_func, validation = validate_and_compile(response.code, _VALIDATION_PARTICLES)

        if not validation.valid:
            logger.warning(f"Generated code invalid: {validation.errors}")
            if second_parent is not None:
                logger.warning("Crossover validation failed, falling back to parametric mutation")
            return _mock_surrogate_generation(base_genome, generation)

        if validation.warnings:
            for warning in validation.warnings:
                logger.debug(f"Code warning: {warning}")

        # Return LLM-generated genome with parentage tracking
        logger.info("✓ LLM code validated and compiled successfully")
        genome = SurrogateGenome(
            theta=BASE_THETA[:],  # Keep default theta as fallback
            description=f"gemini_gen{generation}",
            raw_code=response.code,
            fitness=None,
            accuracy=None,
            speed=None,
            compiled_predict=compiled_func,
        )
        if parent_ids is not None:
            genome.parent_ids = parent_ids
        return genome

    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return _mock_surrogate_generation(base_genome, generation)


def _mock_surrogate_generation(
    base_genome: SurrogateGenome | None, generation: int
) -> SurrogateGenome:
    """Fallback: parametric mutation (no LLM required)."""
    if base_genome is None:
        theta = [
            clamp(base + random.uniform(-0.05, 0.05) * (upper - lower), lower, upper)
            for base, (lower, upper) in zip(BASE_THETA, PARAMETER_BOUNDS, strict=False)
        ]
        return SurrogateGenome(theta=theta, description="mock_seed")

    mutation_scale = 0.12 if generation < 3 else 0.06
    new_theta = mutate_theta(base_genome.theta, mutation_scale)
    return SurrogateGenome(theta=new_theta, description=f"mock_mutant_gen{generation}")


# ------------------------------------------------------------------------------
# Crucible - Physical simulation environment that AI civilizations challenge
# ------------------------------------------------------------------------------
class CosmologyCrucible:
    """
    True 3D N-body gravitational simulation environment.

    Every particle interacts with every other particle via Newton's law of gravitation.
    This is the computationally expensive O(N²) ground truth that surrogate models
    attempt to approximate faster.

    Particles: [x, y, z, vx, vy, vz, mass]
    Physics: F_ij = G * m_i * m_j / r_ij²
    Complexity: O(N²) force calculations per timestep
    """

    def __init__(self, num_particles: int = 50, mass_range: tuple[float, float] = (0.5, 2.0)):
        """Initialize N-body system with random particles.

        Args:
            num_particles: Number of particles in the system (minimum 2 for meaningful N-body physics)
            mass_range: (min, max) range for particle masses

        Raises:
            ValueError: If num_particles < 2 or mass_range is invalid

        Note:
            N-body physics requires at least 2 particles for meaningful gravitational interactions.
            For testing with custom particles, use the with_particles() class method instead.
        """
        # Input validation
        if num_particles < 2:
            raise ValueError(
                f"N-body simulation requires at least 2 particles for meaningful interactions, got {num_particles}. "
                "For testing with custom particles, use CosmologyCrucible.with_particles(particles) instead."
            )
        if mass_range[0] <= 0 or mass_range[1] <= mass_range[0]:
            raise ValueError(f"mass_range must be (min, max) with 0 < min < max, got {mass_range}")

        # List of [x, y, z, vx, vy, vz, mass]
        self.particles = [
            [
                random.uniform(0, 100),  # x position
                random.uniform(0, 100),  # y position
                random.uniform(0, 100),  # z position
                random.uniform(-1, 1),  # vx velocity
                random.uniform(-1, 1),  # vy velocity
                random.uniform(-1, 1),  # vz velocity
                random.uniform(*mass_range),  # mass
            ]
            for _ in range(num_particles)
        ]
        self.G = 1.0  # Gravitational constant (simulation units)
        self.dt = 0.1  # Timestep
        self.epsilon = 0.01  # Softening parameter to avoid singularities

    @classmethod
    def with_particles(cls, particles: list[list[float]]) -> "CosmologyCrucible":
        """Create crucible with specific particles (for testing).

        This factory method bypasses the minimum particle count validation,
        allowing tests to use custom particle configurations including
        edge cases like empty lists, single particles, or specific scenarios.

        Args:
            particles: List of particles, each [x, y, z, vx, vy, vz, mass]

        Returns:
            CosmologyCrucible instance with the specified particles

        Example:
            >>> particles = [[10.0, 20.0, 30.0, 0.1, 0.2, 0.3, 1.0]]
            >>> crucible = CosmologyCrucible.with_particles(particles)
        """
        instance = cls.__new__(cls)
        instance.particles = particles
        instance.G = 1.0
        instance.dt = 0.1
        instance.epsilon = 0.01
        return instance

    def _compute_accelerations(
        self, particles: list[list[float]]
    ) -> list[tuple[float, float, float]]:
        """Compute accelerations for all particles (O(N²)).

        Args:
            particles: List of [x, y, z, vx, vy, vz, mass]

        Returns:
            List of (ax, ay, az) acceleration tuples
        """
        accelerations = []

        for i, p_i in enumerate(particles):
            x_i, y_i, z_i, _, _, _, mass_i = p_i
            ax_total, ay_total, az_total = 0.0, 0.0, 0.0

            # Sum forces from all other particles
            for j, p_j in enumerate(particles):
                if i == j:
                    continue

                x_j, y_j, z_j, _, _, _, mass_j = p_j

                dx = x_j - x_i
                dy = y_j - y_i
                dz = z_j - z_i

                r_sq = dx * dx + dy * dy + dz * dz + self.epsilon * self.epsilon
                r = math.sqrt(r_sq)

                force_magnitude = self.G * mass_j / r_sq

                ax_total += force_magnitude * (dx / r)
                ay_total += force_magnitude * (dy / r)
                az_total += force_magnitude * (dz / r)

            accelerations.append((ax_total, ay_total, az_total))

        return accelerations

    def brute_force_step(self, particles: list[list[float]]) -> list[list[float]]:
        """True N-body physics: every particle interacts with every other particle.

        Uses leapfrog (velocity Verlet) integration for better energy conservation.
        This is the ground truth O(N²) calculation. No artificial delays - the
        computational cost comes from nested loops over all particle pairs.

        Args:
            particles: List of [x, y, z, vx, vy, vz, mass]

        Returns:
            Updated particles after one timestep
        """
        # Leapfrog integration (symplectic, conserves energy better than Euler):
        # 1. v(t + dt/2) = v(t) + a(t) * dt/2       (half-step velocity)
        # 2. x(t + dt) = x(t) + v(t + dt/2) * dt    (full-step position)
        # 3. v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2  (half-step velocity)

        # Step 1: Half-step velocity update using current accelerations
        accelerations_t = self._compute_accelerations(particles)
        particles_half = []

        for p, (ax, ay, az) in zip(particles, accelerations_t, strict=True):
            x, y, z, vx, vy, vz, mass = p
            vx_half = vx + ax * self.dt / 2
            vy_half = vy + ay * self.dt / 2
            vz_half = vz + az * self.dt / 2
            particles_half.append([x, y, z, vx_half, vy_half, vz_half, mass])

        # Step 2: Full-step position update using half-step velocities
        particles_drift = []
        for p in particles_half:
            x, y, z, vx, vy, vz, mass = p
            x_new = x + vx * self.dt
            y_new = y + vy * self.dt
            z_new = z + vz * self.dt
            particles_drift.append([x_new, y_new, z_new, vx, vy, vz, mass])

        # Step 3: Half-step velocity update using new accelerations
        accelerations_t_plus_dt = self._compute_accelerations(particles_drift)
        new_particles = []

        for p, (ax, ay, az) in zip(particles_drift, accelerations_t_plus_dt, strict=True):
            x, y, z, vx, vy, vz, mass = p
            vx_new = vx + ax * self.dt / 2
            vy_new = vy + ay * self.dt / 2
            vz_new = vz + az * self.dt / 2
            new_particles.append([x, y, z, vx_new, vy_new, vz_new, mass])

        # NO artificial delay - real computational cost from O(N²) nested loops
        return new_particles

    def evaluate_surrogate_model(
        self, model: Callable[[list[float], list[list[float]]], list[float]]
    ) -> tuple[float, float]:
        """
        Evaluate the 'accuracy' and 'speed' of the surrogate model.

        Args:
            model: Surrogate model with signature predict(particle, all_particles)

        Returns:
            Tuple of (accuracy, speed) where accuracy is 0-1 and speed is execution time

        Raises:
            ValueError: If particle system is empty

        Note:
            Uses parallel evaluation semantics: all particles see the same initial_state.
            This matches brute_force_step which computes all forces at time t before
            updating any positions. Each model call is independent and makes predictions
            based on the full particle set at the current timestep.
        """
        # Input validation
        if not self.particles:
            raise ValueError("Cannot evaluate model with empty particle system")

        initial_state = [p[:] for p in self.particles]  # Copy state

        # 1. Accuracy evaluation
        ground_truth_next_state = self.brute_force_step(initial_state)

        start_time = time.time()
        try:
            predicted_next_state = []
            for particle in initial_state:
                # Call model with both particle and all_particles
                # NOTE: Parallel evaluation semantics - all particles see the same initial_state.
                # This matches brute_force_step which computes all forces before updating positions.
                # Each model call is independent and makes predictions based on the full particle
                # set at time t.
                prediction = model(particle, initial_state)
                if not isinstance(prediction, (list, tuple)) or len(prediction) != 7:
                    raise ValueError(
                        "Surrogate model output must be a sequence of length 7 "
                        "[x, y, z, vx, vy, vz, mass]."
                    )
                predicted_next_state.append(list(prediction))
        except Exception:
            return 0.0, 999.9  # Invalid code gets worst evaluation

        speed = time.time() - start_time

        # Calculate error using all 3 spatial dimensions
        error = 0.0
        for i in range(len(initial_state)):
            true_p = ground_truth_next_state[i]
            pred_p = predicted_next_state[i]
            # Compare x, y, z positions
            error += (
                (true_p[0] - pred_p[0]) ** 2
                + (true_p[1] - pred_p[1]) ** 2
                + (true_p[2] - pred_p[2]) ** 2
            )

        accuracy = 1.0 / (1.0 + math.sqrt(error))  # Smaller error approaches 1

        return accuracy, speed


# ------------------------------------------------------------------------------
# Evolutionary Engine - Selection pressure that evolves civilizations
# ------------------------------------------------------------------------------
class EvolutionaryEngine:
    """
    Manage the generational succession of civilizations.
    Select superior surrogate models and create next-generation models.
    """

    def __init__(
        self,
        crucible: CosmologyCrucible,
        population_size: int = 10,
        elite_ratio: float = 0.2,
        gemini_client: Optional["GeminiClient"] = None,
        cost_tracker: Optional["CostTracker"] = None,
    ):
        """Initialize the evolutionary engine.

        Args:
            crucible: CosmologyCrucible for evaluating models
            population_size: Number of models per generation
            elite_ratio: Fraction of top performers to keep for breeding (0.0 to 1.0)
            gemini_client: Optional LLM client for code generation
            cost_tracker: Optional cost tracking utility

        Raises:
            ValueError: If elite_ratio is outside valid range [0.0, 1.0]
        """
        if not 0.0 <= elite_ratio <= 1.0:
            raise ValueError(f"elite_ratio must be between 0.0 and 1.0, got {elite_ratio}")

        self.crucible = crucible
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.civilizations: dict[str, dict] = {}
        self.generation = 0
        self.gemini_client = gemini_client
        self.cost_tracker = cost_tracker or (CostTracker() if LLM_AVAILABLE else None)
        self.history: list[dict] = []

    def initialize_population(self):
        """Generate the initial population of civilizations (surrogate models)."""
        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            genome = LLM_propose_surrogate_model(
                None, self.generation, self.gemini_client, self.cost_tracker
            )
            self.civilizations[civ_id] = {"genome": genome, "fitness": 0}

    def run_evolutionary_cycle(self):
        """Execute one generation of evolution (evaluation, selection, reproduction)."""
        print(f"\n===== Generation {self.generation}: Evaluating Surrogate Models =====")

        # Evaluation
        for civ_id, civ_data in self.civilizations.items():
            genome: SurrogateGenome = civ_data["genome"]
            try:
                model_func = genome.build_callable(self.crucible.particles)
                accuracy, speed = self.crucible.evaluate_surrogate_model(model_func)

                # Validate speed is positive and finite
                if not isinstance(speed, (int, float)) or speed <= 0 or not math.isfinite(speed):
                    logger.warning(f"{civ_id}: Invalid speed value {speed}, using fallback")
                    speed = 999.9  # Fallback to worst-case speed

                # Count tokens in LLM-generated code
                token_count = 0
                if genome.raw_code:
                    token_count = count_tokens(genome.raw_code)
                    genome.token_count = token_count

                # Calculate base fitness (accuracy / speed)
                base_fitness = accuracy / (speed + 1e-9)

                # Apply code length penalty if enabled
                fitness = base_fitness
                if LLM_AVAILABLE and settings.enable_code_length_penalty:
                    # Only penalize tokens beyond threshold
                    # Note: If token_count is 0, excess_tokens will be 0 and no penalty applies
                    excess_tokens = max(0, token_count - settings.max_acceptable_tokens)
                    if excess_tokens > 0:
                        # Linear penalty: more excess = lower penalty factor
                        penalty_ratio = excess_tokens / settings.max_acceptable_tokens
                        penalty_factor = 1.0 - (settings.code_length_penalty_weight * penalty_ratio)
                        # Floor at 10% to avoid complete elimination
                        penalty_factor = max(0.1, penalty_factor)

                        fitness = base_fitness * penalty_factor

                        logger.debug(
                            f"{civ_id}: Token penalty applied - {token_count} tokens "
                            f"(excess: {excess_tokens}, factor: {penalty_factor:.3f}, "
                            f"base_fitness: {base_fitness:.2f} -> penalized: {fitness:.2f})"
                        )
                    else:
                        logger.debug(
                            f"{civ_id}: No penalty - {token_count} tokens (below threshold)"
                        )

                self.civilizations[civ_id]["fitness"] = fitness
                self.civilizations[civ_id]["accuracy"] = accuracy
                self.civilizations[civ_id]["speed"] = speed

                # Store in genome for LLM prompts (use penalized fitness)
                genome.fitness = fitness
                genome.accuracy = accuracy
                genome.speed = speed
            except Exception as e:
                logger.error(f"Evaluation failed for {civ_id}: {e}")
                self.civilizations[civ_id]["fitness"] = 0
                self.civilizations[civ_id]["accuracy"] = 0.0
                self.civilizations[civ_id]["speed"] = 999.9
                genome.fitness = 0.0
                genome.accuracy = 0.0
                genome.speed = 999.9

            print(
                f"  Civilization {civ_id}: Fitness={self.civilizations[civ_id]['fitness']:.4f} "
                f"(Acc={self.civilizations[civ_id]['accuracy']:.4f}, Speed={self.civilizations[civ_id]['speed']:.5f}s) | "
                f"Genome: {genome.as_readable()}"
            )

        # Record generation history
        fitness_values = [civ["fitness"] for civ in self.civilizations.values()]
        generation_data = {
            "generation": self.generation,
            "population": [
                {
                    "civ_id": civ_id,
                    "fitness": civ_data["fitness"],
                    "accuracy": civ_data["accuracy"],
                    "speed": civ_data["speed"],
                    "description": civ_data["genome"].description,
                    "token_count": civ_data["genome"].token_count,
                    "parent_ids": civ_data["genome"].parent_ids,
                    # Add raw_code and theta for cross-validation (Task 1 requirement)
                    **(
                        {"raw_code": civ_data["genome"].raw_code}
                        if hasattr(civ_data["genome"], "raw_code")
                        else {}
                    ),
                    **(
                        {"theta": civ_data["genome"].theta}
                        if hasattr(civ_data["genome"], "theta")
                        else {}
                    ),
                }
                for civ_id, civ_data in self.civilizations.items()
            ],
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            "worst_fitness": min(fitness_values) if fitness_values else 0.0,
        }
        self.history.append(generation_data)

        # Selection + Reproduction
        self.breed_next_generation()

    def breed_next_generation(self) -> dict[str, dict]:
        """Generate next generation via selection, crossover, and mutation.

        Returns:
            Dictionary of new civilizations {civ_id: {genome, fitness}}
        """
        # Selection
        sorted_civs = sorted(
            self.civilizations.items(), key=lambda item: item[1].get("fitness", 0), reverse=True
        )
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        elites = sorted_civs[:num_elites]

        print(
            f"\n--- Top performing model in Generation {self.generation}: {elites[0][0]} with fitness {elites[0][1]['fitness']:.2f} ---"
        )

        # Reproduction (crossover + mutation)
        next_generation_civs = {}
        crossover_count = 0
        mutation_count = 0

        for i in range(self.population_size):
            new_civ_id = f"civ_{self.generation + 1}_{i}"

            # Count elites with raw_code for LLM crossover
            llm_elites = [e for e in elites if e[1]["genome"].raw_code is not None]

            # Decide: crossover or mutation?
            use_crossover = (
                settings.enable_crossover
                and len(llm_elites) >= 2
                and random.random() < settings.crossover_rate
            )

            if use_crossover:
                # Crossover: Select two parents
                parent1_civ, parent2_civ = select_crossover_parents(elites)
                parent1_genome = parent1_civ[1]["genome"]
                parent2_genome = parent2_civ[1]["genome"]

                new_genome = LLM_propose_surrogate_model(
                    parent1_genome,
                    self.generation + 1,
                    self.gemini_client,
                    self.cost_tracker,
                    second_parent=parent2_genome,
                    parent_ids=[parent1_civ[0], parent2_civ[0]],
                )
                crossover_count += 1
            else:
                # Mutation: Select single parent
                parent_civ = random.choice(elites)
                parent_genome = parent_civ[1]["genome"]

                new_genome = LLM_propose_surrogate_model(
                    parent_genome,
                    self.generation + 1,
                    self.gemini_client,
                    self.cost_tracker,
                    parent_ids=[parent_civ[0]],
                )
                mutation_count += 1

            next_generation_civs[new_civ_id] = {"genome": new_genome, "fitness": 0}

        logger.info(
            f"Generation {self.generation + 1}: {crossover_count} crossover + {mutation_count} mutation offspring"
        )

        self.civilizations = next_generation_civs
        self.generation += 1

        return next_generation_civs


# ------------------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path

    try:
        from visualization import export_history_json, generate_all_plots

        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        logger.warning("Visualization module not available - install matplotlib to enable plots")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    NUM_GENERATIONS = settings.num_generations if LLM_AVAILABLE and settings else 5
    POPULATION_SIZE = settings.population_size if LLM_AVAILABLE and settings else 10

    # Initialize Gemini client (or None for mock mode)
    gemini_client = None
    cost_tracker = None

    if LLM_AVAILABLE and settings and settings.google_api_key:
        logger.info("=" * 70)
        logger.info(f"Initializing Gemini client: {settings.llm_model}")
        logger.info(f"Temperature: {settings.temperature}")
        logger.info(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
        logger.info("=" * 70)

        gemini_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=settings.enable_rate_limiting,
        )
        cost_tracker = CostTracker(max_cost_usd=1.0)

        estimated_time = settings.estimated_runtime_minutes
        logger.info(f"Estimated runtime: {estimated_time:.1f} minutes")
        logger.info(f"Total API calls needed: {settings.total_requests_needed}")
        logger.info("=" * 70)
        print()
    else:
        logger.info("=" * 70)
        logger.info("Running in MOCK mode (no LLM integration)")
        logger.info(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
        logger.info("=" * 70)
        print()

    # Get initial particles based on test_problem configuration
    test_problem = settings.evolution_test_problem if LLM_AVAILABLE and settings else "plummer"
    num_particles = settings.evolution_num_particles if LLM_AVAILABLE and settings else 50
    initial_particles = get_initial_particles(test_problem, num_particles)

    logger.info(f"Test problem: {test_problem}")
    logger.info(f"Number of particles: {len(initial_particles)}")
    logger.info("=" * 70)
    print()

    # Create crucible and engine with test problem particles
    crucible = CosmologyCrucible.with_particles(initial_particles)
    engine = EvolutionaryEngine(
        crucible,
        population_size=POPULATION_SIZE,
        elite_ratio=settings.elite_ratio,
        gemini_client=gemini_client,
        cost_tracker=cost_tracker,
    )

    # Run evolution
    engine.initialize_population()

    for gen in range(NUM_GENERATIONS):
        engine.run_evolutionary_cycle()

        # Print cost summary after each generation
        if gemini_client and cost_tracker:
            summary = cost_tracker.get_summary()
            logger.info(
                f"Generation {gen} complete - "
                f"Cost so far: ${summary['total_cost_usd']:.4f} "
                f"({summary['budget_used_percent']:.1f}% of budget)"
            )

    # Final summary
    print()
    print("=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)

    if gemini_client and cost_tracker:
        summary = cost_tracker.get_summary()
        print()
        print("LLM Usage Summary:")
        print(f"  Total API calls:     {summary['total_calls']}")
        print(f"  Successful calls:    {summary['successful_calls']}")
        print(f"  Failed calls:        {summary['failed_calls']}")
        print(f"  Total tokens:        {summary['total_tokens']:,}")
        print(f"  Total cost:          ${summary['total_cost_usd']:.4f}")
        print(f"  Avg cost per call:   ${summary['avg_cost_per_call']:.6f}")
        print(f"  Total API time:      {summary['total_time_s']:.1f}s")
        print(f"  Budget remaining:    ${summary['budget_remaining_usd']:.4f}")
        print()

    # Generate visualizations and export data
    if VISUALIZATION_AVAILABLE:
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / f"run_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Saving results to: {output_dir}")

            # Export history as JSON with metadata
            history_path = output_dir / "evolution_history.json"
            metadata = {
                "test_problem": test_problem,
                "num_particles": len(initial_particles),
            }
            export_history_json(engine.history, str(history_path), metadata=metadata)
            print(f"  ✓ Evolution history saved: {history_path}")

            # Generate all plots
            generate_all_plots(engine.history, cost_tracker, str(output_dir))
            print(f"  ✓ Fitness progression plot: {output_dir / 'fitness_progression.png'}")
            print(f"  ✓ Accuracy vs speed plot: {output_dir / 'accuracy_vs_speed.png'}")
            print(f"  ✓ Token progression plot: {output_dir / 'token_progression.png'}")
            if cost_tracker and cost_tracker.calls:
                print(f"  ✓ Cost progression plot: {output_dir / 'cost_progression.png'}")

            print()
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
            print(f"Warning: Could not generate visualizations ({e})")
    else:
        print("Skipping visualization generation (matplotlib not available)")
        print()
        print()

    print("=" * 70)
