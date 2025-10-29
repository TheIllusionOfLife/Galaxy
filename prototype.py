import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

# Import LLM integration modules (will be None if not available)
try:
    from code_validator import validate_and_compile
    from config import settings
    from gemini_client import CostTracker, GeminiClient
    from prompts import get_initial_prompt, get_mutation_prompt

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    settings = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
DEFAULT_ATTRACTOR = [50.0, 50.0]


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
    compiled_predict: Callable[[list[float]], list[float]] | None = None  # Pre-validated callable

    def as_readable(self) -> str:
        if self.raw_code:
            head = self.raw_code.strip().splitlines()
            return head[0] if head else "custom-code"
        coeffs = ", ".join(f"{value:.4f}" for value in self.theta)
        return f"theta=[{coeffs}]"

    def build_callable(self, attractor: list[float]) -> Callable[[list[float]], list[float]]:
        # Prefer the pre-validated callable to avoid TOCTOU security issues
        if self.compiled_predict is not None:
            return self.compiled_predict
        if self.raw_code:
            # Use stricter validator when available; otherwise sandbox fallback
            if LLM_AVAILABLE:
                try:
                    from code_validator import validate_and_compile

                    compiled, validation = validate_and_compile(self.raw_code, attractor)
                    if not validation.valid or compiled is None:
                        raise ValueError(f"Invalid surrogate code: {validation.errors}")
                    self.compiled_predict = compiled
                    return compiled
                except ImportError:
                    pass
            return compile_external_surrogate(self.raw_code, attractor)
        return make_parametric_surrogate(self.theta, attractor)


def make_parametric_surrogate(
    theta: list[float], attractor: list[float]
) -> Callable[[list[float]], list[float]]:
    """Generate a surrogate model defined by theta parameters."""

    g_const, epsilon, dt_velocity, dt_position, velocity_correction, damping = theta

    def model(particle: list[float]) -> list[float]:
        x, y, vx, vy = particle
        dx = attractor[0] - x
        dy = attractor[1] - y
        dist_sq = dx * dx + dy * dy + epsilon
        dist = math.sqrt(dist_sq)
        denom = max(dist_sq * dist, 1e-6)
        ax = g_const * dx / denom
        ay = g_const * dy / denom

        new_vx = (1.0 - damping) * vx + ax * dt_velocity
        new_vy = (1.0 - damping) * vy + ay * dt_velocity
        new_x = x + (vx + velocity_correction * ax) * dt_position
        new_y = y + (vy + velocity_correction * ay) * dt_position

        return [new_x, new_y, new_vx, new_vy]

    return model


def compile_external_surrogate(
    code: str, attractor: list[float]
) -> Callable[[list[float]], list[float]]:
    """Execute LLM-generated code in a safe namespace and retrieve the predict function."""

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
    }
    sandbox_globals = {"__builtins__": allowed_builtins, "math": math}
    local_namespace: dict[str, Callable] = {}
    exec(code, sandbox_globals, local_namespace)

    if "predict" not in local_namespace:
        raise ValueError("Surrogate model code must define predict(particle, attractor) function.")

    predict_func = local_namespace["predict"]

    def wrapped(particle: list[float]) -> list[float]:
        return predict_func(particle, attractor)

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
def LLM_propose_surrogate_model(
    base_genome: SurrogateGenome | None,
    generation: int,
    gemini_client: Optional["GeminiClient"] = None,
    cost_tracker: Optional["CostTracker"] = None,
) -> SurrogateGenome:
    """Generate or evolve a new surrogate model using LLM or Mock fallback.

    Args:
        base_genome: Parent genome (None for initial generation)
        generation: Current generation number
        gemini_client: Gemini API client (None for Mock mode)
        cost_tracker: Cost tracker

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
            cost_tracker.add_call(
                response,
                {
                    "generation": generation,
                    "type": "initial" if base_genome is None else "mutation",
                },
            )

        if not response.success:
            logger.error(f"LLM call failed: {response.error}")
            return _mock_surrogate_generation(base_genome, generation)

        # Validate code
        compiled_func, validation = validate_and_compile(response.code, DEFAULT_ATTRACTOR)

        if not validation.valid:
            logger.warning(f"Generated code invalid: {validation.errors}")
            return _mock_surrogate_generation(base_genome, generation)

        if validation.warnings:
            for warning in validation.warnings:
                logger.debug(f"Code warning: {warning}")

        # Return LLM-generated genome
        logger.info("✓ LLM code validated and compiled successfully")
        return SurrogateGenome(
            theta=BASE_THETA[:],  # Keep default theta as fallback
            description=f"gemini_gen{generation}",
            raw_code=response.code,
            fitness=None,
            accuracy=None,
            speed=None,
            compiled_predict=compiled_func,
        )

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
    Simple N-body simulation environment.
    Compare and evaluate the accurate but slow 'true physical laws'
    with the fast 'surrogate models' proposed by civilizations.
    """

    def __init__(self, num_particles: int = 50, attractor: list[float] | None = None):
        # List of [x, y, vx, vy]
        self.particles = [
            [
                random.uniform(0, 100),
                random.uniform(0, 100),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            ]
            for _ in range(num_particles)
        ]
        self.attractor = attractor or DEFAULT_ATTRACTOR[:]  # Central gravity source

    def brute_force_step(self, particles: list[list[float]]) -> list[list[float]]:
        """True physical laws (Ground Truth). Accurate but intentionally slow."""
        new_particles = []
        for p in particles:
            dx = self.attractor[0] - p[0]
            dy = self.attractor[1] - p[1]
            dist_sq = dx**2 + dy**2 + 1e-6  # Avoid division by zero
            force = 10.0 / dist_sq

            ax = force * dx / math.sqrt(dist_sq)
            ay = force * dy / math.sqrt(dist_sq)

            new_vx = p[2] + ax * 0.1
            new_vy = p[3] + ay * 0.1
            new_x = p[0] + new_vx * 0.1
            new_y = p[1] + new_vy * 0.1
            new_particles.append([new_x, new_y, new_vx, new_vy])

        time.sleep(0.05)  # Simulate heavy computation
        return new_particles

    def evaluate_surrogate_model(
        self, model: Callable[[list[float]], list[float]]
    ) -> tuple[float, float]:
        """
        Evaluate the 'accuracy' and 'speed' of the surrogate model.
        """
        initial_state = [p[:] for p in self.particles]  # Copy state

        # 1. Accuracy evaluation
        ground_truth_next_state = self.brute_force_step(initial_state)

        start_time = time.time()
        try:
            predicted_next_state = []
            for particle in initial_state:
                prediction = model(particle)
                if not isinstance(prediction, (list, tuple)) or len(prediction) != 4:
                    raise ValueError("Surrogate model output must be a sequence of length 4.")
                predicted_next_state.append(list(prediction))
        except Exception:
            return 0.0, 999.9  # Invalid code gets worst evaluation

        speed = time.time() - start_time

        error = 0.0
        for i in range(len(initial_state)):
            true_p = ground_truth_next_state[i]
            pred_p = predicted_next_state[i]
            error += (true_p[0] - pred_p[0]) ** 2 + (true_p[1] - pred_p[1]) ** 2

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
                model_func = genome.build_callable(self.crucible.attractor)
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
                }
                for civ_id, civ_data in self.civilizations.items()
            ],
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            "worst_fitness": min(fitness_values) if fitness_values else 0.0,
        }
        self.history.append(generation_data)

        # Selection
        sorted_civs = sorted(
            self.civilizations.items(), key=lambda item: item[1].get("fitness", 0), reverse=True
        )
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        elites = sorted_civs[:num_elites]

        print(
            f"\n--- Top performing model in Generation {self.generation}: {elites[0][0]} with fitness {elites[0][1]['fitness']:.2f} ---"
        )

        # Reproduction
        next_generation_civs = {}
        for i in range(self.population_size):
            parent_civ = random.choice(elites)
            parent_genome: SurrogateGenome = parent_civ[1]["genome"]

            new_civ_id = f"civ_{self.generation + 1}_{i}"
            new_genome = LLM_propose_surrogate_model(
                parent_genome, self.generation + 1, self.gemini_client, self.cost_tracker
            )
            next_generation_civs[new_civ_id] = {"genome": new_genome, "fitness": 0}

        self.civilizations = next_generation_civs
        self.generation += 1


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

    # Create crucible and engine
    crucible = CosmologyCrucible()
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

            # Export history as JSON
            history_path = output_dir / "evolution_history.json"
            export_history_json(engine.history, str(history_path))
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
