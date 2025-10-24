import math
import random
import time
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Import LLM integration modules (will be None if not available)
try:
    from config import settings
    from gemini_client import GeminiClient, CostTracker, LLMResponse
    from prompts import get_initial_prompt, get_mutation_prompt
    from code_validator import validate_and_compile
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    settings = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
DEFAULT_ATTRACTOR = [50.0, 50.0]


@dataclass
class SurrogateGenome:
    """Parameterised代理モデルの遺伝子表現。"""

    theta: List[float]
    description: str = "parametric"
    raw_code: Optional[str] = None
    fitness: Optional[float] = None      # Store for LLM prompt context
    accuracy: Optional[float] = None     # Store for LLM prompt context
    speed: Optional[float] = None        # Store for LLM prompt context
    compiled_predict: Optional[Callable[[List[float]], List[float]]] = None  # Pre-validated callable

    def as_readable(self) -> str:
        if self.raw_code:
            head = self.raw_code.strip().splitlines()
            return head[0] if head else "custom-code"
        coeffs = ", ".join(f"{value:.4f}" for value in self.theta)
        return f"theta=[{coeffs}]"

    def build_callable(self, attractor: List[float]) -> Callable[[List[float]], List[float]]:
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


def make_parametric_surrogate(theta: List[float], attractor: List[float]) -> Callable[[List[float]], List[float]]:
    """θで定義される代理モデルを生成する。"""

    g_const, epsilon, dt_velocity, dt_position, velocity_correction, damping = theta

    def model(particle: List[float]) -> List[float]:
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


def compile_external_surrogate(code: str, attractor: List[float]) -> Callable[[List[float]], List[float]]:
    """安全な名前空間でLLM生成コードを実行し、predict関数を取得する。"""

    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
    }
    sandbox_globals = {"__builtins__": allowed_builtins, "math": math}
    local_namespace: Dict[str, Callable] = {}
    exec(code, sandbox_globals, local_namespace)

    if "predict" not in local_namespace:
        raise ValueError("代理モデルコードはpredict(particle, attractor)を定義する必要があります。")

    predict_func = local_namespace["predict"]

    def wrapped(particle: List[float]) -> List[float]:
        return predict_func(particle, attractor)

    return wrapped


PARAMETER_BOUNDS = [
    (5.0, 15.0),    # g_const
    (1e-6, 5.0),    # epsilon
    (0.05, 0.2),    # dt_velocity
    (0.05, 0.2),    # dt_position
    (0.0, 0.5),     # velocity_correction
    (0.0, 0.3),     # damping
]

BASE_THETA = [10.0, 0.01, 0.1, 0.1, 0.05, 0.02]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mutate_theta(theta: List[float], mutation_scale: float) -> List[float]:
    mutated: List[float] = []
    for value, (lower, upper) in zip(theta, PARAMETER_BOUNDS):
        span = upper - lower
        jitter = random.gauss(0, span * mutation_scale)
        mutated.append(clamp(value + jitter, lower, upper))
    return mutated


# ------------------------------------------------------------------------------
# LLM-powered代理モデル生成 (Mock fallback付き)
# ------------------------------------------------------------------------------
def LLM_propose_surrogate_model(
    base_genome: Optional[SurrogateGenome],
    generation: int,
    gemini_client: Optional["GeminiClient"] = None,
    cost_tracker: Optional["CostTracker"] = None
) -> SurrogateGenome:
    """LLMまたはMockで新しい代理モデルを生成・進化させる。

    Args:
        base_genome: 親のゲノム (Noneの場合は初期生成)
        generation: 現在の世代数
        gemini_client: Gemini APIクライアント (Noneの場合はMockモード)
        cost_tracker: コスト追跡器

    Returns:
        新しいSurrogateGenome
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
                mutation_type=mutation_type
            )
            # Get adaptive temperature for this generation
            temp_override = settings.get_mutation_temperature(generation)
            logger.info(f"Mutating model (generation {generation}, type {mutation_type}, temp {temp_override:.2f})")

        # Call Gemini with adaptive temperature
        response = gemini_client.generate_surrogate_code(
            prompt,
            temperature=temp_override if base_genome is not None else None
        )

        # Track cost
        if cost_tracker:
            cost_tracker.add_call(response, {
                "generation": generation,
                "type": "initial" if base_genome is None else "mutation"
            })

        if not response.success:
            logger.error(f"LLM call failed: {response.error}")
            return _mock_surrogate_generation(base_genome, generation)

        # Validate code
        compiled_func, validation = validate_and_compile(
            response.code,
            DEFAULT_ATTRACTOR
        )

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
            compiled_predict=compiled_func
        )

    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return _mock_surrogate_generation(base_genome, generation)


def _mock_surrogate_generation(
    base_genome: Optional[SurrogateGenome],
    generation: int
) -> SurrogateGenome:
    """Fallback: parametric mutation (no LLM required)."""
    if base_genome is None:
        theta = [
            clamp(base + random.uniform(-0.05, 0.05) * (upper - lower), lower, upper)
            for base, (lower, upper) in zip(BASE_THETA, PARAMETER_BOUNDS)
        ]
        return SurrogateGenome(theta=theta, description="mock_seed")

    mutation_scale = 0.12 if generation < 3 else 0.06
    new_theta = mutate_theta(base_genome.theta, mutation_scale)
    return SurrogateGenome(
        theta=new_theta,
        description=f"mock_mutant_gen{generation}"
    )

# ------------------------------------------------------------------------------
# るつぼ (Crucible) - AI文明が挑戦する物理シミュレーション環境
# ------------------------------------------------------------------------------
class CosmologyCrucible:
    """
    単純なN体シミュレーション環境。
    正確だが遅い「真の物理法則」と、文明が提案する高速な「代理モデル」を比較評価する。
    """
    def __init__(self, num_particles: int = 50, attractor: Optional[List[float]] = None):
        # [x, y, vx, vy] のリスト
        self.particles = [[random.uniform(0, 100), random.uniform(0, 100), random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(num_particles)]
        self.attractor = attractor or DEFAULT_ATTRACTOR[:]  # 中央の重力源

    def brute_force_step(self, particles: List[List[float]]) -> List[List[float]]:
        """真の物理法則（Ground Truth）。正確だが意図的に遅くしてある。"""
        new_particles = []
        for p in particles:
            dx = self.attractor[0] - p[0]
            dy = self.attractor[1] - p[1]
            dist_sq = dx**2 + dy**2 + 1e-6 # ゼロ除算を避ける
            force = 10.0 / dist_sq
            
            ax = force * dx / math.sqrt(dist_sq)
            ay = force * dy / math.sqrt(dist_sq)
            
            new_vx = p[2] + ax * 0.1
            new_vy = p[3] + ay * 0.1
            new_x = p[0] + new_vx * 0.1
            new_y = p[1] + new_vy * 0.1
            new_particles.append([new_x, new_y, new_vx, new_vy])
        
        time.sleep(0.05) # 重い計算をシミュレート
        return new_particles

    def evaluate_surrogate_model(self, model: Callable[[List[float]], List[float]]) -> tuple[float, float]:
        """
        代理モデルの「精度」と「速度」を評価する。
        """
        initial_state = [p[:] for p in self.particles] # 状態をコピー
        
        # 1. 精度評価
        ground_truth_next_state = self.brute_force_step(initial_state)
        
        start_time = time.time()
        try:
            predicted_next_state = []
            for particle in initial_state:
                prediction = model(particle)
                if not isinstance(prediction, (list, tuple)) or len(prediction) != 4:
                    raise ValueError("代理モデルの出力は長さ4のシーケンスである必要があります。")
                predicted_next_state.append(list(prediction))
        except Exception:
            return 0.0, 999.9 # 不正なコードは最低評価
        
        speed = time.time() - start_time
        
        error = 0.0
        for i in range(len(initial_state)):
            true_p = ground_truth_next_state[i]
            pred_p = predicted_next_state[i]
            error += (true_p[0] - pred_p[0])**2 + (true_p[1] - pred_p[1])**2
        
        accuracy = 1.0 / (1.0 + math.sqrt(error)) # エラーが小さいほど1に近づく
        
        return accuracy, speed

# ------------------------------------------------------------------------------
# 進化的エンジン - 文明を進化させる淘汰圧
# ------------------------------------------------------------------------------
class EvolutionaryEngine:
    """
    文明の世代交代を司る。優れた代理モデルを選択し、次世代のモデルを生み出す。
    """
    def __init__(
        self,
        crucible: CosmologyCrucible,
        population_size: int = 10,
        gemini_client: Optional["GeminiClient"] = None,
        cost_tracker: Optional["CostTracker"] = None
    ):
        self.crucible = crucible
        self.population_size = population_size
        self.civilizations: Dict[str, Dict] = {}
        self.generation = 0
        self.gemini_client = gemini_client
        self.cost_tracker = cost_tracker or (CostTracker() if LLM_AVAILABLE else None)
        self.history: List[Dict] = []

    def initialize_population(self):
        """最初の文明（代理モデル）群を生成する"""
        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            genome = LLM_propose_surrogate_model(
                None,
                self.generation,
                self.gemini_client,
                self.cost_tracker
            )
            self.civilizations[civ_id] = {"genome": genome, "fitness": 0}

    def run_evolutionary_cycle(self):
        """1世代分の進化（評価、選択、繁殖）を実行する"""
        print(f"\n===== Generation {self.generation}: Evaluating Surrogate Models =====")

        # 評価
        for civ_id, civ_data in self.civilizations.items():
            genome: SurrogateGenome = civ_data["genome"]
            try:
                model_func = genome.build_callable(self.crucible.attractor)
                accuracy, speed = self.crucible.evaluate_surrogate_model(model_func)

                # フィットネス = 精度 / 速度 (速度が速いほど良い)
                fitness = accuracy / (speed + 1e-9)
                self.civilizations[civ_id]["fitness"] = fitness
                self.civilizations[civ_id]["accuracy"] = accuracy
                self.civilizations[civ_id]["speed"] = speed

                # Store in genome for LLM prompts
                genome.fitness = fitness
                genome.accuracy = accuracy
                genome.speed = speed
            except Exception as e:
                logger.error(f"Evaluation failed for {civ_id}: {e}")
                self.civilizations[civ_id]["fitness"] = 0
                self.civilizations[civ_id]["accuracy"] = 0.0
                self.civilizations[civ_id]["speed"] = float("inf")
                genome.fitness = 0.0
                genome.accuracy = 0.0
                genome.speed = float("inf")

            print(
                f"  Civilization {civ_id}: Fitness={self.civilizations[civ_id]['fitness']:.4f} "
                f"(Acc={self.civilizations[civ_id]['accuracy']:.4f}, Speed={self.civilizations[civ_id]['speed']:.5f}s) | "
                f"Genome: {genome.as_readable()}"
            )

        # 選択
        sorted_civs = sorted(self.civilizations.items(), key=lambda item: item[1].get('fitness', 0), reverse=True)
        num_elites = max(1, int(self.population_size * 0.2))
        elites = sorted_civs[:num_elites]

        print(f"\n--- Top performing model in Generation {self.generation}: {elites[0][0]} with fitness {elites[0][1]['fitness']:.2f} ---")

        # 繁殖
        next_generation_civs = {}
        for i in range(self.population_size):
            parent_civ = random.choice(elites)
            parent_genome: SurrogateGenome = parent_civ[1]['genome']

            new_civ_id = f"civ_{self.generation + 1}_{i}"
            new_genome = LLM_propose_surrogate_model(
                parent_genome,
                self.generation + 1,
                self.gemini_client,
                self.cost_tracker
            )
            next_generation_civs[new_civ_id] = {"genome": new_genome, "fitness": 0}

        self.civilizations = next_generation_civs
        self.generation += 1

# ------------------------------------------------------------------------------
# メイン実行ブロック
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    NUM_GENERATIONS = settings.num_generations if LLM_AVAILABLE and settings else 5
    POPULATION_SIZE = settings.population_size if LLM_AVAILABLE and settings else 10

    # Initialize Gemini client (or None for mock mode)
    gemini_client = None
    cost_tracker = None

    if LLM_AVAILABLE and settings and settings.google_api_key:
        logger.info(f"=" * 70)
        logger.info(f"Initializing Gemini client: {settings.llm_model}")
        logger.info(f"Temperature: {settings.temperature}")
        logger.info(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
        logger.info(f"=" * 70)

        gemini_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=settings.enable_rate_limiting
        )
        cost_tracker = CostTracker(max_cost_usd=1.0)

        estimated_time = settings.estimated_runtime_minutes
        logger.info(f"Estimated runtime: {estimated_time:.1f} minutes")
        logger.info(f"Total API calls needed: {settings.total_requests_needed}")
        logger.info(f"=" * 70)
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
        gemini_client=gemini_client,
        cost_tracker=cost_tracker
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

    print("=" * 70)
