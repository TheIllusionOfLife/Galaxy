import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# ------------------------------------------------------------------------------
DEFAULT_ATTRACTOR = [50.0, 50.0]


@dataclass
class SurrogateGenome:
    """Parameterised代理モデルの遺伝子表現。"""

    theta: List[float]
    description: str = "parametric"
    raw_code: Optional[str] = None

    def as_readable(self) -> str:
        if self.raw_code:
            head = self.raw_code.strip().splitlines()
            return head[0] if head else "custom-code"
        coeffs = ", ".join(f"{value:.4f}" for value in self.theta)
        return f"theta=[{coeffs}]"

    def build_callable(self, attractor: List[float]) -> Callable[[List[float]], List[float]]:
        if self.raw_code:
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
# Mock LLM - 課題解決のための「代理モデル」を提案する創造主
# ------------------------------------------------------------------------------
def LLM_propose_surrogate_model(base_genome: Optional[SurrogateGenome], generation: int) -> SurrogateGenome:
    """LLMが新しい代理モデル（物理法則を近似する高速な関数）を生成・進化させるプロセスを模倣する。"""

    if base_genome is None:
        theta = [
            clamp(base + random.uniform(-0.05, 0.05) * (upper - lower), lower, upper)
            for base, (lower, upper) in zip(BASE_THETA, PARAMETER_BOUNDS)
        ]
        return SurrogateGenome(theta=theta, description="seed")

    mutation_scale = 0.12 if generation < 3 else 0.06
    new_theta = mutate_theta(base_genome.theta, mutation_scale)
    return SurrogateGenome(theta=new_theta, description=f"mutant of {base_genome.description}")

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

    def evaluate_surrogate_model(self, model: Callable[[List[float]], List[float]]) -> (float, float):
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
    def __init__(self, crucible: CosmologyCrucible, population_size: int = 10):
        self.crucible = crucible
        self.population_size = population_size
        self.civilizations: Dict[str, Dict] = {}
        self.generation = 0

    def initialize_population(self):
        """最初の文明（代理モデル）群を生成する"""
        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            genome = LLM_propose_surrogate_model(None, self.generation)
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
            except Exception as e:
                self.civilizations[civ_id]["fitness"] = 0
                self.civilizations[civ_id]["accuracy"] = 0.0
                self.civilizations[civ_id]["speed"] = float("inf")

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
            new_genome = LLM_propose_surrogate_model(parent_genome, self.generation + 1)
            next_generation_civs[new_civ_id] = {"genome": new_genome, "fitness": 0}

        self.civilizations = next_generation_civs
        self.generation += 1

# ------------------------------------------------------------------------------
# メイン実行ブロック
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_GENERATIONS = 5

    crucible = CosmologyCrucible()
    engine = EvolutionaryEngine(crucible)
    
    engine.initialize_population()
    
    for gen in range(NUM_GENERATIONS):
        engine.run_evolutionary_cycle()
