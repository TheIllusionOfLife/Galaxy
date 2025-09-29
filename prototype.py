import random
import time
import math
from typing import List, Dict, Callable

# ------------------------------------------------------------------------------
# Mock LLM - 課題解決のための「代理モデル」を提案する創造主
# ------------------------------------------------------------------------------
def LLM_propose_surrogate_model(base_model_code: str, generation: int) -> str:
    """
    LLMが新しい代理モデル（物理法則を近似する高速な関数）をPythonコードとして生成・進化させるプロセスを模倣する。
    """
    # 新しい代理モデルを生成（ここでは既存のモデルの係数を変更したり、項を追加することで模倣）
    if generation < 2:
        return f"lambda p: (p[0] + p[2] * {random.uniform(-0.1, 0.1):.2f}, p[1] + p[3] * {random.uniform(-0.1, 0.1):.2f}, p[2], p[3])"
    else:
        new_model_code = base_model_code.replace(
            f"{random.uniform(-0.2, 0.2):.2f}", 
            f"{random.uniform(-0.2, 0.2):.2f}"
        )
        if random.random() < 0.3: # 新しい項を追加する突然変異
            new_model_code = new_model_code.replace(")", f" - p[0] * {random.uniform(0.0, 0.01):.3f})")
        return new_model_code

# ------------------------------------------------------------------------------
# るつぼ (Crucible) - AI文明が挑戦する物理シミュレーション環境
# ------------------------------------------------------------------------------
class CosmologyCrucible:
    """
    単純なN体シミュレーション環境。
    正確だが遅い「真の物理法則」と、文明が提案する高速な「代理モデル」を比較評価する。
    """
    def __init__(self, num_particles: int = 50):
        # [x, y, vx, vy] のリスト
        self.particles = [[random.uniform(0, 100), random.uniform(0, 100), random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(num_particles)]
        self.attractor = [50, 50] # 中央の重力源

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
            predicted_next_state = [model(p) for p in initial_state]
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
            model_code = LLM_propose_surrogate_model("lambda p: (p[0] + p[2]*0.1, p[1] + p[3]*0.1, p[2], p[3])", self.generation)
            self.civilizations[civ_id] = {"model_code": model_code, "fitness": 0}

    def run_evolutionary_cycle(self):
        """1世代分の進化（評価、選択、繁殖）を実行する"""
        print(f"\n===== Generation {self.generation}: Evaluating Surrogate Models =====")
        
        # 評価
        for civ_id, civ_data in self.civilizations.items():
            model_code = civ_data["model_code"]
            try:
                model_func = eval(model_code)
                accuracy, speed = self.crucible.evaluate_surrogate_model(model_func)
                
                # フィットネス = 精度 / 速度 (速度が速いほど良い)
                fitness = accuracy / (speed + 1e-9)
                self.civilizations[civ_id]["fitness"] = fitness
                self.civilizations[civ_id]["accuracy"] = accuracy
                self.civilizations[civ_id]["speed"] = speed
            except Exception as e:
                self.civilizations[civ_id]["fitness"] = 0
            
            print(f"  Civilization {civ_id}: Fitness={self.civilizations[civ_id].get('fitness',0):.2f} "
                  f"(Acc={self.civilizations[civ_id].get('accuracy',0):.3f}, Speed={self.civilizations[civ_id].get('speed',999):.4f}s) | "
                  f"Model: {model_code}")

        # 選択
        sorted_civs = sorted(self.civilizations.items(), key=lambda item: item[1].get('fitness', 0), reverse=True)
        num_elites = max(1, int(self.population_size * 0.2))
        elites = sorted_civs[:num_elites]
        
        print(f"\n--- Top performing model in Generation {self.generation}: {elites[0][0]} with fitness {elites[0][1]['fitness']:.2f} ---")

        # 繁殖
        next_generation_civs = {}
        for i in range(self.population_size):
            parent_civ = random.choice(elites)
            parent_model_code = parent_civ[1]['model_code']
            
            new_civ_id = f"civ_{self.generation + 1}_{i}"
            new_model_code = LLM_propose_surrogate_model(parent_model_code, self.generation + 1)
            next_generation_civs[new_civ_id] = {"model_code": new_model_code, "fitness": 0}

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
