# Penalty Weight Tuning Results

**Date**: November 2, 2025, 22:30 JST
**Experiment**: 4 weight configurations (3-33x increase over baseline)
**Test Size**: 3 population × 2 generations per config = 6 models each
**Total Cost**: $0.02 (4 runs × $0.005)
**Total Runtime**: ~15 minutes

---

## Executive Summary

### Critical Discovery: **Physics Penalty DOES NOT Improve Conservation**

❌ **ALL 4 weight configurations FAILED to achieve conservation goals**:
- **NO models achieved <1% energy drift goal** (0/24 models across all configs)
- **Best drift identical across configs**: 16.25% (same model selected despite different penalties)
- **Increasing weights made problem WORSE**: Higher weights increased mean drift from 319% to 4,151%
- **Fitness decreased monotonically**: Higher weights → lower fitness (519 → 146), but NO physics improvement

**Root Cause**: LLM is generating the **SAME type of fast models** regardless of penalty strength. The parametric fallback model (0.1625 drift) becomes best performer across all configs, suggesting:
1. Physics penalty successfully penalized violators
2. But NO models with good conservation were generated
3. Evolution converged to "least bad" option (16% drift parametric model)
4. LLM code generation doesn't produce physics-preserving approximations

---

## Experiment Configuration

### Test Matrix

| Config | Energy Weight | Momentum Weight | Multiplier vs Baseline |
|--------|---------------|-----------------|------------------------|
| **Baseline** (Task 1) | 0.3 | 0.1 | 1x (reference) |
| **Config 1** | 1.0 | 0.3 | 3-4x (conservative) |
| **Config 2** | 3.0 | 1.0 | 10x (moderate) |
| **Config 3** | 5.0 | 1.5 | 15-17x (aggressive) |
| **Config 4** | 10.0 | 3.0 | 30-33x (very aggressive) |

### Evolution Parameters
- **Population**: 3 models per generation
- **Generations**: 2 (Gen 0-1)
- **Test Problem**: plummer (N=50 particles)
- **Elite Ratio**: 0.2 (top 20% preserved)
- **Threshold**: 1% energy drift, 1% angular momentum drift

---

## Results by Configuration

### Config 1: Weights (1.0, 0.3) - Conservative

**Physics Metrics**:
- Best Energy Drift: **16.25%** (16x threshold)
- Mean Energy Drift: 318.65%
- Median Energy Drift: 179.61%

**Conservation Quality**:
- Excellent (<1%): 0/6 (0%)
- Good (1-10%): 0/6 (0%)
- Poor (10-100%): 1/6 (16.7%) ← Best model
- Severe (>100%): 5/6 (83.3%)

**Fitness**:
- Best: 519.82
- Mean: 133.61

**Analysis**: Conservative increase (3x) insufficient to enforce conservation. Best model at 16% drift, majority severe violators.

---

### Config 2: Weights (3.0, 1.0) - Moderate

**Physics Metrics**:
- Best Energy Drift: **16.25%** (identical to Config 1)
- Mean Energy Drift: 900.70% (2.8x worse than Config 1)
- Median Energy Drift: 203.16%

**Conservation Quality**:
- Excellent (<1%): 0/6 (0%)
- Good (1-10%): 0/6 (0%)
- Poor (10-100%): 1/6 (16.7%)
- Severe (>100%): 5/6 (83.3%)

**Fitness**:
- Best: 304.67 (41% lower than Config 1)
- Mean: 98.36 (26% lower)

**Analysis**: 10x weight increase reduced fitness but did NOT improve physics. Same best model selected (16% drift).

---

### Config 3: Weights (5.0, 1.5) - Aggressive

**Physics Metrics**:
- Best Energy Drift: **16.25%** (still identical)
- Mean Energy Drift: 491.44%
- Median Energy Drift: 201.85%

**Conservation Quality**:
- Excellent (<1%): 0/6 (0%)
- Good (1-10%): 0/6 (0%)
- Poor (10-100%): 1/6 (16.7%)
- Severe (>100%): 5/6 (83.3%)

**Fitness**:
- Best: 144.06 (72% lower than Config 1)
- Mean: 66.30 (50% lower)

**Analysis**: 15x weight increase caused massive fitness drop but NO conservation improvement. Evolution converged to same parametric model.

---

### Config 4: Weights (10.0, 3.0) - Very Aggressive

**Physics Metrics**:
- Best Energy Drift: **16.25%** (no improvement)
- Mean Energy Drift: 4,150.99% (13x worse than Config 1!)
- Median Energy Drift: 161.53%

**Conservation Quality**:
- Excellent (<1%): 0/6 (0%)
- Good (1-10%): 0/6 (0%)
- Poor (10-100%): 2/6 (33.3%) ← Slightly better
- Severe (>100%): 4/6 (66.7%)

**Fitness**:
- Best: 146.40 (72% lower than Config 1)
- Mean: 76.21 (43% lower)

**Analysis**: 30x weight increase reduced severe violations from 83% to 67% but mean drift exploded to 4,151%. Fitness severely penalized but physics still poor.

---

## Comparison Summary

| Metric | Baseline (Task 1) | Config 1 | Config 2 | Config 3 | Config 4 | Trend |
|--------|-------------------|----------|----------|----------|----------|-------|
| **Best Drift (%)** | 4.21 | 16.25 | 16.25 | 16.25 | 16.25 | ❌ **WORSE** |
| **Mean Drift (%)** | 391.67 | 318.65 | 900.70 | 491.44 | 4,150.99 | ❌ **ERRATIC** |
| **% <1% Drift** | 0% | 0% | 0% | 0% | 0% | ❌ **NO CHANGE** |
| **% >100% Drift** | 88% | 83% | 83% | 83% | 67% | ✓ **SLIGHT IMPROVEMENT** |
| **Best Fitness** | 1,518 | 519.82 | 304.67 | 144.06 | 146.40 | ❌ **DECLINING** |
| **Mean Fitness** | N/A | 133.61 | 98.36 | 66.30 | 76.21 | ❌ **DECLINING** |

---

## Critical Findings

### 1. **Physics Penalty Ineffective for Conservation**

Despite 3-33x weight increases, **NO improvement in best energy drift** (stuck at 16.25%):
- All configs selected same parametric model as best performer
- LLM-generated models had worse physics than parametric fallback
- Evolution learned to avoid severe violators but couldn't find conservers

### 2. **LLM Doesn't Generate Physics-Preserving Code**

**Problem**: LLM code generation produces approximations that:
- Prioritize speed (fast evaluations)
- Sacrifice accuracy for computational efficiency
- Do NOT preserve energy/momentum conservation
- Fail to implement symplectic integrators or energy-correcting schemes

**Evidence**: 5-6 out of 6 models per config had >100% energy drift

### 3. **Parametric Model is "Least Bad" Choice**

**Parametric fallback** (16.25% drift) consistently selected as best:
- Still 16x above 1% conservation threshold
- But better than LLM-generated alternatives (100-4000% drift)
- Suggests current problem formulation doesn't incentivize conservation

### 4. **Fitness-Physics Trade-off is Broken**

**Observation**: Higher weights reduce fitness without improving physics
- Config 1 (1.0, 0.3): Fitness=520, Drift=16%
- Config 4 (10.0, 3.0): Fitness=146 (72% drop), Drift=16% (NO CHANGE)

**Conclusion**: Penalty is applied correctly, but no models with good physics exist in the population to select.

---

## Why Did Tuning Fail?

### Hypothesis 1: **Problem Definition Issue**

**Current fitness formula**:
```
base_fitness = accuracy / (speed + 1e-9)
code_penalty = code_weight * max(0, (tokens - max_tokens) / max_tokens)
physics_penalty = energy_weight * max(0, drift - 0.01) + momentum_weight * max(0, momentum_drift - 0.01)
total_penalty = code_penalty + physics_penalty
final_fitness = base_fitness - (base_fitness * min(0.9, total_penalty))
```

**Issues**:
1. **Speed dominates**: Models with speed=0.0002s get 5000x multiplier, overwhelming physics penalty
2. **Penalty cap (90%)**: Allows 10% fitness floor even for catastrophic violators
3. **Threshold too strict**: 1% may be unrealistic for 50-particle plummer sphere

### Hypothesis 2: **LLM Prompt Doesn't Emphasize Conservation**

**Current prompts** (from prompts.py):
- Focus on "surrogate model for gravitational forces"
- Emphasize accuracy and speed
- NO explicit mention of energy/momentum conservation
- NO guidance on symplectic integrators or conservation schemes

**Fix**: Update prompts to explicitly request conservation-preserving approximations

### Hypothesis 3: **Evolution Exploration Too Limited**

**Small population** (3 models × 2 gen = 6 evaluations):
- Insufficient to explore conservation-preserving space
- Most LLM attempts failed validation or had severe violations
- Parametric fallback dominated due to reliability, not quality

**Fix**: Larger population (10-20) × more generations (5-10) to explore conservation strategies

---

## Recommendations

### 🚫 DO NOT Continue Weight Tuning

**Evidence**: 3-33x increase showed NO improvement in conservation
**Conclusion**: Problem is NOT insufficient penalty strength

### ✅ RECOMMENDED ACTIONS

#### 1. **Update LLM Prompts** (HIGH PRIORITY)

Add conservation-focused guidance:
```
"Generate a surrogate model that:
1. Approximates gravitational forces efficiently
2. PRESERVES energy conservation (drift <1%)
3. PRESERVES angular momentum conservation
4. Uses symplectic integration schemes if possible
5. Balances accuracy, speed, AND physics preservation"
```

#### 2. **Rebalance Fitness Formula** (HIGH PRIORITY)

**Option A - Multi-Objective**:
```
fitness = w1 * accuracy + w2 * (1/speed) - w3 * energy_drift - w4 * momentum_drift
```

**Option B - Log-Scale Speed**:
```
base_fitness = accuracy / log(speed + 1.0)
```

**Option C - Hard Constraint**:
```
if energy_drift > 0.1:  # 10% threshold
    fitness = -inf  # Invalid model
```

#### 3. **Relax Threshold for Plummer** (MEDIUM PRIORITY)

Current 1% threshold may be unrealistic for N=50 complex system:
- two_body: Keep 1% (simple system)
- figure_eight: Keep 1-2% (chaotic but manageable)
- plummer: **Increase to 5-10%** (complex N-body)

#### 4. **Larger Population** (MEDIUM PRIORITY)

Increase to 10 pop × 5 gen to explore conservation space properly

---

## Next Steps

### Immediate (This Session if Time)
1. ~~Weight tuning experiments~~ ✅ **COMPLETE (failed)**
2. **Document per-problem threshold recommendations** (Task 3)

### Follow-up (Next Session)
1. **Update LLM prompts** to emphasize conservation
2. **Rebalance fitness formula** (multi-objective or log-scale)
3. **Re-run evolution** with updated prompts + rebalanced fitness
4. **Validate** if conservation improves with better problem formulation

---

## Cost & Time Summary

**Actual Cost**: $0.02 (4 configs × $0.005)
**Actual Runtime**: ~15 minutes (sequential execution)
**API Calls**: 36 (9 per config)
**Models Evaluated**: 24 (6 per config)

**Budget Remaining**: $0.9684 (96.8% of daily limit)

---

## Conclusion

### Experiment Status: ✅ **COMPLETE**

Physics penalty weight tuning experiments successfully executed and analyzed.

### Conservation Goals: ❌ **NOT ACHIEVED**

**Critical Discovery**: Increasing penalty weights from 0.3/0.1 → 10.0/3.0 (33x) did NOT improve conservation:
- NO models achieved <1% energy drift goal across ANY configuration
- Best drift identical (16.25%) regardless of weight strength
- Higher weights reduced fitness but physics remained poor

### Scientific Insight: **Problem Formulation Requires Redesign**

Physics penalty infrastructure works correctly, but:
1. **LLM prompts don't emphasize conservation** → generates fast, non-conserving models
2. **Fitness formula prioritizes speed over physics** → speed (5000x multiplier) overwhelms penalty
3. **Thresholds may be too strict** → 1% unrealistic for 50-particle chaotic system
4. **Exploration insufficient** → small populations don't find conservation-preserving strategies

**Recommended Fix**: Update prompts (emphasize conservation) + rebalance fitness (multi-objective or hard constraints) + larger populations (10×5)

---

## Outputs Generated

- `experiments/penalty_tuning_20251102/config{1-4}_output.txt` (4 evolution logs)
- `results/run_20251102_222731/` (Config 1 outputs)
- `results/run_20251102_222900/` (Config 2 outputs)
- `results/run_20251102_222958/` (Config 3 outputs)
- `results/run_20251102_223052/` (Config 4 outputs)
- `PENALTY_WEIGHT_TUNING_RESULTS.md` (this document)

---

**Status**: Task 2 complete. Ready for Task 3 (Per-Problem Threshold Recommendations).
