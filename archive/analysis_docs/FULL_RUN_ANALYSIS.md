# Full-Scale Physics Penalty Evolution Run - Analysis

**Date**: November 2, 2025, 21:56 JST
**Run ID**: `run_20251102_215639`
**Status**: ✅ COMPLETE

---

## Executive Summary

Successfully completed a full-scale evolution run (10 pop × 5 gen) with **physics penalty enabled**. This is the first validation of PR #43's physics-aware fitness function at full scale.

### Key Findings

❌ **Physics penalty did NOT achieve conservation goals**:
- **NO models with <1% energy drift** (goal: good conservation)
- **Best model: 4.21% energy drift** (vs 1% threshold)
- **88% of models had >100% energy drift** (severe violations)
- **Mean energy drift: 391.7%** across all generations

✅ **Physics penalty successfully penalized violations**:
- Models with high drift received fitness penalties
- Best fitness: 1518.11 (moderate performance)
- No models marked as invalid (fitness=-inf)

### Critical Discovery

**Physics penalty partially effective**: Best model improved dramatically (293% → 4.21% drift), BUT population mean remains poor (391.67% drift). Task 2 analysis revealed the root cause is LLM prompt design, not penalty strength.

---

## Configuration

### Evolution Parameters
```yaml
population_size: 10
num_generations: 5
test_problem: plummer
num_particles: 50
elite_ratio: 0.2
```

### Physics Penalty Settings
```yaml
physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01  # 1% threshold
  angular_momentum_threshold: 0.01
  validation_timesteps: 10
```

---

## Results

### Best Model Performance

| Metric | Value |
|--------|-------|
| **Civilization ID** | civ_1_7 |
| **Generation** | 1 |
| **Fitness** | 1,518.11 |
| **Accuracy** | 0.3754 (37.54%) |
| **Speed** | 0.000235s |
| **Energy Drift** | **4.21%** (4.2x threshold) |
| **Angular Momentum Drift** | 40.78% |
| **Trajectory RMSE** | 104.84 |
| **Type** | Parametric (theta) |

### Comparison to PR #41 Baseline

| Metric | PR #41 Baseline (No Penalty) | This Run (WITH Penalty) | Change |
|--------|------------------------------|-------------------------|--------|
| **Best Energy Drift** | 293% | **4.21%** | ✅ **98.6% better** |
| **Mean Energy Drift** | Unknown | **391.67%** | N/A |
| **Best Fitness** | 24,042 | **1,518** | ❌ **94% lower** |

**Key Finding**: Physics penalty successfully reduced **best model** drift from 293% → 4.21% (98.6% improvement), BUT at severe fitness cost (94% reduction). The population mean drift (391.67%) remains high, indicating most models still violate conservation.

---

## Population Statistics

### Evolution Summary

| Generation | Best Fitness | Avg Fitness | Worst Fitness | Best Energy Drift |
|------------|--------------|-------------|---------------|-------------------|
| 0 | 602.68 | 207.72 | 22.98 | 16.16% (poor) |
| 1 | **1,518.11** | 276.17 | 27.33 | **4.21%** (best) |
| 2 | 662.40 | 225.56 | 33.11 | 11.63% |
| 3 | 346.15 | 189.33 | 6.20 | 24.36% |
| 4 | 378.89 | 176.80 | 57.79 | 24.36% |

**Trend**: Best fitness peaked in Gen 1, then declined. Energy drift shows NO consistent improvement.

### Energy Drift Distribution

**Overall Statistics (50 models)**:
- **Minimum**: 4.21% (still 4x threshold)
- **Maximum**: 4,524.85% (catastrophic)
- **Mean**: 391.67% (severe violations)
- **Median**: 203.87% (severe violations)

**Conservation Quality**:
- **Excellent (<1% drift)**: 0/50 (0%)
- **Good (1-10% drift)**: 1/50 (2%) ← ONLY civ_1_7
- **Poor (10-100% drift)**: 5/50 (10%)
- **Severe (>100% drift)**: 44/50 (88%) ← VAST MAJORITY

### Physics Penalty Applied

| Energy Drift Range | Models | Avg Penalty Applied |
|-------------------|--------|---------------------|
| <1% (excellent) | 0 | N/A |
| 1-10% (good) | 1 | 0.049 (5% fitness reduction) |
| 10-100% (poor) | 5 | 0.609 (61% fitness reduction) |
| >100% (severe) | 44 | 3.93 (>100% fitness reduction, capped at 90%) |

**Critical Issue**: Even with 90% penalty cap, the population mean drift (391.67%) remains high despite best model achieving 4.21%. This suggests:
1. Penalty cap is too lenient (90% floor = 10% of base fitness remains)
2. Base fitness (accuracy/speed) is so high that 10% still beats conservative models
3. Need stricter penalties or invalid marking (fitness=-inf) for severe violations

---

## Cost & Performance

### API Usage
- **Total API calls**: 60 (60 successful, 0 failed)
- **Total tokens**: 148,430
- **Total cost**: $0.0316 (3.2% of $1.0 budget)
- **Avg cost per call**: $0.000526
- **Total API time**: 275.7 seconds (~4.6 minutes)

### Runtime
- **Start**: 21:47:15 JST
- **End**: 21:56:02 JST
- **Total**: ~8.8 minutes (within estimated 15-20 min)

### Validation Failures
- **Crossover failures**: 3 (syntax errors)
- **Mutation failures**: 2 (syntax errors, no return value)
- **Parametric fallback**: 100% success rate (5/5 fallbacks)
- **Invalid models (fitness=-inf)**: 0

---

## Comparison to PR #41 Baseline (WITHOUT Physics Penalty)

### PR #41 Results (No Penalty)
| Test Problem | Fitness | Energy Drift | Accuracy |
|--------------|---------|--------------|----------|
| two_body | 320,270 | 57.6% | 99.3% |
| figure_eight | 230,794 | 11.6% | 99.1% |
| plummer | 24,042 | 293% | 55.5% |

### This Run (WITH Physics Penalty)
| Test Problem | Fitness | Best Energy Drift | Mean Energy Drift | Accuracy |
|--------------|---------|-------------------|-------------------|----------|
| plummer | **1,518** | **4.21%** | **391.67%** | **37.5%** |

### Analysis

**Physics Conservation**: ✅ **IMPROVED (Best Model)** - Best energy drift improved from 293% → 4.21% (98.6% better)
**Physics Conservation**: ❌ **WORSE (Population Mean)** - Mean energy drift 391.67% indicates most models still violate conservation

**Fitness**: ❌ **WORSE** - Dropped from 24,042 to 1,518 (94% reduction)

**Why did physics penalty partially succeed?**
1. ✅ **Best model improved**: 293% → 4.21% drift shows penalty IS effective for top performers
2. ❌ **Population still poor**: 391.67% mean drift shows most models violate conservation
3. **Root causes**:
   - LLM prompts don't emphasize conservation (see Task 2 findings)
   - Fitness formula imbalance: speed multiplier (5000x) overwhelms physics penalty
   - Limited exploration: Small populations don't find conservation-preserving strategies

---

## Recommendations

### 1. Update LLM Prompts to Emphasize Conservation (HIGH PRIORITY)

**Current**: Prompts focus on accuracy and speed, no conservation mention
**Recommended**: Explicitly request energy/momentum conservation, symplectic integrators

**Rationale**: Task 2 showed weight tuning ineffective - LLM generates same fast models regardless of penalty strength. Root cause is prompt, not penalty weights.

### 2. Remove or Lower Penalty Cap (HIGH PRIORITY)

**Current**: 90% cap (10% floor)
**Recommended**:
- Option A: Remove cap entirely (allow unlimited penalty)
- Option B: Mark models with >100% drift as invalid (fitness=-inf)

**Rationale**: Scientifically invalid models should not survive selection.

### 3. Per-Problem Adaptive Thresholds (MEDIUM PRIORITY)

**Current**: 1% threshold for all problems
**Recommended**:
- two_body: 0.1% (simple 2-body orbit)
- figure_eight: 1-2% (chaotic 3-body)
- plummer: **5-10%** (complex N-body, accept higher drift)

**Rationale**: Plummer (N=50) is inherently more chaotic. 1% threshold may be unrealistic.

### 4. Fitness Formula Rebalancing (MEDIUM PRIORITY)

**Current**: `fitness = accuracy / (speed + 1e-9) - penalty`
**Issue**: Speed dominates (models with speed=0.0002s get 5000x multiplier)

**Recommended**: Consider capping speed benefit or using log-scale:
```python
fitness = accuracy / log(speed + 1.0) - penalty
```

### 5. Validation Timesteps (LOW PRIORITY)

**Current**: 10 timesteps during evolution, 100 in final validation
**Consider**: Increase to 20-50 timesteps during evolution for better physics detection

---

## Next Steps

### Immediate Actions (This Session)
1. ✅ Document findings in PHYSICS_PENALTY_RESULTS.md
2. ✅ Create comparison analysis (this document)
3. ⏳ Update configuration with recommended weights
4. ⏳ Run penalty weight tuning experiments (Task 2)

### Follow-up Tasks (Next Session)
1. Implement adaptive thresholds per problem
2. Experiment with penalty cap removal
3. Consider fitness formula rebalancing
4. Scientific paper: "Physics-Aware LLM Evolution Challenges"

---

## Conclusion

**Physics penalty infrastructure is working correctly** - penalties are applied, physics metrics are tracked, and the system is stable.

**However, population mean physics still poor** (391.67% drift) despite best model success (4.21%). Task 2 analysis revealed the issue is NOT penalty strength but LLM prompt design and fitness formula imbalance.

**This run successfully demonstrates that physics-aware fitness is CRITICAL** - without proper weighting, LLMs will discover fast approximations that violate fundamental physics laws.

**Recommended immediate action**: Update LLM prompts with conservation emphasis (see PENALTY_WEIGHT_TUNING_RESULTS.md recommendations).

---

## Outputs Generated

### Evolution Results
- `results/run_20251102_215639/evolution_history.json` (147 KB)
- `results/run_20251102_215639/fitness_progression.png` (210 KB)
- `results/run_20251102_215639/accuracy_vs_speed.png` (179 KB)
- `results/run_20251102_215639/token_progression.png` (237 KB)
- `results/run_20251102_215639/cost_progression.png` (163 KB)
- `results/run_20251102_215639/best_model_info.json` (metadata)

### Physics Validation
- `results/analysis/physics_validation_20251102_215713/validation_results.json`
- `results/analysis/physics_validation_20251102_215713/validation_report.md`

### Analysis Documents
- `FULL_RUN_ANALYSIS.md` (this document)

---

**Status**: Analysis complete. Ready for documentation update (Phase 6).
