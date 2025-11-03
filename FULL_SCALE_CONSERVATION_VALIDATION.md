# Full-Scale Conservation Prompts Validation

> **📌 Resolution**: Issues identified in this validation have been addressed in **[PR #50: Fitness Formula Rebalancing](https://github.com/TheIllusionOfLife/Galaxy/pull/50)**.
>
> **Key Fix**: Hard constraint now eliminates catastrophic physics violators (>10% energy drift, >50% momentum drift) with `fitness=-inf`, preventing models with >1000% drift from surviving selection.

**Date**: November 3, 2025
**Status**: ⚠️ **FAILED TO GENERALIZE** - Conservation prompts did not scale effectively

## Executive Summary

Validated PR #47's conservation-aware LLM prompts at full scale (10 pop × 5 gen) on two_body and plummer problems. **CRITICAL FINDING**: Conservation prompts that achieved 0.10% energy drift in small runs (3 pop × 2 gen) **FAILED to generalize** to full scale.

### Key Findings

❌ **two_body (N=2)**: Best drift **degraded** from 0.10% → 0.16% (+60%)
❌ **plummer (N=50)**: Best drift **degraded** from 4.21% → 16.25% (+286%)
❌ **Conservation rate**: 2% for two_body, 0% for plummer (target was >10%)
⚠️ **Mean drift**: Extremely high (161,061% two_body, 1,658% plummer)

### Cost & Performance

- **Total API calls**: 120 (60 per run)
- **Total cost**: $0.0878 (8.8% of daily budget)
- **Runtime**: 9m 51s total
- **LLM success rate**: 100% (no syntax errors)

---

## Experiment Configuration

### Evolution Parameters
- Model: gemini-2.5-flash-lite
- Population: 10 models per generation
- Generations: 5
- Physics Penalty: Enabled (energy_weight=0.3, momentum_weight=0.1)
- Conservation Prompts: **ENABLED** (PR #47)

### Runs Executed
1. **two_body**: results/run_20251103_111004
2. **plummer**: results/run_20251103_111514

---

## Results

### two_body (N=2 particles)

| Metric | Value | vs PR #47 Small Run | Target |
|--------|-------|---------------------|--------|
| **Best Energy Drift** | **0.16%** | 0.10% → 0.16% (**+60% worse**) | <1% |
| **Mean Energy Drift** | 161,061% | 1.12% → 161,061% (**massive degradation**) | <10% |
| **Median Energy Drift** | 10.15% | N/A | <10% |
| **Models <1% drift** | 1/50 (2%) | 1/6 (17%) → 1/50 (2%) | >10% |
| **Models <10% drift** | 24/50 (48%) | 3/6 (50%) → 24/50 (48%) | >50% |
| **Best Fitness** | 320,157 | 187,907 → 320,157 (+70%) | - |

**Finding**: Gen 0 showed promise (7/10 models with ~1.12% drift), but later generations produced extremely high drift values (>100,000%), suggesting **mutation/crossover broke conservation**.

### plummer (N=50 particles)

| Metric | Value | vs PR #45 Baseline | Target |
|--------|-------|-------------------|--------|
| **Best Energy Drift** | **16.25%** | 4.21% → 16.25% (**+286% worse**) | <10% |
| **Mean Energy Drift** | 1,658% | 391.67% → 1,658% (**+323% worse**) | <100% |
| **Median Energy Drift** | 202.67% | N/A | <100% |
| **Models <1% drift** | 0/49 (0%) | 0/50 (0%) → 0/49 (0%) | >5% |
| **Models <10% drift** | 0/49 (0%) | Unknown → 0/49 (0%) | >10% |
| **Best Fitness** | 1,345 | 1,518 → 1,345 (-11%) | - |
| **Invalid Models** | 1 | 0 → 1 | 0 |

**Finding**: Conservation prompts **completely failed** on complex N-body problem. NO models achieved <10% drift. Baseline (without prompts) was actually better (4.21% vs 16.25%).

---

## Comparison to Baselines

### PR #47 Small Run (WITH Prompts, 3 pop × 2 gen)
- Best drift: 0.10% (two_body)
- Mean drift (Gen 0): 1.12%
- Conservation rate: 17%
- **Result**: ✅ SUCCESS at small scale

### This Run (WITH Prompts, 10 pop × 5 gen)
- Best drift: 0.16% (two_body), 16.25% (plummer)
- Mean drift: 161,061% (two_body), 1,658% (plummer)
- Conservation rate: 2% (two_body), 0% (plummer)
- **Result**: ❌ **COMPLETE FAILURE** at full scale

### PR #45 Baseline (WITHOUT Prompts, 10 pop × 5 gen)
- Best drift: 4.21% (plummer)
- Mean drift: 391.67%
- Conservation rate: 0%
- **Result**: Baseline was actually BETTER than conservation prompts for plummer!

---

## Root Cause Analysis

### Hypothesis 1: Mutation/Crossover breaks Conservation Code ✅ LIKELY
**Evidence**:
- Gen 0 two_body: 7/10 models at ~1.12% drift (good)
- Later generations: Drift explodes to >100,000% (catastrophic)
- **Conclusion**: Initial LLM code conserves well, but mutations destroy conservation properties

### Hypothesis 2: Prompt Insufficient for Complex Problems ✅ CONFIRMED
**Evidence**:
- two_body (simple): 2% conservation rate (poor but not zero)
- plummer (complex, N=50): 0% conservation rate (complete failure)
- **Conclusion**: Prompts don't generalize to many-body interactions

### Hypothesis 3: Physics Penalty Too Weak ✅ CONFIRMED
**Evidence**:
- Models with >1,000% drift still survived selection
- Fitness penalty capped at 90%, allowing severe violators
- **Conclusion**: Even catastrophic physics violations don't eliminate models

### Hypothesis 4: Population Size Effects ⚠️ POSSIBLE
**Evidence**:
- Small run (3 pop): 17% conservation rate
- Full run (10 pop): 2% conservation rate
- **Conclusion**: Larger population introduces more diversity, including poor conservers

---

## Key Findings

### 1. Conservation Prompts Do NOT Scale ❌
**Small scale (6 models)**: 0.10% best drift, 17% conservation rate
**Full scale (100 models)**: 0.16% best drift, 2% conservation rate

**Implication**: PR #47 results were **NOT REPRESENTATIVE** of full-scale performance.

### 2. Mutation/Crossover Destroys Conservation ❌
Gen 0 shows good conservation → Later gens catastrophic
**Implication**: LLM mutations don't preserve physics-aware code patterns.

### 3. Complex Problems Completely Fail ❌
two_body (N=2): 2% conservation rate
plummer (N=50): 0% conservation rate

**Implication**: Prompts designed for simple problems don't transfer.

### 4. Baseline Outperforms on plummer ❌
PR #45 (no prompts): 4.21% best drift
This run (with prompts): 16.25% best drift

**Implication**: Conservation prompts actually HURT performance on complex problems!

---

## Scientific Implications

### 1. **Problem Formulation Complexity Hypothesis VALIDATED**
- Simple conservation guidance insufficient for complex N-body dynamics
- LLMs need problem-specific physics knowledge, not just general principles

### 2. **Evolutionary Preservation Challenge**
- Mutations that improve fitness don't necessarily preserve conservation
- Need explicit conservation constraints in mutation operator

### 3. **Scale-Dependent Effectiveness**
- Small-scale experiments (PR #47) can be **highly misleading**
- Minimum validation: 50+ models across multiple generations

### 4. **Baseline Comparison Critical**
- Conservation prompts can make things WORSE
- Always compare to NO-prompts baseline

---

## Recommendations

### IMMEDIATE (High Priority)

**1. REJECT conservation prompts as currently designed**
- Do NOT merge as production default
- Evidence shows they fail at scale and on complex problems
- Baseline (no prompts) performs better on plummer

**2. Add conservation constraints to mutation operator**
```python
# Instead of free-form mutation prompts
# Add explicit constraints:
if parent_energy_drift < 0.01:
    prompt += "CRITICAL: Your mutation MUST preserve energy conservation."
    prompt += "Verify: E_final - E_initial < 1% at every timestep."
```

**3. Implement hard physics constraint**
```python
# In fitness calculation
if energy_drift > 0.10:  # 10% threshold
    return -inf  # Eliminate completely
```

**4. Problem-specific prompt engineering**
- two_body: Focus on Kepler orbit preservation
- plummer: Multi-body hierarchical approximations, cluster dynamics

### MEDIUM (Research)

**5. Investigate why mutations break conservation**
- Analyze mutation diffs: what code patterns get destroyed?
- Test conservative mutation strategies (smaller changes, lower temperature)

**6. Test intermediate population sizes**
- 3 pop (✓ works), 10 pop (✗ fails)
- Try 5 pop, 7 pop to find threshold

**7. Adaptive threshold by problem complexity**
- two_body: 0.5% threshold (strict)
- plummer: 5% threshold (realistic)

### LONG-TERM (Architecture)

**8. Multi-objective optimization**
- Explicit Pareto front: accuracy × speed × conservation
- Don't let speed dominate

**9. Conservation-preserving crossover**
- Only cross-breed models that BOTH conserve
- Prevents propagation of violators

**10. Meta-learning from failures**
- Use this data to improve prompts
- \"Here's what DOESN'T work: [failed code examples]\"

---

## Validation Status

| Criteria | Status | Notes |
|----------|--------|-------|
| Both runs complete | ✅ | 100 models evaluated |
| Physics metrics collected | ✅ | All models validated |
| Comparison to baselines | ✅ | PR #45 and PR #47 |
| Documentation created | ✅ | This report |
| **Conservation goals met** | ❌ | **FAILED** |
| **Generalization validated** | ❌ | **FAILED** |

---

## Cost & Performance Summary

### API Usage
- Total runs: 2
- Total API calls: 120
- Total cost: $0.0878 (8.8% of budget)
- Budget remaining: $0.9122

### Runtime
- two_body: 5m 10s
- plummer: 4m 41s
- Total: 9m 51s

### LLM Performance
- Successful calls: 120/120 (100%)
- Failed calls: 0
- Syntax error rate: 0% (excellent)

**Note**: LLM code generation worked perfectly. The problem is NOT syntax - it's that mutations destroy conservation properties.

---

## Conclusion

**Conservation-aware LLM prompts (PR #47) FAILED to generalize to full-scale evolution.**

While small-scale experiments (3 pop × 2 gen) showed promise (0.10% drift, 17% conservation rate), full-scale validation (10 pop × 5 gen) revealed **catastrophic failure**:

1. **Best drift degraded**: 0.10% → 0.16% (two_body), 4.21% → 16.25% (plummer)
2. **Mean drift exploded**: 1.12% → 161,061% (two_body), 391% → 1,658% (plummer)
3. **Conservation rate collapsed**: 17% → 2% (two_body), 0% → 0% (plummer)

**Root causes**:
- Mutations break conservation code patterns
- Prompts insufficient for complex N-body dynamics
- Physics penalty too weak (90% cap allows catastrophic violators)
- Scale effects not captured in small experiments

**Critical lesson**: Small-scale LLM experiments can be **highly misleading**. Always validate at production scale (50+ models, 5+ generations) before drawing conclusions.

**Recommendation**: **DO NOT merge** conservation prompts as currently designed. Return to problem formulation (hard constraints, mutation operators, multi-objective optimization) before attempting prompt-based solutions.

---

## Appendices

### A. Run Directories
- two_body: `results/run_20251103_111004`
- plummer: `results/run_20251103_111514`

### B. Validation Reports
- two_body: `results/analysis/physics_validation_20251103_111538/`
- plummer: Validation failed due to invalid model (civ_2_4)

### C. Raw Data
- Analysis results: `validation_results_detailed.json`
- Evolution histories: `[run_dir]/evolution_history.json`

### D. Session Learnings

**Pattern**: Small-scale success ≠ full-scale success
**Evidence**: PR #47 (6 models, 0.10% drift) → This validation (100 models, catastrophic)
**Lesson**: Minimum 50 models × 5 generations for LLM evolution validation

**Pattern**: Mutations destroy emergent properties
**Evidence**: Gen 0 conserves → Gen 1+ catastrophic drift
**Lesson**: LLM mutations need explicit conservation constraints, not just inheritance

**Pattern**: Prompt engineering alone insufficient
**Evidence**: Conservation prompts failed despite perfect syntax
**Lesson**: Architectural changes (hard constraints, multi-objective) required

---

**Next Steps**: See Recommendations section. Priority: Reject current prompts, implement hard physics constraints, investigate mutation strategies.
