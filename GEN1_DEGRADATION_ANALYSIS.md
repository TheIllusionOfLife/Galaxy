# Gen 0 → Gen 1 Degradation Analysis
## Plummer Evolution Investigation (run_20251103_190821)

**Date**: 2025-11-03
**Problem**: Gen 0 had 33% survival (1/3 models), Gen 1 had 0% survival (3/3 eliminated)
**Threshold**: Plummer max_energy_drift = 20%, max_momentum_drift = 20%

---

## Executive Summary

**Root Cause Identified**: **Category A - Prompt Design Issue (Threshold Mismatch)**

The mutation prompt uses a **hardcoded 1% conservation goal** (prompts.py:195) while the actual **plummer threshold is 20%**. This caused the LLM to perceive a working model (16.25% drift, below 20% threshold) as "violating conservation", leading it to over-optimize with complex integrators and introduce critical bugs.

**Impact**: 16.25% drift (Gen 0 survivor) → 218.2% drift (Gen 1 offspring) - **13x degradation**

---

## Detailed Analysis

### Gen 0 Results

| Model | Fitness | Energy Drift | Status | Integration Method | Epsilon |
|-------|---------|--------------|--------|-------------------|---------|
| civ_0_0 | 1.81 | 16.25% | ✅ SURVIVED | Semi-implicit Euler | 1.0 |
| civ_0_1 | null | 248.9% | ❌ ELIMINATED | Semi-implicit Euler | 0.01 |
| civ_0_2 | null | 63.2% | ❌ ELIMINATED | Mock seed | N/A |

**Key Observation**: civ_0_0 survived with 16.25% drift (below 20% threshold)

### Gen 1 Results

| Model | Parent | Fitness | Energy Drift | Status | Integration Method | Epsilon |
|-------|--------|---------|--------------|--------|-------------------|---------|
| civ_1_0 | N/A | null | 1091.7% | ❌ ELIMINATED | Mock mutant | N/A |
| civ_1_1 | civ_0_0 | null | 218.2% | ❌ ELIMINATED | Velocity Verlet | 0.01 |
| civ_1_2 | N/A | null | 762% | ❌ ELIMINATED | Mock mutant | N/A |

**Key Observation**: Only civ_1_1 had real LLM-generated code (offspring of civ_0_0)

---

## Root Cause: Prompt Threshold Mismatch

### Problem in prompts.py:195-198

```python
if energy_drift > 0.01:  # Hardcoded 1% threshold!
    perf_analysis.append(
        f"✗ Energy drift {energy_drift * 100:.2f}% - violates conservation (goal <1%)"
    )
```

### What the LLM Received

For parent civ_0_0 (16.25% drift):
```
✗ Energy drift 16.25% - violates conservation (goal <1%)
```

### The Fundamental Problem

1. **Actual plummer threshold**: 20% (from config.yaml per_problem_thresholds)
2. **Prompt claims goal**: <1%
3. **Parent's 16.25%**: Actually ACCEPTABLE (below 20% threshold)
4. **LLM's perception**: "Parent violates conservation, must fix"

### LLM's "Fix" Attempt (civ_1_1)

The LLM tried to improve conservation by:

1. **Changed integrator**: Semi-implicit Euler → Velocity Verlet
   - Comment in code: "Smaller epsilon for better accuracy, but need a better integrator"
   - Velocity Verlet requires 2 force calculations per step (more complex)

2. **Reduced softening**: epsilon 1.0 → 0.01
   - Less softening = "more accurate" but harder numerical integration
   - Smaller epsilon makes force calculations more sensitive to errors

3. **Introduced critical bug**: Used inconsistent positions in second force loop
   - Line 70-82: Calculates forces from OLD particle positions to NEW position
   - Velocity Verlet requires simultaneous updates (impossible with per-particle function)

**Result**: 16.25% drift → 218.2% drift (**13x worse**)

---

## Code Bug Analysis: civ_1_1 Velocity Verlet Implementation

### The Bug (lines 67-82 in civ_1_1 raw_code)

```python
# Step 3: Calculate new acceleration a(t + dt) at the predicted new positions
next_ax = 0.0
next_ay = 0.0
next_az = 0.0

for other_particle in all_particles:
    if other_particle is particle:
        continue

    opx, opy, opz, opvx, opvy, opvz, omass = other_particle

    # BUG: Using OTHER particle's OLD position (opx) with THIS particle's NEW position (new_px)
    dx = opx - new_px  # ← INCONSISTENT TIME STEPS
    dy = opy - new_py
    dz = opz - new_pz

    r_squared = dx*dx + dy*dy + dz*dz + epsilon*epsilon
    r = math.sqrt(r_squared)

    magnitude = G * omass / r_squared

    next_ax += magnitude * (dx / r)
    next_ay += magnitude * (dy / r)
    next_az += magnitude * (dz / r)
```

### Why This is Wrong

**Velocity Verlet requires**:
- Force at time `t` using positions `x(t)`
- Force at time `t+dt` using positions `x(t+dt)` for **ALL particles**

**What the code does**:
- Force at time `t+dt` for THIS particle at `new_px` (time t+dt)
- But OTHER particles are still at `opx` (time t)
- **Inconsistent**: Mixing t and t+dt positions in the same force calculation

**Why it can't be fixed**: The `predict(particle, all_particles)` function signature requires per-particle updates. Velocity Verlet needs simultaneous updates of all particles, which is architecturally impossible without changing the interface.

---

## Hypothesis Testing Results

### ✅ Hypothesis 1: Prompt Mismatch (CONFIRMED)
- **Evidence**: prompts.py:195 uses hardcoded `energy_drift > 0.01` (1%)
- **Impact**: LLM told 16.25% drift "violates conservation" when threshold is 20%
- **Consequence**: LLM over-optimized, introduced bugs

### ✅ Hypothesis 2: Integration Complexity (CONFIRMED)
- **Evidence**: civ_1_1 attempted Velocity Verlet, introduced critical bug
- **Impact**: Per-particle function signature prevents correct Velocity Verlet
- **Consequence**: Complex integrators beyond LLM capability in this architecture

### ⚠️ Hypothesis 3: Aggressive Mutation (PARTIALLY CONFIRMED)
- **Evidence**: Mutation switched from working semi-implicit Euler to complex Velocity Verlet
- **Note**: This was CAUSED by Hypothesis 1 (prompt mismatch made LLM think it needed fixing)

### ❌ Hypothesis 4: Statistical Variance (REJECTED)
- **Evidence**: Systematic issue - prompt mismatch affects ALL mutations with >1% drift
- **Not random**: Will reproduce on any plummer run where parent has 1-20% drift

---

## Impact Assessment

### Immediate Impact
- **Plummer evolution fails**: Gen 1 has 0% survival
- **No genetic progress**: All offspring eliminated, can't evolve further
- **Wasted API calls**: $0.006/generation for eliminated models

### Scope of Problem
- **Affects all test problems**: two_body (2%), figure_eight (1.5%), plummer (20%)
- **Any parent with >1% drift**: Will receive "violates conservation" message
- **Systematic**: Not a one-time fluke, will reproduce consistently

### Why It Worked in two_body/figure_eight (Hypothesis)
- Smaller N (2-3 particles) → simpler dynamics
- Lower thresholds (2%, 1.5%) closer to the 1% prompt goal
- May have gotten lucky with simpler mutations that didn't break conservation

---

## Solution Proposal

### Primary Fix: Dynamic Threshold Communication (Category A)

**Change prompts.py:194-209** to use actual per-problem threshold:

```python
# Get the actual threshold for this problem
threshold = settings.get_physics_threshold(test_problem, "energy")
momentum_threshold = settings.get_physics_threshold(test_problem, "momentum")

# Categorize drift relative to threshold
if energy_drift > threshold:
    # ELIMINATED by hard constraint
    perf_analysis.append(
        f"✗ Energy drift {energy_drift * 100:.2f}% - ELIMINATED (exceeds {threshold*100:.1f}% threshold)"
    )
elif energy_drift > 0.01:
    # Acceptable but not ideal
    perf_analysis.append(
        f"⚠ Energy drift {energy_drift * 100:.2f}% - acceptable but not ideal (threshold: {threshold*100:.1f}%, ideal: <1%)"
    )
else:
    # Excellent conservation
    perf_analysis.append(
        f"✓ Energy drift {energy_drift * 100:.2f}% - excellent conservation"
    )
```

**Impact**:
- LLM correctly perceives 16.25% as "acceptable" (below 20% threshold)
- LLM less likely to over-optimize with complex integrators
- LLM focuses on other improvements (speed, accuracy) instead

### Secondary Fix: Integration Complexity Constraint (Category B)

**Add guidance to mutation prompt** (prompts.py:249-250):

```python
INTEGRATION METHOD GUIDANCE:
- Semi-implicit Euler: Reliable, good for most cases (RECOMMENDED for first attempts)
- Velocity Verlet: Complex, requires careful implementation, error-prone in this architecture
- RK4/Leapfrog: Advanced, very difficult to implement correctly

If parent achieves acceptable conservation (<{threshold*100:.1f}% for this problem),
focus on tuning parameters (epsilon, force calculations) rather than changing integrator.
```

### Tertiary Fix: Physics Validation (Category C)

**Add to code_validator.py** (future enhancement):
- Smoke test: Run 1 timestep, check energy doesn't explode (>50% change)
- Force direction test: Particles should attract, not repel
- Numerical stability: Check for NaN/inf in test case

---

## Recommended Action Plan

### Phase 1: Implement Primary Fix (HIGH PRIORITY)
1. Update prompts.py lines 194-209 to use dynamic thresholds
2. Pass `test_problem` parameter to `get_mutation_prompt()`
3. Use `settings.get_physics_threshold()` for threshold lookup
4. Test with plummer validation run (3 pop × 2 gen)
5. Expected outcome: Gen 1 survival rate 10-30% (similar to Gen 0)

### Phase 2: Validate Fix (HIGH PRIORITY)
1. Run fresh plummer evolution (10 pop × 5 gen)
2. Monitor Gen 1 survival rate
3. Check if offspring maintain conservation levels
4. Verify fitness improves across generations

### Phase 3: Implement Secondary Fix (MEDIUM PRIORITY)
1. Add integration method guidance to prompts
2. Discourage complex integrators for already-acceptable models
3. Validate with additional runs

### Phase 4: Consider Tertiary Fix (LOW PRIORITY)
1. Add physics validation to code_validator.py
2. Catch bugs earlier in the pipeline
3. Reduce wasted evolution cycles

---

## Conclusion

The Gen 0 → Gen 1 degradation in plummer evolution is caused by a **prompt threshold mismatch**. The mutation prompt uses a hardcoded 1% conservation goal while the actual plummer threshold is 20%, causing the LLM to perceive acceptable models as "violating conservation" and over-optimize with complex, bug-prone integrators.

**Primary fix**: Update prompts.py to use dynamic per-problem thresholds from config.yaml
**Expected impact**: Restore healthy Gen 1 survival rates (10-30%), enable continued evolution
**Implementation time**: 30 minutes (prompt update + validation run)

**Status**: ✅ IMPLEMENTED (commit f44224b)

---

## Post-Fix Validation & Model Upgrade Recommendations

### Validation Results (10 pop × 5 gen)

**Prompt fix was implemented correctly** but revealed a deeper issue:

- **Gen 0**: 1/10 survived (10% survival rate)
  - civ_0_0: 16.08% energy drift ✅ SURVIVED

- **Gen 1-4**: 0/10 each generation (0% survival rate)
  - All offspring eliminated despite dynamic threshold prompts
  - **High syntax error rate**: 50%+ of LLM-generated code had syntax errors
  - Valid code still violated 20% conservation threshold

### Revised Root Cause Assessment

**Primary Root Cause (Category D - LLM Capability Limitation)**:
- Gemini 2.5 Flash Lite cannot reliably generate physically-accurate N-body code for 50-particle Plummer spheres
- Even with correct prompts, generated code violates conservation thresholds
- High syntax error rate (50%+) indicates model struggles with code complexity
- Crossover operations particularly problematic (repeated syntax errors)

**Secondary Issue (Category A - Prompt Design)**:
- Hardcoded 1% threshold was misleading but NOT the primary failure cause
- Fix implemented correctly but insufficient to overcome LLM limitations

### Model Upgrade Strategy

Based on validation showing LLM capability limitations:

**Immediate Action**:
1. **Switch to Gemini 2.5 Pro** for plummer evolution
   - Edit config.yaml: `model.name: gemini-2.5-pro`
   - Expected improvement: 80%+ reduction in syntax errors
   - Cost impact: ~$3 per run (vs $0.03 for flash-lite)

2. **Run comparative experiment**:
   ```bash
   # Baseline (already complete): Flash Lite, 10 pop × 5 gen on plummer
   # Validation: Pro, 10 pop × 5 gen on plummer
   echo "model:\n  name: gemini-2.5-pro" >> config.yaml
   POPULATION_SIZE=10 NUM_GENERATIONS=5 TEST_PROBLEM=plummer uv run python prototype.py
   ```

3. **Decision criteria**:
   - If Pro achieves >20% Gen 1 survival: Use Pro for plummer
   - If Pro still fails: Problem may require simplification (e.g., 20 particles instead of 50)
   - Keep Flash Lite for two_body/figure_eight (sufficient quality)

**Cost-Benefit Analysis**:
- Flash Lite: $0.03/run, 0% Gen 1 survival = wasted compute
- Pro: $3.00/run, potential 20%+ Gen 1 survival = enables evolution
- **ROI**: 100x cost pays off if it enables any evolution progress
- Alternative: Simplify problem (reduce particles) if Pro still fails

**Implementation**:
- Model switching now easy: One line in config.yaml (commit 29bb5f7)
- All costs/rates auto-adjust per model
- See README.md "Switching Between Models" section for details
