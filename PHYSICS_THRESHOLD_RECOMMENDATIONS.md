# Physics Threshold Recommendations by Test Problem

**Date**: November 2, 2025
**Purpose**: Establish problem-specific physics thresholds based on theoretical expectations and empirical baselines
**Status**: Recommendations (Implementation Pending)

---

## Executive Summary

**Current Limitation**: Uniform 1% energy drift threshold applied to ALL test problems, regardless of complexity.

**Problem**: Different N-body problems have fundamentally different conservation characteristics:
- **Simple systems** (two-body): Near-perfect conservation expected (<0.1% drift)
- **Chaotic systems** (figure-eight): Moderate drift acceptable (1-2%)
- **Complex systems** (plummer N=50): Higher drift inevitable (5-15%)

**Recommendation**: Implement problem-specific thresholds that reflect physical reality and computational constraints.

---

## Baseline Physics Metrics (from PR #41)

### Empirical Drift Measurements

**Test Configuration**: Evolved models WITHOUT physics penalty, 100-step validation

| Test Problem | N Particles | Complexity | Energy Drift | Angular Mom. Drift | Fitness |
|--------------|-------------|-----------|--------------|-------------------|---------|
| **two_body** | 2 | Very simple (Kepler orbit) | 57.6% | 46.0% | 320,270 |
| **figure_eight** | 3 | Chaotic (3-body) | 11.6% | ~0% | 230,794 |
| **plummer** | 50 | Complex (N-body cluster) | 293% | 105% | 24,042 |

**Key Observation**: Even models optimized for accuracy/speed (no conservation constraint) showed problem-dependent drift:
- two_body: 57.6% (moderate violation)
- figure_eight: 11.6% (best performer despite chaos)
- plummer: 293% (severe violation)

**Interpretation**:
- figure_eight's low drift surprising → suggests LLM found stable approximation for this specific orbit
- plummer's catastrophic drift expected → 50-particle chaotic system difficult to approximate conservatively
- two_body's moderate drift surprising → simple 2-body should be near-perfect

---

## Theoretical Analysis by Problem Type

### Two-Body Circular Orbit

**Physical Characteristics**:
- **N = 2 particles** in mutual gravitational orbit
- **Closed-form solution**: Kepler's laws (exactly solvable)
- **Conservation**: Perfect energy and angular momentum conservation (symplectic)
- **Integrator expectations**: Leapfrog/Verlet should achieve <0.1% drift over 100 steps

**Expected Drift**:
- **Ideal**: <0.001% (symplectic integrator + simple system)
- **Acceptable**: 0.1-0.5% (good numerical approximation)
- **Poor**: >1% (suggests integration errors or non-conserving approximation)

**Recommended Threshold Options**:
- **Strict**: 0.001 (0.1%) → For scientific simulations requiring high precision
- **Moderate**: 0.002 (0.2%) → Balanced performance vs accuracy
- **Lenient**: 0.005 (0.5%) → Acceptable for fast surrogate models

### Figure-Eight Three-Body Orbit

**Physical Characteristics**:
- **N = 3 particles** in chaotic choreographic orbit (Chenciner-Montgomery solution)
- **Chaotic dynamics**: Small perturbations grow exponentially
- **Conservation**: Energy/momentum conserved but trajectory sensitive to errors
- **Integrator expectations**: 1-2% drift reasonable for chaotic systems

**Expected Drift**:
- **Ideal**: <1% (excellent conservation despite chaos)
- **Acceptable**: 1-2% (standard for chaotic systems)
- **Poor**: >5% (suggests unstable approximation)

**Recommended Threshold Options**:
- **Strict**: 0.010 (1.0%) → High-precision chaotic simulation
- **Moderate**: 0.015 (1.5%) → Balanced chaos management
- **Lenient**: 0.020 (2.0%) → Fast chaotic surrogate

### Plummer Sphere (N=50 Particles)

**Physical Characteristics**:
- **N = 50 particles** in realistic stellar cluster
- **Complexity**: Many-body interactions (O(N²) gravitational pairs)
- **Dynamics**: Mix of relaxation, close encounters, and ejections
- **Conservation challenges**: Approximations introduce cumulative errors

**Expected Drift**:
- **Ideal**: 5-10% (excellent for complex N-body)
- **Acceptable**: 10-20% (standard for surrogate models)
- **Poor**: >50% (suggests severe approximation errors)

**Rationale**: 50-particle system with fast approximations will inevitably sacrifice some conservation for speed. Threshold should reflect this trade-off.

**Recommended Threshold Options**:
- **Strict**: 0.050 (5%) → High-fidelity N-body surrogate
- **Moderate**: 0.100 (10%) → Balanced N-body approximation
- **Lenient**: 0.150 (15%) → Fast N-body prototype

---

## Recommended Threshold Options

### Option A: Strict (Conservative)

**Use Case**: Scientific simulations requiring high precision, publication-quality results

| Problem | Energy Threshold | Rationale |
|---------|------------------|-----------|
| two_body | 0.001 (0.1%) | Near-perfect conservation expected |
| figure_eight | 0.010 (1.0%) | Excellent chaos management |
| plummer | 0.050 (5.0%) | High-fidelity N-body |

**Implementation**:
```yaml
# config.yaml (if per-problem thresholds supported)
physics_penalty:
  per_problem_thresholds:
    two_body:
      energy_drift_threshold: 0.001
      angular_momentum_threshold: 0.001
    figure_eight:
      energy_drift_threshold: 0.010
      angular_momentum_threshold: 0.010
    plummer:
      energy_drift_threshold: 0.050
      angular_momentum_threshold: 0.050
```

---

### Option B: Moderate (Balanced) **← RECOMMENDED**

**Use Case**: General-purpose evolution, balancing conservation vs performance

| Problem | Energy Threshold | Rationale |
|---------|------------------|-----------|
| two_body | 0.002 (0.2%) | Good numerical approximation |
| figure_eight | 0.015 (1.5%) | Balanced chaos management |
| plummer | 0.100 (10%) | Standard N-body approximation |

**Benefits**:
- Realistic expectations for each problem complexity
- Allows surrogate models to prioritize speed when appropriate
- 10% plummer threshold matches "excellent" surrogate criteria

**Implementation**:
```yaml
physics_penalty:
  per_problem_thresholds:
    two_body:
      energy_drift_threshold: 0.002
      angular_momentum_threshold: 0.002
    figure_eight:
      energy_drift_threshold: 0.015
      angular_momentum_threshold: 0.015
    plummer:
      energy_drift_threshold: 0.100
      angular_momentum_threshold: 0.100
```

---

### Option C: Lenient (Permissive)

**Use Case**: Rapid prototyping, proof-of-concept, prioritize speed over precision

| Problem | Energy Threshold | Rationale |
|---------|------------------|-----------|
| two_body | 0.005 (0.5%) | Acceptable for fast surrogates |
| figure_eight | 0.020 (2.0%) | Fast chaotic approximation |
| plummer | 0.150 (15%) | Rapid N-body prototype |

**Use Cases**:
- Early development iterations
- Speed-critical applications
- Exploratory research

**Implementation**:
```yaml
physics_penalty:
  per_problem_thresholds:
    two_body:
      energy_drift_threshold: 0.005
      angular_momentum_threshold: 0.005
    figure_eight:
      energy_drift_threshold: 0.020
      angular_momentum_threshold: 0.020
    plummer:
      energy_drift_threshold: 0.150
      angular_momentum_threshold: 0.150
```

---

## Implementation Plan

### Current Configuration (Uniform Thresholds)

**File**: `config.yaml` (lines 37-43)
```yaml
physics_penalty:
  enabled: true
  energy_weight: 0.3
  momentum_weight: 0.1
  energy_drift_threshold: 0.01      # Uniform 1% for all problems
  angular_momentum_threshold: 0.01  # Uniform 1% for all problems
  validation_timesteps: 10
```

**Issue**: Single threshold applied regardless of problem type.

---

### Option I: Code-Based Implementation (Recommended Long-Term)

**Modify Configuration Schema**:

1. **Update `config.yaml`**:
```yaml
physics_penalty:
  enabled: true
  energy_weight: 5.0
  momentum_weight: 1.5
  validation_timesteps: 10
  per_problem_thresholds:
    two_body:
      energy_drift_threshold: 0.002
      angular_momentum_threshold: 0.002
    figure_eight:
      energy_drift_threshold: 0.015
      angular_momentum_threshold: 0.015
    plummer:
      energy_drift_threshold: 0.100
      angular_momentum_threshold: 0.100
  default_thresholds:  # Fallback for unknown problems
    energy_drift_threshold: 0.01
    angular_momentum_threshold: 0.01
```

2. **Update `config.py`** (Settings class):
```python
# Add per-problem threshold structure
physics_per_problem_thresholds: dict[str, dict[str, float]] = Field(
    default_factory=dict,
    description="Per-problem energy and momentum thresholds"
)

# Helper method to get threshold by problem
def get_physics_threshold(self, problem_name: str, metric: str) -> float:
    if problem_name in self.physics_per_problem_thresholds:
        return self.physics_per_problem_thresholds[problem_name].get(metric, 0.01)
    return 0.01  # Default
```

3. **Update `prototype.py`** (`calculate_physics_penalty()`):
```python
# Line ~917: Use problem-specific threshold
test_problem = settings.test_problem
energy_threshold = settings.get_physics_threshold(test_problem, "energy_drift_threshold")
momentum_threshold = settings.get_physics_threshold(test_problem, "angular_momentum_threshold")

physics_penalty = (
    settings.physics_energy_weight * max(0, energy_drift - energy_threshold) +
    settings.physics_momentum_weight * max(0, momentum_drift - momentum_threshold)
)
```

**Estimated Effort**: 1-2 hours (config schema + code integration + testing)

---

### Option II: Documentation-Based (Quick Solution)

**Create README Section**:

Add to `README.md` under "Configuration" → "Physics Penalty":

```markdown
### Per-Problem Threshold Recommendations

The default 1% threshold may be too strict or too lenient depending on problem complexity.
Recommended thresholds by problem:

**Option B (Balanced - Recommended)**:
- two_body: 0.2% - Simple 2-body orbit (near-perfect conservation)
- figure_eight: 1.5% - Chaotic 3-body (moderate drift acceptable)
- plummer: 10% - Complex N-body (higher drift expected)

To use problem-specific thresholds, manually edit `config.yaml` before running:
```yaml
physics_penalty:
  energy_drift_threshold: 0.002  # For two_body
  # OR
  energy_drift_threshold: 0.100  # For plummer
```

See PHYSICS_THRESHOLD_RECOMMENDATIONS.md for detailed rationale and options.

**Estimated Effort**: 30 minutes (documentation only)

---

## Validation Plan

### Test Per-Problem Thresholds

**Experiment Design**:
1. Run 3 evolution experiments (one per problem)
2. Use Option B (Moderate) thresholds for each
3. Compare conservation quality vs uniform 1% threshold

**Commands**:
```bash
# Two-body with 0.2% threshold
# Edit config.yaml: test_problem=two_body, energy_drift_threshold=0.002
uv run python prototype.py

# Figure-eight with 1.5% threshold
# Edit config.yaml: test_problem=figure_eight, energy_drift_threshold=0.015
uv run python prototype.py

# Plummer with 10% threshold
# Edit config.yaml: test_problem=plummer, energy_drift_threshold=0.100
uv run python prototype.py
```

**Expected Outcomes**:
- **two_body**: More models pass 0.2% threshold than 1% (easier to satisfy)
- **plummer**: Fewer models marked invalid with 10% threshold (more realistic)
- **Overall**: Better balance between conservation goals and achievable performance

---

## Cost Estimate for Code-Based Implementation

**Development**:
- Config schema update: 30 minutes
- Settings class modification: 30 minutes
- Prototype.py integration: 30 minutes
- Testing (3 problems × small run): 15 minutes, $0.015

**Total**: 2 hours development + $0.015 validation

---

## Decision Matrix

| Criteria | Code-Based | Documentation-Based |
|----------|-----------|---------------------|
| **Effort** | 2 hours | 30 minutes |
| **Cost** | $0.015 | $0 |
| **Usability** | Automatic (set once) | Manual (edit per run) |
| **Flexibility** | High (config-driven) | Low (manual edits) |
| **Maintainability** | Excellent (DRY) | Poor (error-prone) |
| **Validation** | Required (integration tests) | Minimal (docs only) |
| **Production Ready** | Yes | Workaround only |

**Recommendation**: **Documentation-Based NOW** (unblocks current work), **Code-Based NEXT SESSION** (proper solution).

---

## Conclusion

### Problem Identified: ✅ **CONFIRMED**

Uniform 1% threshold is unrealistic:
- **Too lenient** for simple two-body (should be 0.1-0.5%)
- **Appropriate** for chaotic figure-eight (1-2%)
- **Too strict** for complex plummer (should be 5-15%)

### Recommendations: ✅ **DOCUMENTED**

Three threshold options (Strict/Moderate/Lenient) provided with clear rationale:
- **Option B (Moderate)** recommended for general use
- Problem-specific thresholds reflect physical reality
- Implementation paths provided (code-based vs doc-based)

### Next Steps

**Immediate** (This Session):
1. ~~Document threshold recommendations~~ ✅ **COMPLETE**
2. Update README with per-problem guidance (documentation-based)

**Follow-Up** (Next Session):
1. Implement code-based per-problem thresholds
2. Run validation experiments (3 problems × moderate thresholds)
3. Update LLM prompts to emphasize conservation (from Task 2 findings)
4. Re-run full evolution with updated prompts + per-problem thresholds

---

## References

- **PR #41**: Physics Validation of Evolved Models (baseline drift measurements)
- **FULL_RUN_ANALYSIS.md**: Task 1 full-scale run (430% plummer drift)
- **PENALTY_WEIGHT_TUNING_RESULTS.md**: Task 2 weight tuning (16% best across all configs)
- **validation_metrics.py**: Physics computation implementation

---

**Status**: Task 3 complete. Per-problem threshold recommendations documented and ready for implementation.
