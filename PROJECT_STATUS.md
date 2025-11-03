# Galaxy Prometheus - Project Status

**Last Updated**: 2025-11-03
**Current Version**: Main branch (commit a4cb175)

## Executive Summary

Galaxy Prometheus is an LLM-driven evolutionary framework for discovering novel N-body simulation algorithms. The system uses Gemini models to evolve surrogate force calculation methods that are faster than traditional approaches while maintaining physical accuracy.

**Current Status**: ✅ **Core system functional and validated**

## Recent Accomplishments (October-November 2025)

### 1. Per-Problem Physics Thresholds (PR #53) ✅ MERGED

**Problem**: Uniform 10% energy drift threshold eliminated 100% of models in complex N=50 plummer simulations, preventing evolution entirely.

**Solution**: Implemented per-problem threshold configuration system allowing different thresholds based on problem complexity:

- **two_body** (N=2): 2.0% threshold - allows parametric baseline models (~1.12% inherent drift)
- **figure_eight** (N=3): 1.5% threshold - moderate for chaotic system
- **plummer** (N=50): 20% threshold - relaxed for complex N-body interactions

**Key Features**:
- Per-problem overrides with graceful fallback to global defaults
- Fail-fast validation with intelligent typo detection using difflib
- Comprehensive test coverage (10 tests)
- Backward compatible with existing configurations

**Validation Results**:

| Problem | Threshold | Survival Rate | Status |
|---------|-----------|---------------|--------|
| plummer | 20% | 16.7% (1/6) | ✅ Working |
| two_body | 2% | 50% (3/6) | ✅ Working |

**Impact**: Critical blocker removed - plummer evolution now possible with healthy population diversity.

### 2. Rebalanced Fitness Formula (PR #50)

**Problem**: Speed multiplier (5000x range) dominated fitness, marginalizing accuracy and conservation penalties.

**Solution**: Logarithmic speed normalization + weighted formula rebalancing:
- `log_speed = log₁₀(1 + speed_ratio)` reduces 5000x → ~4x influence
- Increased energy penalty weight from 0.2 → 0.3
- Momentum penalty weight remains 0.1

**Results**: Conservation violations now properly penalized while maintaining speed incentive.

### 3. Conservation-Aware LLM Prompts (PR #47)

**Status**: ⚠️ **Failed to generalize** (see archive/analysis_docs/CONSERVATION_PROMPTS_RESULTS.md)

**Attempted**: Added physics education to prompts to guide LLM toward conserving implementations.

**Outcome**: Minimal impact on conservation metrics; models still violate physics laws at similar rates.

**Learning**: LLM code generation is largely independent of prompt-based physics guidance. Conservation must be enforced via fitness penalties and hard constraints, not prompt engineering.

## Current Configuration

**Model**: gemini-2.5-flash-lite (temperature: 0.8, max_tokens: 2000)

**Evolution**:
- Population size: 10 models/generation
- Generations: 5
- Elite ratio: 20%
- Particles: 50 (for plummer)
- Test problem: plummer

**Rate Limiting** (Free tier):
- 15 requests/minute
- 50 max requests/run
- Enabled: ✅

**Fitness Formula**:
- Hard constraint: Enabled
  - Energy drift: 10% global, per-problem overrides
  - Momentum drift: 50% global, per-problem overrides
- Speed: Logarithmic normalization (base 10.0)
- Energy penalty weight: 0.3
- Momentum penalty weight: 0.1

**Code Penalty**:
- Enabled: ✅
- Weight: 0.1
- Max tokens: 400

**Crossover**:
- Enabled: ✅
- Rate: 50%
- Temperature: 0.75

## Repository Structure

```
Galaxy/
├── config.yaml                 # Single source of truth for configuration
├── config.py                   # Pydantic settings with validation
├── prototype.py                # Main evolution engine
├── programmer.py               # LLM code generation
├── crucible.py                 # Physics simulation and validation
├── initial_conditions.py       # Test problems (two_body, figure_eight, plummer)
├── baselines.py                # Baseline models (direct N-body, KDTree)
├── tests/
│   ├── test_per_problem_thresholds.py  # Per-problem threshold tests (10 tests)
│   └── ...
├── archive/
│   ├── analysis_docs/          # Old analysis documents
│   ├── old_plans/              # Archived planning documents
│   ├── logs/                   # Old log files
│   └── old_scripts/            # Deprecated test scripts
└── PROJECT_STATUS.md           # This document
```

## Test Coverage

**Core Tests**:
- Per-problem thresholds: 10 tests ✅
- Fitness calculation: Multiple scenarios
- Configuration validation: Comprehensive
- Physics validation: Conservation law checks

**Validation Runs**:
- Plummer (20% threshold): 16.7% survival (1/6 models, Gen 0: 1/3)
- Two_body (2% threshold): 50% survival (3/6 models, Gen 0: 3/3)

## Known Issues and Limitations

1. **Free Tier Rate Limits**: 15 RPM constraint limits evolution speed
2. **Conservation Generalization**: LLM-generated models struggle to maintain strict conservation
3. **Test Coverage**: Integration tests need expansion
4. **Benchmark Suite**: Needs full execution and validation
5. **Figure_eight**: No validation run yet (1.5% threshold untested)

## Next Steps and Roadmap

### Immediate (High Priority)

1. **Full-Scale Plummer Evolution** 🎯
   - Run complete evolution with 20% threshold
   - Validate that population diversity sustains across generations
   - Measure fitness progression and conservation trends

2. **Figure_eight Validation**
   - Run validation with 1.5% threshold
   - Verify chaotic system handling
   - Tune threshold if needed

3. **Benchmark Suite Execution**
   - Run benchmark suite with current baselines
   - Establish performance baseline for comparison
   - Document scaling characteristics

### Medium Priority

4. **Documentation Updates**
   - Update README.md with per-problem threshold feature
   - Add usage examples for configuration
   - Document validation results

5. **Test Coverage Expansion**
   - Add integration tests for full evolution pipeline
   - Add performance regression tests
   - Expand edge case coverage

6. **Threshold Tuning**
   - Collect empirical data from multiple runs
   - Refine thresholds based on actual model performance
   - Document tuning methodology

### Future Exploration

7. **Advanced Mutation Strategies**
   - Temperature annealing refinement
   - Adaptive mutation based on population diversity
   - Novelty search to prevent premature convergence

8. **Alternative Fitness Formulations**
   - Pareto optimization (speed vs conservation)
   - Dynamic penalty weights based on generation
   - Hybrid scoring methods

9. **Model Complexity Analysis**
   - Analyze correlation between code length and performance
   - Investigate optimal token budgets
   - Study code pattern evolution

10. **Production Deployment**
    - Migrate to paid tier for higher rate limits
    - Implement distributed evolution
    - Add result persistence and resumption

## Research Questions

1. **Conservation vs Speed Trade-off**: Can we find models that are both fast AND conserve energy?
2. **Scaling**: How do evolved models perform as N increases beyond training set?
3. **Generalization**: Do models evolved on one problem transfer to others?
4. **LLM Guidance**: Can we improve prompt engineering to bias toward physics-aware code?
5. **Population Diversity**: What strategies maintain genetic diversity in later generations?

## Dependencies and Environment

**Core Dependencies**:
- Python 3.11+
- google-generativeai (Gemini API)
- numpy (numerical computation)
- pydantic (configuration validation)
- pytest (testing)
- uv (package management)

**Optional**:
- scipy (KDTree baseline)
- python-dotenv (environment variables)

**Environment Variables**:
- `GEMINI_API_KEY`: Required for LLM code generation

## CI/CD Status

**GitHub Actions**: Not yet configured
**Pre-commit Hooks**: Not yet configured
**Code Quality Tools**: Ruff (linting), mypy (type checking)

## Contributing

This is a research project exploring LLM-driven algorithm discovery. Key principles:

- **Test-Driven Development**: Write tests before implementation
- **Configuration-Driven**: All parameters in config.yaml
- **Validation-First**: Empirical validation before committing to long runs
- **Branch Workflow**: Feature branches → PR → Merge to main

## Contact and Resources

**Repository**: https://github.com/TheIllusionOfLife/Galaxy
**Documentation**: See README.md
**Analysis Archives**: archive/analysis_docs/

---

*This document is updated regularly as the project evolves. Last major milestone: Per-problem physics thresholds (PR #53, merged 2025-11-03).*
