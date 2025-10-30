# Crossover Effectiveness Analysis - October 31, 2025

## Objective
Validate whether genetic crossover provides evolutionary benefit over mutation-only approach.

## Experimental Design
- **Control**: Crossover disabled (mutation only)
- **Test 1**: Crossover enabled at 30% rate
- **Test 2**: Crossover enabled at 50% rate

## Configuration
- Population size: 10
- Generations: 5
- Model: gemini-2.5-flash-lite
- Temperature: 0.8 (mutation), 0.75 (crossover)
- Elite ratio: 0.2

## Metrics
1. Fitness improvement rate: (Gen 5 best - Gen 0 best) / Gen 0 best
2. Convergence speed: Generation where fitness plateaus
3. Diversity: Average fitness variance per generation
4. LLM success rate: Valid generations / total attempts
5. Cost per run

## Results

### Summary Statistics

| Configuration | Gen 0 Best | Gen 4 Best | Best Overall | Improvement | Avg Variance |
|--------------|------------|------------|--------------|-------------|--------------|
| Control (Disabled) | 24,345.82 | 19,961.36 | 27,191.36 | +11.7% | 26,419,553 |
| Crossover 30% | 23,732.56 | 24,372.33 | 27,123.67 | +14.3% | 30,172,479 |
| Crossover 50% | 23,221.15 | 24,447.09 | **28,628.66** | **+23.3%** | 20,107,307 |

### Key Findings

1. **Best Overall Fitness**:
   - Control: 27,191.36
   - Crossover 30%: 27,123.67 (-0.2% vs control)
   - Crossover 50%: 28,628.66 (+5.3% vs control) ✅

2. **Improvement Rate (Gen 0 → Best)**:
   - Control: 11.7%
   - Crossover 30%: 14.3% (+2.6 percentage points)
   - Crossover 50%: 23.3% (+11.6 percentage points) ✅

3. **Population Diversity**:
   - Crossover 50% showed lowest variance (20M), suggesting faster convergence
   - Crossover 30% showed highest variance (30M), indicating broader exploration

### Observations

1. **Crossover at 50% is most effective**:
   - Achieved highest fitness (28,628.66)
   - Nearly doubled improvement rate (23.3% vs 11.7%)
   - Demonstrates clear benefit over mutation-only approach

2. **Crossover at 30% shows minimal impact**:
   - Slightly lower fitness than control
   - Modest improvement rate increase
   - May not provide sufficient selection pressure

3. **Success Rate**:
   - All runs: 100% LLM success rate (60/60 calls)
   - Validation failures: 2 in 50% run (handled gracefully with fallback)

### Cost Analysis

- Control run: $0.0255
- Crossover 30% run: $0.0252
- Crossover 50% run: $0.0259
- **Total experimental cost**: $0.0766

## Conclusion

**Recommendation**: **KEEP crossover feature, use 50% rate as default**

**Reasoning**:
- Crossover at 50% provides measurable fitness improvement (+5.3%)
- Improvement rate nearly doubled (+11.6 percentage points)
- Cost is comparable to mutation-only approach
- Feature is working as designed with graceful fallback

**Next Steps**:
1. Update default `crossover_rate` in config.yaml from 0.3 to 0.5
2. Keep crossover quality improvements from PR #26 reviews on backlog (low priority)
3. Document crossover effectiveness in README.md
