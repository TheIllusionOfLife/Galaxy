# Benchmark Suite

Comprehensive benchmark suite for systematic performance evaluation of baseline surrogate models across multiple test problems and particle counts.

## Overview

This benchmark suite provides:
- **Systematic Evaluation**: Test baselines against standard N-body test problems
- **Scaling Analysis**: Measure empirical computational complexity (O(N²) vs O(N² log N))
- **Performance Comparison**: Compare accuracy, speed, and physics metrics
- **Publication-Quality Visualizations**: 300 DPI plots for papers and presentations

## Quick Start

```bash
# Run full benchmark suite
python scripts/run_benchmarks.py

# Results saved to:
results/benchmarks/run_YYYYMMDD_HHMMSS/
├── benchmark_results.json      # Complete raw data
├── performance_table.md         # Formatted markdown table
├── scaling_analysis.txt         # Empirical complexity analysis
├── scaling_comparison.png       # Log-log scaling plot
├── accuracy_heatmap.png        # Accuracy by baseline/problem
└── pareto_front.png            # Accuracy vs speed trade-off
```

## Configuration

Benchmarks are configured in `config.yaml`:

```yaml
benchmark:
  enabled: true
  particle_counts: [10, 50, 100, 200]  # Scaling analysis points
  num_timesteps: 100                    # Integration steps per run
  test_problems:                        # From initial_conditions.py
    - two_body      # Two-body circular orbit
    - figure_eight  # Three-body figure-8
    - plummer       # Plummer sphere (realistic cluster)
  baselines:                            # From baselines.py
    - kdtree        # KDTree O(N log N) approximation
    - direct_nbody  # Direct O(N²) ground truth
  kdtree_k_neighbors: 10               # K-NN parameter
```

## Test Problems

### Two-Body Circular Orbit
- **Purpose**: Energy and momentum conservation test
- **Particles**: 2 (fixed)
- **Physics**: Stable Kepler orbit
- **Expected**: High accuracy, perfect conservation

### Three-Body Figure-8
- **Purpose**: Numerical stability and precision test
- **Particles**: 3 (fixed)
- **Physics**: Chenciner-Montgomery choreography
- **Expected**: Challenging, requires high accuracy

### Plummer Sphere
- **Purpose**: Realistic N-body cluster dynamics
- **Particles**: Configurable (10-200)
- **Physics**: Virial equilibrium galaxy cluster
- **Expected**: Scaling analysis, realistic performance

## Baseline Models

### KDTree Baseline (`kdtree`)
- **Algorithm**: K-nearest neighbors using scipy.spatial.KDTree
- **Complexity**: O(N² log N) - O(log N) query per particle
- **Accuracy**: Approximate (depends on k_neighbors parameter)
- **Use Case**: Faster for large N, acceptable accuracy loss

### Direct N-body (`direct_nbody`)
- **Algorithm**: All-pairs gravitational calculation
- **Complexity**: O(N²)
- **Accuracy**: Perfect (1.0) - ground truth reference
- **Use Case**: Baseline for comparison, verifies correctness

## Metrics

### Accuracy
- **Formula**: `1.0 / (1.0 + sqrt(RMSE))`
- **Range**: 0.0-1.0 (higher is better)
- **Meaning**: Overall quality vs ground truth

### Speed
- **Unit**: Seconds for N timesteps
- **Range**: >0 (lower is better)
- **Meaning**: Computational cost

### Energy Drift
- **Formula**: `|E_final - E_initial| / |E_initial|`
- **Range**: ≥0 (lower is better)
- **Meaning**: Energy conservation quality

### Trajectory RMSE
- **Formula**: Root-mean-square position error vs ground truth
- **Range**: ≥0 (lower is better)
- **Meaning**: Spatial accuracy

## Programmatic Usage

```python
from config import Settings
from benchmarks import BenchmarkRunner
from benchmarks.visualization import plot_scaling_comparison
from benchmarks.scaling_analysis import measure_scaling, export_markdown_table

# Load configuration
settings = Settings.load_from_yaml()

# Run benchmarks
runner = BenchmarkRunner(settings)
results = runner.run_all_benchmarks()

# Analyze scaling
kdtree_scaling = measure_scaling(results, "kdtree", "plummer")
print(f"KDTree empirical complexity: O(N^{kdtree_scaling['exponent']:.2f})")

# Export results
table = export_markdown_table(results)
with open("results.md", "w") as f:
    f.write(table)

# Generate plots
plot_scaling_comparison(results, "scaling.png")
```

## Example Results

### Scaling Analysis

```
kdtree on plummer:
  Empirical: O(N^1.40)
  Theoretical: O(N² log N)
  Coefficient: 0.000813

direct_nbody on plummer:
  Empirical: O(N^1.96)
  Theoretical: O(N²)
  Coefficient: 0.000027
```

### Performance Table

| Baseline | Test Problem | N | Accuracy | Speed (s) | Energy Drift | RMSE |
|----------|-------------|---|----------|-----------|--------------|------|
| direct_nbody | plummer     | 200 |    1.000 |  0.867959 |     4.815284 | 0.000000 |
| kdtree       | plummer     | 200 |    0.063 |  1.534345 |     9.527677 | 222.643514 |

## Interpreting Results

### Good Baseline Characteristics
- ✅ Direct N-body: accuracy=1.0, RMSE=0.0 (perfect ground truth)
- ✅ Empirical complexity matches theoretical (±0.1 exponent)
- ✅ KDTree faster than direct N-body for large N
- ✅ Energy drift <10% for good integrators

### Red Flags
- ❌ Direct N-body accuracy <1.0 (implementation bug)
- ❌ Empirical complexity >> theoretical (performance issue)
- ❌ KDTree slower than direct N-body (inefficient implementation)
- ❌ Energy drift >50% (poor conservation)

## Troubleshooting

### Low KDTree Accuracy
**Cause**: k_neighbors too small or problem too complex
**Solution**: Increase `benchmark.kdtree_k_neighbors` in config.yaml

### Slow Execution
**Cause**: Too many timesteps or large particle counts
**Solution**: Reduce `benchmark.num_timesteps` or `particle_counts`

### Missing Plots
**Cause**: matplotlib display issues
**Solution**: Check output directory permissions

## Development

### Adding New Baselines

1. Implement baseline in `baselines.py`:
```python
def create_my_baseline() -> SurrogateGenome:
    \"\"\"My custom baseline model.\"\"\"
    # Implementation
    return SurrogateGenome(...)
```

2. Update `config.yaml`:
```yaml
benchmark:
  baselines:
    - kdtree
    - direct_nbody
    - my_baseline  # Add here
```

3. Update `BenchmarkRunner._get_baseline_model()` in `benchmark_runner.py`

### Adding New Test Problems

1. Implement in `initial_conditions.py`:
```python
def my_test_problem() -> list[list[float]]:
    \"\"\"My custom test problem.\"\"\"
    # Return particles
```

2. Update `config.yaml`:
```yaml
benchmark:
  test_problems:
    - two_body
    - my_problem  # Add here
```

3. Update `BenchmarkRunner._get_test_problem_particles()`

## See Also

- [baselines.py](../baselines.py) - Baseline surrogate implementations
- [initial_conditions.py](../initial_conditions.py) - Test problem generators
- [validation_metrics.py](../validation_metrics.py) - Physics-based metrics
- [MIGRATION_STATUS.md](../MIGRATION_STATUS.md) - Phase 2 infrastructure docs
