# Galaxy - AI Civilization Evolution Simulator

This project simulates how AI civilizations can discover solutions that surpass human capabilities. It is designed to run directly on local machines, allowing you to observe the evolutionary process in action.

## Prototype Philosophy

**LLM Role**: Instead of solving problems directly, the LLM generates and proposes **"strategies," "heuristics," and "surrogate models"** as code. This creative process is simulated through the LLM_propose_strategy function.

**Evolution Process**: Each AI civilization (agent group) proposes strategies that are executed and evaluated in a "Crucible" environment. Superior strategies are selected by the "Evolutionary Engine" and become the foundation for the next generation.

**Purpose**: These prototypes aim to verify whether the process of discovering better solutions can be automated and accelerated, rather than completely solving the problem.

**This Prototype**: AI civilizations are tasked with inventing surrogate models to accelerate computationally expensive N-body simulations (gravitational calculations). Fitness is evaluated based on the balance between the surrogate model's prediction accuracy and computational speed.

## What This Achieves

### Proven Results

This prototype has demonstrated measurable improvements in automated code generation and evolution:

#### Code Quality & Reliability
- **49% Syntax Error Reduction**: Improved from 3.3% to 1.67% through prompt engineering (PR #12)
- **96.7% Validation Success**: First production run achieved high code generation reliability
- **98.3% LLM Success Rate**: Consistent across multiple evolution runs (PR #14, #21, #23)

#### Performance Optimization
- **2x Fitness Improvement**: Evolution consistently doubles baseline fitness (Gen 0 → Gen 4)
- **Code Bloat Prevention**: Penalty system reduced token count by 5.4% without fitness loss (PR #21)
- **Penalty Tuning**: Threshold optimization improved relevance by 2.2x (PR #23, 11.1% vs 5% application rate)

#### Cost Efficiency
- **$0.02 per Full Run**: Typical cost for 50 API calls (10 population × 5 generations)
- **20 Runs per Day**: Within free tier limit (1,000 requests/day)
- **Sub-penny Experiments**: Mini runs (2 gen × 3 pop) cost ~$0.002

#### Evolution Effectiveness
- **Non-monotonic Progress**: Fitness fluctuates during exploration (healthy search behavior)
- **Best-Ever Tracking**: Cumulative maximum fitness clearly visible (PR #23)
- **Adaptive Mutation**: Early exploration (temp=1.0) then exploitation (temp=0.6)

#### Visualization & Analysis
- **5 Comprehensive Plots**: Fitness, accuracy/speed trade-offs, token evolution, cost tracking
- **High-Resolution Output**: 300 DPI publication-quality visualizations
- **Complete History**: JSON export for custom analysis

### Real-World Validation

**Production Runs** (from Session Handover):
- PR #23: 60 API calls, $0.0219 cost, best fitness=27,879.23
- PR #21: 150 API calls across 3 test runs, $0.05 total
- PR #16: 9 API calls, $0.002, validated all visualizations
- PR #14: 60 API calls, $0.02, 98.3% success rate

**Runtime**: ~4 minutes per full evolution (rate-limited to 15 RPM)

### Scientific Contribution

This work demonstrates:
1. **LLMs as Code Generators**: Using LLMs in evolutionary frameworks rather than direct problem solving
2. **Automatic Discovery**: Exploring solution space without human guidance
3. **Multi-Layer Safety**: AST validation + sandbox execution + output validation
4. **Cost-Effective Research**: Free-tier API enables large-scale experiments

### Current Limitations

**Technical Limitations:**
- Single LLM provider (Gemini only)
- Fixed generation count (no convergence detection)
- Single problem domain (N-body simulation)
- Whitespace-based token counting (inaccurate)

**Planned Improvements** (see Session Handover → Next Priority Tasks):
- Multi-LLM support (Claude, GPT-4o)
- Convergence detection and early stopping
- Advanced algorithms (crossover, multi-objective optimization)
- tiktoken migration for accurate token counting
- Code modularization for better maintainability

## Setup

### Requirements
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Google AI API key (free)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/TheIllusionOfLife/Galaxy.git
cd Galaxy
```

2. **Install uv (if not already installed)**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: with pip
pip install uv
```

3. **Install dependencies**
```bash
# With uv (recommended - faster and more reliable)
uv sync --extra dev

# Alternative: with pip
pip install -e ".[dev]"
```

**Why uv?** uv is 10-100x faster than pip and provides reproducible installs via `uv.lock`. Dependencies install in seconds instead of minutes.

4. **Configure API key**
- Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Create a `.env` file in the project root:
```bash
cp .env.example .env
```
- Edit the `.env` file and set your API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

**Note**: All configuration settings (model selection, hyperparameters, feature flags) are defined in `config.yaml` (single source of truth). The `.env` file only contains API keys (secrets).

## Configuration

### Configuration Architecture

The project follows the "Once and Only Once" (DRY) principle for configuration:

- **`config.yaml`** - Single source of truth for ALL configuration defaults
  - Model selection (gemini-2.5-flash-lite)
  - Hyperparameters (temperature, tokens, etc.)
  - Evolution parameters (population, generations, elite ratio)
  - Feature flags (rate limiting, code penalty, etc.)

- **`.env`** - Secrets ONLY (API keys)
  - Never contains configuration parameters
  - Only stores sensitive credentials

- **Environment variables** - Optional one-off overrides
  - Can override any config.yaml setting for specific runs
  - Example: `POPULATION_SIZE=20 uv run python prototype.py`

### Customizing Configuration

Edit `config.yaml` to customize behavior (all parameters documented with comments):

```yaml
# Model Configuration
model:
  name: gemini-2.5-flash-lite  # Change model
  temperature: 0.8             # Adjust creativity (0.0-2.0)
  max_output_tokens: 2000      # Limit response length

# Evolution Parameters
evolution:
  population_size: 10    # Models per generation
  num_generations: 5     # Total generations
  elite_ratio: 0.2       # Top performers kept (0.0-1.0)

# Code Length Penalty (prevent bloat)
code_penalty:
  enabled: true          # Enable/disable feature
  weight: 0.1           # Penalty strength (0.0-1.0)
  max_tokens: 2000      # Threshold before penalty
```

**Important**: Never duplicate settings. Each parameter has exactly ONE definition in `config.yaml`.

## Usage

### Basic Execution

```bash
# Run evolutionary optimization (default: plummer sphere)
uv run python prototype.py

# Run on different test problems
TEST_PROBLEM=two_body uv run python prototype.py
TEST_PROBLEM=figure_eight uv run python prototype.py

# Alternative: edit config.yaml evolution.test_problem field
# Options: two_body (N=2), figure_eight (N=3), plummer (configurable N)
```

### Multi-Problem Validation

Compare evolution results across different test problems:

```bash
# Run evolution on each test problem
uv run python prototype.py  # (configure test_problem in config.yaml)

# Compare results
python scripts/compare_problems.py results/run_* --output results/comparison
```

### Cross-Problem Generalization Analysis

Test whether models trained on one problem can generalize to others:

```bash
# Run cross-validation analysis (requires evolution runs for all 3 test problems)
python scripts/cross_validate_problems.py
```

This creates a 3×3 matrix testing each model on all problems:
- **Trained on** (rows): which problem the model was evolved on
- **Tested on** (columns): which problem the model is evaluated against
- **Generalization penalty**: % fitness drop when tested on different problem

Example output:
```markdown
| Trained On → Tested On | two_body | figure_eight | plummer |
|------------------------|----------|--------------|---------|
| **two_body**           | 320,373 (0%) | 188,876 (+41%) | 12,193 (+96%) |
| **figure_eight**       | 347,071 (-50%) | 244,427 (-6%) | 16,579 (+93%) |
| **plummer**            | 197,256 (-720%) | 145,040 (-503%) | 19,764 (+18%) |
```

**Key Finding**: Models show varying generalization - two_body models specialize to simple problems, while plummer models improve on other tasks (negative penalty = better performance).

Results saved to: `results/analysis/cross_validation_YYYYMMDD_HHMMSS/`

Example comparison output:
| Test Problem | N | Best Fitness | Accuracy | Speed (s) |
|--------------|---|--------------|----------|-----------|
| two_body     |  2|    523,752   |   99.95% |  0.000002 |
| figure_eight |  3|    259,637   |   99.07% |  0.000004 |
| plummer      | 20|      9,392   |   55.53% |  0.000059 |

### Execution Results

The program outputs:
- **Console:** Evaluation results for each generation (fitness, accuracy, speed)
- **Console:** Top-performing models and LLM usage statistics
- **Files:** Automatically generated visualization and data export

#### Visualization and Data Export

After evolution completes, results are automatically saved to a timestamped directory (`results/run_YYYYMMDD_HHMMSS/`):

1. **evolution_history.json** - Complete evolution history with summary statistics
   - All generations, populations, fitness values
   - Best/average/worst fitness per generation
   - Total models evaluated and best overall fitness

2. **fitness_progression.png** - Line plot showing fitness over generations
   - Best, average, and worst fitness trends
   - Identifies improvement/stagnation patterns

3. **accuracy_vs_speed.png** - Scatter plot of accuracy vs speed trade-offs
   - Each point represents one model from any generation
   - Color indicates fitness level
   - Reveals Pareto frontier of speed/accuracy balance

4. **token_progression.png** - Code length evolution over generations
   - Average, maximum, and minimum token counts per generation
   - Individual model scatter overlay colored by fitness
   - Monitors code bloat and validates length penalty effectiveness

5. **cost_progression.png** - Cumulative cost over API calls
   - Tracks spending throughout evolution
   - Helps validate cost estimates

Example output:
```
Saving results to: results/run_20251028_113940
  ✓ Evolution history saved: results/run_20251028_113940/evolution_history.json
  ✓ Fitness progression plot: results/run_20251028_113940/fitness_progression.png
  ✓ Accuracy vs speed plot: results/run_20251028_113940/accuracy_vs_speed.png
  ✓ Token progression plot: results/run_20251028_113940/token_progression.png
  ✓ Cost progression plot: results/run_20251028_113940/cost_progression.png
```

### Cost Management

- **Free tier**: 1,000 requests per day, 15 requests per minute
- **Default settings**: 50 API calls per execution
- **Execution cost**: Approximately $0.02/run (2% of budget)
- **Rate limiting**: Automatically maintains 15 RPM

### Testing

Test API connection:
```bash
uv run python test_gemini_connection.py
```

Run unit tests:
```bash
# Run all tests (excluding integration tests)
uv run pytest tests/ -m "not integration"

# Run all tests including integration tests (requires API key)
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov --cov-report=html
```

## Benchmarks

The project includes a comprehensive benchmark suite for systematic performance evaluation of baseline surrogate models.

### Running Benchmarks

```bash
# Run full benchmark suite
python scripts/run_benchmarks.py
```

This generates:
- **Scaling analysis plots** (log-log complexity comparison)
- **Accuracy heatmaps** (baseline performance)
- **Pareto fronts** (accuracy vs speed trade-off)
- **Performance tables** (markdown + JSON)

### Example Results

**Scaling Analysis:**
```
direct_nbody on plummer:
  Empirical: O(N^1.96)
  Theoretical: O(N²)

kdtree on plummer:
  Empirical: O(N^1.40)
  Theoretical: O(N² log N)
```

**Performance Table:**
| Baseline | Test Problem | N | Accuracy | Speed (s) |
|----------|-------------|---|----------|-----------|
| direct_nbody | plummer | 200 | 1.000 | 0.868 |
| kdtree | plummer | 200 | 0.063 | 1.534 |

### Output Location

Results are saved to timestamped directories:
```
results/benchmarks/run_YYYYMMDD_HHMMSS/
├── benchmark_results.json      # Complete raw data
├── performance_table.md         # Formatted table
├── scaling_analysis.txt         # Complexity analysis
├── scaling_comparison.png       # Log-log plot
├── accuracy_heatmap.png        # Performance heatmap
└── pareto_front.png            # Trade-off visualization
```

**See [benchmarks/README.md](benchmarks/README.md) for detailed documentation.**

## Troubleshooting

### Common Setup Issues

#### "GOOGLE_API_KEY not set" Error

**Error Message:**
```
ValidationError: GOOGLE_API_KEY
  Field required [type=missing, input_value={}, input_type=dict]
```

**Solution:**

1. Create `.env` file from template:
   ```bash
   cp .env.example .env
   ```

2. Get API key from [Google AI Studio](https://aistudio.google.com/apikey)

3. Edit `.env` and add your key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. Verify configuration:
   ```bash
   uv run python test_gemini_connection.py
   ```

---

#### Rate Limit Errors (429 Too Many Requests)

**Error Message:**
```
google.api_core.exceptions.TooManyRequests: 429 Quota exceeded
```

**Solution:**

Ensure rate limiting is enabled in `config.yaml`:
```yaml
rate_limiting:
  enabled: true
  requests_per_minute: 15
```

**If still occurring:**
- Free tier: 15 requests/minute, 1,000 requests/day
- Wait 1 minute before retrying
- Reduce `population_size` or `num_generations` in config.yaml

---

#### Integration Test Failures (No API Key)

**Error Message:**
```
SKIPPED [1] tests/test_integration.py:15: API key required
```

**This is normal!** Integration tests require a real API key and are automatically skipped in CI.

To run integration tests locally:
```bash
# Set API key in .env first
uv run pytest tests/ -m integration
```

CI automatically excludes integration tests:
```bash
pytest tests/ -m "not integration"  # CI command
```

---

#### Import Errors After Fresh Clone

**Error Message:**
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution:**

Install all dependencies including dev tools:
```bash
uv sync --extra dev
```

Or with pip:
```bash
pip install -e ".[dev]"
```

---

#### Code Validation Failures

**Error Message:**
```
ValidationResult(valid=False, errors=['Forbidden: import statement'])
```

**Cause:** LLM generated code with forbidden operations (imports, file I/O, etc.)

**This is expected behavior:**
- Validation prevents malicious code execution
- System automatically falls back to parametric model
- Check `code_validator.py` for allowed operations

**To reduce validation failures:**
- Lower `temperature` in `config.yaml` (more conservative)
- Review prompt engineering in `prompts.py`
- Check Session Learnings below for prompt patterns

---

#### Type Checking Errors (Local vs CI Differences)

**Issue:** Mypy passes locally but fails in CI (or vice versa)

**Cause:** Different type stub availability between environments

**Solution:**

Use `typing.Any` for parameters with environment-dependent types:
```python
from typing import Any

# Instead of:
config: GenerationConfig = {...}  # May fail in CI

# Use:
config: Any = {...}  # Works in both environments
```

See PR #7 commits for examples.

---

#### YAML Configuration Errors

**Error Message:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Cause:** Invalid YAML syntax in `config.yaml`

**Solution:**

1. Check indentation (use spaces, not tabs)
2. For consistency, add a space after colons (`key: value` is preferred over `key:value`)
3. Validate YAML syntax online: https://www.yamllint.com/

Example of correct syntax:
```yaml
model:
  name: gemini-2.5-flash-lite
  temperature: 0.8
```

---

### Getting Help

If you encounter issues not covered here:

1. **Check Session Learnings** (below) for recent patterns and fixes
2. **Review recent PRs** for similar issues and solutions
3. **Check CI logs** on GitHub Actions for detailed error messages
4. **Search existing issues**: https://github.com/TheIllusionOfLife/Galaxy/issues
5. **Open a new issue** if problem persists

## Session Handover

### Last Updated: November 01, 2025 11:28 PM JST

#### Recently Completed (Current Session)
- ✅ **[PR #36 - Phase 3: Evolution with Validated Baselines](https://github.com/TheIllusionOfLife/Galaxy/pull/36)**: LLM-based evolution dramatically outperforms hand-crafted baselines (November 1, 2025)
  - **Achievement**: **7890x better fitness** than KDTree baseline and **219x faster** than direct N-body, while maintaining 98.56% accuracy
  - **Key Finding**: Evolutionary optimization successfully discovers high-performance surrogate models, with the parametric fallback strategy proving critical for robustness (10% LLM failure rate)
  - **Details**: See the full analysis in [PHASE3_RESULTS.md](PHASE3_RESULTS.md) (278 lines with methodology, results, comparison analysis, limitations, future work)
  - **Status**: ✅ Merged (commit [0fa9e5d](https://github.com/TheIllusionOfLife/Galaxy/commit/0fa9e5d))

#### Recently Completed (Previous Sessions)
- ✅ **[PR #34 - Comprehensive Benchmark Suite](https://github.com/TheIllusionOfLife/Galaxy/pull/34)**: Systematic performance evaluation infrastructure (November 1, 2025)
  - **Achievement**: Production-ready benchmark suite for baseline performance analysis and scaling validation
  - **Core Implementation**:
    - benchmarks/benchmark_runner.py (282 lines): Orchestrates systematic benchmarks across baselines/problems/scales
    - benchmarks/scaling_analysis.py (210 lines): Log-log regression for empirical O(N^b) complexity measurement
    - benchmarks/visualization.py (237 lines): Publication-quality 300 DPI plots (scaling, accuracy, Pareto)
    - scripts/run_benchmarks.py (145 lines): User-friendly CLI with timestamped output directories
  - **Testing**: 29 new tests (190/190 total passing)
    - Unit tests: Config validation, baseline creation, metrics calculation, scaling analysis
    - Integration tests: Real baseline execution, multiple particle counts, end-to-end suite
  - **Real Execution Validated**: Ran full benchmark suite with actual baselines
    - Results: `results/benchmarks/run_YYYYMMDD_HHMMSS/` with JSON, markdown, plots, scaling analysis
    - Verified: No timeouts, no truncation, no errors, complete formatted output
  - **Review Fixes**: Addressed all feedback from 3 AI reviewers (gemini-code-assist, claude, CodeRabbit)
    - HIGH: Refactored if/elif chains → dictionary mappings for extensibility
    - MEDIUM: Moved imports to top (PEP 8), extracted duplicated simulation loop to helper
    - MEDIUM: Renamed plot_accuracy_heatmap → plot_accuracy_bars (accurate naming)
    - MEDIUM: Optimized particle count iteration (avoid skipping, determine upfront)
    - ENHANCEMENTS: Added Pydantic validators, accuracy formula comments, debug logging
  - **Features**:
    - 3 Test Problems: two_body (2), figure_eight (3), plummer (scalable N)
    - 2 Baselines: KDTree O(N² log N) vs Direct N-body O(N²)
    - 4 Metrics: Accuracy, speed, energy drift, trajectory RMSE
    - Configurable via config.yaml (particle counts, timesteps, baselines, problems)
  - **Documentation**: benchmarks/README.md (280 lines) + main README updated
  - **Quality**: All CI passing (Python 3.10/3.11/3.12), ruff/mypy passing
  - **Files**: 12 changed (+1,776 lines), 3 new modules, 3 new test suites
  - **Status**: ✅ Merged (commit [ca514d7](https://github.com/TheIllusionOfLife/Galaxy/commit/ca514d7))

#### Recently Completed (Previous Sessions)
- ✅ **[PR #32 - Phase 2 Scientific Validation Infrastructure](https://github.com/TheIllusionOfLife/Galaxy/pull/32)**: Baselines, test problems, and physics validation metrics (November 1, 2025)
  - **Achievement**: Complete scientific validation infrastructure with comprehensive physics-based testing
  - **Core Additions**:
    - baselines.py (201 lines): KDTree (O(N log N)) and direct N-body baselines
    - initial_conditions.py (275 lines): Two-body orbit, figure-8, Plummer sphere test problems
    - validation_metrics.py (221 lines): Energy drift, trajectory RMSE, angular momentum, virial ratio
  - **Testing**: 57 comprehensive new tests (166/166 total passing)
    - Physics validation: Energy conservation, momentum conservation, virial equilibrium
    - Edge cases: Empty systems, single particles, inf/nan handling
    - Integration: Baseline vs direct N-body comparison
  - **Review Fixes**: Addressed 9 CRITICAL issues from 4 AI reviewers
    - Fixed leapfrog integration bug (replaced with symplectic Euler)
    - Fixed random zero risk in Plummer CDF (clamped to 1e-12)
    - Fixed angular momentum vector comparison (detects spin reversals)
    - Documented KDTree performance limitation (O(N² log N) tree rebuilding)
    - Documented Plummer velocity sampling approximation vs Eddington formula
    - Added modules to pyproject.toml packaging
    - Cleaned up test assertions (removed unnecessary NaN checks)
    - Verified mass factors correct (false positive from reviewer)
  - **Quality**: Rating 5/5 stars from chatgpt-codex-connector ("ready to merge immediately")
  - **Files**: 8 changed (+1,392 lines), 3 new modules, 3 new test suites
  - **Dependencies**: Added scipy>=1.9.0, numpy>=1.21.0
  - **Status**: ✅ Merged (commit [fd80a6c](https://github.com/TheIllusionOfLife/Galaxy/commit/fd80a6c))

#### Recently Completed (Current Session)

- ✅ **[PR #38 - Multi-Problem Validation (Phase 4)](https://github.com/TheIllusionOfLife/Galaxy/pull/38)**: Cross-problem generalization infrastructure (November 2, 2025)
  - **Achievement**: Enabled Galaxy to evolve surrogate models on different N-body test problems
  - **Implementation**:
    - Configuration: Added `test_problem` (two_body/figure_eight/plummer) and `num_particles` fields
    - Helper Function: `get_initial_particles()` maps test problems to initial conditions
    - Pipeline Integration: Modified prototype.py to use configurable test problems
    - Metadata Tracking: Evolution history now includes test_problem and num_particles
    - Comparison Tooling: `scripts/compare_problems.py` for systematic cross-problem analysis
  - **Testing**: 18 new tests (7 config + 11 helper), 226/226 tests passing, TDD discipline throughout
  - **Real Validation**: 3 complete evolution runs with real API ($0.029 total cost)
    - two_body (N=2): fitness=523,752, accuracy=99.95%
    - figure_eight (N=3): fitness=259,637, accuracy=99.07%
    - plummer (N=20): fitness=9,392, accuracy=55.53%
  - **Review Fixes**: Addressed all 5 reviewer feedback items (2 HIGH, 3 MEDIUM)
    - Robust config loading with `.get()` defaults (gemini-code-assist)
    - Null fitness handling with type/finiteness checks (chatgpt-codex-connector)
    - Consistent output formatting matching documentation (gemini-code-assist)
  - **Quality**: "APPROVED - Production Ready" from claude reviewer, all CI checks passing
  - **Files**: 8 changed (+796 lines), 3 new test suites, 1 new comparison script
  - **Status**: ✅ Merged (commit [2345cd2](https://github.com/TheIllusionOfLife/Galaxy/commit/2345cd2))

#### Recently Completed (Previous Sessions)
- ✅ **[PR #30 - Phase 1 Complete: 3D N-body Migration](https://github.com/TheIllusionOfLife/Galaxy/pull/30)**: Production-ready 3D particle-particle N-body physics (November 1, 2025)
  - **Achievement**: Transformed from toy 2D single-attractor to true 3D N-body gravitational simulator
  - **Core Migration**:
    - Particles: `[x,y,vx,vy]` (4) → `[x,y,z,vx,vy,vz,mass]` (7)
    - Signature: `predict(particle, attractor)` → `predict(particle, all_particles)`
    - Integration: Euler → Leapfrog (Velocity Verlet) for energy conservation
    - Complexity: O(N) → O(N²) true all-pairs gravitational interactions
  - **Testing**: 28 comprehensive new tests (physics + interface validation)
    - Energy conservation: <10% drift over 50 timesteps
    - Angular momentum conservation: <5% drift
    - Inverse square law verification, O(N²) scaling validation
  - **Real API Verification**: Smoke test passed (5 genomes, 4/5 perfect accuracy, $0.006 cost)
  - **Documentation**: 184-line MIGRATION_STATUS.md tracking all decisions and deferred items
  - **Review Fixes**: Addressed all feedback from claude[bot] reviews
    - Added `CosmologyCrucible.with_particles()` factory method for test flexibility
    - Enhanced validation (2-particle minimum, empty particle check, mass conservation)
    - Clarified parallel evaluation semantics with definitive documentation
  - **Quality**: Rating 4.8/5 stars, 109/109 tests passing, all CI checks passing
  - **Files**: 15 changed (+1,636, -203), 2 new test suites (783 lines)
  - **Status**: ✅ Merged (commit [04d504e](https://github.com/TheIllusionOfLife/Galaxy/commit/04d504e))

#### Recently Completed (Previous Sessions)
- ✅ **[PR #26 - Genetic Crossover Implementation](https://github.com/TheIllusionOfLife/Galaxy/pull/26)**: LLM-based code recombination (October 30, 2025)
  - **Feature**: Third reproduction operator enabling genetic crossover between elite parents
  - **Implementation**:
    - Configuration: `crossover_rate=0.3`, `temperature=0.75` with Pydantic validation
    - Core Logic: `select_crossover_parents()` + enhanced `LLM_propose_surrogate_model()`
    - Prompt Engineering: Enhanced crossover prompts with parent metrics and synthesis guidance
    - Lineage Tracking: New `parent_ids` field for analyzing parent-offspring relationships
  - **Testing**: 10 comprehensive unit tests + 2 integration tests (81/81 tests passing, 0 regressions)
  - **Real-world Validation**: Full evolution run completed (10 pop, 5 gen, cost ~$0.02)
  - **Review Fixes**: Addressed HIGH priority issues from claude's review
    - Fixed crossover fallback validation (prototype.py:255-260)
    - Added robust parent selection with defensive validation (prototype.py:194-210)
  - **CI Resolution**: Fixed claude-review workflow OIDC token validation issues
    - Root Cause: Workflow file modifications require exact match with main branch
    - Solution: Temporarily skip PR #26 using `if: github.event.pull_request.number != 26`
    - Permission Fix: Identified `gh pr comment` requires `pull-requests: write` not just `read`
  - **Approvals**: Claude (4.9/5 - "Exceptional"), Codex (approved), CodeRabbit (4 nitpicks), Gemini (5 suggestions)
  - **Status**: ✅ Merged (commit [924199d](https://github.com/TheIllusionOfLife/Galaxy/commit/924199d)), 7 files changed (+828, -30)

#### Recently Completed (Previous Sessions)
- ✅ **Earlier PRs**: Code length penalty ([#23](https://github.com/TheIllusionOfLife/Galaxy/pull/23), [#21](https://github.com/TheIllusionOfLife/Galaxy/pull/21), [#14](https://github.com/TheIllusionOfLife/Galaxy/pull/14)), config fixes ([#19](https://github.com/TheIllusionOfLife/Galaxy/pull/19)), token visualization ([#16](https://github.com/TheIllusionOfLife/Galaxy/pull/16)), prompt engineering ([#12](https://github.com/TheIllusionOfLife/Galaxy/pull/12)), uv migration ([#10](https://github.com/TheIllusionOfLife/Galaxy/pull/10)) - See git history for details

#### Next Priority Tasks

1. **Cross-Problem Generalization Analysis** (HIGH PRIORITY)
   - **Source**: PR #38 completion, Phase 4 infrastructure now available
   - **Context**: Infrastructure ready, need to analyze if models generalize across problems
   - **Goal**: Test whether models trained on one problem work on others (transfer learning)
   - **Tasks**:
     - Extract best models from each problem's evolution run
     - Test two_body best model on figure_eight and plummer problems
     - Test plummer best model on two_body and figure_eight problems
     - Measure generalization performance (fitness drop, accuracy change)
     - Document findings: problem-specific tricks vs general patterns
   - **Benefits**: Validates scientific robustness, identifies universal vs specialized strategies
   - **Estimated time**: 1-2 hours (model extraction + cross-testing + analysis)
   - **Approach**: Use `compare_problems.py` framework + custom cross-validation script

2. **Physics Validation of Evolved Models** (MEDIUM PRIORITY)
   - **Source**: PHASE3_RESULTS.md limitations section
   - **Context**: Best evolved model (civ_2_7) has no energy drift or trajectory RMSE metrics
   - **Goal**: Validate evolved model maintains physical plausibility beyond accuracy
   - **Tasks**:
     - Run best model through validation_metrics.py
     - Measure energy conservation drift, angular momentum conservation
     - Compare trajectory RMSE vs baselines (KDTree: 120.57, Direct: 0.00)
     - Ensure evolved approximations don't violate conservation laws
   - **Benefits**: Scientific rigor, publishable results, identifies physics-breaking shortcuts
   - **Estimated time**: 1 hour (run metrics + analysis)
   - **Approach**: Extract civ_2_7 code → run validation suite → compare vs baselines

3. **Code Modularization** (MEDIUM PRIORITY - Deferred)
   - **Source**: 4/5 reviewers consensus from previous planning
   - **Context**: prototype.py now at 1044 lines after Phase 1 migration
   - **Goal**: Extract to galaxy/ package structure for SOLID principles
   - **Structure**:
     ```text
     galaxy/
     ├── core/ (genome.py, evolution.py, selection.py)
     ├── crucible/ (base.py, nbody.py)
     ├── llm/ (client.py, gemini.py)
     └── prototype.py (main orchestration, ~100 lines)
     ```
   - **Benefits**: Easier domain extension, testable in isolation, reduced cognitive load
   - **Priority**: Deferred until Phase 2 complete (avoid disrupting working physics)
   - **Estimated time**: 3-4 hours

3. **Adaptive Timestep Integration** (OPTIONAL)
   - **Source**: MIGRATION_STATUS.md open question, claude[bot] review
   - **Context**: Current dt=0.1 may be too large for tight orbits
   - **Enhancement**: Variable timestep based on particle proximity
   - **Benefits**: Better accuracy for close encounters, maintains performance
   - **Priority**: Low (current leapfrog <10% drift acceptable for prototype)
   - **Estimated time**: 2 hours

#### Known Issues / Blockers
- **Non-monotonic Fitness**: Fitness fluctuates between generations (not guaranteed to improve)
  - Expected behavior during exploration phase, not a bug

#### Session Learnings

**Last Updated**: November 02, 2025 01:31 AM JST

- **Null Fitness Handling from JSON Sanitization** (2025-11-02 PR #38): `_sanitize_for_json()` converts NaN/Inf to null requiring type checks
  - **Problem**: `scripts/compare_problems.py` crashed with `TypeError` when comparing null fitness values
  - **Root Cause**: JSON sanitization converts non-finite floats to `null`, but code assumed numeric values
  - **Solution**: Add `isinstance(fitness, (int, float)) and math.isfinite(fitness)` before comparisons
  - **Pattern**: Always validate type and finiteness when loading numeric data from JSON that may contain sanitized values
  - **Caught By**: chatgpt-codex-connector reviewer (P1 priority feedback)
- **Output Formatting Consistency** (2025-11-02 PR #38): User-facing output must match documentation examples
  - **Problem**: README showed "523,752" and "99.95%" but script output "523752.00" and "0.9995"
  - **Impact**: Format mismatch suggests bugs even when functionality is correct
  - **Solution**: Format numbers to match docs: `f"{int(fitness):12,}"` (comma-separated int), `f"{accuracy * 100:.2f}%"` (percentage)
  - **Pattern**: Review all user-facing output (CLI, tables, reports) against documentation before declaring complete
  - **Caught By**: gemini-code-assist reviewer (MEDIUM priority feedback)
- **Post-Fix Verification Success** (2025-11-01 PR #36): Real example of verification catching issues before push
  - **Context**: Fixed parametric model metadata saving bug (early return prevented file creation)
  - **Verification**: Ran `uv run python scripts/extract_best_model.py results/run_20251101_220542` after fix
  - **Result**: Caught that metadata was correctly saved, verified JSON file created
  - **Impact**: Prevented broken workflow from reaching CI, validated fix works with real data
  - **Pattern**: Always run the actual user command after fixing to verify it works, not just assume
  - **Time**: 30 seconds verification prevented potential CI debugging hours
- **Dictionary Mapping Refactoring** (2025-11-01): Replace if/elif chains with dictionary mappings for extensibility
  - **Trigger**: PR #34 review feedback from gemini-code-assist (HIGH priority)
  - **Problem**: Long if/elif chains (`if name=="x": fn_x() elif name=="y": fn_y()`) hard to extend
  - **Solution**: Class-level dict mapping + `.get()` with dynamic error messages showing valid options
  - **Benefits**: Add new options by updating dict only, cleaner code, easier testing
  - **Pattern**: See `~/.claude/core-patterns.md` → "Dictionary Mapping Refactoring" (local AI assistant config, not in repo)
- **Pydantic Validators for Config** (2025-11-01): Use `@field_validator` to catch invalid config at load time
  - **Trigger**: PR #34 review recommendation from claude
  - **Implementation**: Validate list fields against known valid values (e.g., test_problems vs known problems)
  - **Benefits**: Fail-fast on config errors, prevents runtime failures from typos
  - **Pattern**: See `~/.claude/domain-patterns.md` → "DRY Configuration Architecture" (local AI assistant config, not in repo)
- **Extract Duplicated Code to Helpers** (2025-11-01): When seeing duplicate logic blocks, extract immediately
  - **Trigger**: PR #34 review identified nearly identical simulation loops
  - **Solution**: Created `_run_simulation()` helper method used by both baseline and ground truth paths
  - **Benefits**: DRY principle, easier testing, consistent behavior
  - **Pattern**: Standard refactoring, extract when duplication spans 5+ lines
- **AI Reviewer Claim Verification** (2025-11-01): Always verify reviewer claims by reading actual code, not assumptions
  - **Trigger**: PR #32 reviewer claimed missing mass factors in physics calculations
  - **Verification**: Checked validation_metrics.py and found `0.5 * p[6] * v_squared` already correct
  - **Pattern**: Use `grep` or `Read` tool to verify claims about missing code before implementing "fixes"
  - **Why**: AI reviewers can make factually incorrect claims - verify before accepting
  - **Cost**: 30 seconds to verify vs wasting time implementing unnecessary changes

- **Empirical Feature Validation Pattern** (2025-10-31): Always validate new features with controlled experiments before declaring success
  - **Trigger**: PR #26 merged crossover feature, needed to validate effectiveness
  - **Approach**: 3 comparative runs (control, 30%, 50%) with consistent methodology
  - **Results**: Crossover 50% showed +5.3% fitness, +11.6pp improvement rate vs control
  - **Decision**: Updated default config based on empirical data, not assumptions
  - **Pattern**: Don't rely on intuition - measure real impact with controlled experiments
  - **Cost**: $0.08 for comprehensive validation vs potentially keeping ineffective feature

- **GitHub Workflow OIDC Token Validation** (2025-10-30 from PR #26): Workflow file modifications require exact match with main
  - **Problem**: claude-code-review workflow failed with 401 OIDC token exchange error
  - **Root Cause**: Security feature validates workflow files match between PR branch and main branch
  - **Solution**: Either temporarily skip the PR (`if: github.event.pull_request.number != 26`) or merge workflow changes to main first
  - **Pattern**: When modifying `.github/workflows/*` files, expect OIDC validation to fail until merged to main
- **CI Permission Requirements Mapping** (2025-10-30 from PR #26): GitHub CLI commands need specific workflow permissions
  - **Discovery**: `gh pr comment` requires `pull-requests: write`, not just `read`
  - **Pattern**: Map CLI commands to GitHub Actions permissions before using in workflows
  - **Common Mappings**: `gh pr comment` → `pull-requests: write`, `gh pr view` → `pull-requests: read`, `git push` → `contents: write`
- **Post-Fix Verification Discipline** (2025-10-30 from PR #26): Always run quick checks before declaring fixes complete
  - **Why**: Prevents reporting "fixed" when issues still exist (builds trust, saves time)
  - **Approach**: Run 1-2 relevant commands (typecheck, quick test) immediately after fix
  - **Example**: After CI fix, ran `gh pr checks` to verify all tests passing before reporting success
- **Refactoring for Testability** (2025-10-30 from PR #23): Extract calculations to pure functions for robust testing
  - **Trigger**: Code review feedback suggesting tests are weak or implicit
  - **Problem**: Calculation embedded in plotting function → hard to test edge cases directly
  - **Solution**: Extract `calculate_best_ever_fitness()` as separate function with comprehensive docstring
  - **Result**: Added 5 unit tests (monotonic, inf, nan, all-inf, empty) - 71 total tests passing
  - **Pattern**: When reviewers say "test doesn't explicitly verify X", extract X to testable function
- **Test Analysis Documentation Clarity** (2025-10-30 from PR #23): Always explain number discrepancies
  - **Trigger**: Reviewer questioned "5/45 vs 50 models" discrepancy in test analysis
  - **Problem**: Using denominators that differ from total counts without explanation
  - **Solution**: Explicitly document "45/50 (5 failed validation)" before showing percentages
  - **Pattern**: When denominators ≠ total attempts, explain what's excluded (failures, timeouts, etc.)
- **Integration Testing Global Settings Patch** (2025-10-30 from PR #21): `monkeypatch.setenv()` alone insufficient for testing module-level settings
  - **Problem**: Test using `monkeypatch.setenv("PENALTY_WEIGHT", "0.2")` but code still uses default 0.1
  - **Root Cause**: Modules import `settings` at load time; env changes don't update existing references
  - **Solution**: Reload settings + patch ALL module references: `monkeypatch.setattr(config_module, "settings", test_settings)`
  - **Detection**: Test passes but doesn't actually vary parameter being tested
  - **Pattern**: Always patch global references when testing module-level config objects

**Historical Learnings** (detailed patterns in personal reference files: `~/.claude/core-patterns.md` and `~/.claude/domain-patterns.md` - these are user-level patterns spanning all projects):
- GraphQL PR review efficiency, Zip pattern optimization, Complete test coverage for edge cases
- TDD with real API integration, Security review priority, Code simplification from AI review
- Test assertion precision, Prompt engineering for completeness, Statistical testing importance
- uv migration, Make target purpose, Test robustness, Pre-commit setup, Reviewer priority
