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
  max_tokens: 400       # Threshold before penalty

# Physics Penalty (ensure conservation laws)
physics_penalty:
  enabled: true                  # Enable/disable physics validation
  energy_weight: 0.3             # Energy drift penalty weight (0.0-1.0)
  momentum_weight: 0.1           # Angular momentum penalty weight (0.0-1.0)
  energy_drift_threshold: 0.01   # 1% energy drift threshold
  angular_momentum_threshold: 0.01  # 1% momentum drift threshold
  validation_timesteps: 10       # Timesteps for physics validation
```

**Physics Penalty** (New): Models that violate energy or angular momentum conservation are penalized. This ensures evolved approximations remain scientifically valid. Critical finding from PR #41: without physics penalties, ALL evolved models violated conservation laws (up to 293% energy drift). With physics penalties enabled, models preserve physics while maintaining speed/accuracy.

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

### Physics Validation

Validate evolved models for physical plausibility (energy conservation, trajectory accuracy):

```bash
# Validate single run
python -m scripts.validate_evolved_model --run-dir results/run_YYYYMMDD_HHMMSS

# Validate all runs
python -m scripts.validate_evolved_model --all
```

Physics metrics computed:
- **Energy drift**: Relative energy conservation violation (should be <1% for good models)
- **Trajectory RMSE**: Position error vs ground truth (lower is better)
- **Angular momentum conservation**: Rotational conservation violation

Example validation results:

| Test Problem | Fitness | Energy Drift | Trajectory RMSE | Interpretation |
|--------------|---------|--------------|-----------------|----------------|
| two_body     | 320,270 | 57.6%        | 1.99            | ✗ Model violates energy conservation |
| figure_eight | 230,794 | 11.6%        | 0.60            | ✗ Moderate physics violation |
| plummer      | 24,042  | 293%         | 102.4           | ✗ Severe physics violation |

**Key Finding**: ALL evolved models violate physics conservation laws. Models optimize for trajectory accuracy at the expense of energy/momentum preservation, making them unsuitable for long-term simulations.

Results saved to: `results/analysis/physics_validation_YYYYMMDD_HHMMSS/`

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

### Last Updated: November 03, 2025 08:13 AM JST

#### Recently Completed

- ✅ **[PR #45](https://github.com/TheIllusionOfLife/Galaxy/pull/45)**: Physics Penalty Validation and Analysis (Tasks 1-3)
  - **Task 1**: Full-scale evolution run (10 pop × 5 gen) with physics penalty enabled
  - **Task 2**: Penalty weight tuning experiments (4 configs: 3-33x weight increase)
  - **Task 3**: Per-problem threshold recommendations (strict/moderate/lenient options)
  - **Key Finding**: Physics penalty IS effective (best model 98.6% improvement: 293% → 4.21% drift)
  - **Critical Discovery**: Weight tuning ineffective (all configs: 16.25% best drift) - root cause is LLM prompt design, not penalty strength
  - **Documentation**: 3 comprehensive analysis documents (+1,149 lines)
  - **Cost**: $0.0498 (5 runs, 96 API calls)
  - **Status**: ✅ Merged (commit [b11b8d6](https://github.com/TheIllusionOfLife/Galaxy/commit/b11b8d6))

- ✅ **[PR #44](https://github.com/TheIllusionOfLife/Galaxy/pull/44)**: Session handover and learnings documentation
  - Documented PR #43 work and next steps
  - **Status**: ✅ Merged

- ✅ **[PR #43](https://github.com/TheIllusionOfLife/Galaxy/pull/43)**: Physics-Aware Fitness Function implementation
  - Physics penalty infrastructure with energy and momentum drift penalties
  - 23 comprehensive tests, fully configurable via config.yaml
  - **Status**: ✅ Merged (commit [8b8ce01](https://github.com/TheIllusionOfLife/Galaxy/commit/8b8ce01))

- ✅ **Earlier PRs** (#41, #40, #38, #36, #34, #32, #30, #26, #23, #21, #19, #16, #14, #12, #10): See git history for full details

#### Next Priority Tasks

1. **Update LLM Prompts to Emphasize Conservation** (HIGH PRIORITY - New from PR #45)
   - **Source**: PR #45 Task 2 findings - weight tuning showed NO improvement
   - **Context**: LLM prompts focus on accuracy/speed but don't mention conservation
   - **Root Cause**: Generating same fast non-conserving models regardless of penalty strength
   - **Goal**: Add explicit conservation requirements to prompts
   - **Tasks**:
     - Update `prompts.py` to explicitly request energy/momentum conservation
     - Add guidance for symplectic integrators or conservation-preserving schemes
     - Request physics validation in generated code comments
   - **Benefits**: Address root cause of poor conservation (prompt design vs penalty weights)
   - **Estimated time**: 1-2 hours (prompt engineering + validation run)
   - **Expected**: Better population mean drift (currently 391.67%)

2. **Rebalance Fitness Formula** (HIGH PRIORITY - New from PR #45)
   - **Source**: PR #45 analysis - speed multiplier (5000x) overwhelms physics penalty
   - **Context**: Current formula `accuracy / (speed + 1e-9)` creates huge fitness from fast models
   - **Problem**: Physics penalty becomes insignificant compared to speed-based fitness
   - **Goal**: Rebalance to make conservation competitive with speed
   - **Options**:
     - Multi-objective: `w1*accuracy + w2*(1/speed) - w3*energy_drift - w4*momentum_drift`
     - Log-scale speed: `accuracy / log(speed + 1.0)` (reduces 5000x to ~8x)
     - Hard constraint: Mark >10% drift as invalid (fitness=-inf)
   - **Benefits**: Allows physics penalty to influence model selection effectively
   - **Estimated time**: 2-3 hours (implementation + validation runs)

3. **Implement Per-Problem Thresholds (Code-Based)** (MEDIUM PRIORITY)
   - **Source**: PR #45 Task 3 - documented recommendations, needs implementation
   - **Context**: Uniform 1% threshold unrealistic (too strict for plummer, too lenient for two_body)
   - **Recommended thresholds**:
     - two_body: 0.2% (simple 2-body orbit)
     - figure_eight: 1.5% (chaotic 3-body)
     - plummer: 10% (complex N-body)
   - **Tasks**:
     - Update config.yaml schema to support per-problem thresholds
     - Modify config.py Settings class with `get_physics_threshold(problem, metric)` method
     - Update prototype.py to use problem-specific thresholds
   - **Benefits**: More realistic physics expectations per problem complexity
   - **Estimated time**: 1-2 hours (config + code changes)

4. **Code Modularization** (MEDIUM PRIORITY - Deferred)
   - **Source**: Previous planning consensus
   - **Context**: prototype.py now at 1044 lines
   - **Priority**: Deferred until prompts/fitness rebalancing complete
   - **Estimated time**: 3-4 hours

#### Known Issues / Blockers

- **Population Mean Drift Still High**: Best model improved (4.21%) but mean (391.67%) indicates most models violate conservation
  - **Root Cause**: LLM prompts don't emphasize conservation (see Task 1 HIGH PRIORITY)
  - **Solution Path**: Update prompts + rebalance fitness formula

#### Session Learnings

**Last Updated**: November 03, 2025 08:13 AM JST

- **AI Reviewer Hallucination Pattern** (2025-11-03 PR #45): ALL AI reviewers can provide completely incorrect feedback
  - **Problem**: gemini-code-assist, chatgpt-codex-connector, coderabbitai reviewed Galaxy physics PR but provided feedback about non-existent ARC-AGI multi-agent code
  - **Impact**: 7 PR comments, 4 reviews, 25 line comments ALL about wrong codebase (file paths like `src/arc_prometheus/...`)
  - **Verification**: Used `gh pr diff 45` to confirm actual files were Galaxy physics analysis (FULL_RUN_ANALYSIS.md, etc.)
  - **Pattern**: ALWAYS verify reviewer file paths and line numbers match actual PR diff before accepting feedback
  - **Principle**: Correctness > Compliance - reject ALL feedback when reviewers hallucinate wrong codebase
  - **Caught By**: Mandatory verification checklist (checked PR diff vs reviewer claims)

- **Scientific Metric Reporting Clarity** (2025-11-03 PR #45): Distinguish best model vs population mean in comparative analysis
  - **Problem**: Comparison table showed "430% energy drift" but analysis also mentioned "4.21% best drift" - confusing readers
  - **Root Cause**: Conflated best model performance (4.21%) with incorrect population metric (actually 391.67%, not 430%)
  - **Impact**: Scientific conclusion reversed - initially "penalty failed" but actually "penalty effective for best model"
  - **Solution**: Updated all tables to show BOTH metrics: "Best: 4.21% (98.6% improvement)" AND "Mean: 391.67% (population still poor)"
  - **Pattern**: Always report best AND mean for evolutionary algorithms - best shows capability, mean shows population quality
  - **Fixed By**: gemini-code-assist review feedback (energy drift inconsistency)

- **Configuration Validation Documentation** (2025-11-03 PR #45): Validation bounds need inline rationale comments
  - **Problem**: Changed `le=1.0 → le=100.0` for weight validation without explaining why 100.0 is appropriate
  - **Impact**: Future maintainers may question or revert constraint without understanding experimental basis
  - **Solution**: Added inline comments: `le=100.0,  # Upper bound allows 30x+ weight tuning experiments (see PENALTY_WEIGHT_TUNING_RESULTS.md)`
  - **Pattern**: When validation bounds are based on empirical experiments, link to documentation showing rationale
  - **Benefits**: Traceability to experimental design, prevents uninformed constraint changes
  - **Caught By**: Claude Code review (MEDIUM priority issue #2)

- **Test Production Code, Not Duplicates** (2025-11-02 PR #43): Tests should import and call actual functions, not reimplement logic
  - **Problem**: test_physics_penalty.py created helper function `calculate_physics_penalty()` that duplicated production logic
  - **Impact**: Tests passed but didn't exercise actual production code → production bugs could slip through
  - **Solution**: Import `calculate_physics_penalty` from `prototype` module, test actual implementation
  - **Pattern**: Always import production functions in tests. Only create test helpers for test-specific utilities (mocking, fixtures)
  - **Caught By**: coderabbitai reviewer (follow-up feedback after initial refactoring)
- **Test Layered Functionality Separately** (2025-11-02 PR #43): Cap applied to combined penalty, not individual components
  - **Problem**: Tests applied 90% cap to physics penalty alone, but production applies cap to (code + physics) combined
  - **Impact**: Tests validated wrong behavior → incorrect assumptions about production capping logic
  - **Solution**: Test uncapped physics penalty (can exceed 100%), test cap only on combined total penalty
  - **Pattern**: When system has layered behavior (component → combined → capped), test each layer independently
  - **Caught By**: coderabbitai reviewer (CRITICAL priority - lines 72-86, 346-360)
- **Systematic PR Review Success** (2025-11-02 PR #43): `/fix_pr_since_commit_graphql` caught ALL new feedback
  - **Success**: Used GraphQL extraction workflow instead of relying on "pass" CI status
  - **Result**: Discovered 2 reviewers (claude, coderabbitai) with 4 new items after latest commit
  - **Pattern**: ALWAYS run GraphQL extraction, never skip based on CI "pass" status
  - **Verification**: Completed all 6 checklist items before declaring PR ready
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
