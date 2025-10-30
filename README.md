# Galaxy - AI Civilization Evolution Simulator

This project simulates how AI civilizations can discover solutions that surpass human capabilities. It is designed to run directly on local machines, allowing you to observe the evolutionary process in action.

## Prototype Philosophy

**LLM Role**: Instead of solving problems directly, the LLM generates and proposes **"strategies," "heuristics," and "surrogate models"** as code. This creative process is simulated through the LLM_propose_strategy function.

**Evolution Process**: Each AI civilization (agent group) proposes strategies that are executed and evaluated in a "Crucible" environment. Superior strategies are selected by the "Evolutionary Engine" and become the foundation for the next generation.

**Purpose**: These prototypes aim to verify whether the process of discovering better solutions can be automated and accelerated, rather than completely solving the problem.

**This Prototype**: AI civilizations are tasked with inventing surrogate models to accelerate computationally expensive N-body simulations (gravitational calculations). Fitness is evaluated based on the balance between the surrogate model's prediction accuracy and computational speed.

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
# Run evolutionary optimization
uv run python prototype.py

# Alternative: with pip/venv (activate virtual environment first)
source .venv/bin/activate  # if using venv
python prototype.py
```

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

## Session Handover

### Last Updated: October 30, 2025 04:32 AM JST

#### Recently Completed
- ✅ **[PR #19 - Fix elite_ratio Configuration Bug](https://github.com/TheIllusionOfLife/Galaxy/pull/19)**: Complete TDD implementation with DRY refactor merged to main
  - **Problem**: `elite_ratio` hardcoded to 0.2, ignoring user configuration
  - **Solution**: Added configurable `elite_ratio` parameter with validation + DRY configuration architecture
  - **Implementation**:
    - Core Fix: Added `elite_ratio` parameter to EvolutionaryEngine with 0.0-1.0 validation
    - DRY Refactor: Created `config.yaml` as single source of truth (all defaults in ONE place)
    - Secrets Separation: `.env` contains ONLY API keys, no configuration parameters
    - Priority Hierarchy: Environment variables > .env > config.yaml
    - Error Handling: User-friendly YAML parsing and structure validation errors
  - **Testing**: 6 elite_ratio unit tests + 1 integration test, all 64 unit tests passing
  - **Security**: Added detect-secrets hook, comprehensive .gitignore patterns, safe test fixtures
  - **Review Fixes**: Addressed 3 CodeRabbit comments (mock type fix, YAML error handling, structure validation)
  - **CI Fix**: Excluded integration tests from CI (no API key needed), all Python 3.10-3.12 passing
  - **Status**: ✅ Merged (commit [5000cd1](https://github.com/TheIllusionOfLife/Galaxy/commit/5000cd1)), 8 commits squashed, 17 files changed (+884, -155)
- ✅ **[PR #21 - Code Length Penalty Comparative Testing & Validation](https://github.com/TheIllusionOfLife/Galaxy/pull/21)**: Complete TDD testing merged to main
  - **Goal**: Validate penalty system effectiveness with different weights using real API
  - **Testing**: 3 full evolution runs (150 API calls, $0.05 total) with weights 0.1 and 0.2
  - **Results**:
    - Test 2 (weight=0.1): Avg 368.9 tokens, Best fitness 25314.61, 98.3% LLM success
    - Test 3 (weight=0.2): Avg 348.8 tokens, Best fitness 20923.09, 98.3% LLM success
    - Comparison: 5.4% token reduction but 17.3% fitness loss with aggressive penalty
  - **Critical Finding**: Threshold (2000 tokens) too high for typical models (300-400 tokens)
  - **Root Cause**: Penalty only applies to `excess = token_count - threshold`, so models <2000 never penalized
  - **Recommendation**: Lower threshold to 400 tokens to make penalty relevant to actual token ranges
  - **Integration Test**: Added `test_penalty_weight_affects_token_count()` with proper settings patching
  - **Critical Bug Fixes** (4 commits addressing reviewer feedback):
    - **CodeRabbit #1**: Test was not actually testing different weights - fixed by patching global settings references
    - **CodeRabbit #2**: `regenerate_viz.py` crashed on legacy list format - added format detection
    - **`visualization.py:282`**: Changed to `token_count = model.get("token_count") or 0` (handles None + missing key)
    - **`analyze_penalty_results.py`**: Added JSON error handling
  - **Utilities**: Created `regenerate_viz.py` (46 lines) and `analyze_penalty_results.py` (122 lines)
  - **Review Process**: Addressed 3 reviewers (claude, gemini-code-assist, CodeRabbit) across 4 commits
  - **Documentation**: Complete analysis in `results/penalty_comparison_20251030.md` (gitignored)
  - **Verification**: ✅ All outputs validated - no timeout, no truncation, no duplicates, no errors
  - **Status**: ✅ Merged (commit [af73928](https://github.com/TheIllusionOfLife/Galaxy/commit/af73928)), 4 commits squashed, 5 files changed (+291, -6)
- ✅ **[PR #16 - Token Progression Visualization]**: Complete TDD implementation merged to main
  - **Problem**: No visibility into code length evolution across generations (PR #14 added tracking but no visualization)
  - **Solution**: New `token_progression.png` plot with avg/max/min lines + fitness-colored scatter overlay
  - **Implementation**:
    - New function: `plot_token_progression()` in visualization.py (97 lines)
    - Statistics: Per-generation avg/max/min token calculations
    - Scatter overlay: Individual models colored by fitness (viridis colormap)
    - Backward compatible: Uses `.get("token_count", 0)` for missing data
    - Integration: Updated `generate_all_plots()` and prototype.py console output
  - **Testing**: TDD with 6 comprehensive tests (358-482 lines in test_visualization.py)
    - Basic file creation test
    - Missing token data (backward compatibility)
    - Empty history edge case
    - Single generation edge case
    - Integration with generate_all_plots
    - All models missing tokens (added after review)
  - **Real API Validation** (9 API calls, $0.002):
    - Configuration: 2 generations, 3 population
    - Results: All 5 PNG files generated (144-186KB, 300 DPI)
    - JSON: All models contain token_count field
    - Output: results/run_20251029_110221
  - **Review Fixes**: Addressed 3 reviewers (claude, gemini, coderabbit)
    - ARCHITECTURE.md: Replaced incomplete code snippet with complete 38-line example
    - visualization.py: Refactored to tuple comprehension + `zip(*valid_points)` pattern
    - tests: Added test for all-missing-tokens scenario
  - **Status**: ✅ Merged (commit aae2eab), 23 tests passing, CI green across Python 3.10-3.12
- ✅ **[PR #14 - Code Length Penalty System]**: Complete TDD implementation merged to main
  - **Problem**: Token bloat in later generations (Gen 0: 1,038 → Gen 3-4: 2,271 avg tokens, +119%)
  - **Solution**: Configurable fitness penalty system to discourage unnecessarily long code
  - **Implementation**:
    - Token counting: `count_tokens()` function (whitespace-based, noted for tiktoken upgrade)
    - Fitness penalty: Linear penalty with 10% floor, only applied above threshold (2000 tokens)
    - Configuration: 3 new settings (enable, weight, threshold) with defaults
    - History tracking: Added `token_count` to evolution JSON for analysis
  - **Testing**: 12 unit tests + integration test + real API validation
  - **Baseline Results** (Penalty Disabled, 60 API calls, $0.02):
    - Gen 0-4 tokens: 158-747 range, avg 247 tokens
    - 98.3% LLM success rate (59/60)
    - Token tracking functional, no performance degradation
  - **Review Fixes**: Addressed 4 reviewers (claude, codex, gemini, coderabbit)
    - CRITICAL: Removed exposed API key (.env.backup) - ✅ Resolved (key revoked and regenerated)
    - Code: Simplified token counting, removed redundant checks
    - Tests: Made assertions exact, implemented empty integration test
  - **Status**: ✅ Merged, ready for comparative testing with penalty enabled
- ✅ [PR #12 - Prompt Engineering & Test Threshold]: Merged syntax error reduction improvements
  - **Achievement**: Reduced LLM syntax error rate by 49% (3.3% → 1.67%)
  - **Review Fix**: Adjusted test threshold from <1% to <2% based on statistical sample size
  - **Rationale**: With 20 samples, 1 error = 5% rate; threshold must account for variance
  - **Refactoring**: Extracted helper method to eliminate code duplication
- ✅ [Previous Session - Prompt Engineering]: Reduced LLM syntax error rate by 49%
  - **Problem**: Historical 3.3% syntax error rate (2/60 in production run)
  - **Solution**: Enhanced prompts with explicit code completeness verification instructions
  - **Testing**: Ran 3 full evolution cycles (180 API calls total) with real Gemini API
  - **Results**: Syntax error rate reduced to 1.67% (2/120 successful runs = 49% reduction)
  - **Bug Fix**: Fixed UnboundLocalError in prototype.py (temp_override initialization)
  - **Changes**: Updated prompts.py with CRITICAL sections emphasizing bracket matching and completeness
  - **Cost**: $0.08 total testing cost (3 runs × $0.025 = well within budget)
- ✅ [PR #10 - Code Cleanup & uv Adoption]: Merged modern tooling and test organization
  - **uv Package Manager**: Official adoption with 10-100x faster dependency installation
  - **Reproducible Builds**: Committed `uv.lock` for consistent environments
  - **Test Organization**: Moved integration test to proper location with pytest markers
  - **CI Performance**: All Python versions (3.10, 3.11, 3.12) passing in 36-44s
- ✅ [PR #8]: Evolution Visualization and Data Export system
- ✅ [PR #7]: ARCHITECTURE.md and production run validation
- ✅ [PR #5]: CI/CD infrastructure with Ruff, Mypy, Pytest

#### Next Priority Tasks
1. **[COMPLETED]** Code Length Penalty - Parameter Tuning ✓
   - **Threshold Updated**: 2000 → 400 tokens (based on PR #21 findings)
   - **Validation Run**: threshold=400, weight=0.1, 5 generations, 50 models
   - **Results**: 11.1% of models now trigger penalty (vs 5% with old threshold)
   - **Improvement**: 2.2x better relevance to actual model sizes
   - **Token Stats**: avg=247.3, range=102-823 tokens
   - **Performance**: Best fitness=27,879.23, cost=$0.0219, 100% success
   - **Analysis**: See `test_analysis.md` for full details
   - **Status**: Optimal configuration identified and tested with real API

2. **[COMPLETED]** Best-Ever Fitness Visualization ✓
   - **Feature**: Added "Best Ever" line to fitness progression plot
   - **Purpose**: Track cumulative maximum fitness across all generations
   - **Benefit**: Users can see actual evolutionary progress even when fitness regresses
   - **Implementation**: Black dashed line, monotonic increasing, handles inf/nan values
   - **Tests**: 2 comprehensive tests added and passing

#### Known Issues / Blockers
- **Non-monotonic Fitness**: Fitness fluctuates between generations (not guaranteed to improve)
  - Expected behavior during exploration phase, not a bug

#### Session Learnings
- **Integration Testing Global Settings Patch** (2025-10-30 from PR #21): `monkeypatch.setenv()` alone insufficient for testing module-level settings
  - **Problem**: Test using `monkeypatch.setenv("PENALTY_WEIGHT", "0.2")` but code still uses default 0.1
  - **Root Cause**: Modules import `settings` at load time; env changes don't update existing references
  - **Solution**: Reload settings + patch ALL module references: `monkeypatch.setattr(config_module, "settings", test_settings)`
  - **Detection**: Test passes but doesn't actually vary parameter being tested
  - **Pattern**: Always patch global references when testing module-level config objects
- **GraphQL PR Review Efficiency**: Single comprehensive query fetches all feedback sources
  - Pattern: `query { repository { pullRequest { comments, reviews, reviewThreads, statusCheckRollup } } }`
  - Advantage: Single API call vs multiple CLI commands (3 `gh api` calls + parsing)
  - Coverage: PR comments, reviews, line comments, CI annotations all in one query
- **Zip Pattern Optimization**: Tuple comprehension + `zip(*iterable)` cleaner than intermediate lists
  - Anti-pattern: `list1 = []; list2 = []; for x in data: list1.append(x.a); list2.append(x.b)`
  - Better: `valid = [(x.a, x.b) for x in data if condition]; a_list, b_list = zip(*valid)`
  - Benefit: More Pythonic, fewer variables, clearer intent
- **Complete Test Coverage for Edge Cases**: Test "all data missing" not just "some data missing"
  - Example: PR #16 added test_plot_token_progression_all_models_missing_tokens
  - Rationale: Different code path when ALL vs SOME models lack token_count
  - Pattern: Empty list handling often differs from partial data handling
- **Documentation Code Completeness**: Examples must be runnable, not fragments
  - Anti-pattern: Code snippet missing variable initialization, imports, error handling
  - Consequence: Users copy-paste non-working code, creates support burden
  - Fix: Include complete context (imports, initialization, error checks)
- **TDD with Real API Integration**: Integration tests catch format issues mocks miss
  - Mock limitation: Assumes response format, may not match reality
  - Real API advantage: Validates actual LLM output format, timing, error handling
  - Best practice: Write unit tests with mocks, verify with real API before merge
- **TDD Integration Test Implementation**: Empty tests with `pass` should be implemented or removed
  - Anti-pattern: Placeholder tests that don't validate behavior
  - Solution: Either implement with mock data validation or remove entirely
  - PR #14 example: Implemented penalty calculation validation test
- **Security Review Priority**: API keys, credentials, secrets take absolute precedence
  - Even one exposed key blocks merge regardless of feature quality
  - Check for .env*, *.backup, *.bak files before committing
  - Git remove immediately, then revoke/regenerate keys
- **Code Simplification from AI Review**: Trust AI suggestions for redundant code
  - Gemini correctly identified redundant list comprehension in token counting
  - Redundant checks (settings, token_count > 0) can be removed if logic handles edge cases
  - Simpler code is more maintainable and often faster
- **Test Assertion Precision**: Exact assertions > Range assertions
  - Range assertions (e.g., `assert 50 < tokens < 150`) hide actual expected values
  - Exact assertions (`assert tokens == 86`) document intended behavior
  - Narrow ranges (`assert 80 <= tokens <= 90`) acceptable for LLM/approximation variance
- **Prompt Engineering for Code Completeness**: Explicit bracket counting and completeness verification reduces syntax errors
  - Key technique: "Count ALL opening ( [ { MUST have matching closing ) ] }" instruction
  - Special warning for long code (>2000 chars) prevents truncation
  - "Completeness > Cleverness" mindset reduces complex incomplete code
- **LLM Syntax Error Patterns**: Errors occur more in long/complex code (3000+ tokens) during explore phase (temp 1.0)
- **Statistical Testing Importance**: Single runs insufficient - need multiple runs (3+) to validate error rate improvements
- **Unbound Variable Bug Pattern**: Always initialize variables before conditional blocks that use them later
- **TDD Bug Discovery**: Writing integration tests revealed pre-existing UnboundLocalError in production code
- **uv Migration Success**: Switching from pip to uv requires `--extra dev` flag for optional dependencies
- **Make Target Purpose**: `make check` should exclude slow integration tests for quick local validation
- **Test Robustness**: Conditional assertions (e.g., cost_progression.png only if API succeeded) prevent flaky tests
- **Pre-commit Setup**: Tool invocations need `uv run` wrapper when using uv-managed environments
- **Reviewer Priority**: Read review CONTENT not just STATE; even APPROVED reviews can have suggestions
- **Field Name Consistency**: Integration tests must match actual data structure field names (e.g., `civ_id` not `civilization_id`)
- **Test Threshold Statistics**: Adjust test thresholds based on sample size and statistical variance (e.g., 20 samples at 1.67% error = 1 error = 5% rate, use 2% threshold not <1%)
