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
GOOGLE_API_KEY=your_api_key_here
```

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

4. **cost_progression.png** - Cumulative cost over API calls
   - Tracks spending throughout evolution
   - Helps validate cost estimates

Example output:
```
Saving results to: results/run_20251028_113940
  ✓ Evolution history saved: results/run_20251028_113940/evolution_history.json
  ✓ Fitness progression plot: results/run_20251028_113940/fitness_progression.png
  ✓ Accuracy vs speed plot: results/run_20251028_113940/accuracy_vs_speed.png
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

### Last Updated: October 29, 2025 01:47 AM JST

#### Recently Completed
- ✅ [PR #10 - Code Cleanup & uv Adoption]: Merged modern tooling and test organization
  - **uv Package Manager**: Official adoption with 10-100x faster dependency installation
  - **Reproducible Builds**: Committed `uv.lock` for consistent environments
  - **Test Organization**: Moved integration test to proper location with pytest markers
  - **Documentation**: Updated README, CONTRIBUTING, Makefile, CI workflow
  - **PR Review**: Addressed 4 reviewers (1 critical, 2 medium, 1 high-priority issues)
  - **CI Performance**: All Python versions (3.10, 3.11, 3.12) passing in 36-44s
  - **Real API Testing**: Integration test validated with Gemini (33-38s runtime)
- ✅ [Previous Session - PR #8]: Evolution Visualization and Data Export system
- ✅ [Previous Session - PR #7]: ARCHITECTURE.md and production run validation
- ✅ [Previous Session - PR #5]: CI/CD infrastructure with Ruff, Mypy, Pytest

#### Next Priority Tasks
1. **[Prompt Engineering]**: Reduce syntax error rate from 3.3%
   - Source: Generation 2 and 4 had syntax errors (incomplete code blocks)
   - Context: Simple prompt improvement could reduce failures
   - Approach: Add "Ensure code is complete and syntactically valid" to system instruction

2. **[Code Length Penalty]**: Address token bloat in later generations
   - Source: Generation 4 produced 3,576 token functions (vs 726 in Gen 0)
   - Context: Fitness function doesn't penalize code complexity
   - Approach: Add token count to fitness calculation

#### Known Issues / Blockers
- None currently - system validated as production-ready
- **Token Growth**: Later generations produce increasingly complex code (avg 2,271 tokens in Gen 3-4 vs 1,038 in Gen 0)
  - Acceptable for now, but may need fitness penalty for code length
- **Non-monotonic Fitness**: Fitness fluctuates between generations (not guaranteed to improve)
  - Expected behavior during exploration phase, not a bug

#### Session Learnings
- **uv Migration Success**: Switching from pip to uv requires `--extra dev` flag for optional dependencies
- **Make Target Purpose**: `make check` should exclude slow integration tests for quick local validation
- **Test Robustness**: Conditional assertions (e.g., cost_progression.png only if API succeeded) prevent flaky tests
- **Pre-commit Setup**: Tool invocations need `uv run` wrapper when using uv-managed environments
- **GraphQL PR Review**: Single query fetches all feedback sources (comments, reviews, line comments, CI annotations)
- **Reviewer Priority**: Read review CONTENT not just STATE; even APPROVED reviews can have suggestions
- **Field Name Consistency**: Integration tests must match actual data structure field names (e.g., `civ_id` not `civilization_id`)
