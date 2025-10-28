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
uv sync

# Alternative: with pip
pip install -e .
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

### Last Updated: October 28, 2025 03:04 PM JST

#### Recently Completed
- ✅ [PR #8 - Evolution Visualization]: Merged comprehensive visualization and data export system
  - **History Tracking**: Records generation data (fitness, accuracy, speed per model)
  - **JSON Export**: Saves complete evolution history with summary statistics
  - **Fitness Progression Plot**: Line plot showing best/avg/worst fitness over generations
  - **Accuracy vs Speed Plot**: Scatter plot revealing speed/accuracy trade-offs
  - **Cost Progression Plot**: Tracks cumulative API spending
  - **Auto-save**: Timestamped results directories (`results/run_YYYYMMDD_HHMMSS/`)
  - **Tests**: 12 history tracking tests + 17 visualization tests (all passing)
  - **Real API Test**: Verified with 2-gen, 3-pop evolution (all outputs correct)
- ✅ [Production Run]: First successful end-to-end LLM evolution cycle
  - **Validation Success**: 58/60 code samples validated (96.7% success rate)
  - **Cost**: $0.0223 (within estimated $0.02, 2.2% of budget)
  - **Runtime**: 4.0 minutes (vs 3.3 min estimated)
  - **Best Fitness**: 20,846.97 (Generation 3, civ_3_8)
  - **Fitness Improvement**: 2x over baseline (11,114 → 20,846)
  - **Speed Optimization**: 2.25x faster (0.00009s → 0.00004s)
  - **Rate Limiting**: Working correctly (15 RPM enforced, no API errors)
  - **Robustness**: System handled 3.3% validation failures gracefully
- ✅ [Documentation]: ARCHITECTURE.md created with comprehensive system design
  - Documented all core components (Config, Gemini Client, Validator, Prompts, Evolution Engine)
  - Added complete data flow diagrams (text-based ASCII)
  - Documented 3-layer security model (AST → Sandbox → Output validation)
  - Included extension points for future work (new LLMs, fitness functions, problem domains)
  - Production-ready reference for contributors
- ✅ [Analysis]: Detailed evolution analysis document created
  - Token usage trends (726-3,576 tokens, increasing in later generations)
  - Code quality patterns (Euler dominates, adaptive timestep common)
  - Failure analysis (2 syntax errors, both in later generations with complex code)
  - LLM vs mock comparison (LLM 2x better than parametric baseline)
- ✅ [Integration Tests]: Comprehensive integration test suite added (tests/test_integration.py)
  - 11 integration tests covering real Gemini API interactions
  - Tests for code generation, mutation, rate limiting, cost tracking, budget enforcement
  - Tests for error recovery (invalid API key, validation failures)
  - Full evolution cycle test (mini evolution with 2 generations, 3 population)
  - All marked with @pytest.mark.integration for selective execution
- ✅ [Previous Sessions]: PR #5 (CI/CD infrastructure) and PR #2 (Gemini API integration)

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
- **Production Validation**: System works end-to-end with real LLM - infrastructure is solid
- **LLM Code Quality**: Gemini generates physically-correct gravity simulations 96.7% of the time
- **Temperature Effects**: Exploration (1.0) produces creative but riskier code; exploitation (0.6) refines successfully
- **Fitness Trade-offs**: Models that optimize speed too aggressively sacrifice accuracy (87-89% vs 99%)
- **Common Patterns**: LLM independently discovers adaptive timestep, softening parameters, Euler integration
- **Failure Modes**: Syntax errors occur in later generations with complex code (incomplete blocks, missing parens)
- **Cost Accuracy**: Estimated $0.02, actual $0.0223 (+11.5%) - close enough for planning
- **Architecture Documentation**: Comprehensive ARCHITECTURE.md essential for contributor onboarding
- **PR Review Process**: Systematic /fix_pr_graphql workflow caught 10 issues (4 critical) across 4 reviewers
- **Python 3.10 Compatibility**: isinstance() union syntax (`list | tuple`) incompatible with 3.10 - use tuple form
- **JSON Standards**: NaN/Inf values produce non-standard JSON; sanitize recursively with null replacement
- **Package Declaration**: visualization.py module required explicit py-modules declaration for installation
- **Defensive Programming**: Add safety checks even when data is pre-filtered (division by zero, finiteness)
