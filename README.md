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
- Google AI API key (free)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/TheIllusionOfLife/Galaxy.git
cd Galaxy
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -e .
```

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
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Run evolutionary optimization
python prototype.py
```

### Execution Results

The program outputs:
- Evaluation results for each generation (fitness, accuracy, speed)
- Top-performing models
- LLM usage statistics (token count, cost, success rate)

### Cost Management

- **Free tier**: 1,000 requests per day, 15 requests per minute
- **Default settings**: 50 API calls per execution
- **Execution cost**: Approximately $0.02/run (2% of budget)
- **Rate limiting**: Automatically maintains 15 RPM

### Testing

Test API connection:
```bash
python test_gemini_connection.py
```

Run unit tests:
```bash
pytest tests/
```

## Session Handover

### Last Updated: October 28, 2025 05:15 AM JST

#### Recently Completed
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

2. **[Visualization]**: Add evolution progress visualization
   - Source: evolution_analysis.md shows interesting fitness progression
   - Context: Visual plots would help understand trade-offs
   - Approach: matplotlib plots for fitness/generation, accuracy/speed scatter

3. **[Code Length Penalty]**: Address token bloat in later generations
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
