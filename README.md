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

### Last Updated: October 27, 2025 10:04 PM JST

#### Recently Completed
- ✅ [PR #5]: CI/CD infrastructure with automated testing and code quality pipeline
  - Implemented GitHub Actions CI (Python 3.10, 3.11, 3.12 matrix testing)
  - Added Ruff (linting/formatting), Mypy (type checking), Pytest (testing)
  - Created pre-commit hooks, Makefile, and CONTRIBUTING.md
  - Fixed all type errors in critical modules (code_validator, gemini_client, config)
  - Translated all Japanese comments to English per PR review feedback
  - All CI checks passing across all Python versions
- ✅ [PR #2]: Gemini 2.5 Flash Lite API integration with comprehensive security validation
  - Implemented multi-layer code validation (AST-based + runtime sandbox)
  - Added rate limiting (15 RPM), cost tracking, and budget enforcement
  - Fixed TOCTOU vulnerability, rate limiter retry logic, SAFE_BUILTINS security issue

#### Next Priority Tasks
1. **[LLM Code Evolution]**: Run full evolutionary cycle with real Gemini API
   - Source: PR #5 merged, ready for production use with CI/CD
   - Context: Core infrastructure complete with automated testing
   - Approach: Execute `python prototype.py` and monitor LLM-generated surrogate models

2. **[Documentation]**: Add ARCHITECTURE.md explaining system design
   - Source: Previous session recommendation
   - Context: Code is production-ready but lacks architecture overview
   - Approach: Document: LLM → Validator → Sandbox → Evolution Engine flow

3. **[Test Coverage]**: Expand test coverage with more integration tests
   - Source: CI/CD infrastructure now in place
   - Context: Basic tests exist, but could expand coverage
   - Approach: Add more test scenarios, edge cases, error conditions

#### Known Issues / Blockers
- None currently - all critical issues resolved
- **claude-review workflow**: Expected failure due to GitHub Actions workflow validation requiring identical content on default branch. This is non-blocking and normal for newly added workflow files. The workflow functions correctly once merged to main.

#### Session Learnings
- **CI/CD Infrastructure**: Comprehensive setup with Ruff + Mypy + Pytest provides strong foundation
- **Type Safety**: Strict Mypy on critical modules (code_validator, gemini_client, config) catches errors early
- **PR Review Systematic**: `/fix_pr_graphql` command ensures comprehensive feedback coverage with mandatory verification checklist (feedback count, timestamps, author comments, review content, CI status)
- **Japanese→English Translation**: All comments translated improves international collaboration
- **Modern Python Syntax**: Updated isinstance to use union syntax (X | Y) for Python 3.10+
