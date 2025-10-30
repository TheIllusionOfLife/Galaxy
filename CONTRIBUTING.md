# Contributing to Galaxy

Thank you for your interest in contributing to Galaxy! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Git
- Google AI API key (free tier available at [Google AI Studio](https://aistudio.google.com/apikey))

### Initial Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Galaxy.git
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
   make install
   # Or manually:
   uv sync --extra dev
   pre-commit install
   ```

   **Why uv?** uv is 10-100x faster than pip, provides reproducible installs via `uv.lock`, and automatically manages virtual environments.

4. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

## Development Workflow

### 1. Create a Feature Branch
Always work on a feature branch, never directly on `main`:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes
Follow these principles:
- **KISS** (Keep It Simple, Stupid)
- **DRY** (Don't Repeat Yourself)
- **YAGNI** (You Aren't Gonna Need It)
- **SOLID** principles

### 3. Run Quality Checks
Before committing, run:
```bash
make check
```

This runs:
- **Ruff formatter**: Ensures consistent code style
- **Ruff linter**: Catches common bugs and code smells
- **Mypy**: Type checks security-critical modules
- **Pytest**: Runs all tests with coverage

### 4. Commit Changes
Use conventional commit format:
```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in validator"
git commit -m "docs: update README"
git commit -m "test: add integration tests"
git commit -m "refactor: simplify code structure"
```

Pre-commit hooks will automatically:
- Format your code with Ruff
- Run linter checks
- Type check critical files

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
gh pr create --fill  # Or use GitHub web interface
```

## Code Quality Standards

### Formatting and Linting
We use **Ruff** for both linting and formatting:
- Line length: 100 characters
- Target: Python 3.10+
- Security checks enabled (flake8-bandit)

```bash
# Format code
make format

# Run linter
make lint
```

### Type Checking
We use **Mypy** with strict checking for security-critical modules:
- `code_validator.py` - Strict type checking
- `gemini_client.py` - Strict type checking
- `config.py` - Strict type checking

```bash
make typecheck
```

### Testing
We use **Pytest** with these requirements:
- Minimum 80% code coverage
- All tests must pass
- Use `pytest-asyncio` for async tests

```bash
# Run all tests
make test

# Run without integration tests (faster)
make test-fast

# Or manually with uv:
uv run pytest tests/ -m "not integration"  # Fast
uv run pytest tests/                        # All tests
uv run pytest tests/ --cov --cov-report=html  # With coverage
```

### Test Markers
- `@pytest.mark.integration` - Integration tests (use real API, require API key)
- `@pytest.mark.slow` - Slow-running tests (>5 seconds)

Skip integration tests during development:
```bash
uv run pytest -m "not integration"
```

## Test-Driven Development (TDD) Workflow

This project follows strict TDD practices for all feature development and bug fixes.

### The TDD Cycle

**1. Write Test First (RED phase)**

Before writing any implementation code, write a test that defines the expected behavior:

```python
# tests/test_new_feature.py
def test_code_length_penalty_applies_above_threshold():
    """Test that penalty reduces fitness when code exceeds threshold."""
    # Arrange
    code = "def predict(p, a):\n" + "    # comment\n" * 500  # ~2500 tokens
    base_fitness = 1000.0

    # Act
    penalized_fitness = apply_code_length_penalty(
        fitness=base_fitness,
        code=code,
        threshold=2000,
        weight=0.1
    )

    # Assert
    assert penalized_fitness < base_fitness  # Penalty applied
    assert penalized_fitness >= base_fitness * 0.1  # Floor enforced
```

**2. Run Test - Verify Failure (RED phase)**

Ensure the test fails because the feature doesn't exist yet:

```bash
uv run pytest tests/test_new_feature.py::test_code_length_penalty_applies_above_threshold -v
# Expected output: FAILED (function not implemented yet)
```

**3. Write Minimal Implementation (GREEN phase)**

Write the simplest code that makes the test pass:

```python
# code_length_penalty.py
def apply_code_length_penalty(fitness, code, threshold, weight):
    """Apply penalty to fitness based on code length."""
    token_count = count_tokens(code)

    if token_count <= threshold:
        return fitness  # No penalty

    excess_tokens = token_count - threshold
    penalty_factor = max(0.1, 1.0 - (weight * (excess_tokens / threshold)))
    return fitness * penalty_factor
```

**4. Run Test - Verify Success (GREEN phase)**

```bash
uv run pytest tests/test_new_feature.py::test_code_length_penalty_applies_above_threshold -v
# Expected output: PASSED ✓
```

**5. Refactor (REFACTOR phase)**

Improve code quality while keeping tests green:

```python
# Refactored version with type hints and documentation
def apply_code_length_penalty(
    fitness: float,
    code: str,
    threshold: int = 2000,
    weight: float = 0.1
) -> float:
    """
    Apply length penalty to fitness score.

    Args:
        fitness: Base fitness score
        code: Generated code to evaluate
        threshold: Token count before penalty applies
        weight: Penalty strength (0.0-1.0)

    Returns:
        Penalized fitness (minimum 10% of original)
    """
    token_count = count_tokens(code)

    if token_count <= threshold:
        return fitness

    excess_ratio = (token_count - threshold) / threshold
    penalty_factor = max(0.1, 1.0 - (weight * excess_ratio))

    return fitness * penalty_factor
```

**6. Commit at Logical Milestones**

Make atomic commits that represent complete work units:

```bash
# Commit 1: Tests
git add tests/test_code_length_penalty.py
git commit -m "test: add code length penalty tests"

# Commit 2: Implementation
git add code_length_penalty.py
git commit -m "feat: implement code length penalty system"

# Commit 3: Integration
git add prototype.py
git commit -m "feat: integrate penalty into evolution engine"
```

### Real-World TDD Example: Code Length Penalty (PR #14)

This feature demonstrates complete TDD workflow:

**Phase 1: Tests First** (TDD RED phase)
- Wrote 12 unit tests covering:
  - Token counting accuracy
  - Penalty calculation (no penalty, partial, floor)
  - Edge cases (empty code, very long code)
  - Configuration integration
- Created 1 integration test for evolution behavior
- All tests initially failing ❌

**Phase 2: Implementation** (TDD GREEN phase)
- Implemented `count_tokens()` function (whitespace-based)
- Implemented `apply_code_length_penalty()` with floor enforcement
- Updated `SurrogateGenome` to track token count
- Modified fitness calculation in evolution engine
- All tests now passing ✓

**Phase 3: Validation** (Real-world testing)
- Baseline run: 60 API calls, $0.02 cost
- Verified token tracking in evolution history
- Confirmed 98.3% LLM success rate
- No performance degradation

**Phase 4: Tuning** (PR #21, #23)
- Comparative testing with different penalty weights
- Integration test to verify parameter effects
- Threshold optimization based on real data
- All changes test-driven

**Key Learnings:**
- Writing tests first caught edge cases early
- Integration tests revealed threshold too high
- Real API validation essential (mocks missed format issues)
- Incremental commits made review easier

### TDD Best Practices

#### Use Shared Test Fixtures

Leverage `conftest.py` for reusable test setup:

```python
# tests/conftest.py automatically provides:
# - Mock API keys (no .env needed in tests)
# - Test environment configuration

def test_my_feature():
    # API key already set by conftest.py fixture
    client = GeminiClient()
    response = client.generate_surrogate_code(prompt)
    assert response.success
```

Create module-specific fixtures:

```python
@pytest.fixture
def mock_genome():
    """Reusable genome fixture for tests."""
    return SurrogateGenome(
        theta=[],
        raw_code="def predict(p, a): return [p[0]+1, p[1]+1, p[2], p[3]]",
        fitness=100.0,
        accuracy=0.95,
        speed=0.001
    )

def test_elite_selection(mock_genome):
    population = [mock_genome for _ in range(10)]
    # ... test logic
```

#### Integration Tests with Real API

Mark tests requiring actual API calls:

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="API key required")
def test_penalty_weight_affects_token_count():
    """Integration test: Verify penalty weight reduces code bloat."""
    # Use small population to minimize cost
    settings_override = Settings(
        population_size=3,
        num_generations=2,
        enable_code_length_penalty=True,
        penalty_weight=0.2,  # Aggressive penalty
        enable_rate_limiting=False  # Speed up test
    )

    # Run mini evolution
    crucible = CosmologyCrucible(num_test_particles=20)
    engine = EvolutionaryEngine(crucible, settings=settings_override)
    # ... run and verify
```

Run integration tests:

```bash
# Run ALL tests (including integration)
uv run pytest tests/

# Run ONLY integration tests
uv run pytest tests/ -m integration

# Skip integration tests (faster, used in CI)
uv run pytest tests/ -m "not integration"
```

#### Test Organization

Follow this structure:

```python
class TestFeatureName:
    """Group related tests together."""

    def test_normal_case(self):
        """Test expected behavior with valid input."""
        # Arrange, Act, Assert

    def test_edge_case_empty(self):
        """Test behavior with empty input."""
        # Arrange, Act, Assert

    def test_edge_case_invalid(self):
        """Test error handling with invalid input."""
        with pytest.raises(ValueError, match="expected error message"):
            # Act that should raise exception
```

#### Parametric Testing

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("elite_ratio,expected_count", [
    (0.0, 1),   # Floor: minimum 1 elite
    (0.2, 2),   # 20% of 10 = 2
    (0.3, 3),   # 30% of 10 = 3
    (0.5, 5),   # 50% of 10 = 5
    (1.0, 10),  # 100% = all models
])
def test_elite_selection_ratios(elite_ratio, expected_count):
    engine = EvolutionaryEngine(crucible, elite_ratio=elite_ratio)
    # ... verify expected_count elites selected
```

### When to Use TDD vs EPCC

**Use TDD when:**
- ✅ Adding new features (code changes)
- ✅ Fixing bugs (add regression test first)
- ✅ Refactoring (tests protect against breakage)
- ✅ Working with algorithms/logic

**Use EPCC (Explore-Plan-Code-Commit) when:**
- 📝 Writing documentation (no tests to write)
- 🔧 Updating configuration files
- 🎨 Changing UI/visualizations (manual review needed)
- 📊 Adding non-functional improvements

### TDD Anti-Patterns to Avoid

❌ **Don't write implementation before tests**
- Leads to tests that just verify what code does (not what it should do)

❌ **Don't skip the RED phase**
- If test passes immediately, you're not testing new behavior

❌ **Don't write tests after implementation**
- Tests become less thorough, miss edge cases

❌ **Don't test implementation details**
- Test behavior, not internal structure

✅ **Do write tests first**
✅ **Do verify tests fail initially**
✅ **Do commit tests separately** (shows TDD process)
✅ **Do test behavior and contracts**

## Project Structure

```
Galaxy/
├── config.py              # Configuration management (Pydantic)
├── gemini_client.py       # Gemini API client (rate limiting, retry logic)
├── code_validator.py      # Security validation (AST + sandbox)
├── prompts.py            # LLM prompt templates
├── prototype.py          # Main evolution coordinator
├── tests/                # Test suite
│   ├── __init__.py
│   └── test_config.py
├── .github/workflows/    # CI/CD pipelines
│   ├── ci.yml           # Main CI pipeline
│   ├── claude.yml       # Claude Code integration
│   └── claude-code-review.yml
├── pyproject.toml       # Project configuration
├── Makefile             # Developer commands
└── .pre-commit-config.yaml  # Pre-commit hooks
```

## CI/CD Pipeline

### GitHub Actions
Our CI pipeline runs on every push and PR:

1. **Test Job** (Python 3.10, 3.11, 3.12):
   - Ruff linter check
   - Ruff formatter check
   - Mypy type checking
   - Pytest with coverage

2. **Security Job**:
   - Ruff security checks (S rules)
   - Code complexity checks (C90)

### Branch Protection
The `main` branch is protected:
- All CI checks must pass
- At least one approval required
- Branch must be up to date

## Making Your First Contribution

### Good First Issues
Look for issues labeled:
- `good first issue` - Suitable for newcomers
- `help wanted` - Community contributions welcome
- `documentation` - Improve docs

### Areas to Contribute
- **Tests**: Add unit tests for uncovered modules
- **Documentation**: Improve README, add examples
- **Features**: Implement items from Session Handover roadmap
- **Bug Fixes**: Address issues from GitHub issues

## Common Tasks

### Running the Evolution
```bash
make run
# Or with uv:
uv run python prototype.py
```

### Cleaning Cache Files
```bash
make clean
```

### Viewing Available Commands
```bash
make help
```

### Running Pre-commit Manually
```bash
pre-commit run --all-files
```

### Quick Quality Check
```bash
# Run all checks before committing
make check

# Or individually:
make format      # Auto-format code
make lint        # Check for issues
make typecheck   # Type check critical files
make test-fast   # Run fast tests
```

## Security Considerations

### Code Validation
All LLM-generated code goes through:
1. **AST analysis** - Detects forbidden operations
2. **Sandbox execution** - Isolated environment
3. **Output validation** - Checks for NaN/Inf/None

### Sensitive Data
Never commit:
- API keys (use `.env` file)
- Credentials
- Personal data

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/TheIllusionOfLife/Galaxy/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See README.md for usage instructions

## Code Review Process

1. Create PR with descriptive title and body
2. Ensure all CI checks pass
3. Wait for code review
4. Address review comments
5. Get approval
6. Merge (squash and merge preferred)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to Galaxy!
