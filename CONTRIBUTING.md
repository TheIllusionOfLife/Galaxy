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
