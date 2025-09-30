# Plan: Integrate Actual LLM into Prototype

## Phase 1: Project Infrastructure Setup

1. **Create `pyproject.toml`** with dependencies:
   - `anthropic` (Claude API client)
   - `python-dotenv` (environment variable management)
   - `pydantic` (configuration validation)

2. **Create `.env.example`** template for API keys

3. **Create `.gitignore`** to prevent committing secrets

## Phase 2: Configuration & Safety

4. **Create `config.py`** module:
   - Load API keys from environment
   - Define LLM parameters (model, temperature, max_tokens)
   - Add cost tracking constants
   - Add retry/timeout settings

5. **Fix existing bugs in `prototype.py`**:
   - Line 59: Fix return type annotation `(float, float)` → `tuple[float, float]`
   - Lines 17-19: Fix broken mutation logic (replace with regex-based coefficient perturbation)
   - Add named constants for magic numbers
   - Improve error logging

## Phase 3: LLM Integration

6. **Refactor `LLM_propose_surrogate_model`**:
   - Import Anthropic client
   - Design effective prompt for generating surrogate models
   - Parse LLM response to extract lambda code
   - Add validation for generated code safety
   - Implement exponential backoff retry logic
   - Track API costs per generation

7. **Add `llm_client.py`** wrapper module:
   - Centralize API calls
   - Implement rate limiting
   - Add response caching for efficiency
   - Log all requests/responses for observability

## Phase 4: Enhanced Observability

8. **Add structured logging**:
   - Log each generation's models and fitness scores
   - Track API costs in real-time
   - Save evolution history to JSON for analysis
   - Add progress visualization (optional: rich/tqdm)

## Phase 5: Testing & Validation

9. **Create `test_integration.py`**:
   - Test LLM API connectivity
   - Validate generated code is safe to eval
   - Test with mock API responses
   - Verify cost tracking accuracy

10. **Update documentation**:
    - Add setup instructions to README
    - Document API key configuration
    - Add example runs with expected costs

## Expected Outcomes

- Prototype uses real Claude API to generate surrogate models
- Safe code execution with restricted eval namespace
- Cost tracking and rate limiting prevent runaway expenses
- Full observability of evolution process for research analysis
- Maintains backward compatibility (can still run mock mode without API key)

## Notes on Current Implementation

The user has already refactored `prototype.py` with:
- `SurrogateGenome` dataclass for parameterized models
- `compile_external_surrogate()` for safe LLM-generated code execution
- Bounded parameter mutation with `PARAMETER_BOUNDS`
- Improved error handling and validation

This new architecture is well-suited for LLM integration. The `raw_code` field in `SurrogateGenome` can store LLM-generated code, while the parametric approach serves as a baseline.
