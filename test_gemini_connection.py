"""Quick test script to verify Gemini API connectivity and code generation.

This script makes a single API call to test that:
1. API key is valid
2. Gemini can generate surrogate model code
3. Generated code passes validation
4. Generated code can be executed
5. Cost tracking works correctly
"""

import logging
import sys
from config import settings
from gemini_client import GeminiClient
from prompts import get_initial_prompt
from code_validator import validate_and_compile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Gemini API connection test."""
    print("=" * 70)
    print("Testing Gemini 2.5 Flash Lite Connection")
    print("=" * 70)
    print()

    # Check API key
    if not settings.google_api_key:
        print("❌ ERROR: No API key found in .env file")
        print("Please add GOOGLE_API_KEY to .env")
        return 1

    print(f"✓ API key loaded ({len(settings.google_api_key)} characters)")
    print(f"✓ Model: {settings.llm_model}")
    print(f"✓ Temperature: {settings.temperature}")
    print()

    # Initialize client (disable rate limiting for quick test)
    print("Initializing Gemini client...")
    try:
        client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            enable_rate_limiting=False  # Disable for single test call
        )
        print("✓ Client initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return 1

    # Generate surrogate model
    print("Generating surrogate model...")
    print("-" * 70)

    prompt = get_initial_prompt(seed=0)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Approach: Euler integration with adaptive timestep")
    print()

    print("Calling Gemini API (this may take a few seconds)...")
    try:
        response = client.generate_surrogate_code(prompt)
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return 1

    print()
    print("=" * 70)
    print("API Response")
    print("=" * 70)

    if not response.success:
        print(f"❌ API call failed: {response.error}")
        return 1

    print("✅ API call successful!")
    print()
    print(f"Tokens used:      {response.tokens_used:,}")
    print(f"Cost:             ${response.cost_usd:.6f}")
    print(f"Generation time:  {response.generation_time_s:.2f}s")
    print(f"Model:            {response.model}")
    print()

    print("=" * 70)
    print("Generated Code")
    print("=" * 70)
    print(response.code)
    print("=" * 70)
    print()

    # Validate code
    print("Validating generated code...")
    attractor = [50.0, 50.0]
    compiled_func, validation = validate_and_compile(response.code, attractor)

    if not validation.valid:
        print(f"❌ Validation failed:")
        for error in validation.errors:
            print(f"  - {error}")
        return 1

    print("✅ Code passed validation")

    if validation.warnings:
        print("⚠️  Warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")
    print()

    # Test execution
    if compiled_func is None:
        print("❌ Failed to compile function")
        return 1

    print("Testing execution...")
    test_particle = [45.0, 45.0, 1.0, 1.0]
    print(f"Input particle:  {test_particle}")

    try:
        result = compiled_func(test_particle)
        print(f"Output particle: {result}")
        print()
        print("✅ Execution successful!")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return 1

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("✅ API key valid")
    print("✅ Gemini API accessible")
    print("✅ Code generation successful")
    print("✅ Code validation passed")
    print("✅ Code execution successful")
    print("✅ Cost tracking working")
    print()
    print(f"Total cost: ${response.cost_usd:.6f}")
    print()
    print("🎉 All tests passed! Ready to proceed with full integration.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
