"""Shared pytest fixtures and configuration for all tests.

This module provides automatically-applied fixtures that create a safe test
environment with mock API keys and sensible defaults, preventing accidental
exposure of real credentials during testing.
"""

import pytest


@pytest.fixture(autouse=True)
def test_environment(request, monkeypatch):
    """Automatically set safe test environment for all tests.

    This fixture runs automatically for every test, setting safe default
    environment variables that prevent:
    - Accidental use of real API keys
    - Calls to production services during unit tests
    - Exposure of credentials in test logs

    Skips test_config.py to allow those tests to verify default settings.

    Individual tests can override specific values using monkeypatch.setenv().

    Example:
        def test_with_custom_elite_ratio(monkeypatch):
            # Override just elite_ratio for this test
            monkeypatch.setenv("ELITE_RATIO", "0.3")
            # Test runs with ELITE_RATIO=0.3, all other values from this fixture
    """
    # Skip for config tests that need to verify default values
    if "test_config" in request.node.nodeid:
        return

    test_env = {
        "GOOGLE_API_KEY": "test-safe-api-key",  # pragma: allowlist secret
        "ANTHROPIC_API_KEY": "test-safe-anthropic-key",  # pragma: allowlist secret
        "TEMPERATURE": "0.8",
        "POPULATION_SIZE": "5",
        "NUM_GENERATIONS": "2",
        "ELITE_RATIO": "0.2",
        "ENABLE_RATE_LIMITING": "false",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
