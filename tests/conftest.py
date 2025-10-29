"""Shared pytest fixtures and configuration for all tests.

This module provides automatically-applied fixtures that create a safe test
environment with mock API keys, preventing accidental exposure of real
credentials during testing.

Configuration values (model, hyperparameters, feature flags) come from
config.yaml (single source of truth), NOT from environment variables.

NOTE: API keys are set at module level (before config.py import) to avoid
      validation errors when running tests with placeholder .env file.
"""

import os

import pytest

# Set test API keys BEFORE any imports to prevent validation errors
# This runs at module import time, before conftest fixtures or test collection
os.environ["GOOGLE_API_KEY"] = "test-safe-api-key"  # pragma: allowlist secret
os.environ["ANTHROPIC_API_KEY"] = "test-safe-anthropic-key"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def test_environment(request, monkeypatch):
    """Automatically set safe test environment for all tests.

    This fixture runs automatically for every test, ensuring mock API keys
    are available in case they were cleared.

    ONLY overrides secrets (API keys). Configuration values come from
    config.yaml to maintain "Once and Only Once" principle.

    Skips test_config.py to allow those tests to verify config loading.

    Individual tests can override specific values using monkeypatch.setenv().

    Example:
        def test_with_custom_api_key(monkeypatch):
            # Override just API key for this test
            monkeypatch.setenv("GOOGLE_API_KEY", "test-custom-key")
            # Test runs with custom key, all config from config.yaml
    """
    # Skip for config tests that need to verify config loading behavior
    if "test_config" in request.node.nodeid:
        return

    # ONLY set secrets (API keys), NOT configuration parameters
    test_env = {
        "GOOGLE_API_KEY": "test-safe-api-key",  # pragma: allowlist secret
        "ANTHROPIC_API_KEY": "test-safe-anthropic-key",  # pragma: allowlist secret
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
