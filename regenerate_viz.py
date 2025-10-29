#!/usr/bin/env python3
"""Regenerate visualizations for a results directory."""

import json
import sys
from pathlib import Path
from typing import Any

from visualization import generate_all_plots

if len(sys.argv) < 2:
    print("Usage: python regenerate_viz.py <results_directory>")
    sys.exit(1)

results_dir = Path(sys.argv[1])
json_path = results_dir / "evolution_history.json"

if not json_path.exists():
    print(f"Error: {json_path} not found")
    sys.exit(1)

try:
    with open(json_path) as f:
        data: Any = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in {json_path}: {e}")
    sys.exit(1)

# Handle both legacy list format and current dict format
if isinstance(data, dict) and "history" in data:
    history: list[dict[str, Any]] = data["history"]
else:
    history = data  # Legacy list format


# Mock cost tracker
class MockTracker:
    """Mock cost tracker for regenerating visualizations without cost data."""

    def __init__(self) -> None:
        self.calls: list[Any] = []


cost_tracker = MockTracker()
generate_all_plots(history, cost_tracker, str(results_dir))
print(f"✓ Visualizations generated in {results_dir}")
